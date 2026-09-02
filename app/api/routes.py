"""Validated, resource-bounded API routes for image generation."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import uuid
from io import BytesIO
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import settings
from app.core.dependencies import model_service
from app.core.rate_limit import SlidingWindowRateLimiter
from app.services.editing.edit_pipeline import (
    TIER_PROFILES,
)
from app.services.editing.prompt_intent import PromptIntent, analyze_prompt
from app.services.editing.semantic_planner import (
    plan_semantic_layout,
)
from app.services.model_service import (
    MAX_SEED,
    PIXART_ALPHA,
    STABLE_DIFFUSION,
    GenerationCapacityError,
    GenerationError,
    GenerationResult,
    ModelBusyError,
    ModelLoadError,
    ModelUnavailableError,
    OutputSaveError,
    resolve_generation_defaults,
)
from app.services.prompt_enhancer import enhance_prompt

logger = logging.getLogger(__name__)
router = APIRouter()
generation_rate_limiter = SlidingWindowRateLimiter(settings.RATE_LIMIT_PER_MINUTE)
_SOURCE_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
_MASK_IMAGE_FORMATS = frozenset({"PNG"})


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_PROMPT_LENGTH,
        description="Text prompt for image generation",
    )
    negative_prompt: str | None = Field(
        None,
        max_length=settings.MAX_NEGATIVE_PROMPT_LENGTH,
        description="Content to avoid",
    )
    model: Literal[
        "stable-diffusion",
        "pixart-alpha",
        "stable-diffusion-3.5",
        "sd35",
        "sd35_large",
        "flux-dev",
        "flux",
    ] = Field(
        STABLE_DIFFUSION,
        description="Generation backend",
    )
    width: int | None = Field(None, ge=256, le=2048, multiple_of=8)
    height: int | None = Field(None, ge=256, le=2048, multiple_of=8)
    num_inference_steps: int | None = Field(None, ge=1, le=100)
    guidance_scale: float | None = Field(None, ge=1.0, le=20.0)
    num_images: int = Field(1, ge=1, le=4)
    seed: int | None = Field(None, ge=0, le=MAX_SEED)
    enhance_prompt: bool = Field(
        False,
        description="Append randomized style, lighting, and quality modifiers",
    )
    tier: Literal["preview", "final", "ultra"] | None = Field(
        None,
        description="Quality tier (preview=14 steps, final=28 steps, ultra=36 steps+refiner)",
    )
    quality_tier: Literal["preview", "final", "ultra"] | None = Field(
        None,
        description="Alias for tier",
    )
    cfg_rescale: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="CFG rescaling factor phi to prevent contrast blowout",
    )
    guidance_rescale: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Alias for cfg_rescale",
    )
    refiner_enabled: bool | None = Field(
        None,
        description="Optional texture and micro-detail refinement pass",
    )
    refiner_strength: float | None = Field(
        None,
        ge=0.05,
        le=0.60,
        description="Denoise strength for refiner pass",
    )
    layout_override: list[dict[str, Any]] | None = Field(
        None,
        description="Optional custom entity layout bounding boxes/Gaussians",
    )
    plan: dict[str, Any] | None = Field(
        None,
        description="Optional pre-computed semantic layout plan from /api/v1/layout/plan",
    )
    guidance_mode: Literal["gaussian", "box"] | None = Field(
        None,
        description="Spatial guidance prior mode: gaussian or box",
    )
    adaptive_guidance: bool | None = Field(
        None,
        description="Enable adaptive guidance strength",
    )

    @field_validator("prompt", mode="before")
    @classmethod
    def normalize_prompt(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("negative_prompt", mode="before")
    @classmethod
    def normalize_negative_prompt(cls, value):
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def resolve_and_validate_model_defaults(self):
        active_tier = self.tier or self.quality_tier
        if active_tier and active_tier in TIER_PROFILES:
            profile = TIER_PROFILES[active_tier]
            if self.num_inference_steps is None:
                self.num_inference_steps = profile.default_steps
            if self.guidance_scale is None:
                self.guidance_scale = profile.default_guidance_scale
            if self.cfg_rescale is None and self.guidance_rescale is None:
                self.cfg_rescale = profile.cfg_rescale
            if self.refiner_enabled is None:
                self.refiner_enabled = profile.refiner_default

        resolved = resolve_generation_defaults(
            self.model,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )
        self.width = resolved.width
        self.height = resolved.height
        self.num_inference_steps = resolved.num_inference_steps
        self.guidance_scale = resolved.guidance_scale

        if self.model == PIXART_ALPHA and (self.width, self.height) != (512, 512):
            raise ValueError("pixart-alpha currently supports only 512x512 output")

        if self.width * self.height * self.num_images > settings.MAX_BATCH_PIXELS:
            raise ValueError(f"Requested batch exceeds the {settings.MAX_BATCH_PIXELS} pixel limit")
        aspect_ratio = max(self.width / self.height, self.height / self.width)
        if aspect_ratio > settings.MAX_ASPECT_RATIO:
            raise ValueError(
                f"Requested dimensions exceed the {settings.MAX_ASPECT_RATIO}:1 aspect limit"
            )
        if (
            self.width * self.height * self.num_images * self.num_inference_steps
            > settings.MAX_GENERATION_WORK_UNITS
        ):
            raise ValueError("Requested dimensions, batch, and steps are too expensive")
        return self


class GenerationParameters(BaseModel):
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    num_images: int
    seed: int
    seeds: list[int]
    elapsed_seconds: float


class GenerateResponse(BaseModel):
    id: str
    images: list[str]
    prompt: str
    effective_prompt: str
    prompt_enhanced: bool
    model: Literal["stable-diffusion", "pixart-alpha"]
    parameters: GenerationParameters


class EditIntentSummary(BaseModel):
    status: Literal["ok", "assumed"]
    action: str
    target: str | None
    attribute: str | None
    scope: str
    assumption: str | None = None


class EditParameters(BaseModel):
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    strength: float
    seed: int
    elapsed_seconds: float


class EditResponse(BaseModel):
    id: str
    images: list[str]
    prompt: str
    model: Literal["stable-diffusion"]
    intent: EditIntentSummary
    parameters: EditParameters


class PlanLayoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, max_length=settings.MAX_PROMPT_LENGTH)
    visual_context: dict[str, Any] | list[dict[str, Any]] | None = None
    guidance_mode: Literal["gaussian", "box"] = "gaussian"
    adaptive_guidance: bool = True
    manual_guidance_strength: float | None = None
    is_edit: bool = False
    layout_override: list[dict[str, Any]] | None = None


def _client_key(request: Request) -> str:
    """Extract client IP, inspecting trusted forwarding headers if present."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def _enforce_generation_rate_limit(request: Request) -> None:
    retry_after = generation_rate_limiter.check(_client_key(request))
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Generation rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )


def _public_image_path(filepath: str) -> str:
    """Return a host-independent URL path to avoid trusting the Host header."""
    return f"/outputs/{os.path.basename(filepath)}"


def _raise_public_generation_error(exc: Exception, request_id: str) -> None:
    if isinstance(exc, ModelBusyError):
        raise HTTPException(
            status_code=429,
            detail="The generation worker is busy; retry shortly.",
            headers={"Retry-After": "2"},
        ) from exc
    if isinstance(
        exc,
        (ModelUnavailableError, ModelLoadError, GenerationCapacityError),
    ):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, OutputSaveError):
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    if isinstance(exc, GenerationError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.exception("Unexpected generation failure (request_id=%s)", request_id)
    raise HTTPException(
        status_code=500,
        detail=f"Unexpected image generation failure. Request ID: {request_id}",
    ) from exc


def _edit_request_error(
    code: str,
    message: str,
    *,
    status_code: int = 422,
    **details,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, **details},
    )


async def _read_edit_upload(upload: UploadFile, label: str) -> bytes:
    limit = settings.MAX_EDIT_UPLOAD_BYTES
    if upload.size is not None and upload.size > limit:
        raise _edit_request_error(
            "edit_upload_too_large",
            f"{label} exceeds the {limit}-byte upload limit.",
            status_code=413,
            field=label,
        )
    try:
        contents = await upload.read(limit + 1)
    except Exception as exc:
        raise _edit_request_error(
            "invalid_edit_upload",
            f"{label} could not be read.",
            field=label,
        ) from exc
    if len(contents) > limit:
        raise _edit_request_error(
            "edit_upload_too_large",
            f"{label} exceeds the {limit}-byte upload limit.",
            status_code=413,
            field=label,
        )
    if not contents:
        raise _edit_request_error(
            "invalid_edit_upload",
            f"{label} must not be empty.",
            field=label,
        )
    return contents


def _decode_edit_image(
    contents: bytes,
    *,
    label: str,
    allowed_formats: frozenset[str],
) -> Image.Image:
    try:
        with Image.open(BytesIO(contents)) as opened:
            image_format = (opened.format or "").upper()
            if image_format not in allowed_formats:
                expected = ", ".join(sorted(allowed_formats))
                raise _edit_request_error(
                    "unsupported_edit_image_format",
                    f"{label} must be an actual {expected} image.",
                    field=label,
                )
            if getattr(opened, "n_frames", 1) != 1:
                raise _edit_request_error(
                    "animated_edit_image_not_supported",
                    f"{label} must contain exactly one frame.",
                    field=label,
                )
            if opened.width * opened.height > settings.MAX_BATCH_PIXELS:
                raise _edit_request_error(
                    "edit_image_too_large",
                    f"{label} exceeds the {settings.MAX_BATCH_PIXELS} pixel limit.",
                    field=label,
                )
            opened.load()
            return ImageOps.exif_transpose(opened).copy()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise _edit_request_error(
            "invalid_edit_image",
            f"{label} is not a valid decodable image.",
            field=label,
        ) from exc


def _validate_edit_geometry(
    *,
    width: int,
    height: int,
    num_inference_steps: int,
) -> None:
    if width < 256 or height < 256 or width > 2048 or height > 2048 or width % 8 or height % 8:
        raise _edit_request_error(
            "invalid_edit_dimensions",
            "Source dimensions must be between 256 and 2048 pixels and divisible by 8.",
        )
    if width * height > settings.MAX_BATCH_PIXELS:
        raise _edit_request_error(
            "edit_image_too_large",
            f"Source image exceeds the {settings.MAX_BATCH_PIXELS} pixel limit.",
        )
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > settings.MAX_ASPECT_RATIO:
        raise _edit_request_error(
            "invalid_edit_aspect_ratio",
            f"Source dimensions exceed the {settings.MAX_ASPECT_RATIO}:1 aspect limit.",
        )
    if width * height * num_inference_steps > settings.MAX_GENERATION_WORK_UNITS:
        raise _edit_request_error(
            "edit_request_too_expensive",
            "Source dimensions and steps are too expensive.",
        )


def _threshold_edit_mask(mask: Image.Image) -> Image.Image:
    rgba = mask.convert("RGBA")
    black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    grayscale = Image.alpha_composite(black, rgba).convert("L")
    return grayscale.point(([0] * 128) + ([255] * 128), mode="L")


def _summarize_edit_intent(intent: PromptIntent) -> EditIntentSummary:
    instruction = intent.instructions[0]
    return EditIntentSummary(
        status=intent.status,
        action=instruction.action,
        target=instruction.target,
        attribute=instruction.attribute,
        scope=instruction.scope,
        assumption=intent.assumption,
    )


async def _generate_and_save(
    payload: GenerateRequest,
    generation_id: str,
    effective_prompt: str,
    generation_seed: int | None,
) -> tuple[GenerationResult, list[str]]:
    result = await asyncio.to_thread(
        model_service.generate_image,
        prompt=effective_prompt,
        negative_prompt=payload.negative_prompt,
        width=payload.width,
        height=payload.height,
        num_inference_steps=payload.num_inference_steps,
        guidance_scale=payload.guidance_scale,
        num_images=payload.num_images,
        seed=generation_seed,
        model=payload.model,
        wait=False,
    )
    paths = await asyncio.to_thread(
        model_service.save_images,
        result.images,
        generation_id,
    )
    return result, paths


async def execute_generation(payload: GenerateRequest) -> GenerateResponse:
    """Run one validated generation request without applying route admission control."""
    generation_id = str(uuid.uuid4())
    try:
        generation_seed = payload.seed
        if payload.enhance_prompt and generation_seed is None:
            generation_seed = secrets.randbits(63)
        effective_prompt = (
            enhance_prompt(payload.prompt, seed=generation_seed)
            if payload.enhance_prompt
            else payload.prompt
        )
        result, image_paths = await _generate_and_save(
            payload,
            generation_id,
            effective_prompt,
            generation_seed,
        )
        return GenerateResponse(
            id=generation_id,
            images=[_public_image_path(path) for path in image_paths],
            prompt=payload.prompt,
            effective_prompt=effective_prompt,
            prompt_enhanced=payload.enhance_prompt,
            model=payload.model,
            parameters=GenerationParameters(
                width=payload.width,
                height=payload.height,
                num_inference_steps=payload.num_inference_steps,
                guidance_scale=payload.guidance_scale,
                num_images=payload.num_images,
                seed=result.seed,
                seeds=list(result.seeds),
                elapsed_seconds=result.elapsed_seconds,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_public_generation_error(exc, generation_id)


@router.post("/generate", response_model=GenerateResponse)
async def generate_image(payload: GenerateRequest, request: Request):
    _enforce_generation_rate_limit(request)
    return await execute_generation(payload)


@router.post("/edit", response_model=EditResponse)
async def edit_image(
    request: Request,
    prompt: Annotated[
        str,
        Form(min_length=1, max_length=settings.MAX_PROMPT_LENGTH),
    ],
    source_image: Annotated[
        UploadFile,
        File(description="PNG, JPEG, or WebP source image"),
    ],
    mask: Annotated[
        UploadFile,
        File(description="PNG mask; white pixels are edited and black pixels are preserved"),
    ],
    negative_prompt: Annotated[
        str | None,
        Form(max_length=settings.MAX_NEGATIVE_PROMPT_LENGTH),
    ] = None,
    num_inference_steps: Annotated[int | None, Form(ge=1, le=100)] = None,
    guidance_scale: Annotated[float | None, Form(ge=1.0, le=20.0)] = None,
    strength: Annotated[float, Form(gt=0.0, le=1.0)] = 1.0,
    seed: Annotated[int | None, Form(ge=0, le=MAX_SEED)] = None,
):
    _enforce_generation_rate_limit(request)
    edit_id = str(uuid.uuid4())
    try:
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise _edit_request_error(
                "invalid_edit_prompt",
                "prompt must not be blank.",
            )
        normalized_negative_prompt = negative_prompt.strip() if negative_prompt else None
        normalized_negative_prompt = normalized_negative_prompt or None

        intent = analyze_prompt(normalized_prompt, mode="edit")
        instruction_count = len(intent.instructions)
        if instruction_count > 1:
            raise _edit_request_error(
                "compound_edit_not_supported",
                "Submit exactly one edit instruction per request.",
                instruction_count=instruction_count,
            )
        if intent.status == "clarify" or instruction_count == 0:
            raise _edit_request_error(
                "edit_clarification_required",
                intent.clarifying_question or "Clarify what should change before editing.",
                instruction_count=instruction_count,
            )

        source_contents = await _read_edit_upload(source_image, "source_image")
        mask_contents = await _read_edit_upload(mask, "mask")
        source = await asyncio.to_thread(
            _decode_edit_image,
            source_contents,
            label="source_image",
            allowed_formats=_SOURCE_IMAGE_FORMATS,
        )
        decoded_mask = await asyncio.to_thread(
            _decode_edit_image,
            mask_contents,
            label="mask",
            allowed_formats=_MASK_IMAGE_FORMATS,
        )
        if decoded_mask.size != source.size:
            raise _edit_request_error(
                "edit_mask_size_mismatch",
                "mask dimensions must exactly match source_image dimensions after orientation.",
                source_size=list(source.size),
                mask_size=list(decoded_mask.size),
            )

        source = source.convert("RGB")
        binary_mask = await asyncio.to_thread(_threshold_edit_mask, decoded_mask)
        if binary_mask.getbbox() is None:
            raise _edit_request_error(
                "empty_edit_mask",
                "mask must contain at least one editable white pixel after thresholding.",
            )

        resolved = resolve_generation_defaults(
            STABLE_DIFFUSION,
            width=source.width,
            height=source.height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        _validate_edit_geometry(
            width=resolved.width,
            height=resolved.height,
            num_inference_steps=resolved.num_inference_steps,
        )

        result = await asyncio.to_thread(
            model_service.edit_image,
            prompt=normalized_prompt,
            source_image=source,
            mask_image=binary_mask,
            negative_prompt=normalized_negative_prompt,
            num_inference_steps=resolved.num_inference_steps,
            guidance_scale=resolved.guidance_scale,
            strength=strength,
            seed=seed,
            model=STABLE_DIFFUSION,
            wait=False,
        )
        image_paths = await asyncio.to_thread(
            model_service.save_images,
            result.images,
            edit_id,
        )
        return EditResponse(
            id=edit_id,
            images=[_public_image_path(path) for path in image_paths],
            prompt=normalized_prompt,
            model=STABLE_DIFFUSION,
            intent=_summarize_edit_intent(intent),
            parameters=EditParameters(
                width=resolved.width,
                height=resolved.height,
                num_inference_steps=resolved.num_inference_steps,
                guidance_scale=resolved.guidance_scale,
                strength=strength,
                seed=result.seed,
                elapsed_seconds=result.elapsed_seconds,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_public_generation_error(exc, edit_id)
    finally:
        await source_image.close()
        await mask.close()


@router.post("/generate/sample", deprecated=True)
async def generate_sample(request: Request):
    """Generate a unique development smoke-test image."""
    _enforce_generation_rate_limit(request)
    generation_id = f"sample_{uuid.uuid4().hex}"
    payload = GenerateRequest(
        model=STABLE_DIFFUSION,
        prompt="A beautiful sunset over the ocean, photorealistic",
        width=512,
        height=512,
        num_inference_steps=20,
        num_images=1,
    )
    try:
        _, paths = await _generate_and_save(
            payload,
            generation_id,
            payload.prompt,
            payload.seed,
        )
        return FileResponse(paths[0], media_type="image/png", filename="sample.png")
    except HTTPException:
        raise
    except Exception as exc:
        _raise_public_generation_error(exc, generation_id)


@router.post("/layout/plan")
async def plan_layout(payload: PlanLayoutRequest, request: Request):
    """Compute structured semantic layout plan with bounding boxes and Gaussian spatial priors."""
    _enforce_generation_rate_limit(request)
    from app.services.editing.prompt_intent import analyze_prompt
    intent = analyze_prompt(payload.prompt, mode="edit" if payload.is_edit else "generate")
    plan = plan_semantic_layout(
        intent,
        visual_context=payload.visual_context,
        guidance_mode=payload.guidance_mode,
        adaptive_guidance=payload.adaptive_guidance,
        manual_guidance_strength=payload.manual_guidance_strength,
        layout_override=payload.layout_override,
    )
    return plan.to_dict()


@router.get("/health")
async def health_check():
    return {"status": "healthy", **model_service.status()}
