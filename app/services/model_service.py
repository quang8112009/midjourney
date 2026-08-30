"""Thread-safe model lifecycle, inference, and generated-image persistence."""

from __future__ import annotations

import gc
import logging
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoPipelineForInpainting, DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

STABLE_DIFFUSION = "stable-diffusion"
PIXART_ALPHA = "pixart-alpha"
TEXT_TO_IMAGE_TASK = "text-to-image"
INPAINTING_TASK = "inpainting"
MAX_SEED = 2**63 - 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class ModelServiceError(RuntimeError):
    """Base class for safe, expected model-service failures."""


class ModelUnavailableError(ModelServiceError):
    """Raised when a backend cannot run on the configured hardware."""


class ModelLoadError(ModelServiceError):
    """Raised when a checkpoint cannot be loaded."""


class ModelBusyError(ModelServiceError):
    """Raised when the single generation slot is already occupied."""


class GenerationError(ModelServiceError):
    """Raised when inference fails."""


class GenerationCapacityError(GenerationError):
    """Raised when inference exhausts available accelerator memory."""


class OutputSaveError(ModelServiceError):
    """Raised when generated output cannot be committed atomically."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    images: tuple
    model: str
    seed: int
    seeds: tuple[int, ...]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ResolvedGenerationDefaults:
    """Concrete inference controls after applying backend-specific defaults."""

    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float


def resolve_generation_defaults(
    model: str,
    *,
    width: int | None = None,
    height: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> ResolvedGenerationDefaults:
    """Fill omitted generation controls from the selected backend's one policy."""
    if model == PIXART_ALPHA:
        default_width = 512
        default_height = 512
        default_steps = settings.PIXART_DEFAULT_STEPS
        default_guidance_scale = settings.PIXART_DEFAULT_GUIDANCE_SCALE
    else:
        default_width = settings.DEFAULT_WIDTH
        default_height = settings.DEFAULT_HEIGHT
        default_steps = settings.DEFAULT_STEPS
        default_guidance_scale = settings.DEFAULT_GUIDANCE_SCALE

    return ResolvedGenerationDefaults(
        width=default_width if width is None else width,
        height=default_height if height is None else height,
        num_inference_steps=(
            default_steps if num_inference_steps is None else num_inference_steps
        ),
        guidance_scale=(
            default_guidance_scale if guidance_scale is None else guidance_scale
        ),
    )


class ModelService:
    """Own one active pipeline and allow only one inference operation at a time."""

    def __init__(
        self,
        pipeline_loader: Callable | None = None,
        scheduler_loader: Callable | None = None,
        inpainting_pipeline_loader: Callable | None = None,
    ):
        self.pipe = None
        self.active_model: str | None = None
        self.active_task: str | None = None
        self.device = "cpu"
        self._pipeline_loader = pipeline_loader or DiffusionPipeline.from_pretrained
        self._inpainting_pipeline_loader = (
            inpainting_pipeline_loader or AutoPipelineForInpainting.from_pretrained
        )
        self._scheduler_loader = scheduler_loader or DPMSolverMultistepScheduler.from_config
        self._lock = threading.RLock()
        self._output_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._loading_model: str | None = None
        self._loading_task: str | None = None
        self._busy = False
        self._last_load_error: str | None = None
        self._generation_count = 0
        self._last_generation_seconds: float | None = None

    @property
    def is_loaded(self) -> bool:
        return (
            self.pipe is not None
            and self.active_model is not None
            and self.active_task is not None
        )

    @property
    def loading_model(self) -> str | None:
        return self._loading_model

    @property
    def loading_task(self) -> str | None:
        return self._loading_task

    def _model_id(self, model: str) -> str:
        if model == STABLE_DIFFUSION:
            return settings.MODEL_ID
        if model == PIXART_ALPHA:
            return settings.PIXART_MODEL_ID
        raise ValueError(f"Unsupported model '{model}'")

    def _resolve_device(self, model: str) -> str:
        has_cuda = torch.cuda.is_available()
        wants_cuda = settings.DEVICE == "cuda" or (settings.DEVICE == "auto" and has_cuda)
        if settings.DEVICE == "cuda" and not has_cuda:
            raise ModelUnavailableError("CUDA is configured but no CUDA device is available.")
        if model == PIXART_ALPHA and not (wants_cuda and has_cuda):
            raise ModelUnavailableError(
                "PixArt-Alpha requires a CUDA GPU. Use Stable Diffusion on CPU "
                "or deploy this service to a CUDA worker."
            )
        return "cuda" if wants_cuda and has_cuda else "cpu"

    def _resolve_dtype(self, device: str):
        if device == "cpu":
            return torch.float32
        if settings.DTYPE == "float32":
            return torch.float32
        if settings.DTYPE == "float16":
            return torch.float16
        if settings.DTYPE == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                raise ModelUnavailableError(
                    "bfloat16 is configured but unsupported by this CUDA device."
                )
            return torch.bfloat16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _unload_model_locked(self) -> None:
        self.pipe = None
        self.active_model = None
        self.active_task = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_model(self) -> None:
        """Release the active pipeline and accelerator cache."""
        with self._lock:
            self._unload_model_locked()

    def load_model(
        self,
        model: str = STABLE_DIFFUSION,
        *,
        task: str = TEXT_TO_IMAGE_TASK,
    ) -> None:
        """Load a model/task pipeline, replacing a different active identity."""
        with self._lock:
            if task not in (TEXT_TO_IMAGE_TASK, INPAINTING_TASK):
                raise ValueError(f"Unsupported pipeline task '{task}'")
            if task == INPAINTING_TASK and model != STABLE_DIFFUSION:
                raise ValueError("Image editing supports only stable-diffusion")
            if (
                self.active_model == model
                and self.active_task == task
                and self.pipe is not None
            ):
                return

            model_id = self._model_id(model)
            device = self._resolve_device(model)
            dtype = self._resolve_dtype(device)
            with self._state_lock:
                self._loading_model = model
                self._loading_task = task
                self._last_load_error = None
            pipeline = None
            try:
                self._unload_model_locked()
                loader = (
                    self._inpainting_pipeline_loader
                    if task == INPAINTING_TASK
                    else self._pipeline_loader
                )
                pipeline = loader(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    cache_dir=settings.MODEL_CACHE_DIR,
                )

                if model == STABLE_DIFFUSION and task == TEXT_TO_IMAGE_TASK:
                    pipeline.scheduler = self._scheduler_loader(pipeline.scheduler.config)

                if device == "cuda" and settings.MODEL_CPU_OFFLOAD:
                    pipeline.enable_model_cpu_offload()
                else:
                    pipeline = pipeline.to(device)

                if hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()

                self.pipe = pipeline
                self.active_model = model
                self.active_task = task
                self.device = device
                logger.info("Loaded %s/%s (%s) on %s", model, task, model_id, device)
            except ModelServiceError:
                pipeline = None
                self._unload_model_locked()
                with self._state_lock:
                    self._last_load_error = f"Unable to load {model}."
                raise
            except Exception as exc:
                pipeline = None
                self._unload_model_locked()
                with self._state_lock:
                    self._last_load_error = f"Unable to load {model}."
                logger.exception("Failed to load model %s", model)
                raise ModelLoadError(f"Unable to load the {model} checkpoint.") from exc
            finally:
                with self._state_lock:
                    self._loading_model = None
                    self._loading_task = None

    @staticmethod
    def _validate_generation_args(
        model: str,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images: int,
        seed: int | None,
    ) -> None:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        if len(prompt) > settings.MAX_PROMPT_LENGTH:
            raise ValueError("prompt exceeds MAX_PROMPT_LENGTH")
        if (
            negative_prompt is not None
            and len(negative_prompt) > settings.MAX_NEGATIVE_PROMPT_LENGTH
        ):
            raise ValueError("negative_prompt exceeds MAX_NEGATIVE_PROMPT_LENGTH")
        if width < 256 or height < 256 or width % 8 or height % 8:
            raise ValueError("Image dimensions must be >=256 and divisible by 8")
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > settings.MAX_ASPECT_RATIO:
            raise ValueError("Requested dimensions exceed MAX_ASPECT_RATIO")
        if model == PIXART_ALPHA and (width, height) != (512, 512):
            raise ValueError("pixart-alpha supports only 512x512 output")
        if not 1 <= num_inference_steps <= 100:
            raise ValueError("num_inference_steps must be between 1 and 100")
        if not 1.0 <= guidance_scale <= 20.0:
            raise ValueError("guidance_scale must be between 1.0 and 20.0")
        if not 1 <= num_images <= 4:
            raise ValueError("num_images must be between 1 and 4")
        if width * height * num_images > settings.MAX_BATCH_PIXELS:
            raise ValueError("Requested batch exceeds MAX_BATCH_PIXELS")
        if width * height * num_images * num_inference_steps > settings.MAX_GENERATION_WORK_UNITS:
            raise ValueError("Requested batch exceeds MAX_GENERATION_WORK_UNITS")
        if seed is not None and not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be between 0 and {MAX_SEED}")

    @staticmethod
    def _validate_edit_args(
        model: str,
        prompt: str,
        negative_prompt: str | None,
        source_image: Image.Image,
        mask_image: Image.Image,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int | None,
    ) -> None:
        if model != STABLE_DIFFUSION:
            raise ValueError("Image editing supports only stable-diffusion")
        if source_image.size != mask_image.size:
            raise ValueError("source_image and mask_image dimensions must match")
        if mask_image.getbbox() is None:
            raise ValueError("mask_image must select at least one pixel")
        if not 0.0 < strength <= 1.0:
            raise ValueError("strength must be greater than 0 and at most 1")
        width, height = source_image.size
        ModelService._validate_generation_args(
            model,
            prompt,
            negative_prompt,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            1,
            seed,
        )

    def _acquire_inference_slot(self, wait: bool) -> float:
        acquired = self._lock.acquire(blocking=wait)
        if not acquired:
            raise ModelBusyError("The generation worker is busy; retry shortly.")
        started = time.perf_counter()
        with self._state_lock:
            self._busy = True
        return started

    def _release_inference_slot(self) -> None:
        with self._state_lock:
            self._busy = False
        self._lock.release()

    def _record_inference(self, started: float) -> float:
        elapsed = time.perf_counter() - started
        with self._state_lock:
            self._generation_count += 1
            self._last_generation_seconds = elapsed
        return elapsed

    def _invoke_pipeline(
        self,
        *,
        model: str,
        failure_message: str,
        **kwargs,
    ) -> tuple:
        try:
            with torch.inference_mode():
                result = self.pipe(**kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            self._unload_model_locked()
            raise GenerationCapacityError(
                "The requested batch exceeded available GPU memory."
            ) from exc
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.exception("Inference failed for model %s", model)
            raise GenerationError(failure_message) from exc
        return tuple(result.images)

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        num_images: int = 1,
        seed: int | None = None,
        model: str = STABLE_DIFFUSION,
        wait: bool = True,
    ) -> GenerationResult:
        """Generate a reproducible batch, optionally rejecting when the slot is busy."""
        prompt = prompt.strip()
        negative_prompt = negative_prompt.strip() if negative_prompt else None
        resolved = resolve_generation_defaults(
            model,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        width = resolved.width
        height = resolved.height
        num_inference_steps = resolved.num_inference_steps
        guidance_scale = resolved.guidance_scale
        self._validate_generation_args(
            model,
            prompt,
            negative_prompt,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            num_images,
            seed,
        )
        started = self._acquire_inference_slot(wait)
        try:
            self.load_model(model, task=TEXT_TO_IMAGE_TASK)
            resolved_seed = seed if seed is not None else secrets.randbits(63)
            seeds = tuple((resolved_seed + index) % (MAX_SEED + 1) for index in range(num_images))
            generators = [
                torch.Generator(device="cpu").manual_seed(image_seed) for image_seed in seeds
            ]

            images = self._invoke_pipeline(
                model=model,
                failure_message="Image inference failed.",
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generators,
            )
            if len(images) != num_images:
                raise GenerationError("The model returned an unexpected number of images.")
            elapsed = self._record_inference(started)
            return GenerationResult(
                images=images,
                model=model,
                seed=resolved_seed,
                seeds=seeds,
                elapsed_seconds=elapsed,
            )
        finally:
            self._release_inference_slot()

    def edit_image(
        self,
        *,
        prompt: str,
        source_image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        strength: float = 1.0,
        seed: int | None = None,
        model: str = STABLE_DIFFUSION,
        wait: bool = True,
    ) -> GenerationResult:
        """Edit one decoded image inside a mask using a reproducible inpaint pass."""
        if not isinstance(source_image, Image.Image):
            raise ValueError("source_image must be a decoded PIL image")
        if not isinstance(mask_image, Image.Image):
            raise ValueError("mask_image must be a decoded PIL image")

        prompt = prompt.strip()
        negative_prompt = negative_prompt.strip() if negative_prompt else None
        source_image = source_image.convert("RGB")
        mask_image = mask_image.convert("L")
        width, height = source_image.size
        resolved = resolve_generation_defaults(
            model,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        self._validate_edit_args(
            model,
            prompt,
            negative_prompt,
            source_image,
            mask_image,
            resolved.num_inference_steps,
            resolved.guidance_scale,
            strength,
            seed,
        )

        started = self._acquire_inference_slot(wait)
        try:
            self.load_model(model, task=INPAINTING_TASK)
            resolved_seed = seed if seed is not None else secrets.randbits(63)
            generator = torch.Generator(device="cpu").manual_seed(resolved_seed)
            images = self._invoke_pipeline(
                model=model,
                failure_message="Image editing failed.",
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                image=source_image,
                mask_image=mask_image,
                width=resolved.width,
                height=resolved.height,
                num_inference_steps=resolved.num_inference_steps,
                guidance_scale=resolved.guidance_scale,
                strength=strength,
                num_images_per_prompt=1,
                generator=generator,
            )
            if len(images) != 1:
                raise GenerationError("The model returned an unexpected number of images.")
            elapsed = self._record_inference(started)
            return GenerationResult(
                images=images,
                model=model,
                seed=resolved_seed,
                seeds=(resolved_seed,),
                elapsed_seconds=elapsed,
            )
        finally:
            self._release_inference_slot()

    @staticmethod
    def _validate_output_name(name: str) -> None:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("Output name contains unsupported characters")

    @staticmethod
    def _save_png_atomic(image, destination: Path) -> None:
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".tmp",
                prefix=f".{destination.stem}-",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
            image.save(temporary_path, format="PNG")
            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
            raise

    def _prune_outputs_locked(self, protected: set[Path]) -> None:
        output_dir = Path(settings.OUTPUT_DIR).resolve()
        files = [path for path in output_dir.glob("*.png") if path.is_file()]
        files.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
        now = time.time()

        if settings.MAX_OUTPUT_AGE_SECONDS:
            for path in list(files):
                if path in protected:
                    continue
                if now - path.stat().st_mtime > settings.MAX_OUTPUT_AGE_SECONDS:
                    path.unlink(missing_ok=True)
                    files.remove(path)

        total_bytes = sum(path.stat().st_size for path in files)
        while len(files) > settings.MAX_OUTPUT_FILES or total_bytes > settings.MAX_OUTPUT_BYTES:
            candidate = next((path for path in files if path not in protected), None)
            if candidate is None:
                raise OutputSaveError(
                    "The generated batch exceeds configured output storage limits."
                )
            size = candidate.stat().st_size
            candidate.unlink(missing_ok=True)
            files.remove(candidate)
            total_bytes -= size

    def save_images(
        self,
        images: Sequence,
        generation_id: str,
    ) -> list[str]:
        """Atomically save a batch and roll it back if any image fails."""
        self._validate_output_name(generation_id)
        if not images:
            raise OutputSaveError("No generated images were provided for saving.")

        output_dir = Path(settings.OUTPUT_DIR).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        with self._output_lock:
            try:
                for index, image in enumerate(images):
                    destination = output_dir / f"{generation_id}_{index}.png"
                    self._save_png_atomic(image, destination)
                    saved.append(destination)
                self._prune_outputs_locked(set(saved))
            except Exception as exc:
                for path in saved:
                    path.unlink(missing_ok=True)
                logger.exception("Failed to save generated image batch")
                raise OutputSaveError("Generated images could not be saved.") from exc
        return [str(path) for path in saved]

    def save_image(self, image, filename: str | None = None) -> str:
        """Backward-compatible single-image atomic save helper."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"generated_{timestamp}_{uuid.uuid4().hex[:8]}"
        else:
            if not filename.lower().endswith(".png"):
                raise ValueError("Only PNG output filenames are supported")
            name = filename[:-4]
        return self.save_images([image], name)[0]

    def status(self) -> dict:
        with self._state_lock:
            busy = self._busy
            loading_model = self._loading_model
            loading_task = self._loading_task
            last_load_error = self._last_load_error
            generation_count = self._generation_count
            last_generation_seconds = self._last_generation_seconds
        return {
            "model_loaded": self.is_loaded,
            "active_model": self.active_model,
            "active_task": self.active_task,
            "loading_model": loading_model,
            "loading_task": loading_task,
            "configured_device": settings.DEVICE,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "busy": busy,
            "last_load_error": last_load_error,
            "generation_count": generation_count,
            "last_generation_seconds": last_generation_seconds,
        }
