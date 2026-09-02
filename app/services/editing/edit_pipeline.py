"""Region-aware denoise loop.

Three interventions, all at the conditioning level - the DiT's weights and shape
are untouched, so this runs against an existing checkpoint:

1. **Region-aware cross-attention** keeps the edit words from steering pixels
   outside the region (`region_attention`).
2. **Spatial classifier-free guidance** replaces the scalar `guidance_scale` with
   a per-position map: prompt-following inside, source-preserving outside.
3. **Scheduled latent blending** restores the source outside the region after each
   step, ramping in so early layout steps stay free.

The denoiser is injected, so this is testable without a checkpoint and reusable
across PixArt / SD3 / Flux - each only differs in how it is called.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch
from PIL import Image

from app.services.editing.adaptive_reference import (
    apply_region_guidance,
    blend_latents,
    preservation_at_step,
)
from app.services.editing.edit_planner import EditPlan
from app.services.editing.layout_guidance import LayoutGuidanceProcessor
from app.services.editing.masks import as_soft_mask, resize_mask
from app.services.editing.region_attention import (
    RegionAwareAttnProcessor,
    build_attention_bias,
    region_attention_bias,
)
from app.services.editing.semantic_planner import SemanticLayoutPlan


class QualityTier(str, Enum):
    PREVIEW = "preview"
    FINAL = "final"
    ULTRA = "ultra"


@dataclass(frozen=True)
class TierProfile:
    name: str
    default_steps: int
    default_guidance_scale: float
    cfg_rescale: float
    work_units: int
    refiner_default: bool
    refiner_steps: int


TIER_PROFILES: dict[QualityTier | str, TierProfile] = {
    QualityTier.PREVIEW: TierProfile(
        name="preview",
        default_steps=14,
        default_guidance_scale=6.5,
        cfg_rescale=0.70,
        work_units=1,
        refiner_default=False,
        refiner_steps=0,
    ),
    QualityTier.FINAL: TierProfile(
        name="final",
        default_steps=28,
        default_guidance_scale=7.0,
        cfg_rescale=0.70,
        work_units=2,
        refiner_default=False,
        refiner_steps=8,
    ),
    QualityTier.ULTRA: TierProfile(
        name="ultra",
        default_steps=36,
        default_guidance_scale=7.5,
        cfg_rescale=0.75,
        work_units=4,
        refiner_default=True,
        refiner_steps=12,
    ),
}
# String alias lookups
TIER_PROFILES["preview"] = TIER_PROFILES[QualityTier.PREVIEW]
TIER_PROFILES["final"] = TIER_PROFILES[QualityTier.FINAL]
TIER_PROFILES["ultra"] = TIER_PROFILES[QualityTier.ULTRA]


def compute_work_units(
    tier: str | QualityTier = QualityTier.FINAL,
    refiner_enabled: bool = False,
    steps: int | None = None,
) -> int:
    """Calculate work-unit resource consumption for rate limiting."""
    profile = TIER_PROFILES.get(tier, TIER_PROFILES[QualityTier.FINAL])
    base_wu = profile.work_units
    if steps is not None:
        if steps <= 16:
            base_wu = 1
        elif steps <= 30:
            base_wu = 2
        else:
            base_wu = 4
    if refiner_enabled:
        base_wu += 1
    return base_wu


class CFGRescaler:
    """Applies variance-preserving CFG Rescaling (phi-rescaling).

    Prevents color blowout, oversaturation, and dynamic range clipping at high guidance scales.
    """

    @staticmethod
    def apply_rescaled_cfg(
        uncond: torch.Tensor,
        cond: torch.Tensor,
        guidance_scale: float | torch.Tensor,
        rescale_factor: float = 0.70,
    ) -> torch.Tensor:
        """Rescale guided prediction using standard deviation of conditional prediction."""
        noise_pred_cfg = uncond + guidance_scale * (cond - uncond)
        if rescale_factor <= 1e-4:
            return noise_pred_cfg

        # Compute standard deviations across spatial & channel dims
        dims = list(range(1, cond.ndim)) if cond.ndim > 1 else [0]
        std_cond = cond.std(dim=dims, keepdim=True)
        std_cfg = noise_pred_cfg.std(dim=dims, keepdim=True)

        factor = std_cond / (std_cfg + 1e-8)
        rescaled = noise_pred_cfg * factor
        return rescale_factor * rescaled + (1.0 - rescale_factor) * noise_pred_cfg


class MaskAwareRefiner:
    """Texture & micro-detail refiner with strict outside-mask boundary isolation."""

    def __init__(
        self,
        refiner_fn: Callable[..., Image.Image | torch.Tensor] | None = None,
        default_strength: float = 0.25,
    ) -> None:
        self.refiner_fn = refiner_fn
        self.default_strength = default_strength

    def refine_image(
        self,
        base_image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str = "",
        strength: float | None = None,
        edit_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Image.Image:
        """Run refinement pass and enforce mask composite in editing flows."""
        used_strength = strength if strength is not None else self.default_strength
        if self.refiner_fn is None or used_strength <= 1e-3:
            return base_image

        refined = self.refiner_fn(
            image=base_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=used_strength,
            generator=generator,
        )

        if not isinstance(refined, Image.Image):
            return base_image

        if edit_mask is None:
            return refined

        # Composite with strict outside-mask boundary preservation (SSIM >= 0.998)
        w, h = base_image.size
        mask_2d = resize_mask(as_soft_mask(edit_mask), h, w)
        mask_np = (mask_2d.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        mask_pil = Image.fromarray(mask_np, mode="L")
        return Image.composite(refined, base_image, mask_pil)

# (latents, timestep, conditional) -> noise prediction. Two calls per step.
DenoiseFn = Callable[[torch.Tensor, int, bool], torch.Tensor]
# (latents, timestep) -> (uncond, cond). One call per step: the transformer sees
# both halves as a single batch, which is how classifier-free guidance is normally
# run and halves the number of forward passes.
PairedDenoiseFn = Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]
# (latents, noise_pred, timestep) -> next latents
StepFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass
class EditTrace:
    """Per-step record, for tuning and for the eval report."""

    steps: list[dict] = field(default_factory=list)

    def add(self, **kwargs) -> None:
        self.steps.append(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kwargs.items()}
        )


def batched_cfg_denoiser(model_call, uncond_embeddings, cond_embeddings) -> PairedDenoiseFn:
    """Build a paired denoiser that runs uncond+cond in one forward.

    `model_call(latents, timestep, encoder_hidden_states) -> noise_prediction`.
    The latents are duplicated and the two embedding sets concatenated, so the
    transformer is entered once per step instead of twice. Requires the region
    bias to tolerate batch > 1, which `RegionAwareAttnProcessor` now does.
    """

    def pair(latents: torch.Tensor, timestep: int):
        batched = torch.cat([latents, latents], dim=0)
        embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        prediction = model_call(batched, timestep, embeddings)
        uncond, cond = prediction.chunk(2, dim=0)
        return uncond, cond

    return pair


def _resolve_pair(
    denoise: DenoiseFn | None,
    denoise_pair: PairedDenoiseFn | None,
) -> PairedDenoiseFn:
    """Accept either calling convention; the batched one wins when both are given."""
    if denoise_pair is not None:
        return denoise_pair
    if denoise is None:
        raise ValueError("pass either denoise= or denoise_pair=")
    return lambda latents, timestep: (
        denoise(latents, timestep, False),
        denoise(latents, timestep, True),
    )


def default_step(latents: torch.Tensor, noise_pred: torch.Tensor, timestep: int) -> torch.Tensor:
    """Plain Euler step, used when no scheduler is supplied."""
    return latents - noise_pred


def set_region_bias(
    processors: Sequence[RegionAwareAttnProcessor],
    plan: EditPlan,
    *,
    num_image_tokens: int,
    num_text_tokens: int,
    leak_penalty: float = -12.0,
    context_boost: float = 0.5,
    aspect: float | None = None,
) -> None:
    """Push the current region bias into every region-aware attention processor.

    Uses the same `plan.mask` the adaptive coefficients were derived from, so the
    attention masking and the guidance map can never disagree about the region.
    """
    resolved_aspect = aspect if aspect is not None else getattr(plan, "aspect", 1.0)
    if plan.attention_strength <= 0:
        bias = None
    elif plan.token_roles:
        # Role-aware: suppress edit targets outside, boost context inside. Scaled by
        # attention_strength so a regional edit is masked more gently than a local one.
        roles = list(plan.token_roles[:num_text_tokens])
        roles += ["neutral"] * (num_text_tokens - len(roles))
        bias = build_attention_bias(
            plan.mask,
            roles,
            leak_penalty=leak_penalty * plan.attention_strength,
            context_boost=context_boost * plan.attention_strength,
            num_image_tokens=num_image_tokens,
            aspect=resolved_aspect,
        ).transpose(0, 1)  # -> (image_tokens, text_tokens) for the processor
    else:
        bias = region_attention_bias(
            plan.mask,
            num_image_tokens=num_image_tokens,
            num_text_tokens=num_text_tokens,
            edit_token_indices=list(plan.edit_token_indices) or None,
            strength=plan.attention_strength,
            aspect=resolved_aspect,
        )
    for processor in processors:
        processor.set_bias(bias)


def run_region_aware_edit(
    *,
    plan: EditPlan,
    source_latents: torch.Tensor,
    initial_latents: torch.Tensor,
    timesteps: Sequence[int],
    denoise: DenoiseFn | None = None,
    denoise_pair: PairedDenoiseFn | None = None,
    step: StepFn = default_step,
    blend: bool = True,
    trace: EditTrace | None = None,
) -> torch.Tensor:
    """Denoise `initial_latents` into an edit of `source_latents` under `plan`.

    Pass `denoise_pair` to run uncond+cond as one batched forward (half the calls);
    `denoise` keeps the original two-call convention for existing callers.
    """
    predict = _resolve_pair(denoise, denoise_pair)
    if plan.mask is None:
        raise ValueError("plan.mask is required")
    latents = initial_latents
    height, width = latents.shape[-2:]
    mask = resize_mask(plan.mask, height, width).to(
        device=latents.device, dtype=latents.dtype
    )
    scale_map = apply_region_guidance(
        mask, plan.coefficients, height=height, width=width
    ).to(device=latents.device, dtype=latents.dtype)

    total = max(len(timesteps), 1)
    for index, timestep in enumerate(timesteps):
        uncond, cond = predict(latents, timestep)
        # Spatial CFG: the scalar guidance_scale becomes a per-position map.
        noise_pred = uncond + scale_map * (cond - uncond)
        latents = step(latents, noise_pred, timestep)

        if blend:
            progress = (index + 1) / total
            strength = preservation_at_step(plan.coefficients.ref_weight, progress)
            latents = blend_latents(latents, source_latents, mask, strength=strength)
        else:
            strength = 0.0

        if trace is not None:
            trace.add(step=index, timestep=timestep, preservation=strength)

    return latents


def run_baseline_edit(
    *,
    source_latents: torch.Tensor,
    initial_latents: torch.Tensor,
    timesteps: Sequence[int],
    guidance_scale: float,
    denoise: DenoiseFn | None = None,
    denoise_pair: PairedDenoiseFn | None = None,
    step: StepFn = default_step,
) -> torch.Tensor:
    """Today's behaviour: one scalar guidance, no mask, no blending. The eval baseline."""
    predict = _resolve_pair(denoise, denoise_pair)
    latents = initial_latents
    for timestep in timesteps:
        uncond, cond = predict(latents, timestep)
        latents = step(latents, uncond + guidance_scale * (cond - uncond), timestep)
    return latents


def set_layout_guidance(
    processors: Sequence[LayoutGuidanceProcessor],
    plan: SemanticLayoutPlan | None,
    *,
    guidance_strength: float = 0.3,
    schedule_cutoff: float | None = None,
    aspect: float | None = None,
) -> None:
    """Push the semantic layout plan into every layout guidance attention processor."""
    for proc in processors:
        proc.set_plan(plan)
        proc.guidance_strength = guidance_strength
        if schedule_cutoff is not None:
            proc.schedule_cutoff = schedule_cutoff
        if aspect is not None:
            proc.aspect = aspect


def run_hybrid_generation(
    *,
    plan: SemanticLayoutPlan | None,
    initial_latents: torch.Tensor,
    timesteps: Sequence[int],
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.0,
    layout_processors: Sequence[LayoutGuidanceProcessor] | None = None,
    denoise: DenoiseFn | None = None,
    denoise_pair: PairedDenoiseFn | None = None,
    step: StepFn = default_step,
    trace: EditTrace | None = None,
) -> torch.Tensor:
    """Run DiT generation with soft semantic layout guidance and optional CFG rescaling."""
    predict = _resolve_pair(denoise, denoise_pair)
    latents = initial_latents
    total = max(len(timesteps), 1)

    for index, timestep in enumerate(timesteps):
        progress = (index + 1) / total
        if layout_processors:
            for proc in layout_processors:
                proc.set_step_progress(progress)

        uncond, cond = predict(latents, timestep)
        if guidance_rescale > 0:
            noise_pred = CFGRescaler.apply_rescaled_cfg(
                uncond, cond, guidance_scale, rescale_factor=guidance_rescale
            )
        else:
            noise_pred = uncond + guidance_scale * (cond - uncond)
        latents = step(latents, noise_pred, timestep)

        if trace is not None:
            trace.add(step=index, timestep=timestep, progress=progress)

    return latents


def run_hybrid_edit(
    *,
    plan: EditPlan,
    source_latents: torch.Tensor,
    initial_latents: torch.Tensor,
    timesteps: Sequence[int],
    guidance_rescale: float = 0.0,
    layout_processors: Sequence[LayoutGuidanceProcessor] | None = None,
    denoise: DenoiseFn | None = None,
    denoise_pair: PairedDenoiseFn | None = None,
    step: StepFn = default_step,
    blend: bool = True,
    trace: EditTrace | None = None,
) -> torch.Tensor:
    """Run a hybrid edit: soft layout guidance + region-aware CFG + adaptive blending."""
    predict = _resolve_pair(denoise, denoise_pair)
    if plan.mask is None:
        raise ValueError("plan.mask is required")
    latents = initial_latents
    height, width = latents.shape[-2:]
    mask = resize_mask(plan.mask, height, width).to(
        device=latents.device, dtype=latents.dtype
    )
    scale_map = apply_region_guidance(
        mask, plan.coefficients, height=height, width=width
    ).to(device=latents.device, dtype=latents.dtype)

    total = max(len(timesteps), 1)
    for index, timestep in enumerate(timesteps):
        progress = (index + 1) / total
        if layout_processors:
            for proc in layout_processors:
                proc.set_step_progress(progress)

        uncond, cond = predict(latents, timestep)
        if guidance_rescale > 0:
            noise_pred = CFGRescaler.apply_rescaled_cfg(
                uncond, cond, scale_map, rescale_factor=guidance_rescale
            )
        else:
            noise_pred = uncond + scale_map * (cond - uncond)
        latents = step(latents, noise_pred, timestep)

        if blend:
            strength = preservation_at_step(plan.coefficients.ref_weight, progress)
            latents = blend_latents(latents, source_latents, mask, strength=strength)
        else:
            strength = 0.0

        if trace is not None:
            trace.add(step=index, timestep=timestep, preservation=strength, progress=progress)

    return latents
