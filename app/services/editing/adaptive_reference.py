"""Adaptive reference/guidance coefficient for region-aware editing.

Two signals drive everything, as in the reference design:

* **locality**  - how small the edit region is. A small region means most of the
  frame must survive untouched, so preservation should be high.
* **conflict**  - how far the prompt is from what is currently inside that region.
  High conflict means the user asked for a big change, so the prompt should be
  followed harder *inside* the mask.

Deviation from the supplied pseudo-code (deliberate, see docs):
`inside_scale = base * (1 - ref_weight) * 2` couples inside guidance inversely to
the preservation weight, so a *very local* edit (ref_weight -> max) would get the
*weakest* prompt guidance exactly where the user wants the change. That is
backwards. Preservation and edit strength are separated here: `ref_weight`
governs what happens **outside** the mask (and the latent blend), while
`edit_strength` governs **inside**. The original formula is still available via
`legacy_reference_coefficient` so the eval suite can measure the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from app.services.editing.masks import area_ratio, as_soft_mask, resize_mask


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class ReferenceCoefficients:
    """Resolved conditioning strengths for one edit."""

    ref_weight: float
    """Preservation weight in [min_ref_weight, max_ref_weight]. Higher = keep more source."""

    edit_strength: float
    """How hard to follow the prompt inside the mask, in [0, 1]."""

    locality_score: float
    conflict_score: float
    raw_similarity: float
    inside_scale: float
    outside_scale: float
    notes: tuple[str, ...] = field(default=())

    def as_log_dict(self) -> dict:
        return {
            "ref_weight": round(self.ref_weight, 4),
            "edit_strength": round(self.edit_strength, 4),
            "locality_score": round(self.locality_score, 4),
            "conflict_score": round(self.conflict_score, 4),
            "raw_similarity": round(self.raw_similarity, 4),
            "inside_scale": round(self.inside_scale, 4),
            "outside_scale": round(self.outside_scale, 4),
        }


@dataclass(frozen=True)
class CoefficientConfig:
    """Tunable knobs. Defaults are the starting point; tune against the eval suite."""

    locality_weight: float = 0.6
    similarity_weight: float = 0.4
    min_ref_weight: float = 0.2
    max_ref_weight: float = 0.9
    # Cosine similarity between a text and an image embedding occupies a narrow
    # band (CLIP is typically ~0.15-0.35), so the raw value barely moves the
    # result. Calibrate it onto [0,1] across the band actually observed.
    similarity_floor: float = 0.10
    similarity_ceiling: float = 0.40
    inside_gain: float = 0.6
    outside_damp: float = 0.5
    min_scale: float = 1.0
    max_scale: float = 20.0


def calibrate_similarity(similarity: float, config: CoefficientConfig) -> float:
    """Map a raw cosine similarity onto [0,1] across the band the encoder produces."""
    span = config.similarity_ceiling - config.similarity_floor
    if span <= 0:
        return _clamp(similarity, 0.0, 1.0)
    return _clamp((similarity - config.similarity_floor) / span, 0.0, 1.0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two embeddings, pooled over any leading token dims."""
    a = a.reshape(-1, a.shape[-1]).mean(dim=0).float()
    b = b.reshape(-1, b.shape[-1]).mean(dim=0).float()
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def extract_region_embedding(
    image_embedding: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mask-weighted pool of a spatial image embedding (B,C,H,W) -> (C,).

    Falls back to a plain mean for already-pooled embeddings so callers can pass a
    CLIP image vector directly.
    """
    if image_embedding.ndim < 4:
        return image_embedding.reshape(-1, image_embedding.shape[-1]).mean(dim=0)
    height, width = image_embedding.shape[-2:]
    weights = resize_mask(mask, height, width).to(
        device=image_embedding.device, dtype=image_embedding.dtype
    )
    total = weights.sum()
    if float(total) == 0.0:
        return image_embedding.mean(dim=(0, 2, 3))
    weighted = (image_embedding * weights).sum(dim=(0, 2, 3)) / total
    return weighted


def compute_adaptive_reference_coefficient(
    *,
    prompt_embedding: torch.Tensor,
    source_image_embedding: torch.Tensor,
    edit_region_mask: torch.Tensor,
    base_guidance_scale: float = 7.5,
    config: CoefficientConfig | None = None,
) -> ReferenceCoefficients:
    """Derive preservation and edit strength from mask locality and prompt/region conflict."""
    config = config or CoefficientConfig()
    mask = as_soft_mask(edit_region_mask)
    notes: list[str] = []

    # 1. Locality: how much of the frame is being left alone.
    locality_score = 1.0 - _clamp(area_ratio(mask), 0.0, 1.0)

    # 2. Conflict: how far the prompt is from the region's current content.
    region_embedding = extract_region_embedding(source_image_embedding, mask)
    raw_similarity = cosine_similarity(prompt_embedding, region_embedding)
    calibrated = calibrate_similarity(raw_similarity, config)
    conflict_score = 1.0 - calibrated

    # 3. Preservation weight. (1 - conflict) is the calibrated similarity, so this
    #    is the reference formula with the similarity term made responsive.
    weight_sum = config.locality_weight + config.similarity_weight
    if weight_sum <= 0:
        raise ValueError("locality_weight + similarity_weight must be positive")
    raw = config.locality_weight * locality_score + config.similarity_weight * calibrated
    raw /= weight_sum
    ref_weight = config.min_ref_weight + raw * (config.max_ref_weight - config.min_ref_weight)
    ref_weight = _clamp(ref_weight, config.min_ref_weight, config.max_ref_weight)

    # 4. Edit strength is driven by conflict, NOT by (1 - ref_weight): a small
    #    high-conflict edit must still be followed hard inside the mask.
    edit_strength = _clamp(conflict_score, 0.0, 1.0)

    inside_scale = _clamp(
        base_guidance_scale * (1.0 + config.inside_gain * edit_strength),
        config.min_scale,
        config.max_scale,
    )
    outside_scale = _clamp(
        base_guidance_scale * (1.0 - config.outside_damp * ref_weight),
        config.min_scale,
        config.max_scale,
    )

    if area_ratio(mask) == 0.0:
        notes.append("empty_mask")
    if area_ratio(mask) > 0.9:
        notes.append("near_global_edit")

    return ReferenceCoefficients(
        ref_weight=ref_weight,
        edit_strength=edit_strength,
        locality_score=locality_score,
        conflict_score=conflict_score,
        raw_similarity=raw_similarity,
        inside_scale=inside_scale,
        outside_scale=outside_scale,
        notes=tuple(notes),
    )


def legacy_reference_coefficient(
    *,
    prompt_embedding: torch.Tensor,
    source_image_embedding: torch.Tensor,
    edit_region_mask: torch.Tensor,
    min_w: float = 0.2,
    max_w: float = 0.9,
) -> float:
    """The unmodified pseudo-code formula, kept as the eval suite's baseline arm."""
    mask = as_soft_mask(edit_region_mask)
    locality_score = 1.0 - area_ratio(mask)
    region = extract_region_embedding(source_image_embedding, mask)
    similarity = cosine_similarity(prompt_embedding, region)
    conflict_score = 1.0 - similarity
    raw = 0.6 * locality_score + 0.4 * (1.0 - conflict_score)
    return _clamp(min_w + raw * (max_w - min_w), min_w, max_w)


def apply_region_guidance(
    edit_region_mask: torch.Tensor,
    coefficients: ReferenceCoefficients,
    *,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    """Spatial guidance-scale map: prompt-following inside, source-preserving outside."""
    mask = as_soft_mask(edit_region_mask)
    if height is not None and width is not None:
        mask = resize_mask(mask, height, width)
    return mask * coefficients.inside_scale + (1.0 - mask) * coefficients.outside_scale


def preservation_at_step(ref_weight: float, progress: float, *, ramp: float = 0.3) -> float:
    """Preservation strength for a denoise step, `progress` in [0,1].

    Early steps decide layout, so clamping the source too hard there stops the edit
    from forming at all; late steps are where detail leaks, so preservation ramps up.
    """
    progress = _clamp(progress, 0.0, 1.0)
    if ramp <= 0:
        return ref_weight
    scale = _clamp(progress / ramp, 0.0, 1.0) if progress < ramp else 1.0
    return ref_weight * scale


def blend_latents(
    edited_latents: torch.Tensor,
    source_latents: torch.Tensor,
    mask: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Keep the source outside the mask. This is what structurally stops leakage.

    `strength` scales how much of the outside is reclaimed (1.0 = fully restore the
    source outside the mask, 0.0 = leave the model's output alone).
    """
    latent_mask = resize_mask(mask, *edited_latents.shape[-2:]).to(
        device=edited_latents.device, dtype=edited_latents.dtype
    )
    source = source_latents.to(device=edited_latents.device, dtype=edited_latents.dtype)
    keep = (1.0 - latent_mask) * _clamp(strength, 0.0, 1.0)
    return edited_latents * (1.0 - keep) + source * keep


def apply_edge_blending(
    denoised_latent: torch.Tensor,
    source_latent: torch.Tensor,
    edit_mask: torch.Tensor,
    *,
    blend_width: int = 3,
    strength: float = 1.0,
) -> torch.Tensor:
    """Feathered composite of the edit over the source, avoiding a hard seam.

    A binary mask composites with a visible one-pixel step at the boundary; blurring
    the mask first turns that into a gradient. `blend_width` is the feather radius in
    latent cells - 2-3 is usually enough, since one latent cell is 8 image pixels.
    """
    from app.services.editing.masks import feather

    soft = feather(as_soft_mask(edit_mask), radius=max(int(blend_width), 0))
    return blend_latents(denoised_latent, source_latent, soft, strength=strength)
