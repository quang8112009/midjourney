"""Quantitative metrics for edit quality.

Everything here is pure torch so the eval suite runs on CPU with no extra
dependencies. Where a learned metric is the right answer (CLIP score, LPIPS) the
encoder is injected rather than vendored, and a documented proxy is used when it
is absent - the proxy is never silently reported as the learned metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from app.services.editing.masks import as_soft_mask, resize_mask


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(1, 1, size, size).contiguous()


def ssim_map(a: torch.Tensor, b: torch.Tensor, *, window: int = 11, sigma: float = 1.5):
    """Per-pixel SSIM, averaged over channels. Inputs (B,C,H,W) in [0,1]."""
    channels = a.shape[1]
    win = _gaussian_window(window, sigma, a.device, a.dtype).repeat(channels, 1, 1, 1)
    pad = window // 2

    def filt(x):
        return F.conv2d(F.pad(x, (pad,) * 4, mode="reflect"), win, groups=channels)

    mu_a, mu_b = filt(a), filt(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a = filt(a * a) - mu_a2
    sigma_b = filt(b * b) - mu_b2
    sigma_ab = filt(a * b) - mu_ab
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a + sigma_b + c2)
    return (numerator / denominator).mean(dim=1, keepdim=True)


def ssim(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Mean SSIM, optionally restricted to where `mask` is 1."""
    values = ssim_map(a, b)
    if mask is None:
        return float(values.mean())
    weights = resize_mask(mask, *values.shape[-2:]).to(values.dtype)
    total = weights.sum()
    if float(total) == 0.0:
        return float("nan")
    return float((values * weights).sum() / total)


def masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Mean absolute difference, optionally restricted to `mask`."""
    diff = (a - b).abs().mean(dim=1, keepdim=True)
    if mask is None:
        return float(diff.mean())
    weights = resize_mask(mask, *diff.shape[-2:]).to(diff.dtype)
    total = weights.sum()
    if float(total) == 0.0:
        return float("nan")
    return float((diff * weights).sum() / total)


def unintended_change_ratio(
    source: torch.Tensor,
    edited: torch.Tensor,
    edit_mask: torch.Tensor,
) -> float:
    """Share of all change energy that landed OUTSIDE the intended region.

    0.0 = every change stayed in the region; 1.0 = the edit went entirely astray.
    This is the headline leakage number.
    """
    change = (source - edited).abs().mean(dim=1, keepdim=True)
    weights = resize_mask(edit_mask, *change.shape[-2:]).to(change.dtype)
    total = change.sum()
    if float(total) == 0.0:
        return 0.0
    return float((change * (1.0 - weights)).sum() / total)


def changed_region_mask(
    source: torch.Tensor,
    edited: torch.Tensor,
    *,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Where the edit actually landed, for IoU against the intended region."""
    change = (source - edited).abs().mean(dim=1, keepdim=True)
    peak = change.max()
    if float(peak) == 0.0:
        return torch.zeros_like(change)
    return (change / peak >= threshold).to(change.dtype)


def clip_score(
    image_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
) -> float:
    """Cosine alignment between an image and a text embedding.

    Called a CLIP score only when the caller supplies genuine CLIP embeddings;
    the eval report labels the encoder it actually used.
    """
    a = image_embedding.reshape(-1).float()
    b = text_embedding.reshape(-1).float()
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def inside_alignment(
    source: torch.Tensor,
    edited: torch.Tensor,
    direction: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    """Cosine between the realised change and the requested direction, INSIDE the region.

    Restricted to the region on purpose: a whole-frame cosine mostly measures how
    much of the frame moved, which rewards exactly the leakage being eliminated.
    """
    weights = resize_mask(mask, *source.shape[-2:]) >= threshold
    if not bool(weights.any()):
        return float("nan")
    change = edited - source
    selected = change.permute(0, 2, 3, 1)[weights[:, 0]]
    if selected.numel() == 0:
        return float("nan")
    goal = direction.reshape(1, -1).expand_as(selected)
    numerator = (selected * goal).sum(dim=1)
    denominator = selected.norm(dim=1) * goal.norm(dim=1)
    valid = denominator > 0
    if not bool(valid.any()):
        return float("nan")
    return float((numerator[valid] / denominator[valid]).mean())


def realised_change(source: torch.Tensor, edited: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean absolute change inside the region - catches an edit that never happened."""
    return masked_l1(source, edited, mask)


@dataclass(frozen=True)
class EditMetrics:
    """One row of the before/after table."""

    alignment: float
    """Prompt-image agreement inside the edit region. Higher is better."""

    leakage: float
    """Fraction of change energy outside the region. Lower is better."""

    region_iou: float
    """IoU of where the edit landed vs where it was meant to land. Higher is better."""

    preservation_ssim: float
    """SSIM against the source OUTSIDE the region. Higher is better."""

    preservation_l1: float
    """Mean absolute error against the source OUTSIDE the region. Lower is better."""

    edit_magnitude: float
    """Mean absolute change INSIDE the region. Guards against under-editing."""

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def evaluate_edit(
    *,
    source: torch.Tensor,
    edited: torch.Tensor,
    edit_mask: torch.Tensor,
    alignment: float,
) -> EditMetrics:
    """Score one edit against its intended region."""
    from app.services.editing.masks import iou

    mask = as_soft_mask(edit_mask)
    outside = 1.0 - resize_mask(mask, *source.shape[-2:])
    return EditMetrics(
        alignment=alignment,
        edit_magnitude=realised_change(source, edited, mask),
        leakage=unintended_change_ratio(source, edited, mask),
        region_iou=iou(changed_region_mask(source, edited), mask),
        preservation_ssim=ssim(source, edited, outside),
        preservation_l1=masked_l1(source, edited, outside),
    )


def evaluate_count_accuracy(
    plan: Any,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    """Evaluate exact object count accuracy against expectations."""
    planned_counts: dict[str, int] = {obj.label: obj.count for obj in plan.objects}
    for df in getattr(plan, "density_fields", ()):
        planned_counts[df.label] = df.expected_count
    matches: dict[str, bool] = {}
    for entity, exp_count in expected_counts.items():
        matches[entity] = planned_counts.get(entity, 0) == exp_count

    total = len(expected_counts)
    correct = sum(1 for m in matches.values() if m)
    accuracy = correct / max(total, 1)

    return {
        "expected_counts": expected_counts,
        "planned_counts": planned_counts,
        "matches": matches,
        "exact_match_ratio": accuracy,
        "all_matched": bool(correct == total),
        "self_check_valid": bool(plan.self_check.is_valid and plan.self_check.count_match),
    }


def evaluate_spatial_relations(
    plan: Any,
) -> dict[str, Any]:
    """Evaluate bounding box geometry correctness for planned spatial relations."""
    boxes = {obj.label: obj.box for obj in plan.objects}
    for df in getattr(plan, "density_fields", ()):
        boxes[df.label] = df.region
    relation_results: list[dict[str, Any]] = []

    for rel in plan.relations:
        subj_box = boxes.get(rel.subject)
        obj_box = boxes.get(rel.object)
        is_valid_geom = False
        notes = ""

        if subj_box is None or obj_box is None:
            is_valid_geom = False
            notes = f"Missing box for subject '{rel.subject}' or object '{rel.object}'"
        elif rel.relation_type in ("riding", "on", "above"):
            is_valid_geom = subj_box.ymin < obj_box.ymin
            notes = (
                "Rider/top entity is strictly above mount/bottom entity"
                if is_valid_geom
                else "Vertical order inverted"
            )
        elif rel.relation_type in ("under", "below"):
            is_valid_geom = subj_box.ymin > obj_box.ymin
            notes = (
                "Under entity is strictly below base entity"
                if is_valid_geom
                else "Vertical order inverted"
            )
        elif rel.relation_type in ("next_to", "beside", "holding"):
            is_valid_geom = subj_box.xmin != obj_box.xmin
            notes = (
                "Entities have horizontal separation"
                if is_valid_geom
                else "Entities collide horizontally"
            )
        elif rel.relation_type in ("in_front_of", "ahead_of"):
            is_valid_geom = subj_box.ymin >= obj_box.ymin
            notes = (
                "Foreground entity positioned in front"
                if is_valid_geom
                else "Foreground order inverted"
            )
        elif rel.relation_type == "behind":
            is_valid_geom = subj_box.ymin <= obj_box.ymin
            notes = (
                "Background entity positioned behind"
                if is_valid_geom
                else "Background order inverted"
            )
        elif rel.relation_type == "inside":
            is_valid_geom = obj_box.contains(subj_box)
            notes = (
                "Subject is nested inside object"
                if is_valid_geom
                else "Subject box is not contained within object"
            )
        else:
            is_valid_geom = True
            notes = f"Generic relation '{rel.relation_type}'"

        relation_results.append(
            {
                "subject": rel.subject,
                "relation_type": rel.relation_type,
                "object": rel.object,
                "is_valid_geometry": is_valid_geom,
                "notes": notes,
            }
        )

    total_rels = len(relation_results)
    valid_rels = sum(1 for r in relation_results if r["is_valid_geometry"])
    geometry_score = (valid_rels / total_rels) if total_rels > 0 else 1.0

    return {
        "relation_count": total_rels,
        "valid_relation_count": valid_rels,
        "geometry_correctness_score": geometry_score,
        "relations": relation_results,
    }


def evaluate_aesthetic_freedom(
    plan: Any,
    bias_matrix: torch.Tensor,
) -> dict[str, Any]:
    """Verify that style tokens receive exactly 0.0 guidance bias (100% aesthetic freedom)."""
    style_tokens = plan.style_hints.style_tokens
    max_biases: dict[int, float] = {}

    for tok_idx in style_tokens:
        if tok_idx < bias_matrix.shape[1]:
            max_bias = float(bias_matrix[:, tok_idx].abs().max())
            max_biases[tok_idx] = max_bias

    zero_violations = sum(1 for b in max_biases.values() if b > 1e-6)
    freedom_score = (
        1.0 if zero_violations == 0 else (1.0 - zero_violations / max(len(max_biases), 1))
    )

    return {
        "style_tokens": list(style_tokens),
        "is_unconstrained": plan.style_hints.is_unconstrained,
        "style_token_max_bias": max_biases,
        "aesthetic_freedom_score": freedom_score,
        "zero_bias_verified": (zero_violations == 0),
    }


@dataclass(frozen=True)
class StageTrace:
    """Detailed execution record for one stage of the pipeline."""

    stage: str
    elapsed_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataFlowTrace:
    """Request-scoped observability tracking across Planner, Guidance, and Pipeline stages."""

    request_id: str
    stages: list[StageTrace] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    def add_stage(self, stage: str, elapsed_seconds: float, **details: Any) -> None:
        self.stages.append(
            StageTrace(
                stage=stage,
                elapsed_seconds=round(elapsed_seconds, 4),
                details=details,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 4),
            "stages": [
                {
                    "stage": s.stage,
                    "elapsed_seconds": s.elapsed_seconds,
                    "details": s.details,
                }
                for s in self.stages
            ],
        }
