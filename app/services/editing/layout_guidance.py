"""Soft Cross-Attention Layout Guidance for Diffusion Transformers (DiT).

Biaes cross-attention maps of entity and relation tokens toward their planned
spatial regions with gentle positive logit bias (e.g. +0.3).

Crucially, style, mood, lighting, and medium tokens receive ZERO bias, allowing
the DiT's learned aesthetic priors (Midjourney-style creativity) to operate
unconstrained on visual fidelity and texture.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Literal

import torch

from app.services.editing.masks import as_soft_mask, feather, resize_mask
from app.services.editing.region_attention import latent_grid
from app.services.editing.semantic_planner import (
    AdaptiveGuidanceConfig,
    DensityField,
    GaussianSpatialPrior,
    NormalizedBox,
    SemanticLayoutPlan,
)
from app.services.editing.vision_backbone import VisionFeatureProjector

logger = logging.getLogger(__name__)

DEFAULT_GUIDANCE_STRENGTH = 0.3


class GuidanceSchedule(ABC):
    """Abstract reverse-time guidance schedule."""

    @abstractmethod
    def weight(
        self,
        progress: float,
        layer_index: int | None = None,
        total_layers: int | None = None,
        mu_z: float | None = None,
    ) -> float:
        """Compute multiplier in [0.0, 2.0] for guidance at given progress, layer, and depth."""
        pass


BACKBONE_TOKEN_BUDGETS = {
    "sd15": 77,
    "stable-diffusion": 77,
    "pixart": 120,
    "pixart_alpha": 120,
    "pixart-alpha": 120,
    "sd35": 666,
    "sd35_large": 666,
    "stable-diffusion-3.5": 666,
    "flux": 512,
    "flux_dev": 512,
    "flux-dev": 512,
}


def resolve_token_budget(backbone: str = "sd15") -> int:
    """Resolve the maximum token context length for the given model backbone.

    Fixes F6: SD 1.5 uses 77 (CLIP-L), PixArt uses 120 (T5-XXL), SD 3.5 Large uses 666
    (CLIP-L 77 + CLIP-G 77 + T5 512), and FLUX.1 uses 512 (T5-XXL).
    """
    normalized = backbone.lower().replace("_", "-")
    if normalized in BACKBONE_TOKEN_BUDGETS:
        return BACKBONE_TOKEN_BUDGETS[normalized]
    if "3.5" in normalized or "sd3" in normalized:
        return 666
    if "flux" in normalized:
        return 512
    if "pixart" in normalized:
        return 120
    return 77


class MultiEncoderTokenIsolator:
    """Enforces 0.0-bias invariant across CLIP-L, CLIP-G, and T5-XXL token sequences.

    Guarantees zero spatial gradient leakage into style, lighting, camera, and mood tokens
    across multi-stream MMDiT architectures (e.g. SD 3.5 Large).
    """

    def __init__(
        self,
        clip_l_len: int = 77,
        clip_g_len: int = 77,
        t5_len: int = 512,
    ) -> None:
        self.clip_l_len = clip_l_len
        self.clip_g_len = clip_g_len
        self.t5_len = t5_len
        self.t5_offset = clip_l_len + clip_g_len
        self.total_txt_len = clip_l_len + clip_g_len + t5_len

    def map_entity_tokens_to_joint(
        self,
        t5_token_indices: list[int],
    ) -> list[int]:
        """Maps T5-local entity indices to absolute columns in joint attention context."""
        joint_indices = []
        for idx in t5_token_indices:
            if 0 <= idx < self.t5_len:
                joint_indices.append(self.t5_offset + idx)
        return joint_indices

    def build_bias_mask(
        self,
        active_joint_indices: list[int],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Returns boolean mask of length total_txt_len where True = allowed for guidance.

        All non-entity tokens (including 100% of CLIP-L and CLIP-G) are strictly False (0.0 bias).
        """
        mask = torch.zeros(self.total_txt_len, dtype=torch.bool, device=device)
        for idx in active_joint_indices:
            if idx >= self.t5_offset:
                mask[idx] = True
        return mask


class MMDiTJointAttentionHook:
    """Attention hook for joint self/cross-attention blocks in MMDiT (SD 3.5 Large).

    Intercepts the pre-softmax joint attention logits and modifies only the off-diagonal
    Image -> Text slice corresponding to planned spatial entities.
    """

    def __init__(
        self,
        block_idx: int,
        schedule: GuidanceSchedule | None = None,
        isolator: MultiEncoderTokenIsolator | None = None,
        num_patches: int = 4096,
        grid_size: tuple[int, int] = (64, 64),
        guidance_strength: float = DEFAULT_GUIDANCE_STRENGTH,
    ) -> None:
        self.block_idx = block_idx
        self.schedule = schedule or TwoPhaseSchedule()
        self.isolator = isolator or MultiEncoderTokenIsolator()
        self.num_patches = num_patches
        self.grid_h, self.grid_w = grid_size
        self.guidance_strength = guidance_strength
        self.current_step = 0
        self.total_steps = 28
        self.current_progress = 0.0
        self.plan: SemanticLayoutPlan | None = None
        self.enabled = True

    def set_step_context(self, step: int, total_steps: int, progress: float) -> None:
        self.current_step = step
        self.total_steps = total_steps
        self.current_progress = progress

    def set_plan(self, plan: SemanticLayoutPlan | None) -> None:
        self.plan = plan

    def modify_joint_attention_logits(
        self,
        attn_logits: torch.Tensor,
        img_seq_len: int = 4096,
        txt_seq_len: int = 666,
    ) -> torch.Tensor:
        """Modify off-diagonal image->text logits in-place or return biased tensor."""
        if not self.enabled or self.plan is None or self.guidance_strength <= 0:
            return attn_logits

        sched_weight = self.schedule.weight(self.current_progress)
        if sched_weight <= 1e-7:
            return attn_logits

        device = attn_logits.device
        dtype = attn_logits.dtype

        # Off-diagonal slice: Image queries to Text keys
        bias_matrix = build_layout_guidance_bias(
            self.plan,
            num_image_tokens=img_seq_len,
            num_text_tokens=txt_seq_len,
            guidance_strength=self.guidance_strength * sched_weight,
            device=device,
            dtype=dtype,
        )

        cross_slice = attn_logits[..., :img_seq_len, img_seq_len : img_seq_len + txt_seq_len]
        attn_logits[..., :img_seq_len, img_seq_len : img_seq_len + txt_seq_len] = (
            cross_slice + bias_matrix
        )
        return attn_logits


class TwoPhaseSchedule(GuidanceSchedule):
    """Standard two-phase schedule (active 0-cutoff, released cutoff-1.0)."""

    def __init__(self, schedule_cutoff: float = 0.8):
        self.schedule_cutoff = schedule_cutoff

    def weight(
        self,
        progress: float,
        layer_index: int | None = None,
        total_layers: int | None = None,
        mu_z: float | None = None,
    ) -> float:
        return 1.0 if progress < self.schedule_cutoff else 0.0


class DepthAwareSchedule(GuidanceSchedule):
    """Depth-conditioned schedule modulating foreground vs background across steps & layers."""

    def __init__(
        self,
        schedule_cutoff: float = 0.8,
        depth_decay: float = 0.5,
    ):
        self.schedule_cutoff = schedule_cutoff
        self.depth_decay = depth_decay

    def weight(
        self,
        progress: float,
        layer_index: int | None = None,
        total_layers: int | None = None,
        mu_z: float | None = None,
    ) -> float:
        if progress >= self.schedule_cutoff:
            return 0.0

        w = 1.0
        if mu_z is not None:
            if mu_z < 0.4:
                w *= 1.0 + 0.2 * (1.0 - mu_z)
            elif mu_z > 0.6:
                w *= max(0.4, 1.0 - progress * self.depth_decay)

        if layer_index is not None and total_layers and total_layers > 0:
            rel_layer = layer_index / float(total_layers)
            w *= 0.8 + 0.4 * math.sin(rel_layer * math.pi)

        return max(0.0, min(2.0, w))


class LinearSchedule(GuidanceSchedule):
    """Linear decaying guidance schedule."""

    def __init__(self, start_weight: float = 1.0, end_weight: float = 0.0):
        self.start_weight = start_weight
        self.end_weight = end_weight

    def weight(
        self,
        progress: float,
        layer_index: int | None = None,
        total_layers: int | None = None,
        mu_z: float | None = None,
    ) -> float:
        p = max(0.0, min(1.0, progress))
        return self.start_weight * (1.0 - p) + self.end_weight * p


class CosineSchedule(GuidanceSchedule):
    """Cosine smooth transition guidance schedule."""

    def __init__(self, schedule_cutoff: float = 0.8):
        self.schedule_cutoff = schedule_cutoff

    def weight(
        self,
        progress: float,
        layer_index: int | None = None,
        total_layers: int | None = None,
        mu_z: float | None = None,
    ) -> float:
        if progress >= self.schedule_cutoff:
            return 0.0
        rel_p = progress / max(1e-4, self.schedule_cutoff)
        return 0.5 * (1.0 + math.cos(rel_p * math.pi))


def build_layout_guidance_bias(
    plan: SemanticLayoutPlan,
    *,
    num_image_tokens: int,
    num_text_tokens: int,
    guidance_strength: float = DEFAULT_GUIDANCE_STRENGTH,
    aspect: float = 1.0,
    feather_radius: int = 1,
    guidance_mode: Literal["gaussian", "box"] | None = None,
    depth_guidance_enabled: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build additive cross-attention bias matrix (num_image_tokens, num_text_tokens).

    Entries are >= 0. Inside an object's planned region, that object's token columns
    receive +guidance_strength in logit space. Unconstrained style tokens receive 0.0.
    Supports density fields, 3D depth-aware Gaussian heatmaps, and rectangular box masks.
    """
    density_fields = getattr(plan, "density_fields", ())
    if guidance_strength <= 0 or (not plan.objects and not density_fields):
        return torch.zeros(num_image_tokens, num_text_tokens, device=device, dtype=dtype)

    resolved_mode = guidance_mode or getattr(plan, "guidance_mode", "gaussian")
    bias = torch.zeros(num_image_tokens, num_text_tokens, device=device, dtype=dtype)
    grid_h, grid_w = latent_grid(num_image_tokens, max(float(aspect), 1e-3))

    # Pre-render raw spatial weights for all objects to evaluate depth-aware overlap modulation
    obj_weights: list[torch.Tensor] = []
    for obj in plan.objects:
        if resolved_mode == "gaussian" and getattr(obj, "gaussian", None) is not None:
            heatmap = obj.gaussian.to_heatmap(grid_h, grid_w, device=device, dtype=dtype)
            weights = heatmap.reshape(-1)[:num_image_tokens]
        else:
            raw_mask = obj.box.to_mask(grid_h, grid_w, device=device, dtype=dtype)
            if feather_radius > 0:
                soft_mask = feather(raw_mask, radius=feather_radius)
            else:
                soft_mask = raw_mask
            weights = soft_mask.reshape(-1)[:num_image_tokens].to(device=device, dtype=dtype)
        obj_weights.append(weights)

    # Apply soft depth-aware occlusion modulation across overlapping objects
    if depth_guidance_enabled and len(plan.objects) > 1:
        for i in range(len(plan.objects)):
            for j in range(len(plan.objects)):
                if i == j:
                    continue
                z_i = plan.objects[i].gaussian.mu_z if plan.objects[i].gaussian else 0.5
                z_j = plan.objects[j].gaussian.mu_z if plan.objects[j].gaussian else 0.5
                # If object j is in front of object i (z_j < z_i), softly modulate object i
                if z_j < z_i - 0.05:
                    delta_z = z_i - z_j
                    occlusion = torch.clamp(obj_weights[j] * float(delta_z * 1.5), 0.0, 0.70)
                    obj_weights[i] = obj_weights[i] * (1.0 - occlusion)

    # 1. Discrete planned objects
    for idx, obj in enumerate(plan.objects):
        if not obj.token_indices:
            continue
        spatial_weights = obj_weights[idx]
        for token_idx in obj.token_indices:
            if 0 <= token_idx < num_text_tokens:
                bias[:, token_idx] += spatial_weights * float(guidance_strength)

    # 2. Continuous density fields (crowds, swarms, fields, mass entities)
    for df in density_fields:
        if not df.token_indices:
            continue
        heatmap = df.to_heatmap(grid_h, grid_w, device=device, dtype=dtype)
        spatial_weights = heatmap.reshape(-1)[:num_image_tokens]

        for token_idx in df.token_indices:
            if 0 <= token_idx < num_text_tokens:
                bias[:, token_idx] += spatial_weights * float(guidance_strength)

    return bias


def apply_layout_guidance(
    cross_attn_logits: torch.Tensor,
    plan: SemanticLayoutPlan | None = None,
    token_to_region_map: (
        dict[int, NormalizedBox | GaussianSpatialPrior | DensityField | torch.Tensor | None] | None
    ) = None,
    *,
    guidance_strength: float = DEFAULT_GUIDANCE_STRENGTH,
    aspect: float = 1.0,
    feather_radius: int = 1,
    guidance_mode: Literal["gaussian", "box"] | None = None,
    depth_guidance_enabled: bool = True,
    is_image_first: bool | None = None,
) -> torch.Tensor:
    """Apply soft positive bias to cross-attention logits.

    Signature compatible with both image-first and text-first conventions:
    - Image-first: (..., num_image_tokens, num_text_tokens)
    - Text-first:  (..., num_text_tokens, num_image_tokens)
    """
    if guidance_strength <= 0:
        return cross_attn_logits

    device = cross_attn_logits.device
    dtype = cross_attn_logits.dtype

    if cross_attn_logits.ndim < 2:
        return cross_attn_logits

    # Determine layout dimensions and orientation
    if is_image_first is None:
        is_image_first = cross_attn_logits.shape[-2] >= cross_attn_logits.shape[-1]

    if is_image_first:
        num_image_tokens = cross_attn_logits.shape[-2]
        num_text_tokens = cross_attn_logits.shape[-1]
    else:
        num_image_tokens = cross_attn_logits.shape[-1]
        num_text_tokens = cross_attn_logits.shape[-2]

    grid_h, grid_w = latent_grid(num_image_tokens, aspect)
    bias = torch.zeros(num_image_tokens, num_text_tokens, device=device, dtype=dtype)

    # Source 1: If plan is provided
    if plan is not None:
        bias_matrix = build_layout_guidance_bias(
            plan,
            num_image_tokens=num_image_tokens,
            num_text_tokens=num_text_tokens,
            guidance_strength=guidance_strength,
            aspect=aspect,
            feather_radius=feather_radius,
            guidance_mode=guidance_mode,
            depth_guidance_enabled=depth_guidance_enabled,
            device=device,
            dtype=dtype,
        )
        if not is_image_first:
            bias_matrix = bias_matrix.transpose(-1, -2)
        return cross_attn_logits + bias_matrix

    # Source 2: Direct token_to_region_map
    if token_to_region_map:
        for token_idx, region in token_to_region_map.items():
            if region is None or token_idx >= num_text_tokens:
                continue

            if isinstance(region, DensityField):
                heatmap = region.to_heatmap(grid_h, grid_w, device=device, dtype=dtype)
                weights = heatmap.reshape(-1)[:num_image_tokens]
            elif isinstance(region, GaussianSpatialPrior):
                heatmap = region.to_heatmap(grid_h, grid_w, device=device, dtype=dtype)
                weights = heatmap.reshape(-1)[:num_image_tokens]
            elif isinstance(region, NormalizedBox):
                raw_mask = region.to_mask(grid_h, grid_w, device=device, dtype=dtype)
                soft_mask = (
                    feather(raw_mask, radius=feather_radius)
                    if feather_radius > 0
                    else raw_mask
                )
                weights = soft_mask.reshape(-1)[:num_image_tokens]
            elif isinstance(region, torch.Tensor):
                soft_mask = resize_mask(as_soft_mask(region), grid_h, grid_w).to(
                    device=device, dtype=dtype
                )
                weights = soft_mask.reshape(-1)[:num_image_tokens]
            else:
                continue

            bias[:, token_idx] += weights * float(guidance_strength)

    if not is_image_first:
        bias = bias.transpose(-1, -2)

    return cross_attn_logits + bias


class LayoutGuidanceProcessor:
    """Attention processor hook injecting soft layout guidance and visual features into DiT."""

    def __init__(
        self,
        base_processor: Any,
        plan: SemanticLayoutPlan | None = None,
        *,
        guidance_strength: float = DEFAULT_GUIDANCE_STRENGTH,
        schedule_cutoff: float = 0.8,
        aspect: float = 1.0,
        feather_radius: int = 1,
        guidance_mode: Literal["gaussian", "box"] = "gaussian",
        adaptive_guidance: bool = True,
        adaptive_config: AdaptiveGuidanceConfig | None = None,
        schedule: GuidanceSchedule | None = None,
        depth_guidance_enabled: bool = True,
        visual_cross_attn_enabled: bool = True,
        visual_feature_strength: float = 0.25,
    ):
        self.base_processor = base_processor
        self.plan = plan
        self.guidance_strength = guidance_strength
        self.schedule_cutoff = schedule_cutoff
        self.aspect = aspect
        self.feather_radius = feather_radius
        self.guidance_mode = guidance_mode
        self.adaptive_guidance = adaptive_guidance
        self.adaptive_config = adaptive_config or AdaptiveGuidanceConfig()
        self.schedule = schedule or TwoPhaseSchedule(schedule_cutoff=schedule_cutoff)
        self.depth_guidance_enabled = depth_guidance_enabled
        self.visual_cross_attn_enabled = visual_cross_attn_enabled
        self.visual_feature_strength = visual_feature_strength
        self._base_bias: torch.Tensor | None = None
        self._cached_proj_vis: torch.Tensor | None = None
        self._cached_vis_id: int | None = None
        self._active = True
        self._current_progress: float = 0.0
        self._projectors: dict[int, VisionFeatureProjector] = {}

        if self.plan is not None:
            if getattr(self.plan, "guidance_mode", None):
                self.guidance_mode = self.plan.guidance_mode
            if self.adaptive_guidance and getattr(self.plan, "adaptive_gamma", None) is not None:
                self.guidance_strength = self.plan.adaptive_gamma

    def set_plan(self, plan: SemanticLayoutPlan | None) -> None:
        self.plan = plan
        self._base_bias = None
        self._cached_proj_vis = None
        self._cached_vis_id = None
        if plan is not None:
            if getattr(plan, "guidance_mode", None):
                self.guidance_mode = plan.guidance_mode
            if self.adaptive_guidance and getattr(plan, "adaptive_gamma", None) is not None:
                self.guidance_strength = plan.adaptive_gamma

    def set_step_progress(self, progress: float) -> None:
        """Update reverse diffusion progress and modulate schedule."""
        self._current_progress = progress
        weight = self.schedule.weight(progress)
        self._active = weight > 0.0

    def _project_visual_features(
        self,
        features: torch.Tensor,
        target_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project visual spatial features (B, S_vis, D_vis) -> (B, S_vis, D_target)."""
        feat_dim = features.shape[-1]
        if feat_dim not in self._projectors:
            self._projectors[feat_dim] = VisionFeatureProjector(
                vision_dim=feat_dim, cross_attention_dim=target_dim
            ).to(device=device, dtype=dtype)

        projector = self._projectors[feat_dim].to(device=device, dtype=dtype)
        return projector(features.to(device=device, dtype=dtype))

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_index: int | None = None,
        total_layers: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # 1. Multi-Modal Visual Feature Map Cross-Attention Injection
        if (
            self.visual_cross_attn_enabled
            and self.plan is not None
            and self.plan.visual_context is not None
            and encoder_hidden_states is not None
        ):
            vis_ctx = self.plan.visual_context
            vis_features = getattr(vis_ctx, "spatial_features", None)
            if vis_features is None and getattr(vis_ctx, "feature_map", None) is not None:
                vis_features = vis_ctx.feature_map.to_flattened()

            if vis_features is not None:
                target_dim = encoder_hidden_states.shape[-1]
                batch_size = hidden_states.shape[0]
                vis_id = id(vis_features)
                if (
                    self._cached_proj_vis is None
                    or self._cached_vis_id != vis_id
                    or self._cached_proj_vis.shape[0] != batch_size
                    or self._cached_proj_vis.shape[-1] != target_dim
                    or self._cached_proj_vis.device != hidden_states.device
                    or self._cached_proj_vis.dtype != hidden_states.dtype
                ):
                    proj_vis = self._project_visual_features(
                        vis_features, target_dim, hidden_states.device, hidden_states.dtype
                    )
                    if proj_vis.shape[0] == 1 and batch_size > 1:
                        proj_vis = proj_vis.expand(batch_size, -1, -1)
                    self._cached_proj_vis = proj_vis
                    self._cached_vis_id = vis_id
                else:
                    proj_vis = self._cached_proj_vis

                # Concatenate visual reference tokens with text tokens
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, proj_vis * float(self.visual_feature_strength)], dim=1
                )

        # 2. Soft Layout & Gaussian Depth-Aware Guidance Injection
        sched_weight = self.schedule.weight(
            self._current_progress,
            layer_index=layer_index,
            total_layers=total_layers,
        )

        if (
            self._active
            and sched_weight > 0
            and encoder_hidden_states is not None
            and self.plan is not None
            and self.guidance_strength > 0
        ):
            if hidden_states.ndim == 4:
                num_image_tokens = hidden_states.shape[2] * hidden_states.shape[3]
                aspect = hidden_states.shape[3] / max(hidden_states.shape[2], 1)
            else:
                num_image_tokens = hidden_states.shape[1]
                aspect = self.aspect

            num_text_tokens = encoder_hidden_states.shape[1]
            batch_size = hidden_states.shape[0]

            # Pre-compute static normalized base heatmap once; scale dynamically in O(1) per step
            if (
                self._base_bias is None
                or self._base_bias.shape != (num_image_tokens, num_text_tokens)
                or self._base_bias.device != hidden_states.device
                or self._base_bias.dtype != hidden_states.dtype
            ):
                self._base_bias = build_layout_guidance_bias(
                    self.plan,
                    num_image_tokens=num_image_tokens,
                    num_text_tokens=num_text_tokens,
                    guidance_strength=1.0,
                    aspect=aspect,
                    feather_radius=self.feather_radius,
                    guidance_mode=self.guidance_mode,
                    depth_guidance_enabled=self.depth_guidance_enabled,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

            effective_strength = float(self.guidance_strength * sched_weight)
            bias_4d = (self._base_bias * effective_strength).unsqueeze(0).unsqueeze(0)

            if batch_size > 1:
                bias_4d = bias_4d.expand(batch_size, 1, -1, -1)

            if attention_mask is None:
                attention_mask = bias_4d.squeeze(1) if hidden_states.ndim != 4 else bias_4d
            else:
                if attention_mask.dtype == torch.bool:
                    float_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype)
                    float_mask = float_mask.masked_fill(~attention_mask, -10000.0)
                    attention_mask = float_mask

                if attention_mask.ndim == 2:  # (B, K)
                    attention_mask = attention_mask[:, None, None, :] + bias_4d
                elif attention_mask.ndim == 3:  # (B, Q, K)
                    attention_mask = attention_mask + bias_4d.squeeze(1)
                elif attention_mask.ndim == 4:  # (B, H, Q, K)
                    attention_mask = attention_mask + bias_4d

        return self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


def compute_attention_entropy(
    logits: torch.Tensor, dim: int = -1, eps: float = 1e-12
) -> torch.Tensor:
    """Compute Shannon entropy H(p) = -sum(p * log(p)) numerically stable."""
    logits_f32 = logits.float()
    probs = logits_f32.softmax(dim=dim)
    log_probs = logits_f32.log_softmax(dim=dim)
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy.to(dtype=logits.dtype)


def compute_gradient_flow(
    logits: torch.Tensor, target_token_idx: int, dim: int = -1
) -> torch.Tensor:
    """Compute analytical gradient magnitude dp_j / dz_j = p_j * (1 - p_j)."""
    if target_token_idx < 0 or target_token_idx >= logits.shape[dim]:
        raise IndexError(
            f"target_token_idx {target_token_idx} out of bounds for dim {dim}"
        )
    probs = logits.softmax(dim=dim)
    p_target = probs.select(dim=dim, index=target_token_idx)
    return p_target * (1.0 - p_target)


def ablation_soft_vs_hard(
    logits: torch.Tensor,
    box: NormalizedBox,
    target_token_idx: int,
    *,
    soft_strength: float = DEFAULT_GUIDANCE_STRENGTH,
    hard_penalty: float = -12.0,
    aspect: float = 1.0,
    is_image_first: bool = True,
) -> dict[str, float]:
    """Perform quantitative comparison between soft guidance (+0.3) vs hard masking (-12.0).

    Returns a dict with entropy, gradient flow, and retention metrics.
    """
    num_image_tokens = logits.shape[-2] if is_image_first else logits.shape[-1]
    grid_h, grid_w = latent_grid(num_image_tokens, max(float(aspect), 1e-3))
    raw_box_mask = box.to_mask(grid_h, grid_w, device=logits.device, dtype=logits.dtype)
    mask = raw_box_mask.reshape(-1)[:num_image_tokens]
    outside = mask < 0.5

    def safe_outside_mean(tensor: torch.Tensor) -> float:
        if not outside.any():
            return 0.0
        return float(tensor[..., outside].mean().item())

    # 1. Unconstrained Baseline
    token_dim = -1 if is_image_first else -2
    base_entropy = safe_outside_mean(compute_attention_entropy(logits, dim=token_dim))
    base_grad = safe_outside_mean(
        compute_gradient_flow(logits, target_token_idx, dim=token_dim)
    )

    # 2. Soft Guidance (+0.3 inside box, 0 outside)
    soft_logits = logits.clone()
    if is_image_first:
        soft_logits[..., target_token_idx] += mask * soft_strength
    else:
        soft_logits[..., target_token_idx, :] += mask * soft_strength
    soft_entropy = safe_outside_mean(compute_attention_entropy(soft_logits, dim=token_dim))
    soft_grad = safe_outside_mean(
        compute_gradient_flow(soft_logits, target_token_idx, dim=token_dim)
    )

    # 3. Hard Masking (-12 outside box)
    hard_logits = logits.clone()
    if is_image_first:
        hard_logits[..., target_token_idx] += (1.0 - mask) * hard_penalty
    else:
        hard_logits[..., target_token_idx, :] += (1.0 - mask) * hard_penalty
    hard_entropy = safe_outside_mean(compute_attention_entropy(hard_logits, dim=token_dim))
    hard_grad = safe_outside_mean(
        compute_gradient_flow(hard_logits, target_token_idx, dim=token_dim)
    )

    return {
        "baseline_entropy_outside": base_entropy,
        "soft_entropy_outside": soft_entropy,
        "hard_entropy_outside": hard_entropy,
        "soft_entropy_retention": soft_entropy / max(base_entropy, 1e-9),
        "hard_entropy_retention": hard_entropy / max(base_entropy, 1e-9),
        "baseline_gradient_outside": base_grad,
        "soft_gradient_outside": soft_grad,
        "hard_gradient_outside": hard_grad,
        "soft_gradient_retention": soft_grad / max(base_grad, 1e-9),
        "hard_gradient_retention": hard_grad / max(base_grad, 1e-9),
    }
