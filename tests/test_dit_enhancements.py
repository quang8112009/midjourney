"""Tests for modern DiT guidance, token isolation, CFG rescaling, and quality tiers."""

from __future__ import annotations

import torch
from PIL import Image

from app.api.routes import GenerateRequest
from app.services.editing.edit_pipeline import (
    TIER_PROFILES,
    CFGRescaler,
    MaskAwareRefiner,
    QualityTier,
    compute_work_units,
)
from app.services.editing.layout_guidance import (
    MMDiTJointAttentionHook,
    MultiEncoderTokenIsolator,
    TwoPhaseSchedule,
)
from app.services.editing.semantic_planner import (
    NormalizedBox,
    PlannedObject,
    PlanSelfCheck,
    SemanticLayoutPlan,
    StyleHints,
)
from app.services.editing.vision_backbone import (
    BACKBONE_CROSS_DIMS,
    VisionFeatureProjector,
)


def test_multi_encoder_token_isolator_zero_bias_on_style_tokens():
    """Verify Category 5 invariant: non-entity and style tokens receive strictly 0.0 bias."""
    isolator = MultiEncoderTokenIsolator(clip_l_len=77, clip_g_len=77, t5_len=512)
    assert isolator.total_txt_len == 666
    assert isolator.t5_offset == 154

    # Map T5 entity indices (e.g. indices 5 and 6)
    t5_entities = [5, 6]
    joint_indices = isolator.map_entity_tokens_to_joint(t5_entities)
    assert joint_indices == [159, 160]

    mask = isolator.build_bias_mask(joint_indices)
    assert mask.shape == (666,)

    # CLIP-L (0:77) and CLIP-G (77:154) must be strictly False
    assert not mask[0:154].any()
    # Target T5 entity tokens must be True
    assert mask[159].item() is True
    assert mask[160].item() is True
    # Non-target T5 tokens must be False
    assert not mask[161:].any()


def test_mmdit_joint_attention_hook_modifies_only_cross_slice():
    """Verify that MMDiTJointAttentionHook only alters the off-diagonal Image->Text slice."""
    img_seq_len = 256
    txt_seq_len = 666
    total_seq_len = img_seq_len + txt_seq_len

    plan = SemanticLayoutPlan(
        prompt="A photo of a dog in a park",
        objects=(
            PlannedObject(
                label="dog",
                count=1,
                box=NormalizedBox(0.2, 0.2, 0.6, 0.6),
                token_indices=(159,),
            ),
        ),
        relations=(),
        style_hints=StyleHints(),
        self_check=PlanSelfCheck(
            is_valid=True,
            count_match=True,
            relation_match=True,
            ambiguity_detected=False,
        ),
    )

    hook = MMDiTJointAttentionHook(block_idx=10, guidance_strength=0.3)
    hook.set_plan(plan)
    hook.set_step_context(step=0, total_steps=28, progress=0.1)

    logits = torch.zeros(1, 4, total_seq_len, total_seq_len)
    modified = hook.modify_joint_attention_logits(logits.clone(), img_seq_len, txt_seq_len)

    # Self-attention slices (Image->Image and Text->Text) must remain completely untouched (0.0)
    assert torch.all(modified[:, :, :img_seq_len, :img_seq_len] == 0.0)
    assert torch.all(modified[:, :, img_seq_len:, img_seq_len:] == 0.0)
    assert torch.all(modified[:, :, img_seq_len:, :img_seq_len] == 0.0)

    # Cross-attention slice (Image->Text) must have positive bias on target token column 159
    cross_slice = modified[:, :, :img_seq_len, img_seq_len:]
    assert cross_slice[:, :, :, 159].max() > 0.0
    # Other token columns (e.g. style tokens) must remain exactly 0.0
    assert torch.all(cross_slice[:, :, :, 0:154] == 0.0)


def test_vision_feature_projector_cross_dimensions():
    """Verify VisionFeatureProjector projects across SD1.5, PixArt, SD3.5, and Flux."""
    vis_tokens = torch.randn(2, 64, 1024)

    for backbone, expected_dim in BACKBONE_CROSS_DIMS.items():
        projector = VisionFeatureProjector(vision_dim=1024, backbone=backbone)
        out = projector(vis_tokens)
        assert out.shape == (2, 64, expected_dim)

    # Test SwiGLU residual projector
    swiglu_projector = VisionFeatureProjector(
        vision_dim=1024, cross_attention_dim=2048, use_swiglu=True
    )
    swiglu_out = swiglu_projector(vis_tokens)
    assert swiglu_out.shape == (2, 64, 2048)


def test_cfg_rescaler_prevents_variance_blowout():
    """Verify that CFGRescaler stabilizes the standard deviation of guided predictions."""
    uncond = torch.randn(2, 4, 32, 32)
    cond = uncond + 0.5 * torch.randn(2, 4, 32, 32)
    guidance_scale = 10.0

    # Unscaled CFG
    standard_cfg = uncond + guidance_scale * (cond - uncond)
    # Rescaled CFG (phi = 0.70)
    rescaled_cfg = CFGRescaler.apply_rescaled_cfg(
        uncond, cond, guidance_scale=guidance_scale, rescale_factor=0.70
    )

    std_cond = cond.std(dim=(1, 2, 3))
    std_standard = standard_cfg.std(dim=(1, 2, 3))
    std_rescaled = rescaled_cfg.std(dim=(1, 2, 3))

    assert (std_standard > std_cond).all()
    assert (std_rescaled < std_standard).all()


def test_mask_aware_refiner_outside_mask_isolation():
    """Verify Category 4 invariant: Refiner strictly preserves pixels outside the edit mask."""
    # Create distinct base image (red) and refined image (blue)
    base_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    refined_img = Image.new("RGB", (64, 64), color=(0, 0, 255))

    # Center box edit mask (1 in center, 0 outside)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 20:44, 20:44] = 1.0

    refiner = MaskAwareRefiner(refiner_fn=lambda **kwargs: refined_img, default_strength=0.25)
    composite = refiner.refine_image(base_img, "prompt", edit_mask=mask)

    # Pixel (0, 0) is outside mask -> must remain strictly red (255, 0, 0)
    assert composite.getpixel((0, 0)) == (255, 0, 0)
    # Pixel (32, 32) is inside mask -> must be refined blue (0, 0, 255)
    assert composite.getpixel((32, 32)) == (0, 0, 255)


def test_quality_tiers_and_work_units():
    """Verify quality tier profiles and dynamic work-unit calculation."""
    assert TIER_PROFILES[QualityTier.PREVIEW].default_steps == 14
    assert TIER_PROFILES[QualityTier.FINAL].default_steps == 28

    wu_preview = compute_work_units(tier=QualityTier.PREVIEW, refiner_enabled=False)
    wu_final = compute_work_units(tier=QualityTier.FINAL, refiner_enabled=False)
    wu_final_refined = compute_work_units(tier=QualityTier.FINAL, refiner_enabled=True)

    assert wu_preview == 1
    assert wu_final == 2
    assert wu_final_refined == 3


def test_guidance_schedule_phase_transitions_across_steps():
    """Verify TwoPhaseSchedule and DepthAwareSchedule land phase transitions correctly."""
    schedule = TwoPhaseSchedule(schedule_cutoff=0.8)

    # Total steps = 14
    steps_14 = [schedule.weight(step / 14) for step in range(14)]
    # Early steps active, late steps 0.0
    assert steps_14[0] == 1.0
    assert steps_14[-1] == 0.0

    # Total steps = 28
    steps_28 = [schedule.weight(step / 28) for step in range(28)]
    assert steps_28[0] == 1.0
    assert steps_28[-1] == 0.0


def test_generate_request_supports_new_models_and_tiers():
    """Verify GenerateRequest schema parses SD 3.5 Large and quality tiers."""
    req = GenerateRequest(
        prompt="A cinematic landscape",
        model="stable-diffusion-3.5",
        tier="preview",
    )
    assert req.model == "stable-diffusion-3.5"
    assert req.num_inference_steps == 14
    assert req.width == 1024
    assert req.height == 1024

    req_final = GenerateRequest(
        prompt="A cinematic landscape",
        model="pixart-alpha",
        tier="final",
    )
    assert req_final.model == "pixart-alpha"
    assert req_final.num_inference_steps == 28
    assert req_final.width == 512
    assert req_final.height == 512


def test_resolve_token_budget_multibackbone():
    """Verify F6 fix: resolve_token_budget correctly resolves budget for all backbones."""
    from app.services.editing.layout_guidance import resolve_token_budget

    assert resolve_token_budget("stable-diffusion") == 77
    assert resolve_token_budget("sd15") == 77
    assert resolve_token_budget("pixart-alpha") == 120
    assert resolve_token_budget("pixart") == 120
    assert resolve_token_budget("stable-diffusion-3.5") == 666
    assert resolve_token_budget("sd35_large") == 666
    assert resolve_token_budget("flux-dev") == 512
    assert resolve_token_budget("flux") == 512
