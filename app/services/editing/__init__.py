"""Region-aware visual reasoning for DiT-based image editing.

Public surface:

* `plan_edit` - one cheap pre-denoise pass returning region, scope, and strengths
* `run_region_aware_edit` / `run_baseline_edit` - the improved loop and its baseline
* `evaluate_edit` - the metrics used in the before/after report
"""

from app.services.editing.adaptive_reference import (
    CoefficientConfig,
    ReferenceCoefficients,
    apply_region_guidance,
    blend_latents,
    compute_adaptive_reference_coefficient,
    legacy_reference_coefficient,
)
from app.services.editing.alignment import AlignmentReport, check_prompt_image_alignment
from app.services.editing.edit_pipeline import (
    EditTrace,
    run_baseline_edit,
    run_hybrid_edit,
    run_hybrid_generation,
    run_region_aware_edit,
    set_layout_guidance,
)
from app.services.editing.edit_planner import EditPlan, plan_edit
from app.services.editing.layout_guidance import (
    CosineSchedule,
    DepthAwareSchedule,
    GuidanceSchedule,
    LayoutGuidanceProcessor,
    LinearSchedule,
    TwoPhaseSchedule,
    apply_layout_guidance,
    build_layout_guidance_bias,
)
from app.services.editing.metrics import EditMetrics, evaluate_edit
from app.services.editing.prompt_intent import (
    EditInstruction,
    PromptIntent,
    PromptMode,
    SceneObject,
    TargetResolution,
    analyze_prompt,
    parse_instruction,
    resolve_target,
    split_instructions,
)
from app.services.editing.region_attention import (
    AttentionCapture,
    RegionAwareAttnProcessor,
    region_attention_bias,
)
from app.services.editing.semantic_planner import (
    AdaptiveGuidanceConfig,
    DensityField,
    EntityDepthPrior,
    EntityOverlap,
    GaussianSpatialPrior,
    NormalizedBox,
    PlannedObject,
    PlanSelfCheck,
    SemanticLayoutPlan,
    SpatialRelation,
    StyleHints,
    VisualContext,
    VisualEntity,
    compute_adaptive_guidance_strength,
    extract_style_hints,
    plan_semantic_layout,
)
from app.services.editing.vision_backbone import (
    BaseVisionBackbone,
    MockVisionBackbone,
    VisionFeatureProjector,
    VisualFeatureMap,
    get_vision_backbone,
)

__all__ = [
    "AdaptiveGuidanceConfig",
    "AlignmentReport",
    "AttentionCapture",
    "BaseVisionBackbone",
    "CoefficientConfig",
    "CosineSchedule",
    "DensityField",
    "DepthAwareSchedule",
    "EditInstruction",
    "EditMetrics",
    "EditPlan",
    "EditTrace",
    "EntityDepthPrior",
    "EntityOverlap",
    "GaussianSpatialPrior",
    "GuidanceSchedule",
    "LayoutGuidanceProcessor",
    "LinearSchedule",
    "MockVisionBackbone",
    "NormalizedBox",
    "PlanSelfCheck",
    "PlannedObject",
    "PromptIntent",
    "PromptMode",
    "ReferenceCoefficients",
    "SceneObject",
    "SemanticLayoutPlan",
    "SpatialRelation",
    "StyleHints",
    "TargetResolution",
    "TwoPhaseSchedule",
    "VisionFeatureProjector",
    "VisualContext",
    "VisualEntity",
    "VisualFeatureMap",
    "RegionAwareAttnProcessor",
    "analyze_prompt",
    "apply_layout_guidance",
    "apply_region_guidance",
    "blend_latents",
    "build_layout_guidance_bias",
    "check_prompt_image_alignment",
    "compute_adaptive_guidance_strength",
    "compute_adaptive_reference_coefficient",
    "evaluate_edit",
    "extract_style_hints",
    "get_vision_backbone",
    "legacy_reference_coefficient",
    "parse_instruction",
    "plan_edit",
    "plan_semantic_layout",
    "resolve_target",
    "region_attention_bias",
    "run_baseline_edit",
    "run_hybrid_edit",
    "run_hybrid_generation",
    "run_region_aware_edit",
    "set_layout_guidance",
    "split_instructions",
]
