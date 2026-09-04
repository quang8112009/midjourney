"""Tests for Per-Relation Guidance Strength Architecture.

Asserts that:
1. Lateral relations receive high strength (6.0) by default.
2. Depth relations are disabled by default (0.0).
3. Vertical-On relations are disabled by default (0.0).
4. Vertical-Under relations retain default strength (0.3).
5. Legacy DEPTH_GUIDANCE_STRENGTH is preserved as fallback.
"""

from __future__ import annotations

from app.core.config import settings
from app.services.editing.layout_guidance import LayoutGuidanceProcessor
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout


def test_per_relation_guidance_defaults() -> None:
    assert settings.LATERAL_GUIDANCE_STRENGTH == 6.0
    assert settings.DEPTH_RELATION_GUIDANCE_STRENGTH == 0.0
    assert settings.VERTICAL_ON_GUIDANCE_STRENGTH == 0.0
    assert settings.VERTICAL_UNDER_GUIDANCE_STRENGTH == 0.3
    # Legacy global fallback remains accessible
    assert settings.DEPTH_GUIDANCE_STRENGTH == 0.3


def test_get_relation_guidance_strength_mapping() -> None:
    # Lateral
    assert settings.get_relation_guidance_strength("left_of") == 6.0
    assert settings.get_relation_guidance_strength("right_of") == 6.0
    assert settings.get_relation_guidance_strength("beside") == 6.0
    assert settings.get_relation_guidance_strength("next_to") == 6.0

    # Depth (off by default)
    assert settings.get_relation_guidance_strength("in_front_of") == 0.0
    assert settings.get_relation_guidance_strength("behind") == 0.0
    assert settings.get_relation_guidance_strength("far_in_front_of") == 0.0

    # Vertical-On (off by default)
    assert settings.get_relation_guidance_strength("on") == 0.0
    assert settings.get_relation_guidance_strength("on_top_of") == 0.0
    assert settings.get_relation_guidance_strength("resting_on") == 0.0
    assert settings.get_relation_guidance_strength("perched_on") == 0.0

    # Vertical-Under (default 0.3)
    assert settings.get_relation_guidance_strength("under") == 0.3
    assert settings.get_relation_guidance_strength("below") == 0.3
    assert settings.get_relation_guidance_strength("underneath") == 0.3

    # Fallback / None
    assert settings.get_relation_guidance_strength(None) == 0.3


def test_planner_assigns_relation_specific_adaptive_gamma() -> None:
    # 1. Lateral -> 6.0
    intent_lat = analyze_prompt(
        "a yellow banana to the left of a green apple on a table", mode="generate"
    )
    plan_lat = plan_semantic_layout(intent_lat)
    assert plan_lat.adaptive_gamma == 6.0

    # 2. Depth -> 0.0
    intent_depth = analyze_prompt(
        "a red cube in front of a blue sphere on a marble floor", mode="generate"
    )
    plan_depth = plan_semantic_layout(intent_depth)
    assert plan_depth.adaptive_gamma == 0.0

    # 3. Vertical-On -> 0.0
    intent_on = analyze_prompt("a white coffee cup on top of a stack of books", mode="generate")
    plan_on = plan_semantic_layout(intent_on)
    assert plan_on.adaptive_gamma == 0.0

    # 4. Vertical-Under -> 0.3
    intent_under = analyze_prompt("a black cat sitting under a wooden chair", mode="generate")
    plan_under = plan_semantic_layout(intent_under)
    assert plan_under.adaptive_gamma == 0.3


def test_layout_processor_respects_per_relation_strengths() -> None:
    class MockAttnProc:
        def __call__(
            self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs
        ):
            return hidden_states

    # Lateral plan -> processor strength 6.0
    intent_lat = analyze_prompt(
        "a yellow banana to the left of a green apple on a table", mode="generate"
    )
    plan_lat = plan_semantic_layout(intent_lat)
    proc_lat = LayoutGuidanceProcessor(
        base_processor=MockAttnProc(), plan=plan_lat, adaptive_guidance=True
    )
    assert proc_lat.guidance_strength == 6.0

    # Depth plan -> processor strength 0.0 (no-op)
    intent_depth = analyze_prompt("a red cube in front of a blue sphere", mode="generate")
    plan_depth = plan_semantic_layout(intent_depth)
    proc_depth = LayoutGuidanceProcessor(
        base_processor=MockAttnProc(), plan=plan_depth, adaptive_guidance=True
    )
    assert proc_depth.guidance_strength == 0.0

    # Vertical-On plan -> processor strength 0.0 (no-op)
    intent_on = analyze_prompt("a sparrow perched on a fence", mode="generate")
    plan_on = plan_semantic_layout(intent_on)
    proc_on = LayoutGuidanceProcessor(
        base_processor=MockAttnProc(), plan=plan_on, adaptive_guidance=True
    )
    assert proc_on.guidance_strength == 0.0
