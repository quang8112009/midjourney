"""End-to-end data flow audit and round-trip fidelity tests."""

from __future__ import annotations

import time

import torch

from app.services.conversation_store import ConversationStore
from app.services.editing.layout_guidance import (
    LayoutGuidanceProcessor,
    TwoPhaseSchedule,
)
from app.services.editing.metrics import DataFlowTrace
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import (
    DensityField,
    NormalizedBox,
    PlannedObject,
    PlanSelfCheck,
    SemanticLayoutPlan,
    StyleHints,
    plan_semantic_layout,
)


def test_round_trip_discrete_entities_and_3d_gaussians():
    """Verify that discrete entities with 3D Gaussian priors survive serialization without drift."""
    original_box = NormalizedBox(ymin=0.15, xmin=0.25, ymax=0.65, xmax=0.75)
    original_gaussian = original_box.to_gaussian(rotation=0.35, mu_z=0.42, sigma_z=0.18)

    plan = SemanticLayoutPlan(
        prompt="A red dragon perched on a castle tower",
        objects=(
            PlannedObject(
                label="dragon",
                count=1,
                box=original_box,
                token_indices=(2,),
                attributes=("red",),
                gaussian=original_gaussian,
                entity_id="dragon_0001",
                provenance="user_override",
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

    serialized = plan.to_dict()

    # Re-hydrate through planner's layout_override contract
    intent = analyze_prompt(plan.prompt, mode="generate")
    reconstituted = plan_semantic_layout(
        intent=intent,
        layout_override=serialized["objects"],
    )

    assert len(reconstituted.objects) == 1
    rec_obj = reconstituted.objects[0]

    assert rec_obj.label == "dragon"
    assert rec_obj.count == 1
    assert abs(rec_obj.box.ymin - 0.15) < 1e-4
    assert abs(rec_obj.box.xmin - 0.25) < 1e-4
    assert abs(rec_obj.box.ymax - 0.65) < 1e-4
    assert abs(rec_obj.box.xmax - 0.75) < 1e-4

    assert rec_obj.gaussian is not None
    assert abs(rec_obj.gaussian.mu_z - 0.42) < 1e-4
    assert abs(rec_obj.gaussian.theta - 0.35) < 1e-4
    assert abs(rec_obj.gaussian.sigma_z - 0.18) < 1e-4
    assert rec_obj.provenance == "user_override"


def test_round_trip_all_four_density_field_distributions():
    """Verify all 4 density distributions survive serialization without degrading to boxes."""
    distributions = ["gaussian", "uniform", "radial", "elongated"]

    for dist in distributions:
        box = NormalizedBox(ymin=0.2, xmin=0.3, ymax=0.7, xmax=0.8)
        original_df = DensityField(
            entity_id=f"swarm_{dist}",
            label="bees",
            expected_count=50,
            density=1.5,
            center=box.center,
            scale=(0.25, 0.25),
            region=box,
            distribution_type=dist,
            falloff=2.5,
            seed=12345,
            token_indices=(1,),
            mu_z=0.3,
            provenance="user_override",
        )

        plan = SemanticLayoutPlan(
            prompt=f"A swarm of 50 bees with {dist} pattern",
            objects=(),
            relations=(),
            style_hints=StyleHints(),
            self_check=PlanSelfCheck(
                is_valid=True,
                count_match=True,
                relation_match=True,
                ambiguity_detected=False,
            ),
            density_fields=(original_df,),
        )

        serialized = plan.to_dict()
        assert "density_fields" in serialized
        assert len(serialized["density_fields"]) == 1
        assert serialized["density_fields"][0]["is_density_field"] is True

        # Re-hydrate via layout_override
        intent = analyze_prompt(plan.prompt, mode="generate")
        reconstituted = plan_semantic_layout(
            intent=intent,
            layout_override=serialized["density_fields"],
        )

        assert len(reconstituted.density_fields) == 1
        rec_df = reconstituted.density_fields[0]

        assert rec_df.label == "bees"
        assert rec_df.expected_count == 50
        assert rec_df.distribution_type == dist
        assert abs(rec_df.falloff - 2.5) < 1e-4
        assert abs(rec_df.density - 1.5) < 1e-4
        assert rec_df.seed == 12345
        assert abs(rec_df.mu_z - 0.3) < 1e-4
        assert rec_df.provenance == "user_override"


def test_layout_guidance_processor_base_heatmap_reuse():
    """Verify that LayoutGuidanceProcessor reuses the static base heatmap across steps."""
    plan = SemanticLayoutPlan(
        prompt="A cat on a table",
        objects=(
            PlannedObject(
                label="cat",
                count=1,
                box=NormalizedBox(0.2, 0.2, 0.5, 0.5),
                token_indices=(1,),
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

    class DummyBaseProcessor:
        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs,
        ):
            return hidden_states

    proc = LayoutGuidanceProcessor(
        base_processor=DummyBaseProcessor(),
        plan=plan,
        guidance_strength=0.3,
        schedule=TwoPhaseSchedule(schedule_cutoff=0.8),
    )

    hidden = torch.zeros(1, 256, 64)
    encoder_hidden = torch.zeros(1, 77, 64)

    # Step 0 (progress 0.0) -> computes base bias
    proc.set_step_progress(0.0)
    proc(None, hidden, encoder_hidden_states=encoder_hidden)
    assert proc._base_bias is not None
    initial_base_bias_id = id(proc._base_bias)

    # Step 10 (progress 0.4) -> must REUSE the same base bias tensor in memory
    proc.set_step_progress(0.4)
    proc(None, hidden, encoder_hidden_states=encoder_hidden)
    assert id(proc._base_bias) == initial_base_bias_id


def test_session_state_plan_continuity():
    """Verify ConversationStore stores and retrieves semantic plans across turns."""
    store = ConversationStore()
    session_id = store.create_session()

    plan_dict = {
        "prompt": "a blue cube on a wooden desk",
        "objects": [
            {
                "label": "cube",
                "count": 1,
                "box": {"ymin": 0.3, "xmin": 0.2, "ymax": 0.7, "xmax": 0.5},
            }
        ],
    }

    store.set_session_plan(session_id, plan_dict)
    retrieved = store.get_session_plan(session_id)
    assert retrieved == plan_dict

    snapshot = store.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot.session_id == session_id


def test_data_flow_tracing():
    """Verify request-scoped DataFlowTrace records all stages with timing details."""
    trace = DataFlowTrace(request_id="req_test_123")
    start = time.perf_counter()
    time.sleep(0.005)
    trace.add_stage("planner", time.perf_counter() - start, entities=2, density_fields=1)

    start = time.perf_counter()
    time.sleep(0.005)
    trace.add_stage("guidance", time.perf_counter() - start, schedule="two_phase", gamma=0.24)

    trace.total_elapsed_seconds = 0.012

    out = trace.to_dict()
    assert out["request_id"] == "req_test_123"
    assert len(out["stages"]) == 2
    assert out["stages"][0]["stage"] == "planner"
    assert out["stages"][0]["details"]["entities"] == 2
    assert out["stages"][1]["stage"] == "guidance"
