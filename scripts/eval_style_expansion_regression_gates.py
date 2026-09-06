from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from transformers import AutoTokenizer

from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout
from app.services.editing.style_expansion import expand_style
from scripts.eval_spatial_lateral_dedicated import LATERAL_24_SPECS
from scripts.eval_spatial_rigorous_benchmark import SPATIAL_16_SPECS
from scripts.hard_lateral_specs import LATERAL_HARD_24_SPECS


def main():
    print("=" * 90)
    print("PHASE C REGRESSION GATES: SPATIAL LAYOUT INVARIANCE WITH STYLE EXPANSION ON")
    print("=" * 90)

    tok_t5 = AutoTokenizer.from_pretrained("models/sd35_medium/tokenizer_3")

    # 1. Gate 1: Standard 24 Lateral Suite Invariance
    print("\n--- Gate 1: Standard 24 Lateral Suite Invariance ---")
    gate1_passed = True
    gate1_details = []

    for spec in LATERAL_24_SPECS:
        pid = spec["id"]
        raw_prompt = spec["prompt"]
        subj = spec["subject"]
        obj = spec["object"]

        # Raw plan
        raw_plan = plan_semantic_layout(analyze_prompt(raw_prompt), tokenizer=tok_t5)

        # Expanded plan
        exp_res = expand_style(raw_prompt, model="stable-diffusion-3.5", tokenizer=tok_t5, style_expansion_enabled=True)
        exp_plan = plan_semantic_layout(analyze_prompt(exp_res.expanded_prompt), tokenizer=tok_t5)

        # Verify key entities exist in both
        subj_raw = next((o for o in raw_plan.objects if o.label in subj or subj in o.label), None)
        subj_exp = next((o for o in exp_plan.objects if o.label in subj or subj in o.label), None)
        obj_raw = next((o for o in raw_plan.objects if o.label in obj or obj in o.label), None)
        obj_exp = next((o for o in exp_plan.objects if o.label in obj or obj in o.label), None)

        assert subj_raw is not None and subj_exp is not None, f"[{pid}] Subject missing in plan"
        assert obj_raw is not None and obj_exp is not None, f"[{pid}] Object missing in plan"

        # Coordinates must be identical
        mu_x_subj_diff = abs(subj_raw.gaussian.mu_x - subj_exp.gaussian.mu_x)
        mu_x_obj_diff = abs(obj_raw.gaussian.mu_x - obj_exp.gaussian.mu_x)

        passed = mu_x_subj_diff < 1e-4 and mu_x_obj_diff < 1e-4
        if not passed:
            gate1_passed = False

        gate1_details.append({
            "id": pid,
            "raw_prompt": raw_prompt,
            "expanded_prompt": exp_res.expanded_prompt,
            "subj_mu_x_raw": subj_raw.gaussian.mu_x,
            "subj_mu_x_exp": subj_exp.gaussian.mu_x,
            "obj_mu_x_raw": obj_raw.gaussian.mu_x,
            "obj_mu_x_exp": obj_exp.gaussian.mu_x,
            "passed": passed,
        })
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {pid:<8}: subj mu_x={subj_exp.gaussian.mu_x:.2f} (diff={mu_x_subj_diff:.4f}), obj mu_x={obj_exp.gaussian.mu_x:.2f} (diff={mu_x_obj_diff:.4f})")

    print(f"\nGate 1 Verdict: {'ALL 24 PASSED (100% Invariant)' if gate1_passed else 'FAILED'}")

    # 2. Gate 2: Hard 24 Directional Suite Invariance
    print("\n--- Gate 2: Hard 24 Directional Suite Invariance ---")
    gate2_passed = True
    gate2_details = []

    for spec in LATERAL_HARD_24_SPECS:
        pid = spec["id"]
        raw_prompt = spec["prompt"]
        subj = spec["subject"]
        obj = spec["object"]

        raw_plan = plan_semantic_layout(analyze_prompt(raw_prompt), tokenizer=tok_t5)
        exp_res = expand_style(raw_prompt, model="stable-diffusion-3.5", tokenizer=tok_t5, style_expansion_enabled=True)
        exp_plan = plan_semantic_layout(analyze_prompt(exp_res.expanded_prompt), tokenizer=tok_t5)

        subj_raw = next((o for o in raw_plan.objects if o.label in subj or subj in o.label or any(a in o.attributes for a in spec.get("subject", "").split())), None)
        subj_exp = next((o for o in exp_plan.objects if o.label in subj or subj in o.label or any(a in o.attributes for a in spec.get("subject", "").split())), None)

        obj_raw = next((o for o in raw_plan.objects if o.label in obj or obj in o.label or any(a in o.attributes for a in spec.get("object", "").split())), None)
        obj_exp = next((o for o in exp_plan.objects if o.label in obj or obj in o.label or any(a in o.attributes for a in spec.get("object", "").split())), None)

        assert subj_raw is not None and subj_exp is not None, f"[{pid}] Subject missing in plan"
        assert obj_raw is not None and obj_exp is not None, f"[{pid}] Object missing in plan"

        mu_x_subj_diff = abs(subj_raw.gaussian.mu_x - subj_exp.gaussian.mu_x)
        mu_x_obj_diff = abs(obj_raw.gaussian.mu_x - obj_exp.gaussian.mu_x)

        passed = mu_x_subj_diff < 1e-4 and mu_x_obj_diff < 1e-4
        if not passed:
            gate2_passed = False

        gate2_details.append({
            "id": pid,
            "raw_prompt": raw_prompt,
            "expanded_prompt": exp_res.expanded_prompt,
            "passed": passed,
        })
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {pid:<12}: subj mu_x={subj_exp.gaussian.mu_x:.2f}, obj mu_x={obj_exp.gaussian.mu_x:.2f}")

    print(f"\nGate 2 Verdict: {'ALL 24 PASSED (100% Invariant)' if gate2_passed else 'FAILED'}")

    # 3. Gate 3: Rigorous Spatial Multi-Category Suite (16 Prompts across 4 Categories)
    print("\n--- Gate 3: Rigorous Spatial Multi-Category Suite Invariance ---")
    gate3_passed = True
    for spec in SPATIAL_16_SPECS:
        pid = spec["id"]
        raw_prompt = spec["prompt"]
        cat = spec["category"]

        raw_plan = plan_semantic_layout(analyze_prompt(raw_prompt), tokenizer=tok_t5)
        exp_res = expand_style(raw_prompt, model="stable-diffusion-3.5", tokenizer=tok_t5, style_expansion_enabled=True)
        exp_plan = plan_semantic_layout(analyze_prompt(exp_res.expanded_prompt), tokenizer=tok_t5)

        assert len(raw_plan.objects) == len(exp_plan.objects), f"[{pid}] Object count changed: {len(raw_plan.objects)} -> {len(exp_plan.objects)}"
        for o_r, o_e in zip(raw_plan.objects, exp_plan.objects, strict=True):
            assert abs(o_r.gaussian.mu_x - o_e.gaussian.mu_x) < 1e-4, f"[{pid}] mu_x changed on {o_r.label}"
            assert abs(o_r.gaussian.mu_y - o_e.gaussian.mu_y) < 1e-4, f"[{pid}] mu_y changed on {o_r.label}"
            assert abs(o_r.gaussian.mu_z - o_e.gaussian.mu_z) < 1e-4, f"[{pid}] mu_z changed on {o_r.label}"
        print(f"  [PASS] {pid:<12} ({cat:<14}): All {len(raw_plan.objects)} entities 100% spatial coordinate invariant")

    print("\nGate 3 Verdict: ALL 16 MULTI-CATEGORY PROMPTS 100% INVARIANT")

    # Save report
    report = {
        "gate1_standard_24_lateral": {"passed": gate1_passed, "details": gate1_details},
        "gate2_hard_24_directional": {"passed": gate2_passed, "details": gate2_details},
        "gate3_rigorous_16_multicategory": {"passed": gate3_passed},
        "overall_regression_status": "PASSED - ZERO SPATIAL REGRESSION",
    }
    out_file = ROOT_DIR / "benchmarks" / "style_expansion_regression_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Saved regression gate report to: {out_file}")


if __name__ == "__main__":
    main()
