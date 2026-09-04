"""Re-score existing depth images from the N=96 benchmark using Depth Anything V2."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_spatial_depth_dedicated import (  # noqa: E402
    MonocularDepthEvaluator,
    exact_mcnemar_p_value,
)
from scripts.eval_spatial_rigorous_benchmark import (  # noqa: E402
    SEEDS_N96,
    SPATIAL_16_SPECS,
    STRENGTHS,
    StrictSpatialEvaluator,
    wilson_score_interval,
)


def rescore_existing() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator_2d = StrictSpatialEvaluator(device=device)
    evaluator_3d = MonocularDepthEvaluator(device=device)

    img_dir = ROOT_DIR / "benchmarks" / "images" / "rigorous_spatial_n96"
    depth_specs = [s for s in SPATIAL_16_SPECS if s["category"] == "depth"]

    print("=" * 85)
    print("RE-SCORING EXISTING DEPTH IMAGES WITH DEPTH ANYTHING V2 (N=24 per Condition)")
    print("=" * 85)

    results_2d: dict[float, dict[tuple[str, int], bool]] = {s: {} for s in STRENGTHS}
    results_3d: dict[float, dict[tuple[str, int], bool]] = {s: {} for s in STRENGTHS}

    for strength in STRENGTHS:
        for spec in depth_specs:
            p_id = spec["id"]
            subj = spec["subject"]
            obj = spec["object"]
            rel = spec["relation"]

            for seed in SEEDS_N96:
                img_path = img_dir / f"{p_id}_s{seed}_str_{strength:.2f}.png"
                img = Image.open(img_path).convert("RGB")

                s_det, o_det = evaluator_2d.detect_entities(img, subj, obj)
                sat_2d, _ = evaluator_2d.check_relation(s_det, o_det, rel)
                sat_3d, s_d, o_d, _ = evaluator_3d.estimate_relative_depth(img, s_det, o_det, rel)

                results_2d[strength][(p_id, seed)] = sat_2d
                results_3d[strength][(p_id, seed)] = sat_3d

    print("\n--- COMPARISON: 2D GROUND-PLANE PROXY VS TRUE 3D DEPTH ANYTHING V2 ---")
    print(
        f"{'Strength':<8} | {'2D Ground-Plane (N=24)':<22} | "
        f"{'True 3D Depth (N=24)':<22} | {'Discordant 3D (b, c)':<20} | {'McNemar p'}"
    )
    print("-" * 95)

    off_3d = results_3d[0.00]

    for strength in STRENGTHS:
        c_2d = sum(1 for v in results_2d[strength].values() if v)
        c_3d = sum(1 for v in results_3d[strength].values() if v)
        r_2d = (c_2d / 24.0) * 100.0
        r_3d = (c_3d / 24.0) * 100.0

        ci_2d = wilson_score_interval(c_2d, 24)
        ci_3d = wilson_score_interval(c_3d, 24)

        if strength == 0.00:
            disc_str = "Baseline"
            p_str = "—"
        else:
            cur_3d = results_3d[strength]
            b = sum(1 for k in off_3d if not off_3d[k] and cur_3d[k])
            c = sum(1 for k in off_3d if off_3d[k] and not cur_3d[k])
            p_val = exact_mcnemar_p_value(b, c)
            disc_str = f"b={b}, c={c} (net {b - c:+d})"
            p_str = f"p={p_val:.4f}"

        print(
            f"Str {strength:<4.2f} | {c_2d:>2}/24 ({r_2d:>5.1f}%) "
            f"[{ci_2d[0]:.1f}%, {ci_2d[1]:.1f}%] | "
            f"{c_3d:>2}/24 ({r_3d:>5.1f}%) [{ci_3d[0]:.1f}%, {ci_3d[1]:.1f}%] | "
            f"{disc_str:<20} | {p_str}"
        )


if __name__ == "__main__":
    rescore_existing()
