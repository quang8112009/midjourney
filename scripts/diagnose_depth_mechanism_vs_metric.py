"""Diagnostic Study: Distinguish Depth Guidance mechanism vs 2D metric sensitivity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_spatial_rigorous_benchmark import (  # noqa: E402
    SEEDS_N96,
    SPATIAL_16_SPECS,
    StrictSpatialEvaluator,
    compute_image_ssim,
)


def diagnose_depth() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = StrictSpatialEvaluator(device=device)

    img_dir = ROOT_DIR / "benchmarks" / "images" / "rigorous_spatial_n96"

    depth_specs = [s for s in SPATIAL_16_SPECS if s["category"] == "depth"]
    lateral_specs = [s for s in SPATIAL_16_SPECS if s["category"] == "lateral"]

    print("=" * 85)
    print("DEPTH GUIDANCE DIAGNOSTIC: MECHANISM VS METRIC SENSITIVITY")
    print("=" * 85)

    # 1. Compare SSIM / Pixel L1 shift across categories at 0.35, 1.50, 6.00
    strengths = [0.35, 1.50, 6.00]

    print("\n--- 1. STRUCTURAL SHIFT (SSIM & L1 PIXEL SHIFT) BY CATEGORY ---")
    for s in strengths:
        depth_ssims = []
        depth_l1s = []
        lateral_ssims = []
        lateral_l1s = []

        for spec in depth_specs:
            p_id = spec["id"]
            for seed in SEEDS_N96:
                f_off = img_dir / f"{p_id}_s{seed}_str_0.00.png"
                f_on = img_dir / f"{p_id}_s{seed}_str_{s:.2f}.png"
                i_off = np.array(Image.open(f_off).convert("RGB"), dtype=np.float32) / 255.0
                i_on = np.array(Image.open(f_on).convert("RGB"), dtype=np.float32) / 255.0
                depth_ssims.append(compute_image_ssim(Image.open(f_off), Image.open(f_on)))
                depth_l1s.append(float(np.mean(np.abs(i_off - i_on))))

        for spec in lateral_specs:
            p_id = spec["id"]
            for seed in SEEDS_N96:
                f_off = img_dir / f"{p_id}_s{seed}_str_0.00.png"
                f_on = img_dir / f"{p_id}_s{seed}_str_{s:.2f}.png"
                i_off = np.array(Image.open(f_off).convert("RGB"), dtype=np.float32) / 255.0
                i_on = np.array(Image.open(f_on).convert("RGB"), dtype=np.float32) / 255.0
                lateral_ssims.append(compute_image_ssim(Image.open(f_off), Image.open(f_on)))
                lateral_l1s.append(float(np.mean(np.abs(i_off - i_on))))

        m_d_ssim = np.mean(depth_ssims)
        m_d_l1 = np.mean(depth_l1s)
        m_l_ssim = np.mean(lateral_ssims)
        m_l_l1 = np.mean(lateral_l1s)
        print(
            f"Strength {s:.2f}:"
            f"\n  Depth Prompts   (N=24): Mean SSIM = {m_d_ssim:.4f} (Pixel L1 = {m_d_l1:.4f})"
            f"\n  Lateral Prompts (N=24): Mean SSIM = {m_l_ssim:.4f} (Pixel L1 = {m_l_l1:.4f})"
        )

    # 2. Inspect 15 Depth Pairs for Bounding Box Centroid Shifts vs Metric Verdict
    print("\n--- 2. DETAILED INSPECTION OF 15 DEPTH PROMPT PAIRS (OFF vs 6.00) ---")
    depth_pairs = [(spec, seed) for spec in depth_specs for seed in SEEDS_N96][:15]

    for idx, (spec, seed) in enumerate(depth_pairs, 1):
        p_id = spec["id"]
        subj = spec["subject"]
        obj = spec["object"]
        rel = spec["relation"]

        f_off = img_dir / f"{p_id}_s{seed}_str_0.00.png"
        f_on = img_dir / f"{p_id}_s{seed}_str_6.00.png"

        img_off = Image.open(f_off).convert("RGB")
        img_on = Image.open(f_on).convert("RGB")

        ssim_val = compute_image_ssim(img_off, img_on)

        s_det_off, o_det_off = evaluator.detect_entities(img_off, subj, obj)
        sat_off, _ = evaluator.check_relation(s_det_off, o_det_off, rel)

        s_det_on, o_det_on = evaluator.detect_entities(img_on, subj, obj)
        sat_on, _ = evaluator.check_relation(s_det_on, o_det_on, rel)

        # Coordinate shifts
        st_off = "PASS" if sat_off else "FAIL"
        st_on = "PASS" if sat_on else "FAIL"
        if s_det_off and s_det_on and o_det_off and o_det_on:
            s_cy_off, _ = s_det_off["center"]
            s_cy_on, _ = s_det_on["center"]
            o_cy_off, _ = o_det_off["center"]
            o_cy_on, _ = o_det_on["center"]

            dy_off = s_cy_off - o_cy_off
            dy_on = s_cy_on - o_cy_on

            print(
                f"[{idx:02d}] {p_id}_s{seed} ({rel}): SSIM={ssim_val:.3f} | "
                f"OFF ({st_off} dy={dy_off:+.2f}) -> ON ({st_on} dy={dy_on:+.2f}) | "
                f"Subj dY={s_cy_on - s_cy_off:+.2f}, Obj dY={o_cy_on - o_cy_off:+.2f}"
            )
        else:
            print(
                f"[{idx:02d}] {p_id}_s{seed} ({rel}): SSIM={ssim_val:.3f} | "
                f"OFF: {st_off} -> ON: {st_on} (Detection incomplete)"
            )


if __name__ == "__main__":
    diagnose_depth()
