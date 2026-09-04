"""Create a 20-pair visual review contact sheet comparing Lateral OFF (0.00) vs Strength 6.00."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_spatial_lateral_dedicated import (  # noqa: E402
    LATERAL_24_SPECS,
    compute_image_ssim,
)
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator  # noqa: E402

REVIEW_PAIRS = [
    ("lat_01", 42),
    ("lat_01", 100),
    ("lat_02", 42),
    ("lat_02", 2024),
    ("lat_03", 42),
    ("lat_04", 555),
    ("lat_05", 42),
    ("lat_05", 7777),
    ("lat_06", 1024),
    ("lat_07", 42),
    ("lat_08", 42),
    ("lat_09", 2024),
    ("lat_10", 42),
    ("lat_11", 42),
    ("lat_12", 42),
    ("lat_13", 100),
    ("lat_14", 42),
    ("lat_15", 555),
    ("lat_16", 42),
    ("lat_17", 42),
]


def create_contact_sheet() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = StrictSpatialEvaluator(device=device)

    img_dir = ROOT_DIR / "benchmarks" / "images" / "lateral_dedicated_n192"
    out_sheet_path = ROOT_DIR / "benchmarks" / "visual_review_lateral_str6.png"

    spec_map = {s["id"]: s for s in LATERAL_24_SPECS}

    # Dimensions for each pair card: 2 images of 256x256 + margin + header/footer
    thumb_w, thumb_h = 256, 256
    card_w = thumb_w * 2 + 30
    card_h = thumb_h + 80

    # Grid: 5 rows x 4 cols = 20 pairs
    cols = 4
    rows = 5
    sheet_w = cols * card_w + (cols + 1) * 20
    sheet_h = rows * card_h + (rows + 1) * 20 + 80

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    # Title Banner
    draw.text(
        (30, 25),
        "Visual Review: Lateral Spatial Guidance (OFF vs Strength 6.00) — N=20 Sampled Pairs",
        fill=(255, 255, 255),
    )

    audit_summary = []

    for idx, (p_id, seed) in enumerate(REVIEW_PAIRS):
        spec = spec_map[p_id]
        prompt = spec["prompt"]
        subj = spec["subject"]
        obj = spec["object"]
        rel = spec["relation"]

        f_off = img_dir / f"{p_id}_s{seed}_str_0.00.png"
        f_on = img_dir / f"{p_id}_s{seed}_str_6.00.png"

        img_off = Image.open(f_off).convert("RGB")
        img_on = Image.open(f_on).convert("RGB")

        # Evaluation
        s_off, o_off = evaluator.detect_entities(img_off, subj, obj)
        sat_off, _ = evaluator.check_relation(s_off, o_off, rel)

        s_on, o_on = evaluator.detect_entities(img_on, subj, obj)
        sat_on, _ = evaluator.check_relation(s_on, o_on, rel)

        ssim_val = compute_image_ssim(img_off, img_on)

        # Check visual quality / bounding boxes
        r_col = idx % cols
        r_row = idx // cols
        card_x = 20 + r_col * (card_w + 20)
        card_y = 90 + r_row * (card_h + 20)

        # Card Background
        draw.rectangle(
            [card_x, card_y, card_x + card_w, card_y + card_h],
            fill=(36, 38, 44),
            outline=(60, 64, 72),
            width=1,
        )

        # Paste thumbnails
        t_off = img_off.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        t_on = img_on.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)

        sheet.paste(t_off, (card_x + 10, card_y + 40))
        sheet.paste(t_on, (card_x + thumb_w + 20, card_y + 40))

        # Text labels
        status_str = (
            f"OFF: {'PASS' if sat_off else 'FAIL'} -> ON(6.0): {'PASS' if sat_on else 'FAIL'}"
        )
        status_color = (100, 220, 100) if sat_on else (220, 100, 100)

        draw.text(
            (card_x + 10, card_y + 8),
            f"[{p_id} s{seed}] {subj} {rel} {obj}",
            fill=(220, 220, 220),
        )
        draw.text(
            (card_x + 10, card_y + 24),
            f"{status_str} | SSIM: {ssim_val:.3f}",
            fill=status_color,
        )

        # Sub-labels on images
        draw.text((card_x + 14, card_y + 45), "OFF (0.00)", fill=(255, 255, 255))
        draw.text((card_x + thumb_w + 24, card_y + 45), "ON (6.00)", fill=(255, 255, 100))

        audit_summary.append({
            "pair": f"{p_id}_s{seed}",
            "prompt": prompt,
            "relation": rel,
            "off_satisfied": sat_off,
            "on_satisfied": sat_on,
            "ssim": ssim_val,
            "subject_detected_on": s_on is not None,
            "object_detected_on": o_on is not None,
        })

    sheet.save(out_sheet_path)
    print(f"[+] Contact sheet saved to: {out_sheet_path}")

    print("\n" + "=" * 80)
    print("20-PAIR LATERAL VISUAL AUDIT SUMMARY (OFF vs 6.00)")
    print("=" * 80)
    gains = sum(1 for a in audit_summary if not a["off_satisfied"] and a["on_satisfied"])
    losses = sum(1 for a in audit_summary if a["off_satisfied"] and not a["on_satisfied"])
    both_p = sum(1 for a in audit_summary if a["off_satisfied"] and a["on_satisfied"])
    both_f = sum(1 for a in audit_summary if not a["off_satisfied"] and not a["on_satisfied"])

    print(
        f"Distribution in 20 Samples: Both Pass={both_p}, "
        f"Gain(+1)={gains}, Loss(-1)={losses}, Both Fail={both_f}"
    )
    print(f"Mean SSIM: {np.mean([a['ssim'] for a in audit_summary]):.4f}")
    for a in audit_summary:
        st_off = "PASS" if a["off_satisfied"] else "FAIL"
        st_on = "PASS" if a["on_satisfied"] else "FAIL"
        print(f"  [{a['pair']:<12}] SSIM: {a['ssim']:.3f} | {st_off} -> {st_on} | {a['prompt']}")


if __name__ == "__main__":
    create_contact_sheet()
