"""Prepare 120 blinded depth evaluation samples for human validation.

Samples 120 images across OFF (0.00) and Strength 6.00, balanced across in_front_of and behind.
Blinds conditions, prompts, and metric verdicts.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from PIL import Image, ImageDraw

from scripts.eval_spatial_depth_dedicated import (
    DEPTH_24_SPECS,
    MonocularDepthEvaluator,
)
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator

SEEDS_5 = [42, 100, 2024, 7777, 12345]


def main():
    print("=" * 80)
    print("PREPARING 120 BLINDED DEPTH IMAGES FOR HUMAN VALIDATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator_2d = StrictSpatialEvaluator(device=device)
    evaluator_3d = MonocularDepthEvaluator(device=device)

    img_dir = ROOT_DIR / "benchmarks" / "images" / "depth_dedicated_n192"

    front_specs = [s for s in DEPTH_24_SPECS if s["relation"] == "in_front_of"] # 12 specs
    behind_specs = [s for s in DEPTH_24_SPECS if s["relation"] == "behind"] # 12 specs

    # We need:
    # 30 front @ 0.00, 30 front @ 6.00 (from 12 front specs x 5 seeds = 60 items -> sample 30 each)
    # 30 behind @ 0.00, 30 behind @ 6.00 (from 12 behind specs x 5 seeds = 60 items -> sample 30 each)
    rng = random.Random(20260906)

    front_items_0 = [(s, seed, 0.00) for s in front_specs for seed in SEEDS_5]
    front_items_6 = [(s, seed, 6.00) for s in front_specs for seed in SEEDS_5]
    behind_items_0 = [(s, seed, 0.00) for s in behind_specs for seed in SEEDS_5]
    behind_items_6 = [(s, seed, 6.00) for s in behind_specs for seed in SEEDS_5]

    sampled_f0 = rng.sample(front_items_0, 30)
    sampled_f6 = rng.sample(front_items_6, 30)
    sampled_b0 = rng.sample(behind_items_0, 30)
    sampled_b6 = rng.sample(behind_items_6, 30)

    all_sampled = sampled_f0 + sampled_f6 + sampled_b0 + sampled_b6
    rng.shuffle(all_sampled) # Fully randomize order

    print(f"[+] Sampled {len(all_sampled)} balanced depth images.")

    blinded_records = []
    gold_truth_records = []

    print("\nEvaluating 2D and Depth Anything V2 metrics on all 120 samples...")
    for idx, (spec, seed, strength) in enumerate(all_sampled, start=1):
        blind_id = f"blind_{idx:03d}"
        p_id = spec["id"]
        prompt = spec["prompt"]
        subj = spec["subject"]
        obj = spec["object"]
        rel = spec["relation"]

        img_filename = f"{p_id}_s{seed}_str_{strength:.2f}.png"
        img_path = img_dir / img_filename
        img = Image.open(img_path).convert("RGB")

        # 2D Detection & Ground-Plane Evaluation
        s_det, o_det = evaluator_2d.detect_entities(img, subj, obj)
        sat_2d, r_2d = evaluator_2d.check_relation(s_det, o_det, rel)

        # 3D Depth Anything V2 Evaluation
        sat_3d, s_d, o_d, r_3d = evaluator_3d.estimate_relative_depth(img, s_det, o_det, rel)

        # Blind record (for labeler)
        blinded_records.append({
            "blind_id": blind_id,
            "image_filename": img_filename,
            "image_path": str(img_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "prompt": prompt,
            "subject": subj,
            "object": obj,
            "target_relation": rel,
            "question": f"In this image, is the '{subj}' {rel.replace('_', ' ')} the '{obj}'?",
        })

        # Gold truth secret record (for evaluation analysis)
        gold_truth_records.append({
            "blind_id": blind_id,
            "image_filename": img_filename,
            "image_path": str(img_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "spec_id": p_id,
            "seed": seed,
            "strength": strength,
            "prompt": prompt,
            "subject": subj,
            "object": obj,
            "relation": rel,
            "metric_2d_satisfied": sat_2d,
            "metric_2d_reason": r_2d,
            "metric_depth_anything_satisfied": sat_3d,
            "metric_depth_anything_reason": r_3d,
            "depth_subj": s_d,
            "depth_obj": o_d,
        })

    # Save Blinded Sheet and Secret Key
    out_blind_json = ROOT_DIR / "benchmarks" / "human_depth_validation_blinded.json"
    out_secret_json = ROOT_DIR / "benchmarks" / "human_depth_validation_secret_key.json"

    with open(out_blind_json, "w", encoding="utf-8") as f:
        json.dump(blinded_records, f, indent=2)
    with open(out_secret_json, "w", encoding="utf-8") as f:
        json.dump(gold_truth_records, f, indent=2)

    print(f"\n[+] Saved blinded labeling sheet to: {out_blind_json}")
    print(f"[+] Saved secret key with automated metrics to: {out_secret_json}")

    # Generate visual review contact sheets (6 sheets of 20 images each)
    sheets_dir = ROOT_DIR / "benchmarks" / "depth_labeling_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    img_w, img_h = 320, 320
    cols = 4
    rows = 5
    per_sheet = cols * rows # 20

    for sheet_idx in range(6):
        sheet_img = Image.new("RGB", (cols * (img_w + 30) + 30, rows * (img_h + 100) + 60), color=(30, 30, 35))
        draw = ImageDraw.Draw(sheet_img)
        draw.text((30, 15), f"HUMAN DEPTH VALIDATION SHEET {sheet_idx+1}/6 (BLINDED SAMPLES {sheet_idx*20+1:03d} - {(sheet_idx+1)*20:03d})", fill=(255, 255, 255))

        for item_idx in range(per_sheet):
            global_idx = sheet_idx * per_sheet + item_idx
            if global_idx >= len(blinded_records):
                break
            rec = blinded_records[global_idx]
            r = item_idx // cols
            c = item_idx % cols

            x = 30 + c * (img_w + 30)
            y = 50 + r * (img_h + 100)

            item_img = Image.open(ROOT_DIR / rec["image_path"]).convert("RGB").resize((img_w, img_h), Image.Resampling.LANCZOS)
            sheet_img.paste(item_img, (x, y))

            draw.text((x, y + img_h + 5), f"[{rec['blind_id']}] {rec['subject']} -> {rec['target_relation']} -> {rec['object']}", fill=(255, 215, 0))
            draw.text((x, y + img_h + 25), rec['prompt'][:45] + ("..." if len(rec['prompt']) > 45 else ""), fill=(200, 200, 200))
            draw.text((x, y + img_h + 45), f"Q: Is '{rec['subject']}' {rec['target_relation'].replace('_', ' ')} '{rec['object']}'?", fill=(150, 230, 150))

        sheet_path = sheets_dir / f"depth_validation_sheet_{sheet_idx+1}.png"
        sheet_img.save(sheet_path)
        print(f"  [+] Saved visual labeling sheet {sheet_idx+1}/6 to: {sheet_path}")


if __name__ == "__main__":
    main()
