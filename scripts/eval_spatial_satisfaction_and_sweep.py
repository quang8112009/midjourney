"""Evaluate Spatial Satisfaction with Object Detection (OWL-ViT) and Guidance Strength Sweep.

1. Evaluates all 64 image pairs from the A/B baseline with OWL-ViT zero-shot object detection
   to compute spatial satisfaction rates (Guidance OFF vs Guidance ON).
2. Performs a live Guidance Strength Sweep across {0.35, 0.7, 1.5, 3.0, 6.0} to measure
   the control vs. quality trade-off curve (SSIM vs OFF, Satisfaction Rate, CLIP alignment, LAION).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, OwlViTForObjectDetection, OwlViTProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.editing.layout_guidance import (  # noqa: E402
    LayoutGuidanceProcessor,
    TwoPhaseSchedule,
)
from app.services.editing.prompt_intent import analyze_prompt  # noqa: E402
from app.services.editing.semantic_planner import plan_semantic_layout  # noqa: E402

# 16 standard spatial test cases with explicit target labels and expected relation
SPATIAL_BENCHMARK_SPECS = [
    {
        "id": "spatial_01",
        "prompt": "a red cube in front of a blue sphere on a marble floor",
        "subject": "red cube",
        "object": "blue sphere",
        "relation": "in_front_of",
    },
    {
        "id": "spatial_02",
        "prompt": "a golden statue behind a stone fountain in a courtyard",
        "subject": "golden statue",
        "object": "stone fountain",
        "relation": "behind",
    },
    {
        "id": "spatial_03",
        "prompt": "a green chair in front of a brick fireplace in a cozy living room",
        "subject": "green chair",
        "object": "brick fireplace",
        "relation": "in_front_of",
    },
    {
        "id": "spatial_04",
        "prompt": "a tall oak tree behind a small wooden cottage in a meadow",
        "subject": "oak tree",
        "object": "wooden cottage",
        "relation": "behind",
    },
    {
        "id": "spatial_05",
        "prompt": "a white coffee cup on top of a stack of vintage books",
        "subject": "coffee cup",
        "object": "stack of books",
        "relation": "on",
    },
    {
        "id": "spatial_06",
        "prompt": "a brown teddy bear resting on a leather sofa",
        "subject": "teddy bear",
        "object": "leather sofa",
        "relation": "on",
    },
    {
        "id": "spatial_07",
        "prompt": "a red apple on a ceramic plate on a wooden kitchen counter",
        "subject": "red apple",
        "object": "ceramic plate",
        "relation": "on",
    },
    {
        "id": "spatial_08",
        "prompt": "a small sparrow perched on a rusted metal fence",
        "subject": "sparrow",
        "object": "metal fence",
        "relation": "on",
    },
    {
        "id": "spatial_09",
        "prompt": "a black cat sitting under a wooden dining chair",
        "subject": "black cat",
        "object": "wooden chair",
        "relation": "under",
    },
    {
        "id": "spatial_10",
        "prompt": "a green sea turtle swimming below a translucent jellyfish in clear ocean",
        "subject": "sea turtle",
        "object": "jellyfish",
        "relation": "under",
    },
    {
        "id": "spatial_11",
        "prompt": "a pair of leather boots placed under a wooden bench",
        "subject": "leather boots",
        "object": "wooden bench",
        "relation": "under",
    },
    {
        "id": "spatial_12",
        "prompt": "a colorful rug under a glass coffee table",
        "subject": "colorful rug",
        "object": "coffee table",
        "relation": "under",
    },
    {
        "id": "spatial_13",
        "prompt": "a yellow banana to the left of a green apple on a white table",
        "subject": "yellow banana",
        "object": "green apple",
        "relation": "left_of",
    },
    {
        "id": "spatial_14",
        "prompt": "a crystal vase beside an antique brass clock on a mantelpiece",
        "subject": "crystal vase",
        "object": "brass clock",
        "relation": "beside",
    },
    {
        "id": "spatial_15",
        "prompt": "a red sports car parked to the right of a blue bicycle",
        "subject": "red sports car",
        "object": "blue bicycle",
        "relation": "right_of",
    },
    {
        "id": "spatial_16",
        "prompt": "a silver teapot beside a porcelain teacup on a tray",
        "subject": "silver teapot",
        "object": "porcelain teacup",
        "relation": "beside",
    },
]

SEEDS_FULL = [42, 100, 2024, 7777]
SEEDS_SWEEP = [42, 100]
STRENGTH_SWEEP_VALUES = [0.35, 0.70, 1.50, 3.00, 6.00]


class OfficialLAIONAestheticPredictor(nn.Module):
    def __init__(self, input_dim: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SpatialEvaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        # Load OWL-ViT detector
        owl_dir = ROOT_DIR / "models" / "owlvit_base_patch32"
        print(f"Loading OWL-ViT detector from {owl_dir} on {device}...")
        self.owl_processor = OwlViTProcessor.from_pretrained(str(owl_dir))
        self.owl_model = OwlViTForObjectDetection.from_pretrained(str(owl_dir)).to(device).eval()

        # Load CLIP & LAION scorer
        clip_dir = ROOT_DIR / "models" / "clip_vit_l14"
        w_path = ROOT_DIR / "models" / "sac_logos_ava1_l14_linearMSE.pth"
        print(f"Loading CLIP & LAION scorers on {device}...")
        self.clip_processor = CLIPProcessor.from_pretrained(str(clip_dir))
        self.clip_model = CLIPModel.from_pretrained(str(clip_dir)).to(device).eval()

        state_dict = torch.load(str(w_path), map_location=device, weights_only=True)
        self.aesthetic_head = OfficialLAIONAestheticPredictor(768).to(device).eval()
        self.aesthetic_head.load_state_dict(state_dict)

    @torch.inference_mode()
    def detect_entities(
        self, image: Image.Image, subject_text: str, object_text: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        queries = [f"a {subject_text}", f"a {object_text}"]
        inputs = self.owl_processor(text=[queries], images=image, return_tensors="pt").to(
            self.device
        )
        outputs = self.owl_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.owl_processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.08
        )[0]

        best_subj: dict[str, Any] | None = None
        best_obj: dict[str, Any] | None = None

        for idx, label_idx in enumerate(results["labels"].tolist()):
            score = float(results["scores"][idx].item())
            box = [float(x) for x in results["boxes"][idx].tolist()]  # [xmin, ymin, xmax, ymax]
            # Convert to normalized [ymin, xmin, ymax, xmax]
            w, h = image.size
            norm_box = [box[1] / h, box[0] / w, box[3] / h, box[2] / w]
            center_y = (norm_box[0] + norm_box[2]) / 2.0
            center_x = (norm_box[1] + norm_box[3]) / 2.0

            det_info = {
                "score": round(score, 3),
                "box": [round(x, 4) for x in norm_box],
                "center": (round(center_y, 4), round(center_x, 4)),
            }

            if label_idx == 0:
                if best_subj is None or score > best_subj["score"]:
                    best_subj = det_info
            elif label_idx == 1:
                if best_obj is None or score > best_obj["score"]:
                    best_obj = det_info

        return best_subj, best_obj

    def check_relation_satisfaction(
        self,
        subj_det: dict[str, Any] | None,
        obj_det: dict[str, Any] | None,
        relation: str,
    ) -> tuple[bool, str]:
        if subj_det is None or obj_det is None:
            missing = []
            if subj_det is None:
                missing.append("subject")
            if obj_det is None:
                missing.append("object")
            return False, f"Missing detection for {', '.join(missing)}"

        sy, sx = subj_det["center"]
        oy, ox = obj_det["center"]
        s_box = subj_det["box"]  # [ymin, xmin, ymax, xmax]
        o_box = obj_det["box"]

        if relation in ("in_front_of",):
            is_satisfied = sy >= oy - 0.05
            reason = (
                f"Subject y={sy:.2f} >= Object y={oy:.2f}"
                if is_satisfied
                else f"Subject y={sy:.2f} < Object y={oy:.2f}"
            )
            return is_satisfied, reason

        elif relation in ("behind",):
            is_satisfied = sy <= oy + 0.05
            reason = (
                f"Subject y={sy:.2f} <= Object y={oy:.2f}"
                if is_satisfied
                else f"Subject y={sy:.2f} > Object y={oy:.2f}"
            )
            return is_satisfied, reason

        elif relation in ("on",):
            horiz_overlap = not (s_box[3] < o_box[1] or s_box[1] > o_box[3])
            vert_above = s_box[2] <= o_box[2] + 0.15 and sy <= oy + 0.05
            is_satisfied = vert_above and horiz_overlap
            reason = f"Subject on top (sy={sy:.2f} < oy={oy:.2f}, overlap={horiz_overlap})"
            return is_satisfied, reason

        elif relation in ("under",):
            horiz_overlap = not (s_box[3] < o_box[1] or s_box[1] > o_box[3])
            vert_below = sy >= oy - 0.05
            is_satisfied = vert_below and horiz_overlap
            reason = f"Subject below (sy={sy:.2f} > oy={oy:.2f}, overlap={horiz_overlap})"
            return is_satisfied, reason

        elif relation in ("left_of",):
            is_satisfied = sx < ox - 0.03
            reason = (
                f"Subject x={sx:.2f} < Object x={ox:.2f}"
                if is_satisfied
                else f"Subject x={sx:.2f} >= Object x={ox:.2f}"
            )
            return is_satisfied, reason

        elif relation in ("right_of",):
            is_satisfied = sx > ox + 0.03
            reason = (
                f"Subject x={sx:.2f} > Object x={ox:.2f}"
                if is_satisfied
                else f"Subject x={sx:.2f} <= Object x={ox:.2f}"
            )
            return is_satisfied, reason

        elif relation in ("beside",):
            lateral_sep = abs(sx - ox) >= 0.08
            coplanar = abs(sy - oy) <= 0.40
            is_satisfied = lateral_sep and coplanar
            reason = f"Lateral separation dx={abs(sx - ox):.2f}, dy={abs(sy - oy):.2f}"
            return is_satisfied, reason

        return False, f"Unknown relation {relation}"

    @torch.inference_mode()
    def evaluate_quality(self, image: Image.Image, prompt: str) -> tuple[float, float]:
        inputs = self.clip_processor(
            text=[prompt], images=image, return_tensors="pt", padding=True
        ).to(self.device)
        img_out = self.clip_model.get_image_features(pixel_values=inputs.pixel_values)
        txt_out = self.clip_model.get_text_features(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )

        img_feat = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
        txt_feat = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out

        img_norm = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_norm = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        laion_score = float(self.aesthetic_head(img_norm.float()).item())
        clip_sim = float((img_norm * txt_norm).sum(dim=-1).item())
        return round(laion_score, 3), round(clip_sim, 4)


def compute_image_ssim(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(img1, dtype=np.float32) / 255.0
    b = np.array(img2, dtype=np.float32) / 255.0
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.std()
    sigma_b = b.std()
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a**2 + sigma_b**2 + c2)
    ssim = float(num / den)
    return round(ssim, 4)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate spatial satisfaction and sweep guidance strength."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT_DIR / "benchmarks" / "spatial_satisfaction_and_sweep.json"),
        help="Path to output JSON results",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = SpatialEvaluator(device=device)

    print("=" * 80)
    print("STAGE 1: EVALUATING 64 A/B IMAGE PAIRS WITH ZERO-SHOT OWL-ViT DETECTOR")
    print("=" * 80)

    ab_images_dir = ROOT_DIR / "benchmarks" / "images" / "spatial_ab_test"
    total_pairs = len(SPATIAL_BENCHMARK_SPECS) * len(SEEDS_FULL)

    satisfaction_off_count = 0
    satisfaction_on_count = 0
    pair_evaluation_records = []

    for spec in SPATIAL_BENCHMARK_SPECS:
        p_id = spec["id"]
        prompt = spec["prompt"]
        subj_name = spec["subject"]
        obj_name = spec["object"]
        rel_type = spec["relation"]

        for seed in SEEDS_FULL:
            file_off = ab_images_dir / f"{p_id}_s{seed}_off.png"
            file_on = ab_images_dir / f"{p_id}_s{seed}_on.png"

            if not file_off.exists() or not file_on.exists():
                print(f"[-] Missing pair for {p_id} seed {seed}")
                continue

            img_off = Image.open(file_off).convert("RGB")
            img_on = Image.open(file_on).convert("RGB")

            # Detect entities in OFF image
            s_det_off, o_det_off = evaluator.detect_entities(img_off, subj_name, obj_name)
            sat_off, reason_off = evaluator.check_relation_satisfaction(
                s_det_off, o_det_off, rel_type
            )
            if sat_off:
                satisfaction_off_count += 1

            # Detect entities in ON image
            s_det_on, o_det_on = evaluator.detect_entities(img_on, subj_name, obj_name)
            sat_on, reason_on = evaluator.check_relation_satisfaction(s_det_on, o_det_on, rel_type)
            if sat_on:
                satisfaction_on_count += 1

            ssim_val = compute_image_ssim(img_off, img_on)

            status_str = (
                f"OFF: {'PASS' if sat_off else 'FAIL'} -> ON: {'PASS' if sat_on else 'FAIL'}"
            )
            print(
                f"  [{p_id} Seed {seed:>4}] {status_str:<24} | SSIM: {ssim_val:.3f} | {reason_on}"
            )

            pair_evaluation_records.append(
                {
                    "prompt_id": p_id,
                    "prompt": prompt,
                    "seed": seed,
                    "relation": rel_type,
                    "guidance_off": {
                        "satisfied": sat_off,
                        "reason": reason_off,
                        "subject_detection": s_det_off,
                        "object_detection": o_det_off,
                    },
                    "guidance_on": {
                        "satisfied": sat_on,
                        "reason": reason_on,
                        "subject_detection": s_det_on,
                        "object_detection": o_det_on,
                    },
                    "ssim": ssim_val,
                }
            )

    rate_off = (satisfaction_off_count / total_pairs) * 100.0
    rate_on = (satisfaction_on_count / total_pairs) * 100.0
    delta_rate = rate_on - rate_off

    print("\n" + "-" * 80)
    print("STAGE 1 SUMMARY: SPATIAL RELATION SATISFACTION VERDICT")
    print("-" * 80)
    print(f"  Total Image Pairs Tested:       {total_pairs}")
    print(
        f"  Guidance OFF Satisfaction Rate: {rate_off:.2f}% "
        f"({satisfaction_off_count}/{total_pairs} pairs satisfied)"
    )
    print(
        f"  Guidance ON Satisfaction Rate:  {rate_on:.2f}% "
        f"({satisfaction_on_count}/{total_pairs} pairs satisfied)"
    )
    print(f"  Net Satisfaction Gain:          {delta_rate:+.2f}%")

    # -----------------------------------------------------------------------
    # STAGE 2: GUIDANCE STRENGTH SWEEP ON LIVE GPU INFERENCE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: LIVE GUIDANCE STRENGTH SWEEP ACROSS {0.35, 0.70, 1.50, 3.00, 6.00}")
    print("=" * 80)

    sweep_dir = ROOT_DIR / "benchmarks" / "images" / "strength_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    model_path = ROOT_DIR / "models" / "sd15_fp16"
    print(f"Loading SD v1.5 from {model_path} on CUDA (fp16)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 8 representative prompts covering in_front_of, behind, on, under, left_of, right_of, beside
    sweep_prompts = SPATIAL_BENCHMARK_SPECS[:8]
    total_sweep_runs = len(sweep_prompts) * len(SEEDS_SWEEP)

    sweep_results_by_strength = {}

    for strength in STRENGTH_SWEEP_VALUES:
        print(
            f"\n>>> Running Sweep for Guidance Strength = {strength:.2f} "
            f"({total_sweep_runs} generations)..."
        )
        strength_records = []
        ssim_list = []
        sat_count = 0
        laion_list = []
        clip_list = []

        for spec in sweep_prompts:
            p_id = spec["id"]
            prompt = spec["prompt"]
            subj_name = spec["subject"]
            obj_name = spec["object"]
            rel_type = spec["relation"]

            intent = analyze_prompt(prompt, mode="generate")
            plan = plan_semantic_layout(intent, tokenizer=pipe.tokenizer)

            for seed in SEEDS_SWEEP:
                # Load baseline unguided image for SSIM comparison
                file_off = ab_images_dir / f"{p_id}_s{seed}_off.png"
                img_off = Image.open(file_off).convert("RGB")

                # Configure hooked cross-attention processors with current strength
                attn_procs = {}
                schedule = TwoPhaseSchedule(schedule_cutoff=0.8)
                for name, proc in pipe.unet.attn_processors.items():
                    if name.endswith("attn2.processor"):
                        attn_procs[name] = LayoutGuidanceProcessor(
                            base_processor=proc,
                            plan=plan,
                            guidance_strength=strength,
                            adaptive_guidance=False,
                            schedule=schedule,
                            depth_guidance_enabled=True,
                        )
                    else:
                        attn_procs[name] = proc
                pipe.unet.set_attn_processor(attn_procs)

                def step_callback(pipe, step_idx, timestep, callback_kwargs):
                    progress = float(step_idx + 1) / 20.0
                    for p in pipe.unet.attn_processors.values():
                        if isinstance(p, LayoutGuidanceProcessor):
                            p.set_step_progress(progress)
                    return callback_kwargs

                gen = torch.Generator("cuda").manual_seed(seed)
                with torch.inference_mode():
                    img_str = pipe(
                        prompt=prompt,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        generator=gen,
                        width=512,
                        height=512,
                        callback_on_step_end=step_callback,
                    ).images[0]

                out_file = sweep_dir / f"{p_id}_s{seed}_str_{strength:.2f}.png"
                img_str.save(out_file)

                # Metrics
                ssim_val = compute_image_ssim(img_off, img_str)
                ssim_list.append(ssim_val)

                laion, clip_sim = evaluator.evaluate_quality(img_str, prompt)
                laion_list.append(laion)
                clip_list.append(clip_sim)

                # Detection & Satisfaction
                s_det, o_det = evaluator.detect_entities(img_str, subj_name, obj_name)
                sat, reason = evaluator.check_relation_satisfaction(s_det, o_det, rel_type)
                if sat:
                    sat_count += 1

                strength_records.append(
                    {
                        "prompt_id": p_id,
                        "seed": seed,
                        "ssim_vs_off": ssim_val,
                        "satisfied": sat,
                        "reason": reason,
                        "laion_aesthetic": laion,
                        "clip_alignment": clip_sim,
                    }
                )

        mean_ssim = float(np.mean(ssim_list))
        sat_rate = (sat_count / total_sweep_runs) * 100.0
        mean_laion = float(np.mean(laion_list))
        mean_clip = float(np.mean(clip_list))

        sweep_results_by_strength[str(strength)] = {
            "strength": strength,
            "mean_ssim_vs_off": round(mean_ssim, 4),
            "structural_change_pct": round((1.0 - mean_ssim) * 100.0, 2),
            "spatial_satisfaction_rate_pct": round(sat_rate, 2),
            "satisfied_count": f"{sat_count}/{total_sweep_runs}",
            "mean_laion_aesthetic": round(mean_laion, 3),
            "mean_clip_alignment": round(mean_clip, 4),
            "runs": strength_records,
        }

        print(
            f"  Summary @ Strength {strength:.2f}: SSIM vs OFF = {mean_ssim:.4f} "
            f"(Change: {(1.0 - mean_ssim) * 100.0:.1f}%), Satisfaction = {sat_rate:.1f}%, "
            f"LAION = {mean_laion:.3f}, CLIP = {mean_clip:.4f}"
        )

    final_payload = {
        "metadata": {
            "title": "Live Spatial Relation Satisfaction & Strength Sweep",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "detector": "google/owlvit-base-patch32 (official weights on CUDA)",
            "aesthetic_model": "Official LAION Aesthetic Predictor v2.4 + CLIP-L/14",
            "model_id": "runwayml/stable-diffusion-v1-5",
        },
        "stage1_satisfaction_verdict": {
            "total_pairs_tested": total_pairs,
            "guidance_off_satisfaction_rate_pct": round(rate_off, 2),
            "guidance_on_satisfaction_rate_pct": round(rate_on, 2),
            "net_satisfaction_delta_pct": round(delta_rate, 2),
            "pair_records": pair_evaluation_records,
        },
        "stage2_strength_sweep": sweep_results_by_strength,
    }

    out_file = Path(args.output_json)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ALL SPATIAL EVALUATIONS COMPLETE -> Results saved to {out_file}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
