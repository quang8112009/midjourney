"""Dedicated Depth Spatial Relation Benchmark (24 prompts x 8 seeds = 192 pairs/condition).

Evaluates conditions: OFF (0.00), 3.00, 6.00.
Includes:
1. Re-scoring existing depth runs with Depth Anything V2 Monocular Depth Estimator.
2. Full N=192 paired evaluation comparing 2D Bounding-Box predicate vs True 3D Depth Predicate.
3. Exact McNemar tests and Wilson 95% CIs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.editing.layout_guidance import (  # noqa: E402
    LayoutGuidanceProcessor,
    TwoPhaseSchedule,
)
from app.services.editing.prompt_intent import analyze_prompt  # noqa: E402
from app.services.editing.semantic_planner import plan_semantic_layout  # noqa: E402
from scripts.eval_spatial_rigorous_benchmark import (  # noqa: E402
    StrictSpatialEvaluator,
    compute_image_ssim,
    wilson_score_interval,
)

DEPTH_24_SPECS = [
    {
        "id": "dep_01",
        "prompt": "a red cube in front of a blue sphere on a marble floor",
        "subject": "red cube",
        "object": "blue sphere",
        "relation": "in_front_of",
    },
    {
        "id": "dep_02",
        "prompt": "a golden statue behind a stone fountain in a courtyard",
        "subject": "golden statue",
        "object": "stone fountain",
        "relation": "behind",
    },
    {
        "id": "dep_03",
        "prompt": "a green chair in front of a brick fireplace in a living room",
        "subject": "green chair",
        "object": "brick fireplace",
        "relation": "in_front_of",
    },
    {
        "id": "dep_04",
        "prompt": "a tall oak tree behind a wooden cottage in a meadow",
        "subject": "tall oak tree",
        "object": "wooden cottage",
        "relation": "behind",
    },
    {
        "id": "dep_05",
        "prompt": "a ceramic vase in front of a mirror on a dresser",
        "subject": "ceramic vase",
        "object": "mirror",
        "relation": "in_front_of",
    },
    {
        "id": "dep_06",
        "prompt": "a mountain peak behind a log cabin in a forest",
        "subject": "mountain peak",
        "object": "log cabin",
        "relation": "behind",
    },
    {
        "id": "dep_07",
        "prompt": "a wooden stool in front of a bookshelf in a library",
        "subject": "wooden stool",
        "object": "bookshelf",
        "relation": "in_front_of",
    },
    {
        "id": "dep_08",
        "prompt": "a stone wall behind a flowering rose bush in a garden",
        "subject": "stone wall",
        "object": "rose bush",
        "relation": "behind",
    },
    {
        "id": "dep_09",
        "prompt": "a glass bottle in front of a fruit basket on a wooden table",
        "subject": "glass bottle",
        "object": "fruit basket",
        "relation": "in_front_of",
    },
    {
        "id": "dep_10",
        "prompt": "a brick chimney behind a steep roof on a house",
        "subject": "brick chimney",
        "object": "steep roof",
        "relation": "behind",
    },
    {
        "id": "dep_11",
        "prompt": "a bronze sculpture in front of a red curtain in a room",
        "subject": "bronze sculpture",
        "object": "red curtain",
        "relation": "in_front_of",
    },
    {
        "id": "dep_12",
        "prompt": "a lighthouse behind a rocky cliff on a coast",
        "subject": "lighthouse",
        "object": "rocky cliff",
        "relation": "behind",
    },
    {
        "id": "dep_13",
        "prompt": "a red bicycle in front of a wooden door on a house",
        "subject": "red bicycle",
        "object": "wooden door",
        "relation": "in_front_of",
    },
    {
        "id": "dep_14",
        "prompt": "a water tower behind an industrial warehouse on a street",
        "subject": "water tower",
        "object": "warehouse",
        "relation": "behind",
    },
    {
        "id": "dep_15",
        "prompt": "a potted cactus in front of a glass window in a sunroom",
        "subject": "potted cactus",
        "object": "glass window",
        "relation": "in_front_of",
    },
    {
        "id": "dep_16",
        "prompt": "a pine forest behind a rustic wooden fence in a field",
        "subject": "pine forest",
        "object": "wooden fence",
        "relation": "behind",
    },
    {
        "id": "dep_17",
        "prompt": "a leather armchair in front of a floor lamp in a study",
        "subject": "leather armchair",
        "object": "floor lamp",
        "relation": "in_front_of",
    },
    {
        "id": "dep_18",
        "prompt": "a clock tower behind a stone archway in a town square",
        "subject": "clock tower",
        "object": "stone archway",
        "relation": "behind",
    },
    {
        "id": "dep_19",
        "prompt": "a ceramic bowl in front of a coffee machine on a kitchen counter",
        "subject": "ceramic bowl",
        "object": "coffee machine",
        "relation": "in_front_of",
    },
    {
        "id": "dep_20",
        "prompt": "a mountain behind a wooden windmill on a grassy hill",
        "subject": "mountain",
        "object": "wooden windmill",
        "relation": "behind",
    },
    {
        "id": "dep_21",
        "prompt": "a wooden violin in front of a music stand on a stage",
        "subject": "wooden violin",
        "object": "music stand",
        "relation": "in_front_of",
    },
    {
        "id": "dep_22",
        "prompt": "a city skyline behind a suspension bridge over a river",
        "subject": "city skyline",
        "object": "bridge",
        "relation": "behind",
    },
    {
        "id": "dep_23",
        "prompt": "a telephone in front of a desk calendar on an office desk",
        "subject": "telephone",
        "object": "desk calendar",
        "relation": "in_front_of",
    },
    {
        "id": "dep_24",
        "prompt": "a medieval castle behind a stone bridge over a river",
        "subject": "medieval castle",
        "object": "stone bridge",
        "relation": "behind",
    },
]

SEEDS_192 = [42, 100, 555, 1024, 2024, 7777, 9999, 12345]
DEPTH_STRENGTHS = [0.00, 3.00, 6.00]


class MonocularDepthEvaluator:
    """Evaluates relative 3D camera depth ordering using Depth Anything V2."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        model_dir = ROOT_DIR / "models" / "depth_anything_v2_small"
        self.processor = AutoImageProcessor.from_pretrained(str(model_dir))
        self.model = AutoModelForDepthEstimation.from_pretrained(str(model_dir)).to(device).eval()

    @torch.inference_mode()
    def estimate_relative_depth(
        self,
        image: Image.Image,
        subj_det: dict[str, Any] | None,
        obj_det: dict[str, Any] | None,
        relation: str,
    ) -> tuple[bool, float, float, str]:
        """Estimate relative depth between detected subject and object.

        Returns: (is_satisfied, subj_mean_depth, obj_mean_depth, reason)
        In Depth Anything V2, higher depth value = closer to camera.
        """
        if subj_det is None or obj_det is None:
            return False, 0.0, 0.0, "Missing entity detection"

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        depth_raw = outputs.predicted_depth.unsqueeze(1)
        depth_map = F.interpolate(
            depth_raw,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        h, w = image.height, image.width
        s_box = [
            int(subj_det["box"][0] * h),
            int(subj_det["box"][1] * w),
            int(subj_det["box"][2] * h),
            int(subj_det["box"][3] * w),
        ]
        o_box = [
            int(obj_det["box"][0] * h),
            int(obj_det["box"][1] * w),
            int(obj_det["box"][2] * h),
            int(obj_det["box"][3] * w),
        ]

        # Clamp boxes
        s_box = [max(0, s_box[0]), max(0, s_box[1]), min(h, s_box[2]), min(w, s_box[3])]
        o_box = [max(0, o_box[0]), max(0, o_box[1]), min(h, o_box[2]), min(w, o_box[3])]

        s_crop = depth_map[s_box[0]:s_box[2], s_box[1]:s_box[3]]
        o_crop = depth_map[o_box[0]:o_box[2], o_box[1]:o_box[3]]

        s_mean = float(np.mean(s_crop)) if s_crop.size > 0 else 0.0
        o_mean = float(np.mean(o_crop)) if o_crop.size > 0 else 0.0

        delta_depth = s_mean - o_mean

        if relation in ("in_front_of", "far_in_front_of", "front"):
            # Subject must be closer to camera (higher disparity)
            is_sat = delta_depth >= 0.05
            reason = f"s_depth={s_mean:.3f} > o_depth={o_mean:.3f} (delta={delta_depth:+.3f})"
            return is_sat, round(s_mean, 4), round(o_mean, 4), reason
        elif relation in ("behind", "far_behind", "back"):
            # Subject must be further from camera (lower disparity)
            is_sat = delta_depth <= -0.05
            reason = f"s_depth={s_mean:.3f} < o_depth={o_mean:.3f} (delta={delta_depth:+.3f})"
            return is_sat, round(s_mean, 4), round(o_mean, 4), reason

        return False, s_mean, o_mean, f"Unknown depth relation {relation}"


def exact_mcnemar_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, float(p))


def asymptotic_mcnemar_chi2(b: int, c: int) -> tuple[float, float]:
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = ((abs(b - c) - 1.0) ** 2) / float(n)
    p_val = math.erfc(math.sqrt(stat / 2.0))
    return round(stat, 4), round(p_val, 6)


def run_depth_benchmark() -> int:
    parser = argparse.ArgumentParser(description="Dedicated Depth Spatial Benchmark.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT_DIR / "benchmarks" / "depth_dedicated_benchmark.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("DEDICATED DEPTH SPATIAL BENCHMARK (N=192 PAIRS PER CONDITION: 24 PROMPTS x 8 SEEDS)")
    print("=" * 85)

    images_dir = ROOT_DIR / "benchmarks" / "images" / "depth_dedicated_n192"
    images_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 1: GENERATION PASS
    # -----------------------------------------------------------------------
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

    print("\n--- GENERATION PASS (OFF, 3.00, 6.00) ---")
    for strength in DEPTH_STRENGTHS:
        str_label = f"strength_{strength:.2f}"
        print(f"\n>>> Generating Condition: {str_label} (N=192)...")
        cond_t0 = time.time()
        new_gens = 0

        for spec in DEPTH_24_SPECS:
            p_id = spec["id"]
            prompt = spec["prompt"]

            missing_seeds = [
                s for s in SEEDS_192
                if not (images_dir / f"{p_id}_s{s}_str_{strength:.2f}.png").exists()
            ]
            if not missing_seeds:
                continue

            intent = analyze_prompt(prompt, mode="generate")
            plan = plan_semantic_layout(intent, tokenizer=pipe.tokenizer)

            if strength == 0.0:
                pipe.unet.set_attn_processor(
                    {k: AttnProcessor() for k in pipe.unet.attn_processors.keys()}
                )
                for seed in missing_seeds:
                    img_path = images_dir / f"{p_id}_s{seed}_str_{strength:.2f}.png"
                    gen = torch.Generator("cuda").manual_seed(seed)
                    with torch.inference_mode():
                        image = pipe(
                            prompt=prompt,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            generator=gen,
                            width=512,
                            height=512,
                        ).images[0]
                    image.save(img_path)
                    new_gens += 1
            else:
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

                for seed in missing_seeds:
                    img_path = images_dir / f"{p_id}_s{seed}_str_{strength:.2f}.png"
                    gen = torch.Generator("cuda").manual_seed(seed)
                    with torch.inference_mode():
                        image = pipe(
                            prompt=prompt,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            generator=gen,
                            width=512,
                            height=512,
                            callback_on_step_end=step_callback,
                        ).images[0]
                    image.save(img_path)
                    new_gens += 1

        print(f"  {str_label}: Generated {new_gens} new images in {time.time() - cond_t0:.1f}s")

    del pipe
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # STEP 2: EVALUATION PASS (2D Heuristic + True 3D Depth Anything V2)
    # -----------------------------------------------------------------------
    print("\n--- EVALUATION PASS (OWL-ViT + Depth Anything V2 + CLIP) ---")
    evaluator_2d = StrictSpatialEvaluator(device=device)
    evaluator_3d = MonocularDepthEvaluator(device=device)

    # Cache baseline images
    baseline_images: dict[tuple[str, int], Image.Image] = {}
    for spec in DEPTH_24_SPECS:
        p_id = spec["id"]
        for seed in SEEDS_192:
            b_path = images_dir / f"{p_id}_s{seed}_str_0.00.png"
            baseline_images[(p_id, seed)] = Image.open(b_path).convert("RGB")

    condition_records: dict[float, dict[tuple[str, int], dict[str, Any]]] = {
        s: {} for s in DEPTH_STRENGTHS
    }

    for strength in DEPTH_STRENGTHS:
        str_label = f"strength_{strength:.2f}"
        eval_t0 = time.time()
        for spec in DEPTH_24_SPECS:
            p_id = spec["id"]
            prompt = spec["prompt"]
            subj_name = spec["subject"]
            obj_name = spec["object"]
            rel_type = spec["relation"]

            for seed in SEEDS_192:
                img_path = images_dir / f"{p_id}_s{seed}_str_{strength:.2f}.png"
                image = Image.open(img_path).convert("RGB")

                ssim_val = compute_image_ssim(baseline_images[(p_id, seed)], image)
                laion, clip_sim = evaluator_2d.evaluate_quality(image, prompt)

                # 2D detection
                s_det, o_det = evaluator_2d.detect_entities(image, subj_name, obj_name)
                sat_2d, r_2d = evaluator_2d.check_relation(s_det, o_det, rel_type)

                # True 3D Depth estimation
                sat_3d, s_depth, o_depth, r_3d = evaluator_3d.estimate_relative_depth(
                    image, s_det, o_det, rel_type
                )

                condition_records[strength][(p_id, seed)] = {
                    "prompt_id": p_id,
                    "seed": seed,
                    "ssim_vs_off": ssim_val,
                    "laion_aesthetic": laion,
                    "clip_alignment": clip_sim,
                    "sat_2d": sat_2d,
                    "reason_2d": r_2d,
                    "sat_3d": sat_3d,
                    "s_depth_3d": s_depth,
                    "o_depth_3d": o_depth,
                    "reason_3d": r_3d,
                    "subject_det": s_det,
                    "object_det": o_det,
                }

        c_2d = sum(1 for r in condition_records[strength].values() if r["sat_2d"])
        c_3d = sum(1 for r in condition_records[strength].values() if r["sat_3d"])
        rate_2d = (c_2d / 192.0) * 100.0
        rate_3d = (c_3d / 192.0) * 100.0
        ci_2d_l, ci_2d_h = wilson_score_interval(c_2d, 192)
        ci_3d_l, ci_3d_h = wilson_score_interval(c_3d, 192)
        el_time = time.time() - eval_t0

        print(f"\nCondition {str_label} (N=192 evaluated in {el_time:.1f}s):")
        print(
            f"  2D Ground-Plane Metric:   {c_2d}/192 ({rate_2d:.2f}%) "
            f"[95% CI: {ci_2d_l:.1f}%, {ci_2d_h:.1f}%]"
        )
        print(
            f"  True 3D Depth Anything:   {c_3d}/192 ({rate_3d:.2f}%) "
            f"[95% CI: {ci_3d_l:.1f}%, {ci_3d_h:.1f}%]"
        )

    # -----------------------------------------------------------------------
    # STEP 3: STATISTICAL PAIRED ANALYSIS (MCNEMAR TESTS)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("PAIRED MCNEMAR TESTS VS BASELINE OFF (0.00)")
    print("=" * 85)

    off_recs = condition_records[0.00]
    summary_results: dict[str, Any] = {}

    for metric_name in ["sat_2d", "sat_3d"]:
        m_label = (
            "2D Ground-Plane Metric" if metric_name == "sat_2d" else "True 3D Depth Anything V2"
        )
        print(f"\n=== METRIC: {m_label} ===")

        off_sat = sum(1 for r in off_recs.values() if r[metric_name])
        off_r = (off_sat / 192.0) * 100.0
        print(f"  OFF (0.00) Baseline: {off_sat}/192 ({off_r:.2f}%)")

        for strength in [3.00, 6.00]:
            cur_recs = condition_records[strength]
            cur_sat = sum(1 for r in cur_recs.values() if r[metric_name])
            cur_r = (cur_sat / 192.0) * 100.0

            a = sum(
                1 for k in off_recs if off_recs[k][metric_name] and cur_recs[k][metric_name]
            )
            b = sum(
                1 for k in off_recs if not off_recs[k][metric_name] and cur_recs[k][metric_name]
            )
            c = sum(
                1 for k in off_recs if off_recs[k][metric_name] and not cur_recs[k][metric_name]
            )
            d = sum(
                1 for k in off_recs if not off_recs[k][metric_name] and not cur_recs[k][metric_name]
            )

            exact_p = exact_mcnemar_p_value(b, c)
            chi2_stat, chi2_p = asymptotic_mcnemar_chi2(b, c)

            sig_str = "SIGNIFICANT (p < 0.05)" if exact_p < 0.05 else "NOT SIGNIFICANT"
            print(f"\n  Condition Strength {strength:.2f}:")
            print(
                f"    Satisfaction: {cur_sat}/192 ({cur_r:.2f}%) vs "
                f"Baseline {off_sat}/192 ({off_r:.2f}%)"
            )
            print(
                f"    Pairs: a(both pass)={a}, b(gain)={b}, c(loss)={c}, "
                f"d(both fail)={d} -> Net: {b-c:+d}"
            )
            print(
                f"    McNemar Exact p-value: {exact_p:.6e} ({sig_str}) | "
                f"Chi2: {chi2_stat:.4f} (p={chi2_p:.6e})"
            )

            summary_results[f"{metric_name}_str_{strength:.2f}"] = {
                "strength": strength,
                "metric": metric_name,
                "satisfied_count": f"{cur_sat}/192",
                "rate_pct": round(cur_r, 2),
                "discordant_gain_b": b,
                "discordant_loss_c": c,
                "net_gain": b - c,
                "mcnemar_exact_p_value": exact_p,
                "significant": exact_p < 0.05,
            }

    # Dump JSON artifact
    out_payload = {
        "metadata": {
            "title": "Dedicated Depth Spatial Benchmark (N=192 Paired Runs per Condition)",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_id": "runwayml/stable-diffusion-v1-5",
            "depth_estimator": "depth-anything/Depth-Anything-V2-Small-hf",
            "prompts_tested": [s["prompt"] for s in DEPTH_24_SPECS],
            "seeds_tested": SEEDS_192,
        },
        "summary": summary_results,
    }

    out_p = Path(args.output_json)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print("\n" + "=" * 85)
    print(f"[+] Full depth benchmark artifact saved to: {out_p}")
    print("=" * 85)
    return 0


if __name__ == "__main__":
    sys.exit(run_depth_benchmark())
