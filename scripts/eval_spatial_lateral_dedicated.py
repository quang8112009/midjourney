"""Dedicated Lateral-Only Spatial Benchmark (24 prompts x 8 seeds = 192 pairs/condition).

Evaluates 4 conditions: OFF (0.00), 1.50, 3.00, 6.00 (768 total images).
Calculates exact paired McNemar test statistics, discordant pair counts,
and Wilson 95% Confidence Intervals.
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
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image

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

LATERAL_24_SPECS = [
    {
        "id": "lat_01",
        "prompt": "a yellow banana to the left of a green apple on a wooden table",
        "subject": "yellow banana",
        "object": "green apple",
        "relation": "left_of",
    },
    {
        "id": "lat_02",
        "prompt": "a red car to the right of a blue bicycle on a street",
        "subject": "red car",
        "object": "blue bicycle",
        "relation": "right_of",
    },
    {
        "id": "lat_03",
        "prompt": "a crystal vase beside an antique brass clock on a shelf",
        "subject": "crystal vase",
        "object": "brass clock",
        "relation": "beside",
    },
    {
        "id": "lat_04",
        "prompt": "a silver teapot beside a porcelain teacup on a tray",
        "subject": "silver teapot",
        "object": "porcelain teacup",
        "relation": "beside",
    },
    {
        "id": "lat_05",
        "prompt": "a white coffee mug to the left of a black laptop on a wooden desk",
        "subject": "coffee mug",
        "object": "black laptop",
        "relation": "left_of",
    },
    {
        "id": "lat_06",
        "prompt": "a glass bottle to the right of a ceramic bowl on a marble counter",
        "subject": "glass bottle",
        "object": "ceramic bowl",
        "relation": "right_of",
    },
    {
        "id": "lat_07",
        "prompt": "a brown guitar beside a black amplifier in a studio",
        "subject": "brown guitar",
        "object": "black amplifier",
        "relation": "beside",
    },
    {
        "id": "lat_08",
        "prompt": "a green plant to the left of a tall floor lamp in a living room",
        "subject": "green plant",
        "object": "floor lamp",
        "relation": "left_of",
    },
    {
        "id": "lat_09",
        "prompt": "a golden trophy to the right of a framed photo on a wooden shelf",
        "subject": "golden trophy",
        "object": "framed photo",
        "relation": "right_of",
    },
    {
        "id": "lat_10",
        "prompt": "a red apple beside a yellow lemon on a cutting board",
        "subject": "red apple",
        "object": "yellow lemon",
        "relation": "beside",
    },
    {
        "id": "lat_11",
        "prompt": "a blue backpack to the left of a yellow skateboard on a sidewalk",
        "subject": "blue backpack",
        "object": "yellow skateboard",
        "relation": "left_of",
    },
    {
        "id": "lat_12",
        "prompt": "a silver fork to the left of a white plate on a dining table",
        "subject": "silver fork",
        "object": "white plate",
        "relation": "left_of",
    },
    {
        "id": "lat_13",
        "prompt": "a metal knife to the right of a white plate on a dining table",
        "subject": "metal knife",
        "object": "white plate",
        "relation": "right_of",
    },
    {
        "id": "lat_14",
        "prompt": "a plush teddy bear to the left of a toy robot on a carpet",
        "subject": "teddy bear",
        "object": "toy robot",
        "relation": "left_of",
    },
    {
        "id": "lat_15",
        "prompt": "a microscope to the right of a glass beaker on a laboratory bench",
        "subject": "microscope",
        "object": "glass beaker",
        "relation": "right_of",
    },
    {
        "id": "lat_16",
        "prompt": "a tennis racket beside a yellow tennis ball on a court",
        "subject": "tennis racket",
        "object": "tennis ball",
        "relation": "beside",
    },
    {
        "id": "lat_17",
        "prompt": "a leather wallet to the left of a smartphone on a glass table",
        "subject": "leather wallet",
        "object": "smartphone",
        "relation": "left_of",
    },
    {
        "id": "lat_18",
        "prompt": "a pair of sunglasses to the right of a straw hat on a beach towel",
        "subject": "sunglasses",
        "object": "straw hat",
        "relation": "right_of",
    },
    {
        "id": "lat_19",
        "prompt": "a vintage typewriter beside a desk lamp on a mahogany table",
        "subject": "vintage typewriter",
        "object": "desk lamp",
        "relation": "beside",
    },
    {
        "id": "lat_20",
        "prompt": "a blue bird to the left of a brown squirrel on a wooden bench",
        "subject": "blue bird",
        "object": "brown squirrel",
        "relation": "left_of",
    },
    {
        "id": "lat_21",
        "prompt": "a white sneaker to the right of an orange basketball on a gym floor",
        "subject": "white sneaker",
        "object": "orange basketball",
        "relation": "right_of",
    },
    {
        "id": "lat_22",
        "prompt": "a small desk fan beside a computer monitor on an office desk",
        "subject": "desk fan",
        "object": "computer monitor",
        "relation": "beside",
    },
    {
        "id": "lat_23",
        "prompt": "a black cat to the left of a golden dog on a green lawn",
        "subject": "black cat",
        "object": "golden dog",
        "relation": "left_of",
    },
    {
        "id": "lat_24",
        "prompt": "a glass candle holder to the right of a stack of books on a nightstand",
        "subject": "candle holder",
        "object": "stack of books",
        "relation": "right_of",
    },
]

SEEDS_192 = [42, 100, 555, 1024, 2024, 7777, 9999, 12345]
LATERAL_STRENGTHS = [0.00, 1.50, 3.00, 6.00]


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """Compute exact two-sided binomial p-value for McNemar test."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, float(p))


def asymptotic_mcnemar_chi2(b: int, c: int) -> tuple[float, float]:
    """Compute asymptotic McNemar chi-square with Edwards continuity correction."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = ((abs(b - c) - 1.0) ** 2) / float(n)
    # p-value from chi2 survival with df=1: p = 2 * (1 - Phi(sqrt(stat))) = erfc(sqrt(stat/2))
    p_val = math.erfc(math.sqrt(stat / 2.0))
    return round(stat, 4), round(p_val, 6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dedicated Lateral Spatial Relation Benchmark.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT_DIR / "benchmarks" / "lateral_dedicated_benchmark.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("DEDICATED LATERAL SPATIAL BENCHMARK (N=192 PER CONDITION: 24 PROMPTS x 8 SEEDS)")
    print("=" * 85)

    images_dir = ROOT_DIR / "benchmarks" / "images" / "lateral_dedicated_n192"
    images_dir.mkdir(parents=True, exist_ok=True)

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

    total_generations = len(LATERAL_STRENGTHS) * len(LATERAL_24_SPECS) * len(SEEDS_192)
    print(f"Total images: {total_generations} ({len(LATERAL_STRENGTHS)} conds x 192 runs)...")

    t0_start = time.time()

    # -----------------------------------------------------------------------
    # PHASE 1: GENERATE MISSING IMAGES
    # -----------------------------------------------------------------------
    print("\n--- PHASE 1: GENERATING IMAGES ---")
    for strength in LATERAL_STRENGTHS:
        str_label = f"strength_{strength:.2f}"
        print(f"\n>>> Generating Condition: {str_label} (N=192)...")
        cond_t0 = time.time()
        new_gens = 0

        for spec in LATERAL_24_SPECS:
            p_id = spec["id"]
            prompt = spec["prompt"]

            # Check if all seeds for this prompt and strength already exist
            missing_seeds = [
                s
                for s in SEEDS_192
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

    # Unload pipeline to free VRAM for evaluation
    del pipe
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # PHASE 2: EVALUATE QUALITY & SPATIAL RELATION SATISFACTION
    # -----------------------------------------------------------------------
    print("\n--- PHASE 2: EVALUATION (OWL-ViT + CLIP + SSIM) ---")
    evaluator = StrictSpatialEvaluator(device=device)

    # Load baseline images for SSIM
    baseline_images: dict[tuple[str, int], Image.Image] = {}
    for spec in LATERAL_24_SPECS:
        p_id = spec["id"]
        for seed in SEEDS_192:
            b_path = images_dir / f"{p_id}_s{seed}_str_0.00.png"
            loaded = Image.open(b_path).convert("RGB")
            baseline_images[(p_id, seed)] = loaded

    condition_records: dict[float, dict[tuple[str, int], dict[str, Any]]] = {
        s: {} for s in LATERAL_STRENGTHS
    }

    for strength in LATERAL_STRENGTHS:
        str_label = f"strength_{strength:.2f}"
        eval_t0 = time.time()
        for spec in LATERAL_24_SPECS:
            p_id = spec["id"]
            prompt = spec["prompt"]
            subj_name = spec["subject"]
            obj_name = spec["object"]
            rel_type = spec["relation"]

            for seed in SEEDS_192:
                img_path = images_dir / f"{p_id}_s{seed}_str_{strength:.2f}.png"
                image = Image.open(img_path).convert("RGB")

                # SSIM
                base_img = baseline_images[(p_id, seed)]
                ssim_val = compute_image_ssim(base_img, image)

                # Quality
                laion, clip_sim = evaluator.evaluate_quality(image, prompt)

                # Detection & Satisfaction
                s_det, o_det = evaluator.detect_entities(image, subj_name, obj_name)
                is_sat, reason = evaluator.check_relation(s_det, o_det, rel_type)

                condition_records[strength][(p_id, seed)] = {
                    "prompt_id": p_id,
                    "seed": seed,
                    "satisfied": is_sat,
                    "reason": reason,
                    "ssim_vs_off": ssim_val,
                    "laion_aesthetic": laion,
                    "clip_alignment": clip_sim,
                    "subject_det": s_det,
                    "object_det": o_det,
                }

        c_sat_count = sum(1 for rec in condition_records[strength].values() if rec["satisfied"])
        c_rate = (c_sat_count / len(condition_records[strength])) * 100.0
        c_low, c_high = wilson_score_interval(c_sat_count, len(condition_records[strength]))
        el_time = time.time() - eval_t0
        print(
            f"  Evaluated {str_label}: {c_sat_count}/192 ({c_rate:.2f}%) "
            f"[95% Wilson CI: {c_low:.1f}%, {c_high:.1f}%] in {el_time:.1f}s"
        )

    # -----------------------------------------------------------------------
    # PAIRED MCNEMAR ANALYSIS VS OFF (0.00)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("PAIRED MCNEMAR TEST RESULTS VS BASELINE OFF (0.00)")
    print("=" * 85)

    off_records = condition_records[0.00]
    off_sat_count = sum(1 for rec in off_records.values() if rec["satisfied"])
    off_rate = (off_sat_count / 192.0) * 100.0
    off_low, off_high = wilson_score_interval(off_sat_count, 192)

    off_laion = round(float(np.mean([r["laion_aesthetic"] for r in off_records.values()])), 3)
    off_clip = round(float(np.mean([r["clip_alignment"] for r in off_records.values()])), 4)

    summary_results: dict[str, Any] = {
        "strength_0.00": {
            "strength": 0.00,
            "n_samples": 192,
            "satisfied_count": f"{off_sat_count}/192",
            "satisfaction_rate_pct": round(off_rate, 2),
            "wilson_ci_95": [off_low, off_high],
            "mean_ssim_vs_off": 1.0,
            "mean_laion_aesthetic": off_laion,
            "mean_clip_alignment": off_clip,
        }
    }

    for strength in [1.50, 3.00, 6.00]:
        str_label = f"strength_{strength:.2f}"
        cur_records = condition_records[strength]
        cur_sat = sum(1 for rec in cur_records.values() if rec["satisfied"])
        cur_rate = (cur_sat / 192.0) * 100.0
        c_low, c_high = wilson_score_interval(cur_sat, 192)

        # Paired contingency table
        a = sum(
            1 for k in off_records if off_records[k]["satisfied"] and cur_records[k]["satisfied"]
        )
        b = sum(
            1
            for k in off_records
            if not off_records[k]["satisfied"] and cur_records[k]["satisfied"]
        )  # OFF fail -> ON pass
        c = sum(
            1
            for k in off_records
            if off_records[k]["satisfied"] and not cur_records[k]["satisfied"]
        )  # OFF pass -> ON fail
        d = sum(
            1
            for k in off_records
            if not off_records[k]["satisfied"] and not cur_records[k]["satisfied"]
        )

        exact_p = exact_mcnemar_p_value(b, c)
        chi2_stat, chi2_p = asymptotic_mcnemar_chi2(b, c)
        mean_ssim = round(float(np.mean([r["ssim_vs_off"] for r in cur_records.values()])), 4)
        mean_laion = round(float(np.mean([r["laion_aesthetic"] for r in cur_records.values()])), 3)
        mean_clip = round(float(np.mean([r["clip_alignment"] for r in cur_records.values()])), 4)

        summary_results[str_label] = {
            "strength": strength,
            "n_samples": 192,
            "satisfied_count": f"{cur_sat}/192",
            "satisfaction_rate_pct": round(cur_rate, 2),
            "wilson_ci_95": [c_low, c_high],
            "contingency_table": {
                "both_pass_a": a,
                "off_fail_on_pass_b": b,
                "off_pass_on_fail_c": c,
                "both_fail_d": d,
                "total_discordant_b_plus_c": b + c,
                "net_gain_b_minus_c": b - c,
            },
            "mcnemar_exact_p_value": exact_p,
            "mcnemar_chi2_statistic": chi2_stat,
            "mcnemar_chi2_p_value": chi2_p,
            "statistically_significant_alpha_0_05": exact_p < 0.05,
            "statistically_significant_alpha_0_01": exact_p < 0.01,
            "mean_ssim_vs_off": mean_ssim,
            "mean_laion_aesthetic": mean_laion,
            "mean_clip_alignment": mean_clip,
        }

        print(f"\nCondition {str_label}:")
        print(
            f"  Satisfaction: {cur_sat}/192 ({cur_rate:.2f}%) vs "
            f"Baseline {off_sat_count}/192 ({off_rate:.2f}%)"
        )
        print(
            f"  Contingency Table: a(both pass)={a}, b(OFF fail -> ON pass)={b}, "
            f"c(OFF pass -> ON fail)={c}, d(both fail)={d}"
        )
        print(f"  Discordant: b={b}, c={c} -> Net Gain: +{b - c}")
        sig_str = "SIGNIFICANT (p < 0.05)" if exact_p < 0.05 else "NOT SIGNIFICANT"
        print(f"  McNemar Exact p-value: {exact_p:.6e} ({sig_str})")
        print(f"  McNemar Chi2 (corrected): {chi2_stat:.4f} (p = {chi2_p:.6e})")
        print(f"  Quality: SSIM={mean_ssim:.4f}, LAION={mean_laion:.3f}, CLIP={mean_clip:.4f}")

    benchmark_artifact = {
        "metadata": {
            "title": "Dedicated Lateral Spatial Benchmark (24 Prompts x 8 Seeds = 192 Pairs)",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_id": "runwayml/stable-diffusion-v1-5",
            "device": "NVIDIA GeForce RTX 4060 Ti (CUDA fp16)",
            "seeds_tested": SEEDS_192,
            "prompts_tested": [s["prompt"] for s in LATERAL_24_SPECS],
            "total_images_generated": total_generations,
            "total_elapsed_seconds": round(time.time() - t0_start, 2),
        },
        "condition_summary": summary_results,
    }

    out_p = Path(args.output_json)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(benchmark_artifact, f, indent=2)

    print("\n" + "=" * 85)
    print(f"[+] Lateral dedicated benchmark successfully saved to: {out_p}")
    print("=" * 85)
    return 0


if __name__ == "__main__":
    sys.exit(main())
