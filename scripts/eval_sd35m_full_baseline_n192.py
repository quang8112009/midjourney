from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_spatial_lateral_dedicated import LATERAL_24_SPECS, SEEDS_192
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator, wilson_score_interval

LOG_FILE = ROOT_DIR / "baseline_n192.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_full_baseline_n192():
    try:
        log("=" * 80)
        log("MEASURING SD 3.5 MEDIUM UNGUIDED BASELINE (OFF, N=192: 24 prompts x 8 seeds)")
        log("=" * 80)

        out_img_dir = ROOT_DIR / "benchmarks" / "images" / "sd35m_baseline_n192"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        total_runs = len(LATERAL_24_SPECS) * len(SEEDS_192)  # 24 * 8 = 192

        # 1. Pipeline Loading
        log("\n--- [Step 1] Loading SD 3.5 Medium (FP16, 512x512 / 20 steps) ---")
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "D:/midjourney/models/sd35_medium",
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()
        log("[+] Pipeline loaded on CUDA with CPU offload.")

        # 2. Generation Loop
        log(f"\n--- [Step 2] Generating {total_runs} Baseline Images (OFF / Unguided) ---")
        t_gen_start = time.time()
        run_idx = 0

        for spec in LATERAL_24_SPECS:
            pid = spec["id"]
            prompt = spec["prompt"]

            for seed in SEEDS_192:
                run_idx += 1
                img_name = f"{pid}_off_s{seed}.png"
                img_path = out_img_dir / img_name

                if img_path.exists():
                    log(f"  [{run_idx:03d}/{total_runs}] {pid} (s={seed}) | [CACHED]")
                    continue

                gen = torch.Generator(device="cpu").manual_seed(seed)
                t0 = time.time()
                with torch.inference_mode():
                    img = pipe(
                        prompt=prompt,
                        num_inference_steps=20,
                        guidance_scale=4.5,
                        width=512,
                        height=512,
                        generator=gen,
                    ).images[0]
                dt = time.time() - t0
                img.save(img_path)
                log(f"  [{run_idx:03d}/{total_runs}] {pid} (s={seed}) | Generated in {dt:.1f}s")
                gc.collect()
                torch.cuda.empty_cache()

        log(f"\n[+] All {total_runs} images ready in {time.time() - t_gen_start:.1f}s! Freeing pipeline from GPU...")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Strict Evaluation Loop
        log("\n--- [Step 3] Running Strict Spatial & Aesthetic Evaluation (OWL-ViT + CLIP-ViT-L/14) ---")
        evaluator = StrictSpatialEvaluator(device="cuda")

        records = []
        by_relation: dict[str, list[dict]] = {"left_of": [], "right_of": [], "beside": []}
        by_prompt: dict[str, list[dict]] = {s["id"]: [] for s in LATERAL_24_SPECS}

        for spec in LATERAL_24_SPECS:
            pid = spec["id"]
            prompt = spec["prompt"]
            subj = spec["subject"]
            obj = spec["object"]
            rel = spec["relation"]

            for seed in SEEDS_192:
                img_name = f"{pid}_off_s{seed}.png"
                img_path = out_img_dir / img_name
                img = Image.open(img_path).convert("RGB")

                s_det, o_det = evaluator.detect_entities(img, subj, obj)
                is_sat, details = evaluator.check_relation(s_det, o_det, rel)
                clip_sim, aes_score = evaluator.evaluate_quality(img, prompt)

                rec = {
                    "prompt_id": pid,
                    "prompt": prompt,
                    "subject": subj,
                    "object": obj,
                    "relation": rel,
                    "seed": seed,
                    "satisfied": is_sat,
                    "details": details,
                    "subject_detected": s_det is not None,
                    "object_detected": o_det is not None,
                    "subject_score": s_det["score"] if s_det else 0.0,
                    "object_score": o_det["score"] if o_det else 0.0,
                    "subject_box": s_det["box"] if s_det else None,
                    "object_box": o_det["box"] if o_det else None,
                    "clip_similarity": round(float(clip_sim), 4),
                    "laion_aesthetic_score": round(float(aes_score), 4),
                    "image_path": str(img_path),
                }
                records.append(rec)
                by_relation[rel].append(rec)
                by_prompt[pid].append(rec)

        # 4. Statistical Summary & Wilson CI
        total_n = len(records)
        total_sat = sum(1 for r in records if r["satisfied"])
        sat_rate = total_sat / total_n
        ci_low, ci_high = wilson_score_interval(total_sat, total_n, confidence=0.95)

        mean_clip = float(np.mean([r["clip_similarity"] for r in records]))
        mean_aes = float(np.mean([r["laion_aesthetic_score"] for r in records]))

        # Breakdown by relation type
        rel_summary = {}
        for rel_name, rel_recs in by_relation.items():
            rn = len(rel_recs)
            rsat = sum(1 for r in rel_recs if r["satisfied"])
            rrate = rsat / rn if rn > 0 else 0.0
            rci_l, rci_h = wilson_score_interval(rsat, rn, confidence=0.95)
            rel_summary[rel_name] = {
                "total": rn,
                "satisfied": rsat,
                "rate": round(rrate, 4),
                "wilson_ci_95": [round(rci_l, 4), round(rci_h, 4)],
            }

        # Breakdown by individual prompt
        prompt_summary = {}
        for pid, p_recs in by_prompt.items():
            pn = len(p_recs)
            psat = sum(1 for r in p_recs if r["satisfied"])
            prate = psat / pn if pn > 0 else 0.0
            prompt_summary[pid] = {
                "prompt": p_recs[0]["prompt"],
                "relation": p_recs[0]["relation"],
                "total": pn,
                "satisfied": psat,
                "rate": round(prate, 4),
            }

        log("\n" + "=" * 80)
        log("SD 3.5 MEDIUM UNGUIDED BASELINE RESULTS (N=192):")
        log("=" * 80)
        log(f"Overall Spatial Satisfaction: {sat_rate*100:.2f}% ({total_sat}/{total_n})")
        log(f"Wilson 95% Confidence Interval: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
        log(f"Mean CLIP Similarity:         {mean_clip:.4f}")
        log(f"Mean LAION Aesthetic Score:   {mean_aes:.4f}")
        log("\n--- Breakdown by Relation Type ---")
        for r_name, r_data in rel_summary.items():
            log(f"  - {r_name:<10}: {r_data['rate']*100:>5.1f}% ({r_data['satisfied']}/{r_data['total']}) | Wilson 95% CI: [{r_data['wilson_ci_95'][0]*100:.1f}%, {r_data['wilson_ci_95'][1]*100:.1f}%]")

        log("\n--- Prompt-by-Prompt Breakdown ---")
        for pid, p_data in prompt_summary.items():
            log(f"  - {pid} ({p_data['relation']:<8}): {p_data['rate']*100:>5.1f}% ({p_data['satisfied']}/{p_data['total']}) -> \"{p_data['prompt']}\"")

        log("\n--- Direct Comparison: SD 3.5 Medium vs. SD v1.5 Baseline ---")
        log("  - SD v1.5 Lateral Baseline (N=192):  34.90% (67/192)  [95% CI: 28.49%, 41.90%]")
        log(f"  - SD 3.5 Medium Baseline (N=192):    {sat_rate*100:.2f}% ({total_sat}/192) [95% CI: {ci_low*100:.2f}%, {ci_high*100:.2f}%]")

        # Save to JSON
        summary_out = {
            "model": "stabilityai/stable-diffusion-3.5-medium",
            "condition": "OFF (unguided)",
            "total_generations": total_n,
            "satisfied_count": total_sat,
            "satisfaction_rate": round(sat_rate, 4),
            "wilson_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "mean_clip_similarity": round(mean_clip, 4),
            "mean_laion_aesthetic_score": round(mean_aes, 4),
            "sd15_baseline_reference": {
                "satisfaction_rate": 0.3490,
                "satisfied_count": 67,
                "total_generations": 192,
                "wilson_ci_95": [0.2849, 0.4190],
            },
            "relation_breakdown": rel_summary,
            "prompt_breakdown": prompt_summary,
            "details": records,
        }

        out_file = ROOT_DIR / "benchmarks" / "sd35m_lateral_baseline_n192.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary_out, f, indent=2)
        log(f"\n[+] Saved full baseline benchmark to: {out_file}")
    except Exception as exc:
        log(f"[-] ERROR: {exc}")
        traceback.print_exc()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)


if __name__ == "__main__":
    run_full_baseline_n192()
