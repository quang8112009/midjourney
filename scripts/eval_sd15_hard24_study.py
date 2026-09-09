"""Hard 24 Lateral Powered Study on Stable Diffusion v1.5 (UNet + CLIP-L, N=192 Paired Runs).

Evaluates 24 Hard lateral prompts x 8 seeds at OFF (0.00) vs Guided (Strength 6.00).
Uses the established LayoutGuidanceProcessor that dynamically scales across UNet cross-attention block resolutions.
Strict 2-pass architecture:
Pass 1: Load SD v1.5, generate all 384 images to disk, free SD v1.5.
Pass 2: Load Scorers, score all 384 images from disk, calculate exact McNemar and stats.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image

from app.services.editing.layout_guidance import LayoutGuidanceProcessor, TwoPhaseSchedule
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout
from scripts.eval_spatial_lateral_dedicated import (
    SEEDS_192,
    exact_mcnemar_p_value,
    wilson_score_interval,
)
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator, compute_image_ssim
from scripts.hard_lateral_specs import LATERAL_HARD_24_SPECS
from scripts.run_live_aesthetic_baseline import RealAestheticEvaluator, compute_file_sha256

STRENGTHS = [0.0, 6.0]
LOG_FILE = ROOT_DIR / "sd15_hard24_study.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_vram(stage: str):
    alloc = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    log(f"[{stage}] Allocated: {alloc:.2f} GB | Peak: {peak:.2f} GB")


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    t_start = time.time()
    log("=" * 90)
    log("POWERED HARD 24 LATERAL STUDY ON STABLE DIFFUSION v1.5 (N=192 PAIRED RUNS)")
    log("=" * 90)

    device = torch.device("cuda")
    out_dir = ROOT_DIR / "benchmarks" / "images" / "sd15_hard24_n192"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # PASS 1: GENERATION PASS (SD v1.5 Only)
    # -------------------------------------------------------------
    log("\n>>> PASS 1: GENERATION PASS (SD v1.5 Only)")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    log_vram("Pre-Load")

    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        "models/sd15_fp16",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    log(f"[+] SD v1.5 pipeline ready in {time.time() - t0:.2f}s.")
    log_vram("SD 1.5 Loaded")

    plans = {}
    for spec in LATERAL_HARD_24_SPECS:
        intent = analyze_prompt(spec["prompt"], mode="generate")
        plans[spec["id"]] = plan_semantic_layout(intent, tokenizer=pipe.tokenizer)

    for str_val in STRENGTHS:
        log(f"\n--- Generating 192 images at Strength = {str_val:.2f} ---")
        t_str0 = time.time()
        gen_times = []

        for spec in LATERAL_HARD_24_SPECS:
            pid = spec["id"]
            prompt = spec["prompt"]
            plan = plans[pid]

            if str_val == 0.0:
                pipe.unet.set_attn_processor(
                    {k: AttnProcessor() for k in pipe.unet.attn_processors.keys()}
                )
                callback = None
            else:
                attn_procs = {}
                schedule = TwoPhaseSchedule(schedule_cutoff=0.8)
                for name, proc in pipe.unet.attn_processors.items():
                    if name.endswith("attn2.processor"):
                        attn_procs[name] = LayoutGuidanceProcessor(
                            base_processor=proc,
                            plan=plan,
                            guidance_strength=str_val,
                            adaptive_guidance=False,
                            schedule=schedule,
                            depth_guidance_enabled=True,
                        )
                    else:
                        attn_procs[name] = proc
                pipe.unet.set_attn_processor(attn_procs)

                def step_callback(pipe_inst, step_idx, timestep, callback_kwargs):
                    progress = float(step_idx + 1) / 20.0
                    for p in pipe_inst.unet.attn_processors.values():
                        if isinstance(p, LayoutGuidanceProcessor):
                            p.set_step_progress(progress)
                    return callback_kwargs

                callback = step_callback

            for seed in SEEDS_192:
                filename = f"{pid}_s{seed}_str_{str_val:.2f}.png"
                img_path = out_dir / filename

                # Generate new or overwrite corrupted
                gen = torch.Generator(device="cuda").manual_seed(seed)
                t_img0 = time.time()
                with torch.inference_mode():
                    call_kwargs = {
                        "prompt": prompt,
                        "num_inference_steps": 20,
                        "guidance_scale": 7.5,
                        "width": 512,
                        "height": 512,
                        "generator": gen,
                    }
                    if callback is not None:
                        call_kwargs["callback_on_step_end"] = callback
                    img = pipe(**call_kwargs).images[0]
                dt = time.time() - t_img0
                img.save(img_path)
                gen_times.append(dt)

        dt_total = time.time() - t_str0
        mean_dt = sum(gen_times) / max(len(gen_times), 1) if gen_times else 0.0
        log(f"[+] Finished strength {str_val:.2f} in {dt_total:.1f}s (Mean generation: {mean_dt:.3f} s/img).")
        log_vram(f"Post-Gen str={str_val:.2f}")

    # Explicitly unload pipeline
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    log("\n[+] SD v1.5 pipeline freed.")
    log_vram("Post-Free SD 1.5")

    # -------------------------------------------------------------
    # PASS 2: SCORING PASS (Scorers ONLY)
    # -------------------------------------------------------------
    log("\n>>> PASS 2: SCORING FROM DISK (Scorers Only)")
    torch.cuda.reset_peak_memory_stats()
    evaluator_spatial = StrictSpatialEvaluator(device=device)
    evaluator_aes = RealAestheticEvaluator(device=device)
    log_vram("Scorers Loaded")

    results_by_strength: dict[float, dict[tuple[str, int], dict[str, Any]]] = {s: {} for s in STRENGTHS}
    baseline_images: dict[tuple[str, int], Image.Image] = {}

    for spec in LATERAL_HARD_24_SPECS:
        pid = spec["id"]
        for seed in SEEDS_192:
            base_path = out_dir / f"{pid}_s{seed}_str_0.00.png"
            baseline_images[(pid, seed)] = Image.open(base_path).convert("RGB")

    t_score0 = time.time()
    for str_val in STRENGTHS:
        log(f"Scoring 192 images at Strength = {str_val:.2f}...")
        for spec in LATERAL_HARD_24_SPECS:
            pid = spec["id"]
            prompt = spec["prompt"]
            subj = spec["subject"]
            obj = spec["object"]
            rel = spec["relation"]

            for seed in SEEDS_192:
                filename = f"{pid}_s{seed}_str_{str_val:.2f}.png"
                img_path = out_dir / filename
                img = Image.open(img_path).convert("RGB")
                sha256 = compute_file_sha256(img_path)

                s_det, o_det = evaluator_spatial.detect_entities(img, subj, obj)
                sat, reason = evaluator_spatial.check_relation(s_det, o_det, rel)
                both_present = bool(s_det and o_det)

                base_img = baseline_images[(pid, seed)]
                ssim_val = compute_image_ssim(base_img, img)
                aes_scores = evaluator_aes.evaluate(img, prompt)

                results_by_strength[str_val][(pid, seed)] = {
                    "id": pid,
                    "seed": seed,
                    "prompt": prompt,
                    "subject": subj,
                    "object": obj,
                    "relation": rel,
                    "satisfied": bool(sat),
                    "reason": reason,
                    "both_present": both_present,
                    "s_det": s_det,
                    "o_det": o_det,
                    "ssim_vs_off": ssim_val,
                    "laion_aesthetic": aes_scores["laion_aesthetic_v2_4"],
                    "clip_alignment": aes_scores["clip_alignment"],
                    "image_path": str(img_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                    "sha256": sha256,
                }

    log(f"[+] Finished scoring in {time.time() - t_score0:.1f}s.")
    log_vram("Post-Scoring")

    del evaluator_spatial
    del evaluator_aes
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Statistical Analysis & McNemar Calculations
    conditions_summary = {}
    base_records = results_by_strength[0.0]

    log("\n" + "=" * 115)
    log("SD v1.5 HARD 24 SUMMARY & STATISTICAL BREAKDOWN:")
    log("=" * 115)

    for str_val in STRENGTHS:
        cur_records = results_by_strength[str_val]
        total_runs = len(cur_records)
        sat_runs = sum(1 for r in cur_records.values() if r["satisfied"])
        pres_runs = sum(1 for r in cur_records.values() if r["both_present"])

        sat_rate = sat_runs / total_runs
        pres_rate = pres_runs / total_runs
        ci_sat = wilson_score_interval(sat_runs, total_runs)
        ci_pres = wilson_score_interval(pres_runs, total_runs)

        left_runs = [r for r in cur_records.values() if r["relation"] == "left_of"]
        left_sat = sum(1 for r in left_runs if r["satisfied"])
        left_rate = left_sat / len(left_runs)

        right_runs = [r for r in cur_records.values() if r["relation"] == "right_of"]
        right_sat = sum(1 for r in right_runs if r["satisfied"])
        right_rate = right_sat / len(right_runs)

        omissions = sum(1 for r in cur_records.values() if not r["both_present"])
        misplacements = sum(1 for r in cur_records.values() if r["both_present"] and not r["satisfied"])

        mean_ssim = float(np.mean([r["ssim_vs_off"] for r in cur_records.values()]))
        mean_laion = float(np.mean([r["laion_aesthetic"] for r in cur_records.values()]))
        mean_clip = float(np.mean([r["clip_alignment"] for r in cur_records.values()]))

        mcnemar_data = None
        if str_val > 0.0:
            both_pass = 0
            gain_b = 0
            loss_c = 0
            both_fail = 0
            for k in cur_records.keys():
                off_s = base_records[k]["satisfied"]
                on_s = cur_records[k]["satisfied"]
                if off_s and on_s:
                    both_pass += 1
                elif not off_s and on_s:
                    gain_b += 1
                elif off_s and not on_s:
                    loss_c += 1
                else:
                    both_fail += 1

            p_val = exact_mcnemar_p_value(gain_b, loss_c)
            is_sig = bool(p_val < 0.05)
            net_gain = gain_b - loss_c
            mcnemar_data = {
                "contingency_table": {
                    "both_pass_a": both_pass,
                    "gain_b": gain_b,
                    "loss_c": loss_c,
                    "both_fail_d": both_fail,
                },
                "net_gain": net_gain,
                "exact_mcnemar_p_value": p_val,
                "statistically_significant": is_sig,
            }

        conditions_summary[f"{str_val:.1f}"] = {
            "strength": str_val,
            "total_runs": total_runs,
            "satisfied_count": sat_runs,
            "satisfaction_rate": round(sat_rate, 4),
            "wilson_ci_95": [round(ci_sat[0] * 100, 2), round(ci_sat[1] * 100, 2)],
            "presence_count": pres_runs,
            "presence_rate": round(pres_rate, 4),
            "presence_ci_95": [round(ci_pres[0] * 100, 2), round(ci_pres[1] * 100, 2)],
            "directional": {
                "total": total_runs,
                "satisfied": sat_runs,
                "rate": round(sat_rate, 4),
                "wilson_ci_95": [round(ci_sat[0] * 100, 2), round(ci_sat[1] * 100, 2)],
                "left_of": {"total": len(left_runs), "satisfied": left_sat, "rate": round(left_rate, 4)},
                "right_of": {"total": len(right_runs), "satisfied": right_sat, "rate": round(right_rate, 4)},
            },
            "failure_modes": {
                "omissions": omissions,
                "misplacements": misplacements,
            },
            "mean_ssim": round(mean_ssim, 4),
            "mean_laion_aesthetic": round(mean_laion, 3),
            "mean_clip_similarity": round(mean_clip, 4),
            "mcnemar_vs_off": mcnemar_data,
        }

        p_str = f"p={mcnemar_data['exact_mcnemar_p_value']:.4e}" if mcnemar_data else "Baseline (OFF)"
        net_str = f"Net: +{mcnemar_data['net_gain']}" if mcnemar_data else ""
        log(f"Strength {str_val:4.1f} | Directional: {sat_rate*100:5.2f}% ({sat_runs:3d}/192) [{ci_sat[0]*100:.1f}%, {ci_sat[1]*100:.1f}%] [L:{left_rate*100:.1f}%, R:{right_rate*100:.1f}%] | Pres: {pres_rate*100:5.2f}% | SSIM: {mean_ssim:.4f} | {p_str} {net_str}")

    log("=" * 115)

    # 5. Save Full Dataset
    payload = {
        "model": "runwayml/stable-diffusion-v1-5",
        "benchmark": "Hard 24 Lateral Powered Study",
        "resolution": "512x512",
        "steps": 20,
        "seeds": SEEDS_192,
        "conditions": conditions_summary,
        "details": {
            str(s): [r for r in results_by_strength[s].values()]
            for s in STRENGTHS
        },
    }

    out_json = ROOT_DIR / "benchmarks" / "sd15_hard24_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    total_time = time.time() - t_start
    log(f"\n[+] Powered Hard 24 study on SD v1.5 completed in {total_time:.1f}s ({total_time/60:.2f} min). Saved to {out_json}")


if __name__ == "__main__":
    main()
