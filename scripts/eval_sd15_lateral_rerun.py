from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image

from app.services.editing.layout_guidance import LayoutGuidanceProcessor, TwoPhaseSchedule
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout
from scripts.eval_spatial_lateral_dedicated import (
    LATERAL_24_SPECS,
    SEEDS_192,
    exact_mcnemar_p_value,
)
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator, wilson_score_interval

LOG_FILE = ROOT_DIR / "sd15_rerun.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 80)
    log("RE-RUNNING SD v1.5 LATERAL BENCHMARK (N=192/condition, OFF vs 6.00)")
    log("Using Fixed Semantic Planner with Compound & Same-Class Disambiguation")
    log("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_img_dir = ROOT_DIR / "benchmarks" / "images" / "sd15_lateral_fixed"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    model_path = ROOT_DIR / "models" / "sd15_fp16"
    log(f"Loading SD v1.5 from {model_path} on CUDA (fp16)...")

    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    strengths = [0.0, 6.0]
    total_images = len(strengths) * len(LATERAL_24_SPECS) * len(SEEDS_192)
    log(f"Total images to evaluate: {total_images} (2 conditions x 192 runs)...")

    t_start = time.time()
    run_idx = 0

    log("\n--- PHASE 1: GENERATING SD v1.5 IMAGES ---")
    for strength in strengths:
        log(f"\n>>> Condition: Strength = {strength:.2f} (N=192)...")
        for spec in LATERAL_24_SPECS:
            pid = spec["id"]
            prompt = spec["prompt"]

            intent = analyze_prompt(prompt, mode="generate")
            plan = plan_semantic_layout(intent, tokenizer=pipe.tokenizer)

            if strength == 0.0:
                pipe.unet.set_attn_processor({k: AttnProcessor() for k in pipe.unet.attn_processors.keys()})
                for seed in SEEDS_192:
                    run_idx += 1
                    img_path = out_img_dir / f"{pid}_str_{strength:.2f}_s{seed}.png"
                    if img_path.exists():
                        continue
                    gen = torch.Generator(device).manual_seed(seed)
                    with torch.inference_mode():
                        img = pipe(
                            prompt=prompt,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            generator=gen,
                            width=512,
                            height=512,
                        ).images[0]
                    img.save(img_path)
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

                def step_cb(pipe, step_idx, timestep, callback_kwargs):
                    prog = float(step_idx + 1) / 20.0
                    for p in pipe.unet.attn_processors.values():
                        if isinstance(p, LayoutGuidanceProcessor):
                            p.set_step_progress(prog)
                    return callback_kwargs

                for seed in SEEDS_192:
                    run_idx += 1
                    img_path = out_img_dir / f"{pid}_str_{strength:.2f}_s{seed}.png"
                    if img_path.exists():
                        continue
                    gen = torch.Generator(device).manual_seed(seed)
                    with torch.inference_mode():
                        img = pipe(
                            prompt=prompt,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            generator=gen,
                            width=512,
                            height=512,
                            callback_on_step_end=step_cb,
                        ).images[0]
                    img.save(img_path)

    log(f"[+] All {total_images} images generated in {time.time() - t_start:.1f}s! Freeing pipeline...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # PHASE 2: STRICT EVALUATION
    # -----------------------------------------------------------------------
    log("\n--- PHASE 2: RUNNING STRICT SPATIAL EVALUATION ---")
    evaluator = StrictSpatialEvaluator(device="cuda")

    eval_by_cond: dict[str, list[dict[str, Any]]] = {}

    for strength in strengths:
        str_key = str(strength)
        eval_by_cond[str_key] = []
        log(f"  Evaluating condition str={strength:.2f} ...")

        for spec in LATERAL_24_SPECS:
            pid = spec["id"]
            prompt = spec["prompt"]
            subj = spec["subject"]
            obj = spec["object"]
            rel = spec["relation"]

            for seed in SEEDS_192:
                img_path = out_img_dir / f"{pid}_str_{strength:.2f}_s{seed}.png"
                img = Image.open(img_path).convert("RGB")

                s_det, o_det = evaluator.detect_entities(img, subj, obj)
                is_sat, details = evaluator.check_relation(s_det, o_det, rel)
                clip_sim, aes_score = evaluator.evaluate_quality(img, prompt)

                both_present = s_det is not None and o_det is not None
                if is_sat:
                    failure_cat = "success"
                elif not both_present:
                    if s_det is None and o_det is None:
                        failure_cat = "omission_both"
                    elif s_det is None:
                        failure_cat = "omission_subject"
                    else:
                        failure_cat = "omission_object"
                else:
                    failure_cat = "misplacement"

                eval_by_cond[str_key].append({
                    "prompt_id": pid,
                    "prompt": prompt,
                    "subject": subj,
                    "object": obj,
                    "relation": rel,
                    "seed": seed,
                    "strength": strength,
                    "satisfied": is_sat,
                    "both_present": both_present,
                    "failure_category": failure_cat,
                    "details": details,
                    "subject_detected": s_det is not None,
                    "object_detected": o_det is not None,
                    "subject_score": s_det["score"] if s_det else 0.0,
                    "object_score": o_det["score"] if o_det else 0.0,
                    "clip_similarity": round(float(clip_sim), 4),
                    "laion_aesthetic_score": round(float(aes_score), 4),
                })

    # -----------------------------------------------------------------------
    # PHASE 3: COMPUTE METRICS & CONTINGENCY TABLES
    # -----------------------------------------------------------------------
    log("\n" + "=" * 95)
    log("SD v1.5 RE-EVALUATION SUMMARY (WITH PLANNER FIX):")
    log("=" * 95)
    log(f"{'Condition':<10} | {'Overall Sat':<16} | {'Directional (N=136)':<21} | {'Symmetric (N=56)':<18} | {'Presence':<10} | {'McNemar vs OFF'}")
    log("-" * 95)

    off_recs = eval_by_cond["0.0"]

    for strength in strengths:
        str_key = str(strength)
        cur_recs = eval_by_cond[str_key]
        n_tot = len(cur_recs)
        n_sat = sum(1 for r in cur_recs if r["satisfied"])
        n_pres = sum(1 for r in cur_recs if r["both_present"])
        sat_rate = n_sat / n_tot
        ci_sat = wilson_score_interval(n_sat, n_tot)

        dir_recs = [r for r in cur_recs if r["relation"] in ("left_of", "right_of")]
        dir_sat = sum(1 for r in dir_recs if r["satisfied"])
        dir_rate = dir_sat / len(dir_recs)

        sym_recs = [r for r in cur_recs if r["relation"] == "beside"]
        sym_sat = sum(1 for r in sym_recs if r["satisfied"])
        sym_rate = sym_sat / len(sym_recs)

        mcnemar_str = "Baseline Ref"
        if strength > 0.0:
            gain_b = sum(
                1
                for r_off, r_on in zip(off_recs, cur_recs, strict=True)
                if (not r_off["satisfied"]) and r_on["satisfied"]
            )
            loss_c = sum(
                1
                for r_off, r_on in zip(off_recs, cur_recs, strict=True)
                if r_off["satisfied"] and (not r_on["satisfied"])
            )
            p_val = exact_mcnemar_p_value(gain_b, loss_c)
            net_gain = gain_b - loss_c
            mcnemar_str = f"Net +{net_gain:02d} (p={p_val:.6f})"

        log(f"str={strength:<6.2f} | {sat_rate*100:5.1f}% ({n_sat:03d}/{n_tot}) [{ci_sat[0]:4.1f}%, {ci_sat[1]:4.1f}%] | {dir_rate*100:5.1f}% ({dir_sat:03d}/136)       | {sym_rate*100:5.1f}% ({sym_sat:02d}/56)    | {n_pres/n_tot*100:5.1f}%     | {mcnemar_str}")

    # Save results json
    out_json = ROOT_DIR / "benchmarks" / "sd15_lateral_fixed_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_by_cond, f, indent=2)
    log(f"\n[+] Saved SD v1.5 fixed results to: {out_json}")


if __name__ == "__main__":
    main()
