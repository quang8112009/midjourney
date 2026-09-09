"""Powered Standard 24 study on PixArt-Alpha (DiT + T5-XXL, N=192 paired runs).

Evaluates 24 Standard lateral prompts x 8 seeds at OFF (0.00), Strength 1.5, and Strength 3.0.
Metrics match other backbones: directional split (left/right), symmetric, dual presence, omissions/misplacements,
SSIM vs OFF, LAION, CLIP alignment, and exact paired McNemar test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, PixArtAlphaPipeline
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from app.services.editing.layout_guidance import build_layout_guidance_bias
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.region_attention import RegionAwareAttnProcessor
from app.services.editing.semantic_planner import plan_semantic_layout
from scripts.eval_spatial_lateral_dedicated import (
    LATERAL_24_SPECS,
    SEEDS_192,
    exact_mcnemar_p_value,
    wilson_score_interval,
)
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator, compute_image_ssim
from scripts.run_live_aesthetic_baseline import RealAestheticEvaluator, compute_file_sha256

STRENGTHS = [0.0, 1.5, 3.0]
LOG_FILE = ROOT_DIR / "pixart_standard24_study.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    t_start = time.time()
    log("=" * 90)
    log("POWERED STANDARD 24 LATERAL STUDY ON PIXART-ALPHA (N=192 PAIRED RUNS)")
    log("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator_spatial = StrictSpatialEvaluator(device=device)
    evaluator_aes = RealAestheticEvaluator(device=device)

    out_dir = ROOT_DIR / "benchmarks" / "images" / "pixart_standard24_n192"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Pipeline
    log("\nLoading PixArt-Alpha Pipeline...")
    t0 = time.time()
    tok_t5 = AutoTokenizer.from_pretrained("models/sd35_medium/tokenizer_3")
    text_enc = T5EncoderModel.from_pretrained("models/sd35_medium/text_encoder_3", torch_dtype=torch.float16)

    pipe = PixArtAlphaPipeline.from_pretrained(
        "models/pixart_alpha",
        text_encoder=text_enc,
        tokenizer=tok_t5,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # Attach RegionAwareAttnProcessor
    cross_attn_names = [n for n in pipe.transformer.attn_processors.keys() if "attn2" in n or "cross" in n]
    new_processors = {}
    for name, proc in pipe.transformer.attn_processors.items():
        if name in cross_attn_names:
            new_processors[name] = RegionAwareAttnProcessor(proc)
        else:
            new_processors[name] = proc
    pipe.transformer.set_attn_processor(new_processors)
    log(f"[+] Pipeline ready in {time.time() - t0:.2f}s.")

    # 2. Pre-compute plans
    plans = {}
    for spec in LATERAL_24_SPECS:
        intent = analyze_prompt(spec["prompt"], mode="generate")
        plans[spec["id"]] = plan_semantic_layout(intent, tokenizer=tok_t5)

    # 3. Generate & Evaluate across all conditions
    total_gens = len(LATERAL_24_SPECS) * len(SEEDS_192) * len(STRENGTHS)
    log(f"\nGenerating {total_gens} images ({len(LATERAL_24_SPECS)} prompts x {len(SEEDS_192)} seeds x 3 strengths)...")

    results_by_strength: dict[float, dict[tuple[str, int], dict[str, Any]]] = {s: {} for s in STRENGTHS}
    baseline_images: dict[tuple[str, int], Image.Image] = {}

    for str_val in STRENGTHS:
        log(f"\n--- Running Strength = {str_val:.2f} ---")
        t_str0 = time.time()

        for spec in LATERAL_24_SPECS:
            pid = spec["id"]
            prompt = spec["prompt"]
            subj = spec["subject"]
            obj = spec["object"]
            rel = spec["relation"]
            plan = plans[pid]

            # Build bias
            if str_val > 0.0:
                bias_tensor = build_layout_guidance_bias(
                    plan=plan,
                    num_image_tokens=1024,
                    num_text_tokens=120,
                    guidance_strength=str_val,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                for name in cross_attn_names:
                    pipe.transformer.attn_processors[name].set_bias(bias_tensor.to("cuda", dtype=torch.float16))
            else:
                for name in cross_attn_names:
                    pipe.transformer.attn_processors[name].set_bias(None)

            for seed in SEEDS_192:
                filename = f"{pid}_s{seed}_str_{str_val:.2f}.png"
                img_path = out_dir / filename

                if not img_path.exists():
                    gen = torch.Generator(device="cpu").manual_seed(seed)
                    with torch.inference_mode():
                        img = pipe(
                            prompt=prompt,
                            num_inference_steps=20,
                            guidance_scale=4.5,
                            width=512,
                            height=512,
                            generator=gen,
                            clean_caption=False,
                        ).images[0]
                    img.save(img_path)
                else:
                    img = Image.open(img_path).convert("RGB")

                sha256 = compute_file_sha256(img_path)

                if str_val == 0.0:
                    baseline_images[(pid, seed)] = img

                # Evaluation
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

        log(f"[+] Finished strength {str_val:.2f} in {time.time() - t_str0:.1f}s.")

    # 4. Statistical Analysis & McNemar Calculations
    conditions_summary = {}

    log("\n" + "=" * 115)
    log("PIXART-ALPHA STANDARD 24 SUMMARY & STATISTICAL BREAKDOWN:")
    log("=" * 115)

    base_records = results_by_strength[0.0]

    for str_val in STRENGTHS:
        cur_records = results_by_strength[str_val]
        total_runs = len(cur_records)
        sat_runs = sum(1 for r in cur_records.values() if r["satisfied"])
        pres_runs = sum(1 for r in cur_records.values() if r["both_present"])

        sat_rate = sat_runs / total_runs
        pres_rate = pres_runs / total_runs
        ci_sat = wilson_score_interval(sat_runs, total_runs)
        ci_pres = wilson_score_interval(pres_runs, total_runs)

        # Directional subset (136 pairs: 72 left_of, 64 right_of)
        dir_runs = [r for r in cur_records.values() if r["relation"] in ("left_of", "right_of")]
        dir_sat = sum(1 for r in dir_runs if r["satisfied"])
        dir_rate = dir_sat / len(dir_runs)
        dir_ci = wilson_score_interval(dir_sat, len(dir_runs))


        left_runs = [r for r in cur_records.values() if r["relation"] == "left_of"]
        left_sat = sum(1 for r in left_runs if r["satisfied"])
        left_rate = left_sat / len(left_runs)

        right_runs = [r for r in cur_records.values() if r["relation"] == "right_of"]
        right_sat = sum(1 for r in right_runs if r["satisfied"])
        right_rate = right_sat / len(right_runs)

        # Symmetric subset (56 pairs: beside)
        sym_runs = [r for r in cur_records.values() if r["relation"] == "beside"]
        sym_sat = sum(1 for r in sym_runs if r["satisfied"])
        sym_rate = sym_sat / len(sym_runs)
        sym_ci = wilson_score_interval(sym_sat, len(sym_runs))

        # Failure modes: omissions vs misplacements
        omissions = sum(1 for r in cur_records.values() if not r["both_present"])
        misplacements = sum(1 for r in cur_records.values() if r["both_present"] and not r["satisfied"])

        # Perceptual & Aesthetic metrics
        mean_ssim = float(np.mean([r["ssim_vs_off"] for r in cur_records.values()]))
        mean_laion = float(np.mean([r["laion_aesthetic"] for r in cur_records.values()]))
        mean_clip = float(np.mean([r["clip_alignment"] for r in cur_records.values()]))

        # McNemar vs OFF
        mcnemar_data = None
        if str_val > 0.0:
            both_pass = 0
            gain_b = 0 # off=fail, on=pass
            loss_c = 0 # off=pass, on=fail
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
                "total": len(dir_runs),
                "satisfied": dir_sat,
                "rate": round(dir_rate, 4),
                "wilson_ci_95": [round(dir_ci[0] * 100, 2), round(dir_ci[1] * 100, 2)],
                "left_of": {"total": len(left_runs), "satisfied": left_sat, "rate": round(left_rate, 4)},
                "right_of": {"total": len(right_runs), "satisfied": right_sat, "rate": round(right_rate, 4)},
            },
            "symmetric": {
                "total": len(sym_runs),
                "satisfied": sym_sat,
                "rate": round(sym_rate, 4),
                "wilson_ci_95": [round(sym_ci[0] * 100, 2), round(sym_ci[1] * 100, 2)],
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
        log(f"Strength {str_val:4.1f} | Overall: {sat_rate*100:5.2f}% ({sat_runs:3d}/192) [{ci_sat[0]*100:.1f}%, {ci_sat[1]*100:.1f}%] | Dir: {dir_rate*100:5.2f}% ({dir_sat:3d}/136) [L:{left_rate*100:.1f}%, R:{right_rate*100:.1f}%] | Pres: {pres_rate*100:5.2f}% | SSIM: {mean_ssim:.4f} | {p_str} {net_str}")

    log("=" * 115)

    # 5. Save Full Dataset
    payload = {
        "model": "PixArt-alpha/PixArt-XL-2-512x512",
        "benchmark": "Standard 24 Lateral Powered Study",
        "resolution": "512x512",
        "steps": 20,
        "seeds": SEEDS_192,
        "conditions": conditions_summary,
        "details": {
            str(s): [r for r in results_by_strength[s].values()]
            for s in STRENGTHS
        },
    }

    out_json = ROOT_DIR / "benchmarks" / "pixart_standard24_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    total_time = time.time() - t_start
    log(f"\n[+] Powered Standard 24 study on PixArt completed in {total_time:.1f}s ({total_time/60:.2f} min). Saved to {out_json}")


if __name__ == "__main__":
    main()
