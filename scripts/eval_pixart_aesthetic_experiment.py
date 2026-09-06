"""PixArt-Alpha (DiT + T5-XXL) Aesthetic Baseline and Paired Style Expansion A/B Study (N=320 Pairs).

Evaluates 40 style prompts x 8 seeds ([42, 100, 2024, 7777, 123, 999, 4321, 8888])
at 512x512 / 20 steps / guidance 4.5 across all 5 real pretrained aesthetic scorers.
"""

from __future__ import annotations

import gc
import json
import math
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

from app.services.editing.style_expansion import expand_style
from scripts.run_live_aesthetic_baseline import (
    AESTHETIC_40_PROMPTS,
    RealAestheticEvaluator,
    compute_file_sha256,
)

SEEDS_8 = [42, 100, 2024, 7777, 123, 999, 4321, 8888]
LOG_FILE = ROOT_DIR / "pixart_aesthetic_experiment.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def wilcoxon_signed_rank(diff: np.ndarray) -> tuple[float, float, float]:
    d = diff[diff != 0]
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0
    ranks = np.argsort(np.abs(d)) + 1
    w_pos = np.sum(ranks[d > 0])
    w_neg = np.sum(ranks[d < 0])
    w = min(w_pos, w_neg)
    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mean_w) / std_w
    p_val = math.erfc(abs(z) / math.sqrt(2))
    return float(w), float(z), float(p_val)


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    t_start = time.time()
    log("=" * 90)
    log("EXPERIMENT 1: PIXART-ALPHA AESTHETIC BASELINE & DESCRIPTOR EXPANSION A/B (N=320 PAIRS)")
    log("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_off = ROOT_DIR / "benchmarks" / "images" / "pixart_baseline_512x512"
    dir_on = ROOT_DIR / "benchmarks" / "images" / "pixart_expanded_512x512"
    dir_off.mkdir(parents=True, exist_ok=True)
    dir_on.mkdir(parents=True, exist_ok=True)

    # 1. Load PixArt Pipeline
    log("\nLoading local T5 and PixArt-Alpha Pipeline...")
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
    log(f"[+] Pipeline ready in {time.time() - t0:.2f}s.")

    # 2. Prepare Prompts
    prompts_map = []
    for idx, p in enumerate(AESTHETIC_40_PROMPTS, start=1):
        pid = f"aes_{idx:02d}"
        res = expand_style(p, model="pixart-alpha", tokenizer=tok_t5, style_expansion_enabled=True)
        prompts_map.append((pid, p, res.expanded_prompt))

    # 3. Generate 320 OFF images and 320 ON images
    total_gens = len(prompts_map) * len(SEEDS_8)
    log(f"\nGenerating {total_gens} OFF and {total_gens} ON images (512x512, 20 steps, guidance 4.5)...")

    gen_off_count = 0
    gen_on_count = 0
    t_gen0 = time.time()

    for pid, orig_p, exp_p in prompts_map:
        for seed in SEEDS_8:
            # OFF image
            f_off = dir_off / f"{pid}_seed_{seed}.png"
            if not f_off.exists():
                gen_off_count += 1
                gen = torch.Generator(device="cpu").manual_seed(seed)
                with torch.inference_mode():
                    img_off = pipe(
                        prompt=orig_p,
                        num_inference_steps=20,
                        guidance_scale=4.5,
                        width=512,
                        height=512,
                        generator=gen,
                        clean_caption=False,
                    ).images[0]
                img_off.save(f_off)

            # ON image
            f_on = dir_on / f"{pid}_seed_{seed}.png"
            if not f_on.exists():
                gen_on_count += 1
                gen = torch.Generator(device="cpu").manual_seed(seed)
                with torch.inference_mode():
                    img_on = pipe(
                        prompt=exp_p,
                        num_inference_steps=20,
                        guidance_scale=4.5,
                        width=512,
                        height=512,
                        generator=gen,
                        clean_caption=False,
                    ).images[0]
                img_on.save(f_on)

    log(f"[+] Finished image generation in {time.time() - t_gen0:.2f}s (Generated new: {gen_off_count} OFF, {gen_on_count} ON).")

    # Free Pipeline memory before evaluation
    del pipe
    del text_enc
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Run Multi-Metric Real Aesthetic Evaluator
    log("\nLoading Real Aesthetic Evaluator on CUDA...")
    evaluator = RealAestheticEvaluator(device=device)

    eval_off: list[dict[str, Any]] = []
    eval_on: list[dict[str, Any]] = []
    off_records_json = []

    log("\nEvaluating 320 image pairs across all 5 real scorers...")
    eval_t0 = time.time()

    for pid, orig_p, exp_p in prompts_map:
        prompt_off_runs = []
        for seed in SEEDS_8:
            f_off = dir_off / f"{pid}_seed_{seed}.png"
            f_on = dir_on / f"{pid}_seed_{seed}.png"

            img_off = Image.open(f_off).convert("RGB")
            img_on = Image.open(f_on).convert("RGB")

            sha_off = compute_file_sha256(f_off)
            sha_on = compute_file_sha256(f_on)

            sc_off = evaluator.evaluate(img_off, orig_p)
            sc_off["prompt_id"] = pid
            sc_off["seed"] = seed
            sc_off["sha256"] = sha_off
            eval_off.append(sc_off)
            prompt_off_runs.append(sc_off)

            sc_on = evaluator.evaluate(img_on, orig_p)
            sc_on["prompt_id"] = pid
            sc_on["seed"] = seed
            sc_on["sha256"] = sha_on
            eval_on.append(sc_on)

        off_records_json.append({
            "prompt_id": pid,
            "prompt": orig_p,
            "expanded_prompt": exp_p,
            "seeds": prompt_off_runs,
        })

    log(f"[+] Evaluation finished in {time.time() - eval_t0:.2f}s.")

    # 5. Paired Statistical Analysis (N=320 Pairs)
    metrics = [
        ("laion_aesthetic_v2_4", "LAION v2.4"),
        ("imagereward", "ImageReward"),
        ("hps_v2_1", "HPS v2.1"),
        ("clip_alignment", "CLIP Alignment"),
        ("pickscore_v1", "PickScore v1"),
    ]

    log("\n" + "=" * 135)
    log("PIXART-ALPHA PAIRED STYLE EXPANSION RESULTS (N=320 PAIRS, 40 PROMPTS x 8 SEEDS):")
    log("=" * 135)
    log(
        f"{'Metric':<16} | {'OFF (Mean+-std)':<20} | {'ON (Mean+-std)':<20} | "
        f"{'Paired Diff (d_bar)':<20} | {'95% CI of Diff':<18} | {'Paired t-test':<18} | {'Wilcoxon Test':<18}"
    )
    log("-" * 135)

    stats_summary = {}

    for m_key, m_name in metrics:
        v_off = np.array([r[m_key] for r in eval_off])
        v_on = np.array([r[m_key] for r in eval_on])
        diff = v_on - v_off
        n = len(diff)

        m_off = float(np.mean(v_off))
        std_off = float(np.mean([np.std(v_off[i * 8 : (i + 1) * 8]) for i in range(40)]))

        m_on = float(np.mean(v_on))
        std_on = float(np.mean([np.std(v_on[i * 8 : (i + 1) * 8]) for i in range(40)]))

        d_mean = float(np.mean(diff))
        d_std = float(np.std(diff, ddof=1))
        d_se = d_std / math.sqrt(n)

        t_stat = d_mean / d_se
        p_t = math.erfc(abs(t_stat) / math.sqrt(2))

        ci_low = d_mean - 1.96 * d_se
        ci_high = d_mean + 1.96 * d_se

        w_stat, w_z, p_w = wilcoxon_signed_rank(diff)
        ratio_to_noise = abs(d_mean) / std_off

        stats_summary[m_key] = {
            "off_mean": round(m_off, 4),
            "off_seed_std": round(std_off, 4),
            "on_mean": round(m_on, 4),
            "on_seed_std": round(std_on, 4),
            "d_mean": round(d_mean, 4),
            "d_se": round(d_se, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "ratio_to_noise": round(ratio_to_noise, 2),
            "t_stat": round(t_stat, 3),
            "t_pvalue": p_t,
            "wilcoxon_z": round(w_z, 3),
            "wilcoxon_pvalue": p_w,
        }

        p_t_str = f"p={p_t:.2e}" if p_t < 1e-4 else f"p={p_t:.4f}"
        p_w_str = f"p={p_w:.2e}" if p_w < 1e-4 else f"p={p_w:.4f}"
        log(
            f"{m_name:<16} | {m_off:6.4f} +- {std_off:6.4f}   | {m_on:6.4f} +- {std_on:6.4f}   | "
            f"{d_mean:+8.4f} (r={ratio_to_noise:.2f}x) | [{ci_low:+7.4f}, {ci_high:+7.4f}] | "
            f"t={t_stat:+6.2f} ({p_t_str}) | z={w_z:+6.2f} ({p_w_str})"
        )

    log("=" * 135)

    # 6. Save Artifacts
    baseline_payload = {
        "metadata": {
            "model": "PixArt-alpha/PixArt-XL-2-512x512",
            "architecture": "DiT + T5-XXL",
            "resolution": "512x512",
            "steps": 20,
            "guidance_scale": 4.5,
            "seeds": SEEDS_8,
            "num_pairs": 320,
        },
        "overall_summary": stats_summary,
        "per_prompt_results": off_records_json,
    }

    out_base_json = ROOT_DIR / "benchmarks" / "pixart_aesthetic_baseline.json"
    out_ab_json = ROOT_DIR / "benchmarks" / "pixart_style_expansion_ab_results.json"

    with open(out_base_json, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, indent=2)
    with open(out_ab_json, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)

    total_time = time.time() - t_start
    log(f"\n[+] Exp 1 completed in {total_time:.1f}s ({total_time/60.0:.2f} min). Saved datasets to benchmarks/")


if __name__ == "__main__":
    main()
