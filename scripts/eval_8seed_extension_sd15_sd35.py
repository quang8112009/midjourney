"""Extend SD v1.5 and SD 3.5 Medium Aesthetic & Style Expansion A/B datasets to 8 seeds.

Seeds: [42, 100, 2024, 7777, 123, 999, 4321, 8888] (N=320 pairs each).
Evaluates all 5 real scorers and recomputes paired statistics and 95% CIs.
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
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)
from PIL import Image
from transformers import AutoTokenizer

from app.services.editing.style_expansion import expand_style
from scripts.run_live_aesthetic_baseline import (
    AESTHETIC_40_PROMPTS,
    RealAestheticEvaluator,
    compute_file_sha256,
)

SEEDS_8 = [42, 100, 2024, 7777, 123, 999, 4321, 8888]
LOG_FILE = ROOT_DIR / "8seed_extension.log"


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


def pre_encode_sd35(model_path: str, prompts: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        transformer=None,
        vae=None,
        torch_dtype=torch.float16,
    ).to("cuda")

    bank: dict[str, dict[str, torch.Tensor]] = {}
    for prompt in prompts:
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                do_classifier_free_guidance=True,
            )
        bank[prompt] = {
            "prompt_embeds": prompt_embeds.cpu(),
            "negative_prompt_embeds": negative_prompt_embeds.cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds.cpu(),
        }
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return bank


def run_sd15_8seed(evaluator: RealAestheticEvaluator) -> dict[str, Any]:
    log("\n" + "=" * 80)
    log("PART 1: SD v1.5 EXTENSION TO 8 SEEDS (N=320 PAIRS)")
    log("=" * 80)

    dir_off = ROOT_DIR / "benchmarks" / "images" / "sd15_baseline"
    dir_on = ROOT_DIR / "benchmarks" / "images" / "sd15_expanded_512x512_20steps"
    dir_off.mkdir(parents=True, exist_ok=True)
    dir_on.mkdir(parents=True, exist_ok=True)

    prompts_map = []
    for idx, p in enumerate(AESTHETIC_40_PROMPTS, start=1):
        pid = f"aes_{idx:02d}"
        res = expand_style(p, model="stable-diffusion", style_expansion_enabled=True)
        prompts_map.append((pid, p, res.expanded_prompt))

    # Check if any new images need generation
    model_path = ROOT_DIR / "models" / "sd15_fp16"
    pipe = None

    for pid, orig_p, exp_p in prompts_map:
        for seed in SEEDS_8:
            f_off = dir_off / f"{pid}_seed_{seed}.png"
            f_on = dir_on / f"{pid}_seed_{seed}.png"

            if not f_off.exists() or not f_on.exists():
                if pipe is None:
                    log("Loading SD v1.5 pipeline on CUDA...")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        str(model_path),
                        variant="fp16",
                        use_safetensors=True,
                        torch_dtype=torch.float16,
                        safety_checker=None,
                    ).to("cuda")
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

                if not f_off.exists():
                    gen = torch.Generator("cuda").manual_seed(seed)
                    with torch.inference_mode():
                        img_off = pipe(
                            prompt=orig_p,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            generator=gen,
                            width=512,
                            height=512,
                        ).images[0]
                    img_off.save(f_off)

                if not f_on.exists():
                    gen = torch.Generator("cuda").manual_seed(seed)
                    with torch.inference_mode():
                        img_on = pipe(
                            prompt=exp_p,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            generator=gen,
                            width=512,
                            height=512,
                        ).images[0]
                    img_on.save(f_on)

    if pipe is not None:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Evaluate 320 pairs
    eval_off = []
    eval_on = []
    records_off = []

    log("Evaluating SD v1.5 320 pairs across 5 real scorers...")
    for pid, orig_p, exp_p in prompts_map:
        seed_runs_off = []
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
            seed_runs_off.append(sc_off)

            sc_on = evaluator.evaluate(img_on, orig_p)
            sc_on["prompt_id"] = pid
            sc_on["seed"] = seed
            sc_on["sha256"] = sha_on
            eval_on.append(sc_on)

        records_off.append({
            "prompt_id": pid,
            "prompt": orig_p,
            "expanded_prompt": exp_p,
            "seeds": seed_runs_off,
        })

    return analyze_and_format_stats("SD v1.5", eval_off, eval_on, records_off)


def run_sd35_8seed(evaluator: RealAestheticEvaluator) -> dict[str, Any]:
    log("\n" + "=" * 80)
    log("PART 2: SD 3.5 MEDIUM EXTENSION TO 8 SEEDS (N=320 PAIRS)")
    log("=" * 80)

    dir_off = ROOT_DIR / "benchmarks" / "images" / "sd35m_aesthetic_512x512_20steps"
    dir_on = ROOT_DIR / "benchmarks" / "images" / "sd35m_aesthetic_expanded_512x512_20steps"
    dir_off.mkdir(parents=True, exist_ok=True)
    dir_on.mkdir(parents=True, exist_ok=True)

    tok_t5 = AutoTokenizer.from_pretrained("models/sd35_medium/tokenizer_3")

    prompts_map = []
    all_prompts = []
    for idx, p in enumerate(AESTHETIC_40_PROMPTS, start=1):
        pid = f"aes_{idx:02d}"
        res = expand_style(p, model="stable-diffusion-3.5", tokenizer=tok_t5, style_expansion_enabled=True)
        prompts_map.append((pid, p, res.expanded_prompt))
        all_prompts.append(p)
        all_prompts.append(res.expanded_prompt)

    prompt_bank = pre_encode_sd35("models/sd35_medium", list(set(all_prompts)))

    # Check if new images need generation
    pipe = None
    for pid, orig_p, exp_p in prompts_map:
        for seed in SEEDS_8:
            f_off = dir_off / f"{pid}_seed_{seed}.png"
            f_on = dir_on / f"{pid}_seed_{seed}.png"

            if not f_off.exists() or not f_on.exists():
                if pipe is None:
                    log("Loading SD 3.5 Medium generator pipeline on CUDA...")
                    pipe = StableDiffusion3Pipeline.from_pretrained(
                        "models/sd35_medium",
                        text_encoder=None,
                        text_encoder_2=None,
                        text_encoder_3=None,
                        tokenizer=None,
                        tokenizer_2=None,
                        tokenizer_3=None,
                        torch_dtype=torch.float16,
                    ).to("cuda")
                    pipe.set_progress_bar_config(disable=True)

                if not f_off.exists():
                    p_data = prompt_bank[orig_p]
                    gen = torch.Generator("cpu").manual_seed(seed)
                    with torch.inference_mode():
                        img_off = pipe(
                            prompt_embeds=p_data["prompt_embeds"].to("cuda"),
                            pooled_prompt_embeds=p_data["pooled_prompt_embeds"].to("cuda"),
                            negative_prompt_embeds=p_data["negative_prompt_embeds"].to("cuda"),
                            negative_pooled_prompt_embeds=p_data["negative_pooled_prompt_embeds"].to("cuda"),
                            num_inference_steps=20,
                            guidance_scale=4.5,
                            width=512,
                            height=512,
                            generator=gen,
                        ).images[0]
                    img_off.save(f_off)

                if not f_on.exists():
                    p_data = prompt_bank[exp_p]
                    gen = torch.Generator("cpu").manual_seed(seed)
                    with torch.inference_mode():
                        img_on = pipe(
                            prompt_embeds=p_data["prompt_embeds"].to("cuda"),
                            pooled_prompt_embeds=p_data["pooled_prompt_embeds"].to("cuda"),
                            negative_prompt_embeds=p_data["negative_prompt_embeds"].to("cuda"),
                            negative_pooled_prompt_embeds=p_data["negative_pooled_prompt_embeds"].to("cuda"),
                            num_inference_steps=20,
                            guidance_scale=4.5,
                            width=512,
                            height=512,
                            generator=gen,
                        ).images[0]
                    img_on.save(f_on)

    if pipe is not None:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Evaluate 320 pairs
    eval_off = []
    eval_on = []
    records_off = []

    log("Evaluating SD 3.5 Medium 320 pairs across 5 real scorers...")
    for pid, orig_p, exp_p in prompts_map:
        seed_runs_off = []
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
            seed_runs_off.append(sc_off)

            sc_on = evaluator.evaluate(img_on, orig_p)
            sc_on["prompt_id"] = pid
            sc_on["seed"] = seed
            sc_on["sha256"] = sha_on
            eval_on.append(sc_on)

        records_off.append({
            "prompt_id": pid,
            "prompt": orig_p,
            "expanded_prompt": exp_p,
            "seeds": seed_runs_off,
        })

    return analyze_and_format_stats("SD 3.5 Medium", eval_off, eval_on, records_off)


def analyze_and_format_stats(
    model_name: str,
    eval_off: list[dict[str, Any]],
    eval_on: list[dict[str, Any]],
    records_off: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = [
        ("laion_aesthetic_v2_4", "LAION v2.4"),
        ("imagereward", "ImageReward"),
        ("hps_v2_1", "HPS v2.1"),
        ("clip_alignment", "CLIP Alignment"),
        ("pickscore_v1", "PickScore v1"),
    ]

    log("\n" + "=" * 135)
    log(f"{model_name.upper()} 8-SEED PAIRED STYLE EXPANSION RESULTS (N=320 PAIRS):")
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
    return {
        "stats_summary": stats_summary,
        "records_off": records_off,
        "eval_off": eval_off,
        "eval_on": eval_on,
    }


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    t_start = time.time()
    log("=" * 90)
    log("EXPERIMENT 2: 8-SEED EXTENSION FOR SD v1.5 & SD 3.5 MEDIUM (N=320 PAIRS EACH)")
    log("=" * 90)

    evaluator = RealAestheticEvaluator(device=torch.device("cuda"))

    res_sd15 = run_sd15_8seed(evaluator)
    res_sd35 = run_sd35_8seed(evaluator)

    # 3. Direct Paired Backbone Comparison at 8 Seeds (SD 3.5 M vs SD 1.5, N=320 Pairs)
    log("\n" + "=" * 135)
    log("DIRECT MULTI-BACKBONE PAIRED COMPARISON AT 8 SEEDS (SD 3.5 M vs SD v1.5, N=320 PAIRS):")
    log("=" * 135)
    log(
        f"{'Metric':<16} | {'SD v1.5 (Mean+-std)':<20} | {'SD 3.5 M (Mean+-std)':<20} | "
        f"{'Paired Diff (d_bar)':<20} | {'95% CI of Diff':<18} | {'Paired t-test':<18} | {'Wilcoxon Test':<18}"
    )
    log("-" * 135)

    backbone_stats = {}
    metrics = [
        ("laion_aesthetic_v2_4", "LAION v2.4"),
        ("imagereward", "ImageReward"),
        ("hps_v2_1", "HPS v2.1"),
        ("clip_alignment", "CLIP Alignment"),
        ("pickscore_v1", "PickScore v1"),
    ]

    for m_key, m_name in metrics:
        v15 = np.array([r[m_key] for r in res_sd15["eval_off"]])
        v35 = np.array([r[m_key] for r in res_sd35["eval_off"]])
        diff = v35 - v15
        n = len(diff)

        m_15 = float(np.mean(v15))
        s_15 = float(np.mean([np.std(v15[i * 8 : (i + 1) * 8]) for i in range(40)]))

        m_35 = float(np.mean(v35))
        s_35 = float(np.mean([np.std(v35[i * 8 : (i + 1) * 8]) for i in range(40)]))

        d_mean = float(np.mean(diff))
        d_std = float(np.std(diff, ddof=1))
        d_se = d_std / math.sqrt(n)

        t_stat = d_mean / d_se
        p_t = math.erfc(abs(t_stat) / math.sqrt(2))

        ci_low = d_mean - 1.96 * d_se
        ci_high = d_mean + 1.96 * d_se

        w_stat, w_z, p_w = wilcoxon_signed_rank(diff)
        ratio_to_noise = abs(d_mean) / s_15

        backbone_stats[m_key] = {
            "sd15_mean": round(m_15, 4),
            "sd15_seed_std": round(s_15, 4),
            "sd35_mean": round(m_35, 4),
            "sd35_seed_std": round(s_35, 4),
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
            f"{m_name:<16} | {m_15:6.4f} +- {s_15:6.4f}   | {m_35:6.4f} +- {s_35:6.4f}   | "
            f"{d_mean:+8.4f} (r={ratio_to_noise:.2f}x) | [{ci_low:+7.4f}, {ci_high:+7.4f}] | "
            f"t={t_stat:+6.2f} ({p_t_str}) | z={w_z:+6.2f} ({p_w_str})"
        )

    log("=" * 135)

    # 4. Save Updated Benchmark Datasets
    out_sd15_json = ROOT_DIR / "benchmarks" / "sd15_8seed_aesthetic_and_ab.json"
    out_sd35_json = ROOT_DIR / "benchmarks" / "sd35m_8seed_aesthetic_and_ab.json"
    out_paired_json = ROOT_DIR / "benchmarks" / "paired_backbone_8seed_stats.json"

    with open(out_sd15_json, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {"seeds": SEEDS_8, "num_pairs": 320},
            "stats_summary": res_sd15["stats_summary"],
            "records_off": res_sd15["records_off"],
        }, f, indent=2)

    with open(out_sd35_json, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {"seeds": SEEDS_8, "num_pairs": 320},
            "stats_summary": res_sd35["stats_summary"],
            "records_off": res_sd35["records_off"],
        }, f, indent=2)

    with open(out_paired_json, "w", encoding="utf-8") as f:
        json.dump(backbone_stats, f, indent=2)

    total_time = time.time() - t_start
    log(f"\n[+] Exp 2 completed in {total_time:.1f}s ({total_time/60.0:.2f} min).")


if __name__ == "__main__":
    main()
