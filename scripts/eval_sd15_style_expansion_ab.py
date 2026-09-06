from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image

from app.services.editing.style_expansion import expand_style
from scripts.run_live_aesthetic_baseline import (
    AESTHETIC_40_PROMPTS,
    FIXED_SEEDS,
    RealAestheticEvaluator,
    compute_file_sha256,
)

LOG_FILE = ROOT_DIR / "sd15_style_expansion_ab.log"


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

    log("=" * 90)
    log("SD v1.5 AESTHETIC STYLE EXPANSION PAIRED A/B EVALUATION (N=160 PAIRS)")
    log("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ROOT_DIR / "models" / "sd15_fp16"
    images_dir = ROOT_DIR / "benchmarks" / "images" / "sd15_expanded_512x512_20steps"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SD v1.5 unexpanded baseline
    with open(ROOT_DIR / "benchmarks" / "aesthetic_baseline.json") as f:
        sd15_base_data = json.load(f)

    prompts_sd15_base = sd15_base_data.get("per_prompt_results") or sd15_base_data.get("results_per_prompt", [])

    # 2. Expand all 40 aesthetic prompts within 77-token CLIP-L budget
    prompts_map = []
    for idx, p in enumerate(AESTHETIC_40_PROMPTS, start=1):
        pid = f"aes_{idx:02d}"
        res = expand_style(p, model="stable-diffusion", style_expansion_enabled=True)
        prompts_map.append((pid, p, res.expanded_prompt, res))

    log(f"[+] Prepared {len(prompts_map)} style-expanded prompts for SD v1.5.")

    # 3. Load Pipeline
    log(f"\nLoading SD v1.5 from {model_path} on CUDA (fp16)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 4. Generate all 160 expanded images
    total_images = len(prompts_map) * len(FIXED_SEEDS)
    log(f"\nGenerating {total_images} expanded images (512x512, 20 DPM steps, guidance 7.5)...")

    evaluator = RealAestheticEvaluator(device=device)

    eval_on = []
    sample_hashes = []
    count = 0
    t_start = time.time()

    for pid, orig_prompt, exp_prompt, _res in prompts_map:
        for seed in FIXED_SEEDS:
            count += 1
            filename = f"{pid}_seed_{seed}.png"
            image_path = images_dir / filename

            if not image_path.exists():
                gen = torch.Generator(device="cuda").manual_seed(seed)
                with torch.inference_mode():
                    img = pipe(
                        prompt=exp_prompt,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        generator=gen,
                        width=512,
                        height=512,
                    ).images[0]
                img.save(image_path)
            else:
                img = Image.open(image_path).convert("RGB")


            sha256_hash = compute_file_sha256(image_path)
            if len(sample_hashes) < 8:
                sample_hashes.append((filename, sha256_hash))

            # Evaluate with original prompt for fair semantic alignment & aesthetic comparison
            scores = evaluator.evaluate(img, orig_prompt)
            scores["prompt_id"] = pid
            scores["seed"] = seed
            scores["sha256"] = sha256_hash
            eval_on.append(scores)

            log(f"  [{count:03d}/{total_images}] {filename} -> LAION: {scores['laion_aesthetic_v2_4']:.3f}, CLIP: {scores['clip_alignment']:.4f}, HPS: {scores['hps_v2_1']:.4f}")

    total_time = time.time() - t_start
    log(f"[+] Finished generation & evaluation in {total_time:.1f}s.")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Extract Baseline (OFF) runs
    eval_off = []
    for p_entry in prompts_sd15_base:
        seeds_runs = p_entry.get("seeds") or p_entry.get("per_seed_runs", [])
        for s_run in seeds_runs:
            eval_off.append(s_run)

    assert len(eval_off) == len(eval_on) == 160

    # 6. Paired Statistical Analysis (N=160 Pairs)
    metrics = [
        ("laion_aesthetic_v2_4", "LAION v2.4"),
        ("imagereward", "ImageReward"),
        ("hps_v2_1", "HPS v2.1"),
        ("clip_alignment", "CLIP Alignment"),
        ("pickscore_v1", "PickScore v1"),
    ]

    log("\n" + "=" * 135)
    log("SD v1.5 PAIRED STYLE EXPANSION RESULTS (N=160 PAIRS, OFF vs ON):")
    log("=" * 135)
    log(
        f"{'Metric':<16} | {'OFF (Mean+-std)':<20} | {'ON (Mean+-std)':<20} | "
        f"{'Paired Diff (d_bar)':<20} | {'95% CI of Diff':<18} | {'Paired t-test':<18} | {'Wilcoxon Test':<18}"
    )
    log("-" * 135)

    stats_results = {}

    for m_key, m_name in metrics:
        v_off = np.array([r[m_key] for r in eval_off])
        v_on = np.array([r[m_key] for r in eval_on])
        diff = v_on - v_off
        n = len(diff)

        m_off = float(np.mean(v_off))
        std_off = float(np.mean([np.std(v_off[i * 4 : (i + 1) * 4]) for i in range(40)]))

        m_on = float(np.mean(v_on))
        std_on = float(np.mean([np.std(v_on[i * 4 : (i + 1) * 4]) for i in range(40)]))

        d_mean = float(np.mean(diff))
        d_std = float(np.std(diff, ddof=1))
        d_se = d_std / math.sqrt(n)

        t_stat = d_mean / d_se
        p_t = math.erfc(abs(t_stat) / math.sqrt(2))

        ci_low = d_mean - 1.96 * d_se
        ci_high = d_mean + 1.96 * d_se

        w_stat, w_z, p_w = wilcoxon_signed_rank(diff)
        ratio_to_noise = abs(d_mean) / std_off

        stats_results[m_key] = {
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

    out_json = ROOT_DIR / "benchmarks" / "sd15_style_expansion_ab_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=2)
    log(f"[+] Saved SD v1.5 paired style expansion statistics to: {out_json}")


if __name__ == "__main__":
    main()
