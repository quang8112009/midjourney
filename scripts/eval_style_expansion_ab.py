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
from diffusers import StableDiffusion3Pipeline
from PIL import Image, ImageDraw

from app.services.editing.style_expansion import expand_style
from scripts.run_live_aesthetic_baseline import (
    AESTHETIC_40_PROMPTS,
    FIXED_SEEDS,
    RealAestheticEvaluator,
)

LOG_FILE = ROOT_DIR / "phase_c_style_expansion.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def pre_encode_prompts(model_path: str, prompts: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    log("\nLoading SD 3.5 Medium Text Encoders on CUDA...")
    t0 = time.time()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        transformer=None,
        vae=None,
        torch_dtype=torch.float16,
    ).to("cuda")

    bank: dict[str, dict[str, torch.Tensor]] = {}
    for _idx, prompt in enumerate(prompts, start=1):
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
    log(f"[+] Pre-encoded {len(prompts)} prompts in {time.time() - t0:.2f}s.")
    return bank


def generate_images(
    prompts_map: list[tuple[str, str, str]], # (pid, orig_prompt, exp_prompt)
    seeds: list[int],
    prompt_bank: dict[str, dict[str, torch.Tensor]],
    model_path: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_images = len(prompts_map) * len(seeds)
    log(f"\nGenerating {total_images} images with Style Expansion ON (512x512, 20 steps)...")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
        torch_dtype=torch.float16,
    ).to("cuda")

    t_start = time.time()
    run_idx = 0

    for pid, _, exp_prompt in prompts_map:
        p_data = prompt_bank[exp_prompt]
        p_emb = p_data["prompt_embeds"].to("cuda")
        neg_p_emb = p_data["negative_prompt_embeds"].to("cuda")
        pool_emb = p_data["pooled_prompt_embeds"].to("cuda")
        neg_pool_emb = p_data["negative_pooled_prompt_embeds"].to("cuda")

        for seed in seeds:
            run_idx += 1
            filename = f"{pid}_seed_{seed}.png"
            img_path = out_dir / filename

            if img_path.exists():
                log(f"  [{run_idx:03d}/{total_images}] {pid} (s={seed}) | [CACHED]")
                continue

            gen = torch.Generator(device="cpu").manual_seed(seed)
            t0 = time.time()
            with torch.inference_mode():
                img = pipe(
                    prompt_embeds=p_emb,
                    pooled_prompt_embeds=pool_emb,
                    negative_prompt_embeds=neg_p_emb,
                    negative_pooled_prompt_embeds=neg_pool_emb,
                    num_inference_steps=20,
                    guidance_scale=4.5,
                    width=512,
                    height=512,
                    generator=gen,
                ).images[0]
            dt = time.time() - t0
            img.save(img_path)
            log(f"  [{run_idx:03d}/{total_images}] {pid} (s={seed}) | Generated in {dt:.2f}s ({dt/20.0:.3f}s/step)")

    total_time = time.time() - t_start
    log(f"[+] Generated in {total_time:.1f}s. Freeing pipeline...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def create_contact_sheet(
    prompts_map: list[tuple[str, str, str]],
    dir_off: Path,
    dir_on: Path,
    out_sheet_path: Path,
    num_pairs: int = 20,
) -> None:
    log(f"\nCreating 20-pair visual review contact sheet at {out_sheet_path}...")
    img_w, img_h = 256, 256
    sheet_w = 1200
    sheet_h = 60 + num_pairs * (img_h + 50)

    canvas = Image.new("RGB", (sheet_w, sheet_h), color=(25, 25, 30))
    draw = ImageDraw.Draw(canvas)

    y_offset = 20
    draw.text((30, y_offset), "PHASE C: AESTHETIC STYLE EXPANSION VISUAL REVIEW (20 PAIRS: OFF vs ON)", fill=(255, 255, 255))
    y_offset += 40

    for i in range(num_pairs):
        pid, orig_p, exp_p = prompts_map[i]
        seed = FIXED_SEEDS[0] # seed 42

        f_off = dir_off / f"{pid}_seed_{seed}.png"
        f_on = dir_on / f"{pid}_seed_{seed}.png"

        img_off = Image.open(f_off).resize((img_w, img_h), Image.Resampling.LANCZOS)
        img_on = Image.open(f_on).resize((img_w, img_h), Image.Resampling.LANCZOS)

        canvas.paste(img_off, (40, y_offset))
        canvas.paste(img_on, (320, y_offset))

        draw.text((40, y_offset + img_h + 5), f"OFF (s={seed})", fill=(180, 180, 180))
        draw.text((320, y_offset + img_h + 5), f"ON (s={seed})", fill=(100, 220, 100))

        # Text prompt
        text_x = 600
        draw.text((text_x, y_offset), f"[{pid}] ORIGINAL PROMPT:", fill=(255, 215, 0))
        # Wrap orig prompt
        draw.text((text_x, y_offset + 20), orig_p[:70] + ("..." if len(orig_p) > 70 else ""), fill=(220, 220, 220))
        
        draw.text((text_x, y_offset + 50), "EXPANDED CLAUSE:", fill=(100, 220, 100))
        draw.text((text_x, y_offset + 70), exp_p[len(orig_p):][:85] + ("..." if len(exp_p[len(orig_p):]) > 85 else ""), fill=(180, 240, 180))

        y_offset += img_h + 50

    canvas.save(out_sheet_path)
    log(f"[+] Contact sheet saved successfully to: {out_sheet_path}")


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 80)
    log("PHASE C: LLM AESTHETIC STYLE EXPANSION PAIRED EVALUATION & REGRESSION GATES")
    log("=" * 80)

    model_path = "models/sd35_medium"
    prompts = AESTHETIC_40_PROMPTS
    seeds = FIXED_SEEDS

    # 1. Expand all 40 aesthetic prompts
    prompts_map: list[tuple[str, str, str]] = []
    expanded_prompts: list[str] = []
    for idx, p in enumerate(prompts, start=1):
        pid = f"aes_{idx:02d}"
        res = expand_style(p, model="stable-diffusion-3.5", style_expansion_enabled=True)
        prompts_map.append((pid, p, res.expanded_prompt))
        expanded_prompts.append(res.expanded_prompt)

    log(f"[+] Generated {len(prompts_map)} expanded prompt specifications.")

    # 2. Pre-encode expanded prompts
    prompt_bank = pre_encode_prompts(model_path, expanded_prompts)

    # 3. Generate 160 expanded images (512x512, 20 steps)
    dir_off = ROOT_DIR / "benchmarks" / "images" / "sd35m_aesthetic_512x512_20steps"
    dir_on = ROOT_DIR / "benchmarks" / "images" / "sd35m_aesthetic_expanded_512x512_20steps"
    generate_images(prompts_map, seeds, prompt_bank, model_path, dir_on)

    # 4. Evaluate all 160 ON images
    log("\nLoading Real Aesthetic Evaluator...")
    evaluator = RealAestheticEvaluator(device=torch.device("cuda"))

    eval_off: list[dict[str, Any]] = []
    eval_on: list[dict[str, Any]] = []

    for pid, orig_p, _exp_p in prompts_map:
        for seed in seeds:
            f_off = dir_off / f"{pid}_seed_{seed}.png"
            f_on = dir_on / f"{pid}_seed_{seed}.png"

            img_off = Image.open(f_off).convert("RGB")
            img_on = Image.open(f_on).convert("RGB")

            sc_off = evaluator.evaluate(img_off, orig_p)
            # Evaluate ON image with original prompt to test semantic alignment retention and genuine aesthetic score
            sc_on = evaluator.evaluate(img_on, orig_p)

            eval_off.append(sc_off)
            eval_on.append(sc_on)

    # 5. Paired Statistical Analysis (N=160 Pairs)
    metrics = [
        ("laion_aesthetic_v2_4", "LAION v2.4"),
        ("clip_alignment", "CLIP Alignment"),
        ("pickscore_v1", "PickScore v1"),
        ("hps_v2_1", "HPS v2.1"),
        ("imagereward", "ImageReward"),
    ]

    log("\n" + "=" * 115)
    log("PAIRED STYLE EXPANSION RESULTS ON SD 3.5 MEDIUM (N=160 PAIRS):")
    log("=" * 115)
    log(f"{'Metric':<18} | {'OFF (Mean+-std)':<18} | {'ON (Mean+-std)':<18} | {'Mean Diff':<12} | {'95% CI of Diff':<18} | {'Paired t-stat':<14} | {'p-value':<12}")
    log("-" * 115)

    stats_results = {}

    for m_key, m_name in metrics:
        v_off = np.array([r[m_key] for r in eval_off])
        v_on = np.array([r[m_key] for r in eval_on])
        diff = v_on - v_off
        n = len(diff)

        m_off = float(np.mean(v_off))
        std_off = float(np.mean([np.std(v_off[i*4:(i+1)*4]) for i in range(40)]))

        m_on = float(np.mean(v_on))
        std_on = float(np.mean([np.std(v_on[i*4:(i+1)*4]) for i in range(40)]))

        d_mean = float(np.mean(diff))
        d_std = float(np.std(diff, ddof=1))
        d_se = d_std / math.sqrt(n)

        t_stat = d_mean / d_se
        from math import erfc
        p_val = erfc(abs(t_stat) / math.sqrt(2))

        ci_low = d_mean - 1.96 * d_se
        ci_high = d_mean + 1.96 * d_se

        stats_results[m_key] = {
            "off_mean": round(m_off, 4),
            "off_seed_std": round(std_off, 4),
            "on_mean": round(m_on, 4),
            "on_seed_std": round(std_on, 4),
            "d_mean": round(d_mean, 4),
            "d_se": round(d_se, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "t_stat": round(t_stat, 3),
            "p_value": p_val,
        }

        p_str = f"p={p_val:.2e}" if p_val < 1e-4 else f"p={p_val:.4f}"
        log(f"{m_name:<18} | {m_off:6.4f} +- {std_off:6.4f}   | {m_on:6.4f} +- {std_on:6.4f}   | {d_mean:+8.4f}   | [{ci_low:+7.4f}, {ci_high:+7.4f}] | t={t_stat:+7.2f}     | {p_str:<12}")

    log("=" * 115)

    # 6. Generate 20-pair Contact Sheet
    sheet_path = ROOT_DIR / "benchmarks" / "visual_review_style_expansion_ab.png"
    create_contact_sheet(prompts_map, dir_off, dir_on, sheet_path, num_pairs=20)

    # 7. Save JSON Results
    out_json = ROOT_DIR / "benchmarks" / "style_expansion_ab_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=2)
    log(f"\n[+] Saved paired style expansion statistics to: {out_json}")


if __name__ == "__main__":
    main()
