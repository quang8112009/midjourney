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
from diffusers import StableDiffusion3Pipeline
from PIL import Image

from scripts.run_live_aesthetic_baseline import (
    AESTHETIC_40_PROMPTS,
    FIXED_SEEDS,
    RealAestheticEvaluator,
    compute_file_sha256,
)

LOG_FILE = ROOT_DIR / "sd35m_aesthetic_eval.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def pre_encode_aesthetic_prompts(model_path: str, prompts: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    log("\n[Step 1] Loading SD 3.5 Medium Text Encoders on CUDA...")
    t0 = time.time()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        transformer=None,
        vae=None,
        torch_dtype=torch.float16,
    ).to("cuda")

    bank: dict[str, dict[str, torch.Tensor]] = {}
    log(f"[+] Encoding {len(prompts)} aesthetic prompts...")
    for idx, prompt in enumerate(prompts, start=1):
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
        if idx % 10 == 0 or idx == len(prompts):
            log(f"  Encoded [{idx:02d}/{len(prompts)}] prompts in {time.time() - t0:.2f}s")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    log(f"[+] All {len(prompts)} prompts pre-encoded in {time.time() - t0:.2f}s. Encoders freed from VRAM.")
    return bank


def generate_configuration_images(
    config_name: str,
    width: int,
    height: int,
    num_steps: int,
    guidance_scale: float,
    prompts: list[str],
    seeds: list[int],
    prompt_bank: dict[str, dict[str, torch.Tensor]],
    model_path: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_images = len(prompts) * len(seeds)
    log("\n================================================================================")
    log(f"GENERATING CONFIGURATION: [{config_name.upper()}] ({width}x{height} / {num_steps} steps, N={total_images})")
    log("================================================================================")

    log("Loading SD 3.5 Medium Transformer & VAE on CUDA (fp16)...")
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

    for idx, prompt in enumerate(prompts, start=1):
        prompt_id = f"aes_{idx:02d}"
        p_data = prompt_bank[prompt]
        p_emb = p_data["prompt_embeds"].to("cuda")
        neg_p_emb = p_data["negative_prompt_embeds"].to("cuda")
        pool_emb = p_data["pooled_prompt_embeds"].to("cuda")
        neg_pool_emb = p_data["negative_pooled_prompt_embeds"].to("cuda")

        for seed in seeds:
            run_idx += 1
            filename = f"{prompt_id}_seed_{seed}.png"
            img_path = out_dir / filename

            if img_path.exists():
                log(f"  [{run_idx:03d}/{total_images}] {prompt_id} (s={seed}) | [CACHED]")
                continue

            gen = torch.Generator(device="cpu").manual_seed(seed)
            t0 = time.time()
            with torch.inference_mode():
                img = pipe(
                    prompt_embeds=p_emb,
                    pooled_prompt_embeds=pool_emb,
                    negative_prompt_embeds=neg_p_emb,
                    negative_pooled_prompt_embeds=neg_pool_emb,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=gen,
                ).images[0]
            dt = time.time() - t0
            img.save(img_path)
            log(f"  [{run_idx:03d}/{total_images}] {prompt_id} (s={seed}) | Generated in {dt:.2f}s ({dt/num_steps:.3f}s/step)")

    total_time = time.time() - t_start
    log(f"\n[+] Configuration [{config_name}] completed in {total_time:.1f}s ({total_time/total_images:.2f}s/image). Freeing pipeline...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_image_set(
    config_name: str,
    prompts: list[str],
    seeds: list[int],
    images_dir: Path,
    evaluator: RealAestheticEvaluator,
) -> dict[str, Any]:
    log(f"\nEvaluating aesthetic metrics for [{config_name}] across {len(prompts)*len(seeds)} images...")
    results_per_prompt = []
    all_laions = []
    all_clips = []
    all_picks = []
    all_hpss = []
    all_rewards = []

    # Per-prompt cross-seed standard deviations
    prompt_laion_stds = []
    prompt_clip_stds = []

    sample_hashes = []

    for idx, prompt in enumerate(prompts, start=1):
        prompt_id = f"aes_{idx:02d}"
        seed_results = []
        p_laions = []
        p_clips = []
        p_picks = []
        p_hpss = []
        p_rewards = []

        for seed in seeds:
            filename = f"{prompt_id}_seed_{seed}.png"
            img_path = images_dir / filename
            img = Image.open(img_path).convert("RGB")
            sha = compute_file_sha256(img_path)

            scores = evaluator.evaluate(img, prompt)
            scores["seed"] = seed
            scores["filename"] = filename
            scores["sha256"] = sha
            seed_results.append(scores)

            p_laions.append(scores["laion_aesthetic_v2_4"])
            p_clips.append(scores["clip_alignment"])
            p_picks.append(scores["pickscore_v1"])
            p_hpss.append(scores["hps_v2_1"])
            p_rewards.append(scores["imagereward"])

            all_laions.append(scores["laion_aesthetic_v2_4"])
            all_clips.append(scores["clip_alignment"])
            all_picks.append(scores["pickscore_v1"])
            all_hpss.append(scores["hps_v2_1"])
            all_rewards.append(scores["imagereward"])

            if idx <= 3:
                sample_hashes.append({"filename": filename, "sha256": sha})

        prompt_laion_stds.append(float(np.std(p_laions)))
        prompt_clip_stds.append(float(np.std(p_clips)))

        results_per_prompt.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "mean_laion": round(float(np.mean(p_laions)), 3),
            "std_laion": round(float(np.std(p_laions)), 4),
            "mean_clip": round(float(np.mean(p_clips)), 4),
            "std_clip": round(float(np.std(p_clips)), 4),
            "mean_pickscore": round(float(np.mean(p_picks)), 4),
            "mean_hpsv2": round(float(np.mean(p_hpss)), 4),
            "mean_imagereward": round(float(np.mean(p_rewards)), 4),
            "seeds": seed_results,
        })

    # Overall cross-seed variance: average within-prompt cross-seed standard deviation
    avg_within_prompt_seed_std_laion = float(np.mean(prompt_laion_stds))
    avg_within_prompt_seed_std_clip = float(np.mean(prompt_clip_stds))

    summary = {
        "config_name": config_name,
        "overall_metrics": {
            "laion_aesthetic_v2_4": {
                "mean": round(float(np.mean(all_laions)), 3),
                "cross_seed_std": round(avg_within_prompt_seed_std_laion, 4),
                "total_std": round(float(np.std(all_laions)), 4),
            },
            "clip_alignment": {
                "mean": round(float(np.mean(all_clips)), 4),
                "cross_seed_std": round(avg_within_prompt_seed_std_clip, 4),
                "total_std": round(float(np.std(all_clips)), 4),
            },
            "pickscore_v1": {
                "mean": round(float(np.mean(all_picks)), 4),
            },
            "hps_v2_1": {
                "mean": round(float(np.mean(all_hpss)), 4),
            },
            "imagereward": {
                "mean": round(float(np.mean(all_rewards)), 4),
            },
        },
        "sample_sha256_hashes": sample_hashes,
        "per_prompt_results": results_per_prompt,
    }
    return summary


def main() -> int:
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 80)
    log("PHASE A: STABLE DIFFUSION 3.5 MEDIUM REAL AESTHETIC BASELINE STUDY")
    log("Evaluating Matched Baseline (512x512/20s) and Native Setting (1024x1024/28s)")
    log("=" * 80)

    model_path = "models/sd35_medium"
    prompts = AESTHETIC_40_PROMPTS
    seeds = FIXED_SEEDS

    # 1. Pre-encode all 40 prompts
    prompt_bank = pre_encode_aesthetic_prompts(model_path, prompts)

    # 2. Config A: Matched Baseline (512x512 / 20 steps, guidance_scale=4.5)
    dir_matched = ROOT_DIR / "benchmarks" / "images" / "sd35m_aesthetic_512x512_20steps"
    generate_configuration_images(
        config_name="sd35m_matched_512x512_20steps",
        width=512,
        height=512,
        num_steps=20,
        guidance_scale=4.5,
        prompts=prompts,
        seeds=seeds,
        prompt_bank=prompt_bank,
        model_path=model_path,
        out_dir=dir_matched,
    )

    # 3. Config B: Native Resolution (1024x1024 / 28 steps, guidance_scale=4.5)
    dir_native = ROOT_DIR / "benchmarks" / "images" / "sd35m_aesthetic_1024x1024_28steps"
    generate_configuration_images(
        config_name="sd35m_native_1024x1024_28steps",
        width=1024,
        height=1024,
        num_steps=28,
        guidance_scale=4.5,
        prompts=prompts,
        seeds=seeds,
        prompt_bank=prompt_bank,
        model_path=model_path,
        out_dir=dir_native,
    )

    # 4. Initialize Real Evaluator (Pretrained CLIP-L/14 + LAION Head)
    log("\n[Step 4] Loading Real Aesthetic Evaluator (CLIP-ViT-L/14 + LAION Predictor)...")
    evaluator = RealAestheticEvaluator(device=torch.device("cuda"))

    # 5. Evaluate both configurations
    summary_matched = evaluate_image_set(
        config_name="sd35m_matched_512x512_20steps",
        prompts=prompts,
        seeds=seeds,
        images_dir=dir_matched,
        evaluator=evaluator,
    )

    summary_native = evaluate_image_set(
        config_name="sd35m_native_1024x1024_28steps",
        prompts=prompts,
        seeds=seeds,
        images_dir=dir_native,
        evaluator=evaluator,
    )

    # 6. Load Committed SD v1.5 Baseline for Comparison
    with open(ROOT_DIR / "benchmarks" / "aesthetic_baseline.json") as f:
        sd15_baseline = json.load(f)

    # 7. Print Master Summary Table
    log("\n" + "=" * 90)
    log("PHASE A: COMPREHENSIVE AESTHETIC BASELINE COMPARISON TABLE")
    log("=" * 90)
    log(f"{'Backbone / Configuration':<35} | {'LAION v2.4':<14} | {'CLIP Align':<12} | {'PickScore':<10} | {'HPS v2.1':<10} | {'ImageReward':<12}")
    log("-" * 90)

    # SD v1.5
    sd15_m = sd15_baseline["overall_metrics"]
    log(f"{'SD v1.5 (512x512, 20 steps)':<35} | {sd15_m['laion_aesthetic_v2_4']['mean']:.3f} ± {sd15_m['laion_aesthetic_v2_4']['cross_seed_std']:.3f} | {sd15_m['clip_alignment']['mean']:.4f}     | {sd15_m['pickscore_v1']['mean']:.4f}   | {sd15_m['hps_v2_1']['mean']:.4f}   | {sd15_m['imagereward']['mean']:.4f}")

    # SD 3.5 M Matched
    m_m = summary_matched["overall_metrics"]
    log(f"{'SD 3.5 M (512x512, 20 steps)':<35} | {m_m['laion_aesthetic_v2_4']['mean']:.3f} ± {m_m['laion_aesthetic_v2_4']['cross_seed_std']:.3f} | {m_m['clip_alignment']['mean']:.4f}     | {m_m['pickscore_v1']['mean']:.4f}   | {m_m['hps_v2_1']['mean']:.4f}   | {m_m['imagereward']['mean']:.4f}")

    # SD 3.5 M Native
    n_m = summary_native["overall_metrics"]
    log(f"{'SD 3.5 M (1024x1024, 28 steps)':<35} | {n_m['laion_aesthetic_v2_4']['mean']:.3f} ± {n_m['laion_aesthetic_v2_4']['cross_seed_std']:.3f} | {n_m['clip_alignment']['mean']:.4f}     | {n_m['pickscore_v1']['mean']:.4f}   | {n_m['hps_v2_1']['mean']:.4f}   | {n_m['imagereward']['mean']:.4f}")
    log("=" * 90)

    # 8. Save output JSON
    full_output = {
        "metadata": {
            "title": "Phase A: SD 3.5 Medium Real Aesthetic Baseline Study",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware": "NVIDIA GeForce RTX 4060 Ti (16GB)",
            "prompts_count": len(prompts),
            "seeds": seeds,
        },
        "matched_512x512_20steps": summary_matched,
        "native_1024x1024_28steps": summary_native,
        "sd15_baseline_reference": sd15_baseline["overall_metrics"],
    }

    out_file = ROOT_DIR / "benchmarks" / "sd35m_aesthetic_baseline.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    log(f"\n[+] Saved full aesthetic baseline results to: {out_file}")

    return 0


if __name__ == "__main__":
    main()
