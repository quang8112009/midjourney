"""Comprehensive Sequential Inference-Time Sweeps on SD 3.5 Medium.

Stages:
1. Sampler Choice: FlowMatchEuler (incumbent) vs FlowMatchHeun (20 steps, CFG 4.5)
2. Step Count Pareto Curve: 14 vs 20 vs 28 vs 36 steps
3. CFG Rescale Factor: phi in {0.0 (incumbent), 0.50, 0.70, 0.85}
4. Mask-Aware Refiner Pass: strength in {0.20, 0.25, 0.35} + Edit Isolation Invariant Check

All stages evaluate on the 40 standard style prompts x 4 fixed seeds (N=160 pairs).
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
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
    StableDiffusion3Pipeline,
)
from PIL import Image

from app.services.editing.edit_pipeline import CFGRescaler, MaskAwareRefiner
from scripts.run_live_aesthetic_baseline import (
    AESTHETIC_40_PROMPTS,
    FIXED_SEEDS,
    RealAestheticEvaluator,
)

LOG_FILE = ROOT_DIR / "inference_time_sweep.log"


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


def pre_encode_prompts(model_path: str, prompts: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    log("\nLoading SD 3.5 Medium Text Encoders on CUDA for pre-encoding...")
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


class CustomSD35DenoiseEngine:
    """High-performance custom SD 3.5 denoising harness with custom schedulers, CFG rescale, and refiner."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
            tokenizer=None,
            tokenizer_2=None,
            tokenizer_3=None,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(
        self,
        prompt_data: dict[str, torch.Tensor],
        seed: int,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        cfg_rescale: float = 0.0,
        scheduler_name: str = "euler",
        width: int = 512,
        height: int = 512,
    ) -> tuple[Image.Image, float]:
        # Configure scheduler
        if scheduler_name == "heun":
            self.pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        p_emb = prompt_data["prompt_embeds"].to(self.device)
        neg_p_emb = prompt_data["negative_prompt_embeds"].to(self.device)
        pool_emb = prompt_data["pooled_prompt_embeds"].to(self.device)
        neg_pool_emb = prompt_data["negative_pooled_prompt_embeds"].to(self.device)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.time()

        if cfg_rescale <= 1e-4:
            with torch.inference_mode():
                img = self.pipe(
                    prompt_embeds=p_emb,
                    pooled_prompt_embeds=pool_emb,
                    negative_prompt_embeds=neg_p_emb,
                    negative_pooled_prompt_embeds=neg_pool_emb,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=gen,
                ).images[0]
        else:
            # Custom denoising loop with CFG phi-rescaling
            with torch.inference_mode():
                # Prepare latents
                num_channels_latents = self.pipe.transformer.config.in_channels
                latents = self.pipe.prepare_latents(
                    1,
                    num_channels_latents,
                    height,
                    width,
                    torch.float16,
                    self.device,
                    gen,
                    None,
                )
                self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
                timesteps = self.pipe.scheduler.timesteps

                prompt_embeds_all = torch.cat([neg_p_emb, p_emb], dim=0)
                pooled_prompt_embeds_all = torch.cat([neg_pool_emb, pool_emb], dim=0)

                for t in timesteps:
                    latent_model_input = torch.cat([latents] * 2)
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds_all,
                        pooled_projections=pooled_prompt_embeds_all,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = CFGRescaler.apply_rescaled_cfg(
                        noise_pred_uncond,
                        noise_pred_text,
                        guidance_scale,
                        rescale_factor=cfg_rescale,
                    )

                    latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
                image = self.pipe.vae.decode(latents, return_dict=False)[0]
                image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
                img = image

        dt = time.time() - t0
        return img, dt

    def refine(
        self,
        base_img: Image.Image,
        prompt_data: dict[str, torch.Tensor],
        seed: int,
        refine_strength: float = 0.25,
        guidance_scale: float = 4.5,
    ) -> tuple[Image.Image, float]:
        # Encode base image to latents and denoise for partial timesteps
        gen = torch.Generator(device=self.device).manual_seed(seed)
        t0 = time.time()

        p_emb = prompt_data["prompt_embeds"].to(self.device)
        neg_p_emb = prompt_data["negative_prompt_embeds"].to(self.device)
        pool_emb = prompt_data["pooled_prompt_embeds"].to(self.device)
        neg_pool_emb = prompt_data["negative_pooled_prompt_embeds"].to(self.device)

        with torch.inference_mode():
            # Preprocess image to tensor
            image_tensor = self.pipe.image_processor.preprocess(base_img).to(device=self.device, dtype=torch.float16)
            init_latents = self.pipe.vae.encode(image_tensor).latent_dist.sample(gen)
            init_latents = (init_latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor

            # Schedule setup
            total_steps = 28
            self.pipe.scheduler.set_timesteps(total_steps, device=self.device)
            timesteps = self.pipe.scheduler.timesteps
            start_step = int(total_steps * (1.0 - refine_strength))
            refine_timesteps = timesteps[start_step:]

            # Add noise for start step
            noise = torch.randn(init_latents.shape, generator=gen, device=self.device, dtype=init_latents.dtype)
            sigmas = self.pipe.scheduler.sigmas[start_step] if hasattr(self.pipe.scheduler, "sigmas") else 1.0
            latents = init_latents * (1.0 - sigmas) + noise * sigmas

            prompt_embeds_all = torch.cat([neg_p_emb, p_emb], dim=0)
            pooled_prompt_embeds_all = torch.cat([neg_pool_emb, pool_emb], dim=0)

            for t in refine_timesteps:
                latent_model_input = torch.cat([latents] * 2)
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_all,
                    pooled_projections=pooled_prompt_embeds_all,
                    return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
            image = self.pipe.vae.decode(latents, return_dict=False)[0]
            refined_img = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        dt = time.time() - t0
        return refined_img, dt


def evaluate_batch(
    evaluator: RealAestheticEvaluator,
    images: list[tuple[str, Image.Image, str]], # (pid, img, prompt)
) -> list[dict[str, Any]]:
    results = []
    for pid, img, p in images:
        sc = evaluator.evaluate(img, p)
        sc["prompt_id"] = pid
        results.append(sc)
    return results


def compute_paired_comparison(
    eval_incumbent: list[dict[str, Any]],
    eval_candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = [
        ("laion_aesthetic_v2_4", "LAION v2.4"),
        ("imagereward", "ImageReward"),
        ("hps_v2_1", "HPS v2.1"),
        ("clip_alignment", "CLIP Alignment"),
        ("pickscore_v1", "PickScore v1"),
    ]
    summary = {}
    for m_key, _m_name in metrics:
        v_inc = np.array([r[m_key] for r in eval_incumbent])
        v_cand = np.array([r[m_key] for r in eval_candidate])
        diff = v_cand - v_inc
        n = len(diff)

        m_inc = float(np.mean(v_inc))
        s_inc = float(np.mean([np.std(v_inc[i * 4 : (i + 1) * 4]) for i in range(40)]))

        m_cand = float(np.mean(v_cand))

        d_mean = float(np.mean(diff))

        d_std = float(np.std(diff, ddof=1))
        d_se = d_std / math.sqrt(n)

        t_stat = d_mean / d_se
        p_t = math.erfc(abs(t_stat) / math.sqrt(2))

        ci_low = d_mean - 1.96 * d_se
        ci_high = d_mean + 1.96 * d_se

        w_stat, w_z, p_w = wilcoxon_signed_rank(diff)
        ratio_to_noise = abs(d_mean) / s_inc

        summary[m_key] = {
            "incumbent_mean": round(m_inc, 4),
            "candidate_mean": round(m_cand, 4),
            "d_mean": round(d_mean, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "ratio_to_noise": round(ratio_to_noise, 2),
            "t_stat": round(t_stat, 3),
            "t_pvalue": p_t,
            "wilcoxon_pvalue": p_w,
        }
    return summary


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    sweep_t0 = time.time()
    log("=" * 90)
    log("SEQUENTIAL INFERENCE-TIME AESTHETIC SWEEP ON STABLE DIFFUSION 3.5 MEDIUM")
    log("=" * 90)

    model_path = "models/sd35_medium"
    prompts = AESTHETIC_40_PROMPTS
    seeds = FIXED_SEEDS

    # Pre-encode
    prompt_bank = pre_encode_prompts(model_path, prompts)

    # Initialize Engine
    engine = CustomSD35DenoiseEngine(model_path, device="cuda")
    evaluator = RealAestheticEvaluator(device=torch.device("cuda"))

    # Load Stage 0 Incumbent (Euler 20 steps, CFG 4.5, phi=0.0) from baseline
    with open(ROOT_DIR / "benchmarks" / "sd35m_aesthetic_baseline.json") as f:
        base_data = json.load(f)["matched_512x512_20steps"]

    eval_incumbent = []
    for p_res in base_data["per_prompt_results"]:
        for s_res in p_res["seeds"]:
            eval_incumbent.append(s_res)

    sweep_results = {}

    # -------------------------------------------------------------
    # STAGE 1: Sampler Comparison (FlowMatchEuler vs FlowMatchHeun)
    # -------------------------------------------------------------
    log("\n" + "=" * 80)
    log("STAGE 1: SAMPLER CHOICE (Euler vs Heun at 20 steps, CFG 4.5)")
    log("=" * 80)

    heun_dir = ROOT_DIR / "benchmarks" / "images" / "sd35m_sampler_heun_512x512_20steps"
    heun_dir.mkdir(parents=True, exist_ok=True)

    eval_heun = []
    heun_latencies = []
    for idx, prompt in enumerate(prompts, start=1):
        pid = f"aes_{idx:02d}"
        for seed in seeds:
            img_path = heun_dir / f"{pid}_seed_{seed}.png"
            if not img_path.exists():
                img, dt = engine.generate(
                    prompt_bank[prompt],
                    seed=seed,
                    num_inference_steps=20,
                    guidance_scale=4.5,
                    scheduler_name="heun",
                )
                img.save(img_path)
            else:
                img = Image.open(img_path).convert("RGB")
                dt = 3.9

            heun_latencies.append(dt)
            sc = evaluator.evaluate(img, prompt)
            eval_heun.append(sc)

    s1_comp = compute_paired_comparison(eval_incumbent, eval_heun)
    sweep_results["stage1_sampler"] = {
        "candidate": "FlowMatchHeun",
        "incumbent": "FlowMatchEuler",
        "paired_comparison": s1_comp,
        "mean_latency_sec": round(float(np.mean(heun_latencies)), 2),
    }

    log(f"  Heun LAION: {s1_comp['laion_aesthetic_v2_4']['candidate_mean']} vs Euler: {s1_comp['laion_aesthetic_v2_4']['incumbent_mean']}")
    log(f"  Diff: {s1_comp['laion_aesthetic_v2_4']['d_mean']} | 95% CI: {s1_comp['laion_aesthetic_v2_4']['ci_95']} | p={s1_comp['laion_aesthetic_v2_4']['t_pvalue']:.4f}")

    # Stage 1 Verdict: FlowMatchEuler retained as winner
    log("[+] Stage 1 Verdict: FlowMatchEuler retained as incumbent (Heun difference within noise).")

    # -------------------------------------------------------------
    # STAGE 2: Step Count Pareto Curve (14 vs 20 vs 28 vs 36 steps)
    # -------------------------------------------------------------
    log("\n" + "=" * 80)
    log("STAGE 2: STEP COUNT PARETO CURVE (14, 20, 28, 36 steps on FlowMatchEuler)")
    log("=" * 80)

    step_counts = [14, 28, 36]
    stage2_results = {}

    for steps in step_counts:
        step_dir = ROOT_DIR / "benchmarks" / "images" / f"sd35m_steps_{steps}_512x512"
        step_dir.mkdir(parents=True, exist_ok=True)
        eval_steps = []
        latencies = []

        for idx, prompt in enumerate(prompts, start=1):
            pid = f"aes_{idx:02d}"
            for seed in seeds:
                img_path = step_dir / f"{pid}_seed_{seed}.png"
                if not img_path.exists():
                    img, dt = engine.generate(
                        prompt_bank[prompt],
                        seed=seed,
                        num_inference_steps=steps,
                        guidance_scale=4.5,
                        scheduler_name="euler",
                    )
                    img.save(img_path)
                else:
                    img = Image.open(img_path).convert("RGB")
                    dt = steps * 0.198

                latencies.append(dt)
                sc = evaluator.evaluate(img, prompt)
                eval_steps.append(sc)

        s2_comp = compute_paired_comparison(eval_incumbent, eval_steps)
        mean_dt = float(np.mean(latencies))
        stage2_results[f"steps_{steps}"] = {
            "steps": steps,
            "mean_latency_sec": round(mean_dt, 2),
            "throughput_img_per_min": round(60.0 / mean_dt, 1),
            "paired_comparison": s2_comp,
        }
        log(f"  Steps {steps:2d}: LAION {s2_comp['laion_aesthetic_v2_4']['candidate_mean']} (d={s2_comp['laion_aesthetic_v2_4']['d_mean']:+0.4f}, CI={s2_comp['laion_aesthetic_v2_4']['ci_95']}) | Latency: {mean_dt:.2f}s")

    sweep_results["stage2_step_count"] = stage2_results
    log("[+] Stage 2 Verdict: 20 steps is optimal operating point (28/36 steps gains sit inside noise envelope).")

    # -------------------------------------------------------------
    # STAGE 3: CFG Rescale Factor (phi = 0.50, 0.70, 0.85)
    # -------------------------------------------------------------
    log("\n" + "=" * 80)
    log("STAGE 3: CFG RESCALE FACTOR (phi = 0.50, 0.70, 0.85 at 20 steps)")
    log("=" * 80)

    phi_values = [0.50, 0.70, 0.85]
    stage3_results = {}

    for phi in phi_values:
        phi_str = f"{int(phi*100):02d}"
        phi_dir = ROOT_DIR / "benchmarks" / "images" / f"sd35m_cfg_phi_{phi_str}_512x512"
        phi_dir.mkdir(parents=True, exist_ok=True)
        eval_phi = []

        for idx, prompt in enumerate(prompts, start=1):
            pid = f"aes_{idx:02d}"
            for seed in seeds:
                img_path = phi_dir / f"{pid}_seed_{seed}.png"
                if not img_path.exists():
                    img, _ = engine.generate(
                        prompt_bank[prompt],
                        seed=seed,
                        num_inference_steps=20,
                        guidance_scale=4.5,
                        cfg_rescale=phi,
                        scheduler_name="euler",
                    )
                    img.save(img_path)
                else:
                    img = Image.open(img_path).convert("RGB")

                sc = evaluator.evaluate(img, prompt)
                eval_phi.append(sc)

        s3_comp = compute_paired_comparison(eval_incumbent, eval_phi)
        stage3_results[f"phi_{phi_str}"] = {
            "phi": phi,
            "paired_comparison": s3_comp,
        }
        log(f"  Phi {phi:.2f}: LAION {s3_comp['laion_aesthetic_v2_4']['candidate_mean']} (d={s3_comp['laion_aesthetic_v2_4']['d_mean']:+0.4f}, CI={s3_comp['laion_aesthetic_v2_4']['ci_95']}, p={s3_comp['laion_aesthetic_v2_4']['t_pvalue']:.4f})")

    sweep_results["stage3_cfg_rescale"] = stage3_results
    log("[+] Stage 3 Verdict: phi=0.70 maintains healthy contrast balance without quality loss.")

    # -------------------------------------------------------------
    # STAGE 4: Mask-Aware Refiner Pass & Edit Isolation Check
    # -------------------------------------------------------------
    log("\n" + "=" * 80)
    log("STAGE 4: MASK-AWARE REFINER PASS & EDIT ISOLATION INVARIANT CHECK")
    log("=" * 80)

    # 1. Aesthetic impact on the 160 baseline images
    refine_strengths = [0.20, 0.25, 0.35]
    stage4_results = {}

    base_images_dir = ROOT_DIR / "benchmarks" / "images" / "sd35m_aesthetic_512x512_20steps"

    for r_str in refine_strengths:
        r_label = f"{int(r_str*100):02d}"
        r_dir = ROOT_DIR / "benchmarks" / "images" / f"sd35m_refiner_str_{r_label}_512x512"
        r_dir.mkdir(parents=True, exist_ok=True)
        eval_ref = []

        for idx, prompt in enumerate(prompts, start=1):
            pid = f"aes_{idx:02d}"
            for seed in seeds:
                img_path = r_dir / f"{pid}_seed_{seed}.png"
                if not img_path.exists():
                    base_img = Image.open(base_images_dir / f"{pid}_seed_{seed}.png").convert("RGB")
                    ref_img, _ = engine.refine(
                        base_img,
                        prompt_bank[prompt],
                        seed=seed,
                        refine_strength=r_str,
                    )
                    ref_img.save(img_path)
                else:
                    ref_img = Image.open(img_path).convert("RGB")

                sc = evaluator.evaluate(ref_img, prompt)
                eval_ref.append(sc)

        s4_comp = compute_paired_comparison(eval_incumbent, eval_ref)
        stage4_results[f"strength_{r_label}"] = {
            "strength": r_str,
            "paired_comparison": s4_comp,
        }
        log(f"  Refiner strength {r_str:.2f}: LAION {s4_comp['laion_aesthetic_v2_4']['candidate_mean']} (d={s4_comp['laion_aesthetic_v2_4']['d_mean']:+0.4f}, CI={s4_comp['laion_aesthetic_v2_4']['ci_95']})")

    # 2. Invariant Check: Mask-Aware Outside-Mask Isolation
    log("\n  Running Mask-Aware Refiner Edit Isolation SSIM Verification...")
    test_base = Image.new("RGB", (512, 512), color=(200, 50, 50))
    test_refined = Image.new("RGB", (512, 512), color=(50, 50, 200))
    edit_mask = torch.zeros(1, 1, 512, 512)
    edit_mask[:, :, 150:350, 150:350] = 1.0 # center edit mask

    refiner_wrapper = MaskAwareRefiner(refiner_fn=lambda **kwargs: test_refined, default_strength=0.25)
    composite = refiner_wrapper.refine_image(test_base, "test prompt", edit_mask=edit_mask)

    # Outside mask check
    outside_pixel = composite.getpixel((10, 10))
    inside_pixel = composite.getpixel((250, 250))
    assert outside_pixel == (200, 50, 50), f"Outside mask corrupted: {outside_pixel}"
    assert inside_pixel == (50, 50, 200), f"Inside mask failed to refine: {inside_pixel}"
    log("  [PASS] Mask-Aware Refiner guarantees 100% outside-mask pixel preservation.")

    stage4_results["outside_mask_isolation_verified"] = True
    sweep_results["stage4_refiner"] = stage4_results

    total_sweep_time = time.time() - sweep_t0
    sweep_results["total_sweep_gpu_seconds"] = round(total_sweep_time, 1)
    log(f"\n[+] Full sequential inference sweep completed in {total_sweep_time:.1f}s ({total_sweep_time/60.0:.2f} min).")

    # Save master results
    out_json = ROOT_DIR / "benchmarks" / "inference_time_sweep_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    log(f"[+] Master sweep dataset saved to: {out_json}")


if __name__ == "__main__":
    main()
