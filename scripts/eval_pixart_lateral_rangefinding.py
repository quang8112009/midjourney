"""Range-finding sweep on PixArt-Alpha across lateral guidance strengths.

Sweeps strength in {0.0, 1.5, 3.0, 6.0, 10.0} on 8 lateral prompts x 2 seeds (N=16 per strength).
Explicitly marked as underpowered range-finding to bracket operating strength for the powered study.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
from scripts.eval_spatial_lateral_dedicated import LATERAL_24_SPECS
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator, compute_image_ssim
from scripts.run_live_aesthetic_baseline import RealAestheticEvaluator

STRENGTHS = [0.0, 1.5, 3.0, 6.0, 10.0]
SEEDS_2 = [42, 100]
SWEEP_SPECS_8 = LATERAL_24_SPECS[:8] # 8 prompts

def main():
    print("=" * 85)
    print("PIXART-ALPHA LATERAL STRENGTH RANGE-FINDING SWEEP (UNDERPOWERED, N=16 per condition)")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator_spatial = StrictSpatialEvaluator(device=device)
    evaluator_aes = RealAestheticEvaluator(device=device)

    out_dir = ROOT_DIR / "benchmarks" / "images" / "pixart_strength_rangefinding"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Pipeline
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

    # 2. Pre-compute plans
    plans = {}
    for spec in SWEEP_SPECS_8:
        intent = analyze_prompt(spec["prompt"], mode="generate")
        plans[spec["id"]] = plan_semantic_layout(intent, tokenizer=tok_t5)

    # 3. Generate images across strengths
    results_by_strength = {}
    baseline_images = {} # (pid, seed) -> PIL image

    t0_all = time.time()

    for str_val in STRENGTHS:
        print(f"\n--- Generating & Evaluating Strength = {str_val:.1f} ---")
        str_records = []
        ssim_list = []
        laion_list = []
        clip_list = []
        sat_list = []
        presence_list = []

        for spec in SWEEP_SPECS_8:
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

            for seed in SEEDS_2:
                img_path = out_dir / f"{pid}_s{seed}_str_{str_val:.1f}.png"
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

                if str_val == 0.0:
                    baseline_images[(pid, seed)] = img

                # Evaluation
                s_det, o_det = evaluator_spatial.detect_entities(img, subj, obj)
                sat, reason = evaluator_spatial.check_relation(s_det, o_det, rel)
                both_present = bool(s_det and o_det)

                base_img = baseline_images.get((pid, seed), img)
                ssim_val = compute_image_ssim(base_img, img)

                aes_scores = evaluator_aes.evaluate(img, prompt)

                sat_list.append(bool(sat))
                presence_list.append(both_present)
                ssim_list.append(ssim_val)
                laion_list.append(aes_scores["laion_aesthetic_v2_4"])
                clip_list.append(aes_scores["clip_alignment"])

                str_records.append({
                    "id": pid,
                    "seed": seed,
                    "satisfied": bool(sat),
                    "both_present": both_present,
                    "ssim": ssim_val,
                    "laion": aes_scores["laion_aesthetic_v2_4"],
                    "clip": aes_scores["clip_alignment"],
                })

        sat_rate = float(np.mean(sat_list)) * 100.0
        pres_rate = float(np.mean(presence_list)) * 100.0
        mean_ssim = float(np.mean(ssim_list))
        mean_laion = float(np.mean(laion_list))
        mean_clip = float(np.mean(clip_list))

        results_by_strength[str_val] = {
            "strength": str_val,
            "satisfaction_pct": round(sat_rate, 2),
            "dual_presence_pct": round(pres_rate, 2),
            "ssim_vs_off": round(mean_ssim, 4),
            "laion_aesthetic": round(mean_laion, 3),
            "clip_alignment": round(mean_clip, 4),
            "num_runs": len(str_records),
        }

        print(f"Strength {str_val:4.1f}: Sat = {sat_rate:5.1f}% | Presence = {pres_rate:5.1f}% | SSIM = {mean_ssim:.4f} | LAION = {mean_laion:.3f} | CLIP = {mean_clip:.4f}")

    total_time = time.time() - t0_all
    print("\n" + "=" * 85)
    print(f"[+] Range-finding complete in {total_time:.1f}s ({total_time/60:.2f} min).")
    
    with open(ROOT_DIR / "benchmarks" / "pixart_lateral_rangefinding.json", "w", encoding="utf-8") as f:
        json.dump(results_by_strength, f, indent=2)

if __name__ == "__main__":
    main()
