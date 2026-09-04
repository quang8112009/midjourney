"""Evaluate Guidance Layer Effect: A/B Spatial Evaluation (Guidance OFF vs ON).

Measures pixel-level delta, perceptual distance, CLIP alignment, and LAION aesthetic
across 16 spatial relation prompts and 4 fixed seeds on live CUDA SD v1.5.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.editing.layout_guidance import (  # noqa: E402
    LayoutGuidanceProcessor,
    TwoPhaseSchedule,
)
from app.services.editing.prompt_intent import analyze_prompt  # noqa: E402
from app.services.editing.semantic_planner import plan_semantic_layout  # noqa: E402

SPATIAL_16_PROMPTS = [
    # 1. in front of / behind
    "a red cube in front of a blue sphere on a marble floor",
    "a golden statue behind a stone fountain in a courtyard",
    "a green chair in front of a brick fireplace in a cozy living room",
    "a tall oak tree behind a small wooden cottage in a meadow",
    # 2. on / riding
    "a white coffee cup on top of a stack of vintage books",
    "a brown teddy bear resting on a leather sofa",
    "a red apple on a ceramic plate on a wooden kitchen counter",
    "a small sparrow perched on a rusted metal fence",
    # 3. under / below
    "a black cat sitting under a wooden dining chair",
    "a green sea turtle swimming below a translucent jellyfish in clear ocean",
    "a pair of leather boots placed under a wooden bench",
    "a colorful rug under a glass coffee table",
    # 4. next to / beside / lateral relations
    "a yellow banana to the left of a green apple on a white table",
    "a crystal vase beside an antique brass clock on a mantelpiece",
    "a red sports car parked to the right of a blue bicycle",
    "a silver teapot beside a porcelain teacup on a tray",
]

FIXED_SEEDS = [42, 100, 2024, 7777]


class OfficialLAIONAestheticPredictor(nn.Module):
    def __init__(self, input_dim: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MetricEvaluator:
    def __init__(
        self,
        clip_dir: Path | None = None,
        weights_path: Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_path = clip_dir or (ROOT_DIR / "models" / "clip_vit_l14")
        w_path = weights_path or (ROOT_DIR / "models" / "sac_logos_ava1_l14_linearMSE.pth")

        self.processor = CLIPProcessor.from_pretrained(str(clip_path))
        self.clip_model = CLIPModel.from_pretrained(str(clip_path)).to(self.device).eval()

        state_dict = torch.load(str(w_path), map_location=self.device, weights_only=True)
        self.aesthetic_head = OfficialLAIONAestheticPredictor(768).to(self.device).eval()
        self.aesthetic_head.load_state_dict(state_dict)

    @torch.inference_mode()
    def evaluate_aesthetic_and_clip(
        self, image: Image.Image, prompt: str
    ) -> tuple[float, float, torch.Tensor]:
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(
            self.device
        )
        img_out = self.clip_model.get_image_features(pixel_values=inputs.pixel_values)
        txt_out = self.clip_model.get_text_features(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )

        img_feat = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
        txt_feat = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out

        img_norm = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_norm = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        laion_score = self.aesthetic_head(img_norm.float()).item()
        clip_sim = (img_norm * txt_norm).sum(dim=-1).item()
        return laion_score, clip_sim, img_norm


def compute_pixel_and_perceptual_metrics(
    img_off: Image.Image,
    img_on: Image.Image,
    feat_off: torch.Tensor,
    feat_on: torch.Tensor,
) -> dict[str, float]:
    arr_off = np.array(img_off, dtype=np.float32) / 255.0
    arr_on = np.array(img_on, dtype=np.float32) / 255.0

    l1_diff = float(np.mean(np.abs(arr_off - arr_on)))
    l1_diff_255 = l1_diff * 255.0

    rmse = float(np.sqrt(np.mean((arr_off - arr_on) ** 2)))

    if rmse < 1e-10:
        psnr = 100.0
    else:
        psnr = float(20 * math.log10(1.0 / rmse))

    mu_x = arr_off.mean()
    mu_y = arr_on.mean()
    sigma_x = arr_off.std()
    sigma_y = arr_on.std()
    sigma_xy = np.mean((arr_off - mu_x) * (arr_on - mu_y))
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2)
    ssim = float(num / den)
    structural_dissimilarity = max(0.0, 1.0 - ssim)

    perceptual_dist = float(1.0 - (feat_off * feat_on).sum(dim=-1).item())

    return {
        "l1_pixel_diff": round(l1_diff, 4),
        "l1_diff_scale_255": round(l1_diff_255, 2),
        "rmse": round(rmse, 4),
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim, 4),
        "structural_dissimilarity": round(structural_dissimilarity, 4),
        "perceptual_distance_clip": round(perceptual_dist, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Spatial Guidance A/B Evaluation (OFF vs ON).")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT_DIR / "benchmarks" / "spatial_ab_results.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    out_dir = ROOT_DIR / "benchmarks" / "images" / "spatial_ab_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("SPATIAL GUIDANCE A/B EVALUATION: GUIDANCE OFF vs. GUIDANCE ON")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ROOT_DIR / "models" / "sd15_fp16"

    print(f"Loading SD v1.5 from {model_path} on CUDA (fp16)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    evaluator = MetricEvaluator(device=device)

    total_runs = len(SPATIAL_16_PROMPTS) * len(FIXED_SEEDS)
    print(f"Executing {total_runs} paired A/B generations (16 prompts x 4 seeds)...")

    results_per_prompt = []
    all_l1_diffs = []
    all_rmse_diffs = []
    all_perceptual_dists = []
    all_ssims = []

    all_laion_off = []
    all_laion_on = []
    all_clip_off = []
    all_clip_on = []

    count = 0
    t0_all = time.time()

    for p_idx, prompt in enumerate(SPATIAL_16_PROMPTS, start=1):
        prompt_id = f"spatial_{p_idx:02d}"
        intent = analyze_prompt(prompt, mode="generate")
        plan = plan_semantic_layout(intent, tokenizer=pipe.tokenizer)

        prompt_pairs = []

        for seed in FIXED_SEEDS:
            count += 1
            file_off = out_dir / f"{prompt_id}_s{seed}_off.png"
            file_on = out_dir / f"{prompt_id}_s{seed}_on.png"

            # ----------------------------------------------------
            # RUN A: Guidance OFF (clean standard UNet processors)
            # ----------------------------------------------------
            pipe.unet.set_attn_processor(
                {k: AttnProcessor() for k in pipe.unet.attn_processors.keys()}
            )
            gen_off = torch.Generator("cuda").manual_seed(seed)
            with torch.inference_mode():
                img_off = pipe(
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=gen_off,
                    width=512,
                    height=512,
                ).images[0]
            img_off.save(file_off)

            # ----------------------------------------------------
            # RUN B: Guidance ON (hooked cross-attention processors)
            # ----------------------------------------------------
            attn_procs = {}
            schedule = TwoPhaseSchedule(schedule_cutoff=0.8)
            for name, proc in pipe.unet.attn_processors.items():
                if name.endswith("attn2.processor"):
                    attn_procs[name] = LayoutGuidanceProcessor(
                        base_processor=proc,
                        plan=plan,
                        guidance_strength=0.35,
                        schedule=schedule,
                        depth_guidance_enabled=True,
                    )
                else:
                    attn_procs[name] = proc
            pipe.unet.set_attn_processor(attn_procs)

            def step_callback(pipe, step_idx, timestep, callback_kwargs):
                progress = float(step_idx + 1) / 20.0
                for p in pipe.unet.attn_processors.values():
                    if isinstance(p, LayoutGuidanceProcessor):
                        p.set_step_progress(progress)
                return callback_kwargs

            gen_on = torch.Generator("cuda").manual_seed(seed)
            with torch.inference_mode():
                img_on = pipe(
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=gen_on,
                    width=512,
                    height=512,
                    callback_on_step_end=step_callback,
                ).images[0]
            img_on.save(file_on)

            # ----------------------------------------------------
            # EVALUATION & METRICS
            # ----------------------------------------------------
            laion_off, clip_off, feat_off = evaluator.evaluate_aesthetic_and_clip(img_off, prompt)
            laion_on, clip_on, feat_on = evaluator.evaluate_aesthetic_and_clip(img_on, prompt)

            pixel_metrics = compute_pixel_and_perceptual_metrics(img_off, img_on, feat_off, feat_on)

            pair_res = {
                "seed": seed,
                "files": {
                    "off": str(file_off.relative_to(ROOT_DIR)).replace("\\", "/"),
                    "on": str(file_on.relative_to(ROOT_DIR)).replace("\\", "/"),
                },
                "guidance_off": {
                    "laion_aesthetic": round(laion_off, 3),
                    "clip_alignment": round(clip_off, 4),
                },
                "guidance_on": {
                    "laion_aesthetic": round(laion_on, 3),
                    "clip_alignment": round(clip_on, 4),
                },
                "divergence": pixel_metrics,
            }
            prompt_pairs.append(pair_res)

            all_l1_diffs.append(pixel_metrics["l1_pixel_diff"])
            all_rmse_diffs.append(pixel_metrics["rmse"])
            all_perceptual_dists.append(pixel_metrics["perceptual_distance_clip"])
            all_ssims.append(pixel_metrics["ssim"])

            all_laion_off.append(laion_off)
            all_laion_on.append(laion_on)
            all_clip_off.append(clip_off)
            all_clip_on.append(clip_on)

            l1_val = pixel_metrics["l1_pixel_diff"]
            l1_255 = pixel_metrics["l1_diff_scale_255"]
            ssim_val = pixel_metrics["ssim"]
            p_dist = pixel_metrics["perceptual_distance_clip"]
            msg = (
                f"[{count:02d}/{total_runs:02d}] {prompt_id} (Seed {seed}) -> "
                f"Pixel L1: {l1_val:.4f} ({l1_255:.1f}/255), "
                f"SSIM: {ssim_val:.3f}, Perceptual Dist: {p_dist:.4f}, "
                f"LAION: {laion_off:.2f}->{laion_on:.2f}"
            )
            print(f"  {msg}")

        results_per_prompt.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "planned_entities": [(o.label, o.box.center, o.attributes) for o in plan.objects],
                "planned_relations": [
                    (r.subject, r.relation_type, r.object) for r in plan.relations
                ],
                "pairs": prompt_pairs,
            }
        )

    elapsed = time.time() - t0_all
    mean_l1 = float(np.mean(all_l1_diffs))
    mean_rmse = float(np.mean(all_rmse_diffs))
    mean_perceptual = float(np.mean(all_perceptual_dists))
    mean_ssim = float(np.mean(all_ssims))

    mean_laion_off_val = float(np.mean(all_laion_off))
    mean_laion_on_val = float(np.mean(all_laion_on))
    mean_clip_off_val = float(np.mean(all_clip_off))
    mean_clip_on_val = float(np.mean(all_clip_on))

    summary = {
        "metadata": {
            "title": "Empirical Spatial Guidance A/B Benchmark (Live Pixel Delta)",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_id": "runwayml/stable-diffusion-v1-5",
            "guidance_strength": 0.35,
            "guidance_schedule": "TwoPhaseSchedule (cutoff=0.8)",
            "total_prompts": len(SPATIAL_16_PROMPTS),
            "seeds": FIXED_SEEDS,
            "total_image_pairs": total_runs,
            "total_images_generated": total_runs * 2,
            "elapsed_seconds": round(elapsed, 2),
        },
        "aggregate_comparison": {
            "pixel_divergence": {
                "mean_l1_pixel_difference": round(mean_l1, 4),
                "mean_l1_scale_255": round(mean_l1 * 255.0, 2),
                "mean_rmse": round(mean_rmse, 4),
                "mean_ssim_between_off_and_on": round(mean_ssim, 4),
                "mean_clip_perceptual_distance": round(mean_perceptual, 4),
                "is_pixel_modified": bool(mean_l1 > 0.01 and mean_ssim < 0.99),
            },
            "aesthetic_and_alignment": {
                "laion_aesthetic_off": round(mean_laion_off_val, 3),
                "laion_aesthetic_on": round(mean_laion_on_val, 3),
                "laion_delta": round(mean_laion_on_val - mean_laion_off_val, 3),
                "clip_alignment_off": round(mean_clip_off_val, 4),
                "clip_alignment_on": round(mean_clip_on_val, 4),
                "clip_alignment_delta": round(mean_clip_on_val - mean_clip_off_val, 4),
            },
        },
        "per_prompt_results": results_per_prompt,
    }

    out_json = Path(args.output_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("A/B SPATIAL EVALUATION SUMMARY")
    print("=" * 75)
    print(f"  Total Image Pairs Evaluated:   {total_runs} (Generated in {elapsed:.1f}s)")
    print(f"  Mean Pixel L1 Difference:      {mean_l1:.4f} ({mean_l1 * 255.0:.1f} / 255 levels)")
    print(f"  Mean Structural Similarity:    {mean_ssim:.4f} (SSIM between OFF and ON)")
    print(f"  Mean Perceptual Distance:      {mean_perceptual:.4f} (CLIP visual distance)")
    l_off = mean_laion_off_val
    l_on = mean_laion_on_val
    l_del = l_on - l_off
    c_off = mean_clip_off_val
    c_on = mean_clip_on_val
    c_del = c_on - c_off
    print(
        f"  LAION Aesthetic:               OFF = {l_off:.3f} -> ON = {l_on:.3f} "
        f"(Delta: {l_del:+.3f})"
    )
    print(
        f"  CLIP Text Alignment:           OFF = {c_off:.4f} -> ON = {c_on:.4f} "
        f"(Delta: {c_del:+.4f})"
    )
    print(f"[+] Full results saved to: {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
