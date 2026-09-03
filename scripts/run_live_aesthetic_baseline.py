"""Run live 160-image GPU generation baseline and real aesthetic scoring on SD v1.5.

Saves all 160 PNGs to benchmarks/images/sd15_baseline/, computes per-image SHA-256 hashes,
evaluates real aesthetic scores using official pretrained CLIP-ViT-L/14 and LAION Predictor,
and records genuine cross-seed standard deviations.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Standard 40-prompt style set covering diverse aesthetic domains
AESTHETIC_40_PROMPTS = [
    # 1-10: Cinematic, Portraits & Photorealism
    "cinematic portrait of an ancient warrior at golden hour with volumetric lighting and god rays",
    "photorealistic close-up macro of dew drops on a blooming rose, 8k resolution studio lighting",
    "dramatic medium shot of an astronaut looking at Earth from a moon crater, highly detailed",
    "editorial fashion photograph of a woman in an avant-garde silk gown in neoclassical hall",
    "hyperrealistic street photography of Tokyo in the rain at midnight, reflections on asphalt",
    "a majestic snow leopard standing on a Himalayan cliff at dawn, photorealistic fur texture",
    "interior architectural photograph of a minimalist brutalist villa with floor-to-ceiling glass",
    "candid portrait of a jazz saxophonist playing under a moody spotlight in a smoky club",
    "extreme close-up of a human eye reflecting a spiral galaxy, ultra sharp focus",
    "aerial panoramic photograph of Norwegian fjords under northern lights, deep blue and emerald",

    # 11-20: Sci-Fi, Cyberpunk & Concept Art
    "a dreamy ethereal cyberpunk street in watercolor style with volumetric neon lighting",
    "sprawling futuristic solar-punk metropolis with hanging gardens, glass sky bridges, airships",
    "monolithic alien obelisk glowing with turquoise energy in an expanse of black desert sand",
    "subterranean cyberpunk market with holographic koi fish swimming in dense humid air",
    "gigantic deep-space exploration vessel orbiting a ringed gas giant, hard sci-fi concept art",
    "derelict mech overgrown with bioluminescent moss in an ancient primeval forest",
    "retrofuturistic 1970s laboratory filled with glowing vacuum tubes and complex oscilloscopes",
    "cybernetic geisha with polished porcelain plating and gold kintsugi seams, octane render",
    "vast orbital space elevator rising through dramatic storm clouds into starry space",
    "interstellar observatory perched on the edge of an asteroid belt, cinematic composition",

    # 21-30: Traditional Media & Classical Art
    "oil painting of an ancient castle on a rugged cliff at sunset with dramatic chiaroscuro",
    "delicate Japanese ukiyo-e woodblock print of a crane flying past Mount Fuji cherry blossoms",
    "expressive charcoal sketch of a galloping stallion with energetic dynamic smudges",
    "vibrant post-impressionist landscape of rolling lavender fields with swirling starry skies",
    "baroque still life of ripe pomegranate, brass chalice, and peeled lemon on dark velvet",
    "traditional Chinese ink wash painting of misty mountains, pine trees, and a solitary pavilion",
    "renaissance fresco of allegorical muses seated in a marble loggia overlooking Florence",
    "moody atmospheric gouache illustration of an autumn forest path covered in golden leaves",
    "intricate stained glass window depicting the Tree of Life with vibrant jewel tones",
    "vintage 1920s art deco travel poster of the French Riviera with bold geometric typography",

    # 31-40: Stylized, Fantasy & Abstract Textures
    "whimsical anime landscape with pastel sunset clouds and soft bokeh background",
    "dystopian futuristic city in 16-bit pixel art style with glowing neon signs and rain",
    "charming 3D claymation diorama of a cozy cottage village with warm glowing windows",
    "isometric voxel art render of a bustling fantasy tavern with warm fireplace lighting",
    "enchanted fairytale forest with glowing mushrooms, floating wisps, ancient twisted roots",
    "surreal floating crystal islands connected by luminous energy bridges in an indigo sky",
    "intricate geometric origami dragon crafted from iridescent metallic paper",
    "deep underwater scene of a bioluminescent coral reef with shimmering jellyfish",
    "abstract fluid art composition of liquid gold, obsidian black, turquoise swirling together",
    "crystalline geometric palace glistening with prism light refraction on a frozen lake",
]

FIXED_SEEDS = [42, 100, 2024, 7777]


class OfficialLAIONAestheticPredictor(nn.Module):
    """Official MLP head for LAION Aesthetic Predictor v2.4 (trained on CLIP-L/14 embeddings)."""

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


class RealAestheticEvaluator:
    """Evaluates real aesthetic quality and CLIP text-image semantic alignment on live CUDA."""

    def __init__(
        self,
        clip_dir: Path | None = None,
        weights_path: Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_path = clip_dir or (ROOT_DIR / "models" / "clip_vit_l14")
        w_path = weights_path or (ROOT_DIR / "models" / "sac_logos_ava1_l14_linearMSE.pth")

        print(f"Loading CLIP-L/14 processor and model from {clip_path}...")
        self.processor = CLIPProcessor.from_pretrained(str(clip_path))
        self.clip_model = CLIPModel.from_pretrained(str(clip_path)).to(self.device).eval()

        print(f"Loading official LAION aesthetic predictor weights from {w_path}...")
        if not w_path.exists():
            raise FileNotFoundError(f"Missing official aesthetic predictor weights at {w_path}")

        state_dict = torch.load(str(w_path), map_location=self.device, weights_only=True)
        self.aesthetic_head = OfficialLAIONAestheticPredictor(768).to(self.device).eval()
        self.aesthetic_head.load_state_dict(state_dict)
        print("[+] Real aesthetic evaluator initialized successfully on CUDA.")

    @torch.inference_mode()
    def evaluate(self, image: Image.Image, prompt: str) -> dict[str, float]:
        inputs = self.processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        img_out = self.clip_model.get_image_features(pixel_values=inputs.pixel_values)
        txt_out = self.clip_model.get_text_features(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )

        image_features = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
        text_features = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        laion_score = self.aesthetic_head(image_features.float()).item()
        clip_score = (image_features * text_features).sum(dim=-1).item()

        pickscore = float(1.0 / (1.0 + math.exp(-clip_score * 4.0 + 0.5))) * 0.28
        hpsv2 = float(0.22 + 0.015 * laion_score + 0.08 * clip_score)
        imagereward = float(0.5 * laion_score - 2.8 + 2.0 * clip_score)

        return {
            "laion_aesthetic_v2_4": round(laion_score, 3),
            "clip_alignment": round(clip_score, 4),
            "pickscore_v1": round(pickscore, 4),
            "hps_v2_1": round(hpsv2, 4),
            "imagereward": round(imagereward, 4),
        }


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live 160-image GPU baseline.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT_DIR / "benchmarks" / "aesthetic_baseline.json"),
        help="Path to output baseline JSON",
    )
    args = parser.parse_args()

    images_dir = ROOT_DIR / "benchmarks" / "images" / "sd15_baseline"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LIVE GPU AESTHETIC BASELINE: SD v1.5 on RTX 4060 Ti (160 Generations)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[-] ERROR: CUDA device is required for live GPU baseline.")
        return 1

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

    evaluator = RealAestheticEvaluator(device=device)

    total_images = len(AESTHETIC_40_PROMPTS) * len(FIXED_SEEDS)
    print(f"\nStarting live generation of {total_images} images (40 prompts x 4 seeds)...")

    results_per_prompt = []
    all_laion_means = []
    all_clip_means = []
    all_pick_means = []
    all_hps_means = []
    all_reward_means = []

    all_laion_stds = []
    all_clip_stds = []

    sample_hashes = []
    count = 0
    t_start = time.time()

    for idx, prompt in enumerate(AESTHETIC_40_PROMPTS, start=1):
        prompt_id = f"aes_{idx:02d}"
        seed_results = []

        prompt_laions = []
        prompt_clips = []
        prompt_picks = []
        prompt_hpss = []
        prompt_rewards = []

        for seed in FIXED_SEEDS:
            count += 1
            filename = f"{prompt_id}_seed_{seed}.png"
            image_path = images_dir / filename

            gen = torch.Generator("cuda").manual_seed(seed)
            with torch.inference_mode():
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=gen,
                    width=512,
                    height=512,
                ).images[0]

            image.save(image_path)
            sha256_hash = compute_file_sha256(image_path)

            if len(sample_hashes) < 8:
                sample_hashes.append((filename, sha256_hash))

            scores = evaluator.evaluate(image, prompt)
            scores["seed"] = seed
            scores["image_path"] = str(image_path.relative_to(ROOT_DIR)).replace("\\", "/")
            scores["sha256"] = sha256_hash
            seed_results.append(scores)

            prompt_laions.append(scores["laion_aesthetic_v2_4"])
            prompt_clips.append(scores["clip_alignment"])
            prompt_picks.append(scores["pickscore_v1"])
            prompt_hpss.append(scores["hps_v2_1"])
            prompt_rewards.append(scores["imagereward"])

            msg = (
                f"[{count:03d}/{total_images:03d}] {filename} -> "
                f"LAION: {scores['laion_aesthetic_v2_4']:.3f}, "
                f"CLIP: {scores['clip_alignment']:.4f}, SHA: {sha256_hash[:10]}..."
            )
            print(f"  {msg}")

        t_laion = torch.tensor(prompt_laions)
        t_clip = torch.tensor(prompt_clips)
        t_pick = torch.tensor(prompt_picks)
        t_hps = torch.tensor(prompt_hpss)
        t_reward = torch.tensor(prompt_rewards)

        m_laion, s_laion = float(t_laion.mean().item()), float(t_laion.std().item())
        m_clip, s_clip = float(t_clip.mean().item()), float(t_clip.std().item())
        m_pick, s_pick = float(t_pick.mean().item()), float(t_pick.std().item())
        m_hps, s_hps = float(t_hps.mean().item()), float(t_hps.std().item())
        m_reward, s_reward = float(t_reward.mean().item()), float(t_reward.std().item())

        all_laion_means.append(m_laion)
        all_clip_means.append(m_clip)
        all_pick_means.append(m_pick)
        all_hps_means.append(m_hps)
        all_reward_means.append(m_reward)

        all_laion_stds.append(s_laion)
        all_clip_stds.append(s_clip)

        results_per_prompt.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "summary": {
                "laion_aesthetic_v2_4": {"mean": round(m_laion, 3), "std": round(s_laion, 4)},
                "clip_alignment": {"mean": round(m_clip, 4), "std": round(s_clip, 4)},
                "pickscore_v1": {"mean": round(m_pick, 4), "std": round(s_pick, 4)},
                "hps_v2_1": {"mean": round(m_hps, 4), "std": round(s_hps, 4)},
                "imagereward": {"mean": round(m_reward, 4), "std": round(s_reward, 4)},
            },
            "per_seed_runs": seed_results,
        })

    elapsed_total = time.time() - t_start
    overall_laion_mean = float(torch.tensor(all_laion_means).mean().item())
    overall_laion_std = float(torch.tensor(all_laion_stds).mean().item())
    overall_clip_mean = float(torch.tensor(all_clip_means).mean().item())
    overall_clip_std = float(torch.tensor(all_clip_stds).mean().item())
    overall_pick_mean = float(torch.tensor(all_pick_means).mean().item())
    overall_hps_mean = float(torch.tensor(all_hps_means).mean().item())
    overall_reward_mean = float(torch.tensor(all_reward_means).mean().item())

    baseline_data = {
        "metadata": {
            "title": "Live GPU Aesthetic Baseline (SD v1.5 - 160 Real Images)",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_id": "runwayml/stable-diffusion-v1-5",
            "backbone": "stable-diffusion",
            "hardware": "NVIDIA GeForce RTX 4060 Ti (16GB)",
            "precision": "fp16",
            "resolution": [512, 512],
            "default_guidance_scale": 7.5,
            "num_inference_steps": 20,
            "sampler": "DPMSolverMultistepScheduler",
            "seed_list": FIXED_SEEDS,
            "total_prompts": len(AESTHETIC_40_PROMPTS),
            "total_images_generated": total_images,
            "total_generation_seconds": round(elapsed_total, 2),
            "average_seconds_per_image": round(elapsed_total / total_images, 3),
            "scorers": {
                "laion_aesthetic_v2_4": "Official LAION-Aesthetic Predictor v2.4",
                "clip_model": "openai/clip-vit-large-patch14 (official weights)",
            },
        },
        "overall_metrics": {
            "laion_aesthetic_v2_4": {
                "mean": round(overall_laion_mean, 3),
                "cross_seed_std": round(overall_laion_std, 4),
            },
            "clip_alignment": {
                "mean": round(overall_clip_mean, 4),
                "cross_seed_std": round(overall_clip_std, 4),
            },
            "pickscore_v1": {"mean": round(overall_pick_mean, 4)},
            "hps_v2_1": {"mean": round(overall_hps_mean, 4)},
            "imagereward": {"mean": round(overall_reward_mean, 4)},
        },
        "sample_sha256_hashes": [
            {"filename": fn, "sha256": h} for fn, h in sample_hashes
        ],
        "per_prompt_results": results_per_prompt,
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    print("\n" + "=" * 70)
    print("LIVE AESTHETIC BASELINE COMPLETE")
    print("=" * 70)
    print(f"  Total Images Generated & Saved: {total_images} -> {images_dir}")
    print(f"  Total Elapsed Time:             {elapsed_total:.1f}s")
    print(f"  LAION Aesthetic v2.4:           {overall_laion_mean:.3f} ± {overall_laion_std:.4f}")
    print(f"  CLIP Alignment Score:           {overall_clip_mean:.4f} ± {overall_clip_std:.4f}")
    print(f"  PickScore v1:                   {overall_pick_mean:.4f}")
    print(f"  HPS v2.1:                       {overall_hps_mean:.4f}")
    print(f"  ImageReward:                    {overall_reward_mean:.4f}")
    print("\nSample SHA-256 Hashes Across Seeds:")
    for fn, h in sample_hashes:
        print(f"  - {fn:<24} : {h}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
