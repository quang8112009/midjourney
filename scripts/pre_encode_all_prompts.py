from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from diffusers import StableDiffusion3Pipeline

from scripts.eval_spatial_lateral_dedicated import LATERAL_24_SPECS
from scripts.hard_lateral_specs import LATERAL_HARD_24_SPECS


def main():
    print("=" * 80)
    print("PRE-ENCODING ALL BENCHMARK PROMPTS FOR HIGH-SPEED PHASE 5 STUDY")
    print("=" * 80)

    out_file = ROOT_DIR / "benchmarks" / "sd35m_prompt_embeddings.pt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_specs = LATERAL_24_SPECS + LATERAL_HARD_24_SPECS
    unique_prompts = list(dict.fromkeys(s["prompt"] for s in all_specs))
    print(f"Total Unique Prompts to Encode: {len(unique_prompts)}")

    print("\nLoading Text Encoding Pipeline on CUDA...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "models/sd35_medium",
        transformer=None,
        vae=None,
        torch_dtype=torch.float16,
    ).to("cuda")

    prompt_bank = {}
    t_start = time.time()

    for idx, p in enumerate(unique_prompts, 1):
        t0 = time.time()
        with torch.inference_mode():
            p_emb, neg_p_emb, pool_emb, neg_pool_emb = pipe.encode_prompt(
                prompt=p,
                prompt_2=p,
                prompt_3=p,
                device="cuda",
            )
        prompt_bank[p] = {
            "prompt_embeds": p_emb.cpu(),
            "negative_prompt_embeds": neg_p_emb.cpu(),
            "pooled_prompt_embeds": pool_emb.cpu(),
            "negative_pooled_prompt_embeds": neg_pool_emb.cpu(),
        }
        dt = time.time() - t0
        print(f"  [{idx:02d}/{len(unique_prompts)}] Encoded '{p[:40]}...' in {dt:.2f}s", flush=True)

    torch.save(prompt_bank, out_file)
    print(f"\n[+] Successfully saved {len(prompt_bank)} prompt embeddings to: {out_file}")
    print(f"[+] Total time: {time.time() - t_start:.2f}s ({((time.time() - t_start)/len(unique_prompts)):.2f}s per prompt)")


if __name__ == "__main__":
    main()
