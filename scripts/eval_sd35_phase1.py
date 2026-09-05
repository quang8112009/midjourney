from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import snapshot_download

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def get_dir_size_bytes(path: Path) -> int:
    """Calculate the total physical size in bytes for a directory."""
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string (GB and MB)."""
    gb = size_bytes / (1024**3)
    mb = size_bytes / (1024**2)
    return f"{gb:.2f} GB ({mb:.2f} MB / {size_bytes:,} bytes)"


def main():
    print("=" * 80)
    print("PHASE 1: STABLE DIFFUSION 3.5 LARGE BACKBONE BENCHMARK & SMOKE TEST")
    print("=" * 80)

    # 1. Environment & Hardware Inspection
    print("\n--- [Step 1.1] Hardware & CUDA Inspection ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if not cuda_available:
        print("[-] ERROR: CUDA is not available. Aborting.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_vram_bytes / (1024**3)
    print(f"Device Name: {device_name}")
    print(f"Total GPU VRAM: {total_vram_gb:.2f} GB ({total_vram_bytes:,} bytes)")

    # 2. Download / Cache Checkpoint
    model_id = "stabilityai/stable-diffusion-3.5-large"
    print(f"\n--- [Step 1.2] Downloading / Verifying Checkpoint: {model_id} ---")
    print("Downloading FP16 checkpoint files from Hugging Face...")
    download_start = time.perf_counter()

    # Ignore unnecessary fp32 duplicates & standalone monolithic files
    ignore_patterns = [
        "sd3.5_large.safetensors",
        "text_encoders/*",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
        "text_encoder_3/model-00001-of-00002.safetensors",
        "text_encoder_3/model-00002-of-00002.safetensors",
        "text_encoder_3/model.safetensors.index.json",
    ]

    snapshot_path = snapshot_download(
        repo_id=model_id,
        ignore_patterns=ignore_patterns,
        max_workers=4,
    )
    download_elapsed = time.perf_counter() - download_start
    print(f"[+] Download / snapshot sync completed in {download_elapsed:.2f} seconds.")
    print(f"Snapshot directory: {snapshot_path}")

    # Measure disk usage in HF cache
    hf_cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    repo_cache_dir = hf_cache_home / f"models--{model_id.replace('/', '--')}"
    blobs_dir = repo_cache_dir / "blobs"
    total_cache_size = get_dir_size_bytes(blobs_dir) if blobs_dir.exists() else get_dir_size_bytes(repo_cache_dir)
    print(f"Hugging Face Cache Location: {repo_cache_dir}")
    print(f"Actual Disk Usage on Disk: {format_bytes(total_cache_size)}")

    # 3. Load Pipeline at FP16 with CPU Offload
    print("\n--- [Step 1.3] Pipeline Loading & VRAM / Offload Analysis ---")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    print("Loading StableDiffusion3Pipeline from snapshot at fp16...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        snapshot_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    # Calculate model component parameter sizes
    component_sizes = {}
    total_params = 0
    for name, module in [
        ("transformer", pipe.transformer),
        ("text_encoder", pipe.text_encoder),
        ("text_encoder_2", pipe.text_encoder_2),
        ("text_encoder_3", pipe.text_encoder_3),
        ("vae", pipe.vae),
    ]:
        if module is not None:
            num_params = sum(p.numel() for p in module.parameters())
            component_sizes[name] = num_params
            total_params += num_params
            print(f"  - {name}: {num_params:,} parameters ({num_params * 2 / (1024**3):.2f} GB in fp16)")
    
    total_model_vram_fp16 = total_params * 2 / (1024**3)
    print(f"Total Model Parameters: {total_params:,}")
    print(f"Estimated Minimum FP16 Weights Size: {total_model_vram_fp16:.2f} GB")

    offload_required = total_model_vram_fp16 > (total_vram_gb * 0.85)
    print(f"Total Physical VRAM: {total_vram_gb:.2f} GB vs Required Weights: {total_model_vram_fp16:.2f} GB")
    print(f"CPU Offload Required: {offload_required} (Model weights exceed 16GB VRAM limit)")

    if offload_required:
        print("Enabling model CPU offload (enable_model_cpu_offload)...")
        pipe.enable_model_cpu_offload()
    else:
        print("Loading full pipeline directly to CUDA...")
        pipe = pipe.to("cuda")
    
    load_time = time.perf_counter() - load_start

    post_load_allocated = torch.cuda.memory_allocated() / (1024**3)
    post_load_reserved = torch.cuda.memory_reserved() / (1024**3)
    post_load_peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[+] Pipeline load completed in {load_time:.2f}s")
    print(f"  Post-load VRAM allocated: {post_load_allocated:.2f} GB")
    print(f"  Post-load VRAM reserved: {post_load_reserved:.2f} GB")
    print(f"  Post-load Peak VRAM: {post_load_peak:.2f} GB")

    # 4. Generate Single Unguided Image
    prompt = "a peaceful mountain landscape at sunset with a crystal clear lake"
    seed = 42
    num_steps = 28
    guidance_scale = 4.5
    width = 1024
    height = 1024

    print("\n--- [Step 1.4] Generating Single Unguided Image ---")
    print(f"Prompt: \"{prompt}\"")
    print(f"Seed: {seed}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {num_steps} (FlowMatchEuler)")
    print(f"Guidance Scale: {guidance_scale}")

    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    gen_start = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
    gen_elapsed = time.perf_counter() - gen_start

    inference_peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    inference_peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

    print("[+] Image Generation Completed!")
    print(f"  Per-Image Generation Time: {gen_elapsed:.2f} seconds")
    print(f"  Peak VRAM during generation (Allocated): {inference_peak_allocated:.2f} GB")
    print(f"  Peak VRAM during generation (Reserved): {inference_peak_reserved:.2f} GB")

    # Save output images
    output_dirs = [
        ROOT_DIR / "outputs",
        ROOT_DIR / "benchmarks",
        ROOT_DIR / "benchmarks" / "smoke_tests",
    ]
    saved_paths = []
    for out_d in output_dirs:
        out_d.mkdir(parents=True, exist_ok=True)
        img_path = out_d / "phase1_sd35_unguided_s42.png"
        image.save(img_path)
        saved_paths.append(img_path)
        print(f"[+] Saved PNG to: {img_path}")

    # 5. Phase 5 Study Feasibility Assessment
    print("\n--- [Step 1.5] Phase 5 Feasibility Assessment ---")
    total_images_phase5 = 576
    est_total_seconds = gen_elapsed * total_images_phase5
    est_total_hours = est_total_seconds / 3600.0

    print(f"Phase 5 Total Generations: {total_images_phase5}")
    print(f"Single Image Latency: {gen_elapsed:.2f} s")
    print(f"Estimated Phase 5 Runtime: {est_total_hours:.2f} hours ({est_total_seconds:.1f} seconds)")

    flag_threshold = 60.0
    if gen_elapsed > flag_threshold:
        print(f"\n[!] WARNING: Single image generation time ({gen_elapsed:.2f}s) exceeds the {flag_threshold}s threshold!")
        print(f"[!] Full Phase 5 study of 576 images would take approximately {est_total_hours:.2f} hours.")
    else:
        print(f"\n[✓] PASSED: Single image generation time ({gen_elapsed:.2f}s) is within acceptable bounds (<= {flag_threshold}s).")

    print("\n" + "=" * 80)
    print("PHASE 1 EXECUTION COMPLETE - STOPPING FOR REVIEW")
    print("=" * 80)


if __name__ == "__main__":
    main()
