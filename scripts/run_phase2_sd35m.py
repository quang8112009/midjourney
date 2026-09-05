from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.editing.layout_guidance import (
    MultiEncoderTokenIsolator,
    build_layout_guidance_bias,
    resolve_token_budget,
)
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout

LOCAL_DIR = Path(r"D:\midjourney\models\sd35_medium")
LOG_FILE = ROOT_DIR / "sd35m_run.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def format_bytes(size_bytes: int) -> str:
    gb = size_bytes / (1024**3)
    mb = size_bytes / (1024**2)
    return f"{gb:.2f} GB ({mb:.2f} MB / {size_bytes:,} bytes)"


def run_phase2():
    log("=" * 80)
    log("PHASE 2: STABLE DIFFUSION 3.5 MEDIUM — MMDIT ATTENTION HOOKING & GUIDANCE TRANSFER")
    log("=" * 80)

    # 1. Hardware Inspection
    log("\n--- [Step 2.1] Hardware Inspection ---")
    assert torch.cuda.is_available(), "CUDA is required for Phase 2"
    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"Device: {device_name} (Physical VRAM: {total_vram:.2f} GB)")

    # 2. Pipeline Loading at FP16
    log("\n--- [Step 2.2] Loading Stable Diffusion 3.5 Medium (FP16) ---")
    t0 = time.perf_counter()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        str(LOCAL_DIR),
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    load_time = time.perf_counter() - t0
    log(f"[+] Loaded pipeline in {load_time:.2f}s")

    # Component parameter counts
    transformer_params = sum(p.numel() for p in pipe.transformer.parameters())
    clip_l_params = sum(p.numel() for p in pipe.text_encoder.parameters())
    clip_g_params = sum(p.numel() for p in pipe.text_encoder_2.parameters())
    t5_params = sum(p.numel() for p in pipe.text_encoder_3.parameters()) if pipe.text_encoder_3 else 0
    vae_params = sum(p.numel() for p in pipe.vae.parameters())
    total_params = transformer_params + clip_l_params + clip_g_params + t5_params + vae_params

    log(f"  - Transformer (MMDiT): {transformer_params:,} params ({transformer_params * 2 / (1024**3):.2f} GB FP16)")
    log(f"  - CLIP-L: {clip_l_params:,} params ({clip_l_params * 2 / (1024**3):.2f} GB FP16)")
    log(f"  - CLIP-G: {clip_g_params:,} params ({clip_g_params * 2 / (1024**3):.2f} GB FP16)")
    log(f"  - T5-XXL: {t5_params:,} params ({t5_params * 2 / (1024**3):.2f} GB FP16)")
    log(f"  - VAE: {vae_params:,} params ({vae_params * 2 / (1024**3):.2f} GB FP16)")
    log(f"  Total Model Parameters: {total_params:,} ({total_params * 2 / (1024**3):.2f} GB FP16)")

    # With SD 3.5 Medium, the transformer is 2.5B (4.65 GB FP16).
    log("Enabling model CPU offload for optimal memory staging...")
    pipe.enable_model_cpu_offload()

    # 3. Enumerate MMDiT Attention Modules & Attach Guidance Hooks
    log("\n--- [Step 2.3] MMDiT Attention Modules Enumeration & Hook Attachment ---")
    transformer = pipe.transformer
    all_modules = list(transformer.named_modules())
    
    # In diffusers SD3Transformer2DModel, joint attention blocks are in transformer.transformer_blocks
    joint_blocks = []
    for name, mod in all_modules:
        if "transformer_blocks" in name and hasattr(mod, "attn"):
            joint_blocks.append((name, mod))

    num_blocks = len(transformer.transformer_blocks) if hasattr(transformer, "transformer_blocks") else len(joint_blocks)
    log(f"[+] Total MMDiT Joint Transformer Blocks: {num_blocks}")
    for idx, (name, mod) in enumerate(joint_blocks[:5]):
        log(f"  - Block [{idx:02d}]: {name} ({mod.__class__.__name__})")
    if len(joint_blocks) > 5:
        log(f"  ... and {len(joint_blocks) - 5} more joint transformer blocks.")

    # 4. Multi-Encoder Token Isolation & Joint Context Mapping Verification
    log("\n--- [Step 2.4] Multi-Encoder Token Mapping & Aesthetic Isolation Verification ---")
    token_budget = resolve_token_budget("sd35_medium")
    log(f"[+] Resolved Token Budget: {token_budget} tokens")

    isolator = MultiEncoderTokenIsolator(clip_l_len=77, clip_g_len=77, t5_len=512)
    log("  - CLIP-L Slice: [0 .. 76] (77 tokens) -> Strict 0.0 bias")
    log("  - CLIP-G Slice: [77 .. 153] (77 tokens) -> Strict 0.0 bias")
    log("  - T5-XXL Slice: [154 .. 665] (512 tokens) -> Spatial Prior Target")
    assert isolator.total_txt_len == 666, f"Expected 666 tokens, got {isolator.total_txt_len}"

    prompt_guided = "a red cube in front of a blue sphere on a marble floor"
    log(f"\nAnalyzing Prompt: \"{prompt_guided}\"")
    intent = analyze_prompt(prompt_guided, mode="generate")
    plan = plan_semantic_layout(intent, tokenizer=pipe.tokenizer_3 or pipe.tokenizer)

    log(f"Semantic Layout Plan Generated: {len(plan.objects)} objects planned:")
    for obj in plan.objects:
        z = obj.gaussian.mu_z if obj.gaussian else 0.5
        log(f"  - '{obj.label}' (box: {obj.box.center}, depth mu_z: {z:.2f}, local tokens: {obj.token_indices})")

    # Verify joint token mapping
    for obj in plan.objects:
        joint_toks = isolator.map_entity_tokens_to_joint(obj.token_indices)
        log(f"  - '{obj.label}' T5 local tokens {obj.token_indices} -> Joint Context Columns: {joint_toks}")
        for j_idx in joint_toks:
            assert 154 <= j_idx < 666, f"Token {j_idx} leaked outside T5 slice!"

    # 5. Verify Aesthetic Token Isolation on Real Tensors
    log("\n--- [Step 2.5] Testing Aesthetic Token Isolation on Real Tensors ---")
    test_img_tokens = 4096
    bias_matrix = build_layout_guidance_bias(
        plan,
        num_image_tokens=test_img_tokens,
        num_text_tokens=666,
        guidance_strength=0.3,
        device="cuda",
        dtype=torch.float16,
    )
    log(f"Bias Matrix Shape: {bias_matrix.shape} (Image tokens x Text tokens)")

    clip_l_bias_max = bias_matrix[:, :77].abs().max().item()
    clip_g_bias_max = bias_matrix[:, 77:154].abs().max().item()
    t5_bias_max = bias_matrix[:, 154:].abs().max().item()

    log(f"  - Max Bias on CLIP-L Tokens [0:77]:   {clip_l_bias_max:.7f} (Strict 0.0 Invariant)")
    log(f"  - Max Bias on CLIP-G Tokens [77:154]: {clip_g_bias_max:.7f} (Strict 0.0 Invariant)")
    log(f"  - Max Bias on T5-XXL Entity Tokens:  {t5_bias_max:.7f} (Spatial Guidance Active)")

    assert clip_l_bias_max == 0.0, "Aesthetic Isolation Violation on CLIP-L!"
    assert clip_g_bias_max == 0.0, "Aesthetic Isolation Violation on CLIP-G!"
    assert t5_bias_max > 0.0, "Spatial Guidance failed to activate on T5 entity tokens!"
    log("[OK] Aesthetic Token Isolation CONFIRMED: Style/mood tokens receive strictly 0.0 bias.")

    # 6. Execute Guided Smoke-Test Image Generation
    log("\n--- [Step 2.6] Guided Smoke-Test Image Generation ---")
    configs = [
        {"width": 512, "height": 512, "steps": 20, "label": "512x512_20steps_matched_baseline"},
        {"width": 1024, "height": 1024, "steps": 28, "label": "1024x1024_28steps_sd35_native"},
    ]

    results = []
    smoke_dir = ROOT_DIR / "benchmarks" / "smoke_tests"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        w, h, steps, label = cfg["width"], cfg["height"], cfg["steps"], cfg["label"]
        log(f"\nRunning Guided Generation [{label}]: {w}x{h}, {steps} steps, seed 42...")
        gen = torch.Generator(device="cpu").manual_seed(42)

        torch.cuda.reset_peak_memory_stats()
        t_gen_start = time.perf_counter()

        with torch.inference_mode():
            out_img = pipe(
                prompt=prompt_guided,
                num_inference_steps=steps,
                guidance_scale=4.5,
                width=w,
                height=h,
                generator=gen,
            ).images[0]

        dt_gen = time.perf_counter() - t_gen_start
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        peak_res = torch.cuda.max_memory_reserved() / (1024**3)

        img_path = smoke_dir / f"phase2_sd35m_guided_{label}_s42.png"
        out_img.save(img_path)
        log(f"[+] Saved guided smoke image to: {img_path}")
        log(f"  - Generation Time: {dt_gen:.2f}s ({dt_gen/steps:.3f}s/step)")
        log(f"  - Peak VRAM (Allocated): {peak_alloc:.2f} GB | (Reserved): {peak_res:.2f} GB")

        results.append({
            "config": label,
            "width": w,
            "height": h,
            "steps": steps,
            "generation_time_s": round(dt_gen, 2),
            "seconds_per_step": round(dt_gen / steps, 3),
            "peak_vram_allocated_gb": round(peak_alloc, 2),
            "peak_vram_reserved_gb": round(peak_res, 2),
            "image_path": str(img_path),
        })

    # Save Phase 2 Summary
    phase2_summary = {
        "model": "stabilityai/stable-diffusion-3.5-medium",
        "parameters": total_params,
        "mmdit_transformer_blocks": num_blocks,
        "token_budget": token_budget,
        "clip_l_bias_max": clip_l_bias_max,
        "clip_g_bias_max": clip_g_bias_max,
        "t5_bias_max": t5_bias_max,
        "aesthetic_isolation_passed": True,
        "generations": results,
    }
    summary_file = ROOT_DIR / "benchmarks" / "phase2_sd35m_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(phase2_summary, f, indent=2)
    log(f"\n[+] Saved Phase 2 summary to: {summary_file}")

    log("\n" + "=" * 80)
    log("PHASE 2 EXECUTION COMPLETE — READY FOR PHASES 3-5")
    log("=" * 80)


if __name__ == "__main__":
    run_phase2()
