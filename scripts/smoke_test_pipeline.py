from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.editing.layout_guidance import (  # noqa: E402
    LayoutGuidanceProcessor,
    TwoPhaseSchedule,
)
from app.services.editing.prompt_intent import analyze_prompt  # noqa: E402
from app.services.editing.semantic_planner import plan_semantic_layout  # noqa: E402


def run_smoke_test():
    smoke_dir = ROOT_DIR / "benchmarks" / "smoke_tests"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: CUDA Environment Check")
    print("=" * 70)
    cuda_ok = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {cuda_ok}")
    if not cuda_ok:
        print("[-] ERROR: CUDA is not available. Halting smoke test.")
        return False
    device_name = torch.cuda.get_device_name(0)
    print(f"  Device: {device_name}")
    device = torch.device("cuda")

    print("\n" + "=" * 70)
    print("STEP 2: Live SD v1.5 Generation (Guidance DISABLED)")
    print("=" * 70)
    model_path = ROOT_DIR / "models" / "sd15_fp16"
    print(f"Loading local pipeline from {model_path} on {device} (fp16)...")
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print(f"Pipeline loaded in {time.time() - t0:.2f}s")

    prompt_base = "a peaceful mountain landscape at sunset with a crystal clear lake"
    print(f"Generating image (20 steps, seed 42): '{prompt_base}'...")
    gen = torch.Generator("cuda").manual_seed(42)
    t0 = time.time()
    with torch.inference_mode():
        img_base = pipe(
            prompt=prompt_base,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=gen,
            width=512,
            height=512,
        ).images[0]
    dt_base = time.time() - t0
    base_out_path = smoke_dir / "smoke_base.png"
    img_base.save(base_out_path)
    print(f"[+] Unguided image generated in {dt_base:.2f}s and saved to {base_out_path}")

    print("\n" + "=" * 70)
    print("STEP 3: Live SD v1.5 Generation (Guidance ENABLED)")
    print("=" * 70)
    prompt_guided = "a red cube in front of a blue sphere on a marble floor"
    print(f"Prompt: '{prompt_guided}'")

    print("Planning semantic layout...")
    intent = analyze_prompt(prompt_guided, mode="generate")
    plan = plan_semantic_layout(intent, tokenizer=pipe.tokenizer)
    print(f"Plan generated: {len(plan.objects)} objects planned:")
    for obj in plan.objects:
        z = obj.gaussian.mu_z if obj.gaussian else 0.5
        msg = (
            f"  - '{obj.label}' (count={obj.count}) at box center {obj.box.center}, "
            f"depth mu_z={z:.2f}, tokens={obj.token_indices}"
        )
        print(msg)

    print("Attaching LayoutGuidanceProcessor hooks to UNet cross-attention modules...")
    attn_procs = {}
    matched_count = 0
    total_procs = len(pipe.unet.attn_processors)
    schedule = TwoPhaseSchedule(schedule_cutoff=0.8)

    for name, proc in pipe.unet.attn_processors.items():
        if name.endswith("attn2.processor"):  # Cross-attention processor in UNet
            attn_procs[name] = LayoutGuidanceProcessor(
                base_processor=proc,
                plan=plan,
                guidance_strength=0.3,
                schedule=schedule,
                depth_guidance_enabled=True,
            )
            matched_count += 1
        else:
            attn_procs[name] = proc

    pipe.unet.set_attn_processor(attn_procs)
    hook_msg = (
        f"[+] Successfully hooked {matched_count} cross-attention modules "
        f"out of {total_procs} total UNet processors."
    )
    print(hook_msg)

    # Step callback to update progress in LayoutGuidanceProcessors
    def step_callback(pipe, step_idx, timestep, callback_kwargs):
        progress = float(step_idx + 1) / 20.0
        for p in pipe.unet.attn_processors.values():
            if isinstance(p, LayoutGuidanceProcessor):
                p.set_step_progress(progress)
        return callback_kwargs

    print("Running guided diffusion inference (20 steps, seed 42)...")
    gen_guided = torch.Generator("cuda").manual_seed(42)
    t0 = time.time()
    try:
        with torch.inference_mode():
            img_guided = pipe(
                prompt=prompt_guided,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=gen_guided,
                width=512,
                height=512,
                callback_on_step_end=step_callback,
            ).images[0]
        dt_guided = time.time() - t0
        guided_out_path = smoke_dir / "smoke_guided.png"
        img_guided.save(guided_out_path)
        print(f"[+] Guided image generated successfully in {dt_guided:.2f}s with 0 shape errors!")
        print(f"[+] Saved to: {guided_out_path}")
        return True
    except Exception as exc:
        print(f"[-] ERROR during guided inference: {exc}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
