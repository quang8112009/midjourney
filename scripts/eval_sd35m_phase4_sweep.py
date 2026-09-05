from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.editing.layout_guidance import (
    MMDiTJointAttentionHook,
    TwoPhaseSchedule,
    build_layout_guidance_bias,
)
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout
from scripts.eval_spatial_lateral_dedicated import LATERAL_24_SPECS
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator

SWEEP_STRENGTHS = [0.0, 1.5, 3.0, 6.0, 10.0, 15.0]
SWEEP_PROMPTS = LATERAL_24_SPECS[:8]  # Representative subset of 8 lateral prompts across 2 seeds = 16 runs per condition
SWEEP_SEEDS = [42, 100]

class MMDiTJointAttnProcessor:
    def __init__(self, key_name: str, hook: MMDiTJointAttentionHook):
        self.key_name = key_name
        self.hook = hook

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            n_img = query.shape[2]
            n_txt = encoder_hidden_states_query_proj.shape[2]

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

            if self.hook.enabled and self.hook.plan is not None and self.hook.guidance_strength > 0:
                sched_w = self.hook.schedule.weight(self.hook.current_progress)
                if sched_w > 1e-7:
                    bias = build_layout_guidance_bias(
                        self.hook.plan,
                        num_image_tokens=n_img,
                        num_text_tokens=n_txt,
                        guidance_strength=self.hook.guidance_strength * sched_w,
                        device=query.device,
                        dtype=query.dtype,
                    )
                    total_len = n_img + n_txt
                    attn_bias = torch.zeros((batch_size, 1, total_len, total_len), device=query.device, dtype=query.dtype)
                    attn_bias[:, :, :n_img, n_img : n_img + n_txt] = bias.unsqueeze(0).unsqueeze(0)
                    if attention_mask is not None:
                        attention_mask = attention_mask + attn_bias
                    else:
                        attention_mask = attn_bias

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def run_phase4_sweep():
    print("=" * 80, flush=True)
    print("PHASE 4: RANGE-FINDING STRENGTH SWEEP ON SD 3.5 MEDIUM (512x512 / 20 steps)", flush=True)
    print("=" * 80, flush=True)

    out_img_dir = ROOT_DIR / "benchmarks" / "images" / "sd35m_phase4_sweep"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(SWEEP_STRENGTHS) * len(SWEEP_PROMPTS) * len(SWEEP_SEEDS)

    # 1. Image Generation Loop
    print("\n1. Loading SD 3.5 Medium Pipeline (FP16)...", flush=True)
    pipe = StableDiffusion3Pipeline.from_pretrained("D:/midjourney/models/sd35_medium", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    hooks = []
    procs = {}
    keys = list(pipe.transformer.attn_processors.keys())
    for key in keys:
        hook = MMDiTJointAttentionHook(block_idx=0, guidance_strength=0.0, schedule=TwoPhaseSchedule(0.8))
        hooks.append(hook)
        procs[key] = MMDiTJointAttnProcessor(key_name=key, hook=hook)
    pipe.transformer.set_attn_processor(procs)

    def step_cb(pipe, step_idx, timestep, callback_kwargs):
        prog = (step_idx + 1) / 20.0
        for h in hooks:
            h.set_step_context(step=step_idx, total_steps=20, progress=prog)
        return callback_kwargs

    run_idx = 0
    t_gen_start = time.time()

    for strength in SWEEP_STRENGTHS:
        print(f"\n>>> Condition: Guidance Strength = {strength:.2f} <<<", flush=True)
        for h in hooks:
            h.guidance_strength = strength

        for spec in SWEEP_PROMPTS:
            pid = spec["id"]
            prompt = spec["prompt"]

            intent = analyze_prompt(prompt, mode="generate")
            plan = plan_semantic_layout(intent, tokenizer=pipe.tokenizer_3)
            for h in hooks:
                h.set_plan(plan)

            for seed in SWEEP_SEEDS:
                run_idx += 1
                img_name = f"{pid}_str_{strength:.2f}_s{seed}.png"
                img_path = out_img_dir / img_name

                if img_path.exists():
                    print(f"  [{run_idx:02d}/{total_runs}] {pid} (s={seed}) str={strength:.1f} | [CACHED]", flush=True)
                    continue

                gen = torch.Generator(device="cpu").manual_seed(seed)
                t0 = time.time()
                with torch.inference_mode():
                    img = pipe(
                        prompt=prompt,
                        num_inference_steps=20,
                        guidance_scale=4.5,
                        width=512,
                        height=512,
                        generator=gen,
                        callback_on_step_end=step_cb,
                    ).images[0]
                dt = time.time() - t0
                img.save(img_path)
                print(f"  [{run_idx:02d}/{total_runs}] {pid} (s={seed}) str={strength:.1f} | Generated in {dt:.1f}s", flush=True)

    print(f"\n[+] All images generated in {time.time() - t_gen_start:.1f}s! Freeing pipeline from GPU...", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Evaluation Loop
    print("\n2. Initializing Strict Spatial Evaluator...", flush=True)
    evaluator = StrictSpatialEvaluator(device="cuda")

    results_by_strength: dict[str, list[dict]] = {str(s): [] for s in SWEEP_STRENGTHS}

    for strength in SWEEP_STRENGTHS:
        print(f"\nEvaluating Condition str={strength:.2f} ...", flush=True)
        for spec in SWEEP_PROMPTS:
            pid = spec["id"]
            prompt = spec["prompt"]
            subj = spec["subject"]
            obj = spec["object"]
            rel = spec["relation"]

            for seed in SWEEP_SEEDS:
                img_name = f"{pid}_str_{strength:.2f}_s{seed}.png"
                img_path = out_img_dir / img_name
                img = Image.open(img_path).convert("RGB")

                s_det, o_det = evaluator.detect_entities(img, subj, obj)
                is_sat, details = evaluator.check_relation(s_det, o_det, rel)
                clip_sim, aes_score = evaluator.evaluate_quality(img, prompt)

                record = {
                    "prompt_id": pid,
                    "prompt": prompt,
                    "relation": rel,
                    "seed": seed,
                    "strength": strength,
                    "satisfied": is_sat,
                    "details": details,
                    "subject_detected": s_det is not None,
                    "object_detected": o_det is not None,
                    "subject_score": s_det["score"] if s_det else 0.0,
                    "object_score": o_det["score"] if o_det else 0.0,
                    "aesthetic_score": round(aes_score, 3),
                    "clip_score": round(clip_sim, 3),
                    "image_path": str(img_path),
                }
                results_by_strength[str(strength)].append(record)

    summary_table = []
    for strength in SWEEP_STRENGTHS:
        recs = results_by_strength[str(strength)]
        n = len(recs)
        n_sat = sum(1 for r in recs if r["satisfied"])
        sat_rate = n_sat / n if n > 0 else 0.0
        mean_aes = float(np.mean([r["aesthetic_score"] for r in recs]))
        mean_clip = float(np.mean([r["clip_score"] for r in recs]))

        summary_table.append({
            "strength": strength,
            "total_runs": n,
            "satisfied_count": n_sat,
            "satisfaction_rate": round(sat_rate, 4),
            "mean_aesthetic": round(mean_aes, 3),
            "mean_clip_similarity": round(mean_clip, 3),
        })

    print("\n" + "=" * 80)
    print("PHASE 4 SWEEP SUMMARY TABLE (SD 3.5 Medium @ 512x512 / 20 steps):")
    print("=" * 80)
    print(f"{'Strength':<10} | {'Satisfaction':<14} | {'Mean AES':<10} | {'Mean CLIP':<10}")
    print("-" * 55)
    for row in summary_table:
        print(f"{row['strength']:<10.2f} | {row['satisfaction_rate']*100:>5.1f}% ({row['satisfied_count']}/{row['total_runs']}) | {row['mean_aesthetic']:<10.3f} | {row['mean_clip_similarity']:<10.3f}")

    out_json = ROOT_DIR / "benchmarks" / "phase4_sd35m_sweep_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": "stabilityai/stable-diffusion-3.5-medium",
            "resolution": "512x512",
            "steps": 20,
            "summary": summary_table,
            "details": results_by_strength,
        }, f, indent=2)
    print(f"\n[+] Saved Phase 4 sweep results to: {out_json}")


if __name__ == "__main__":
    run_phase4_sweep()
