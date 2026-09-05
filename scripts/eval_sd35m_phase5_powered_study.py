from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.editing.layout_guidance import (
    MMDiTJointAttentionHook,
    TwoPhaseSchedule,
    build_layout_guidance_bias,
)
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout
from scripts.eval_spatial_lateral_dedicated import (
    LATERAL_24_SPECS,
    SEEDS_192,
    exact_mcnemar_p_value,
)
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator, wilson_score_interval
from scripts.hard_lateral_specs import LATERAL_HARD_24_SPECS

LOG_FILE = ROOT_DIR / "phase5_study.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


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


def run_benchmark_eval(
    benchmark_name: str,
    specs: list[dict],
    prompt_bank: dict[str, dict[str, torch.Tensor]],
    tokenizer_t5,
    strengths: list[float] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    if strengths is None:
        strengths = [0.0, 3.0, 6.0]
    if seeds is None:
        seeds = SEEDS_192
    log("\n" + "=" * 80)
    log(f"RUNNING POWERED STUDY ON [{benchmark_name.upper()}] (N={len(specs)*len(seeds)} runs/condition)")
    log("=" * 80)

    out_img_dir = ROOT_DIR / "benchmarks" / "images" / f"sd35m_phase5_{benchmark_name}"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pipeline Generation (Transformer & VAE on CUDA)
    log("\n1. Loading SD 3.5 Medium Generation Pipeline (Transformer & VAE on CUDA)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "models/sd35_medium",
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
        torch_dtype=torch.float16,
    ).to("cuda")

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

    total_images = len(strengths) * len(specs) * len(seeds)
    run_idx = 0
    t_gen_start = time.time()

    for str_val in strengths:
        log(f"\n>>> Condition: Guidance Strength = {str_val:.2f} <<<")
        for h in hooks:
            h.guidance_strength = str_val

        for spec in specs:
            pid = spec["id"]
            prompt = spec["prompt"]

            intent = analyze_prompt(prompt, mode="generate")
            plan = plan_semantic_layout(intent, tokenizer=tokenizer_t5)
            for h in hooks:
                h.set_plan(plan)

            p_data = prompt_bank[prompt]
            p_emb = p_data["prompt_embeds"].to("cuda")
            neg_p_emb = p_data["negative_prompt_embeds"].to("cuda")
            pool_emb = p_data["pooled_prompt_embeds"].to("cuda")
            neg_pool_emb = p_data["negative_pooled_prompt_embeds"].to("cuda")

            for seed in seeds:
                run_idx += 1
                img_name = f"{pid}_str_{str_val:.2f}_s{seed}.png"
                img_path = out_img_dir / img_name

                if img_path.exists():
                    log(f"  [{run_idx:03d}/{total_images}] {pid} (s={seed}) str={str_val:.2f} | [CACHED]")
                    continue

                gen = torch.Generator(device="cpu").manual_seed(seed)
                t0 = time.time()
                with torch.inference_mode():
                    img = pipe(
                        prompt_embeds=p_emb,
                        pooled_prompt_embeds=pool_emb,
                        negative_prompt_embeds=neg_p_emb,
                        negative_pooled_prompt_embeds=neg_pool_emb,
                        num_inference_steps=20,
                        guidance_scale=4.5,
                        width=512,
                        height=512,
                        generator=gen,
                        callback_on_step_end=step_cb,
                    ).images[0]
                dt = time.time() - t0
                img.save(img_path)
                log(f"  [{run_idx:03d}/{total_images}] {pid} (s={seed}) str={str_val:.2f} | Generated in {dt:.2f}s ({dt/20.0:.3f}s/step)")

    log(f"\n[+] All {total_images} images generated in {time.time() - t_gen_start:.1f}s! Freeing generation pipeline...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Strict Evaluation Loop
    log("\n2. Running Strict Evaluation (OWL-ViT + CLIP-ViT-L/14)...")
    evaluator = StrictSpatialEvaluator(device="cuda")

    eval_by_cond: dict[str, list[dict]] = {str(s): [] for s in strengths}

    for str_val in strengths:
        log(f"  Evaluating condition str={str_val:.2f} ...")
        for spec in specs:
            pid = spec["id"]
            prompt = spec["prompt"]
            subj = spec["subject"]
            obj = spec["object"]
            rel = spec["relation"]

            for seed in seeds:
                img_name = f"{pid}_str_{str_val:.2f}_s{seed}.png"
                img_path = out_img_dir / img_name
                img = Image.open(img_path).convert("RGB")

                s_det, o_det = evaluator.detect_entities(img, subj, obj)
                is_sat, details = evaluator.check_relation(s_det, o_det, rel)
                clip_sim, aes_score = evaluator.evaluate_quality(img, prompt)

                both_present = s_det is not None and o_det is not None

                if is_sat:
                    failure_cat = "success"
                elif not both_present:
                    if s_det is None and o_det is None:
                        failure_cat = "omission_both"
                    elif s_det is None:
                        failure_cat = "omission_subject"
                    else:
                        failure_cat = "omission_object"
                else:
                    failure_cat = "misplacement"

                rec = {
                    "prompt_id": pid,
                    "prompt": prompt,
                    "subject": subj,
                    "object": obj,
                    "relation": rel,
                    "seed": seed,
                    "strength": str_val,
                    "satisfied": is_sat,
                    "both_present": both_present,
                    "failure_category": failure_cat,
                    "details": details,
                    "subject_detected": s_det is not None,
                    "object_detected": o_det is not None,
                    "subject_score": s_det["score"] if s_det else 0.0,
                    "object_score": o_det["score"] if o_det else 0.0,
                    "clip_similarity": round(float(clip_sim), 4),
                    "laion_aesthetic_score": round(float(aes_score), 4),
                    "image_path": str(img_path),
                }
                eval_by_cond[str(str_val)].append(rec)

    # 3. Statistical Summaries & McNemar Paired Tests
    cond_summaries = {}
    off_recs = eval_by_cond["0.0"]

    for str_val in strengths:
        cur_recs = eval_by_cond[str(str_val)]
        n = len(cur_recs)
        n_sat = sum(1 for r in cur_recs if r["satisfied"])
        n_pres = sum(1 for r in cur_recs if r["both_present"])
        sat_rate = n_sat / n if n > 0 else 0.0
        pres_rate = n_pres / n if n > 0 else 0.0
        ci_sat = wilson_score_interval(n_sat, n)
        ci_pres = wilson_score_interval(n_pres, n)

        # Directional vs symmetric breakdown
        dir_recs = [r for r in cur_recs if r["relation"] in ("left_of", "right_of")]
        sym_recs = [r for r in cur_recs if r["relation"] == "beside"]

        dir_sat = sum(1 for r in dir_recs if r["satisfied"])
        dir_n = len(dir_recs)
        dir_rate = dir_sat / dir_n if dir_n > 0 else 0.0
        dir_ci = wilson_score_interval(dir_sat, dir_n) if dir_n > 0 else (0.0, 0.0)

        sym_sat = sum(1 for r in sym_recs if r["satisfied"])
        sym_n = len(sym_recs)
        sym_rate = sym_sat / sym_n if sym_n > 0 else 0.0
        sym_ci = wilson_score_interval(sym_sat, sym_n) if sym_n > 0 else (0.0, 0.0)

        left_recs = [r for r in cur_recs if r["relation"] == "left_of"]
        left_sat = sum(1 for r in left_recs if r["satisfied"])
        left_n = len(left_recs)
        left_rate = left_sat / left_n if left_n > 0 else 0.0

        right_recs = [r for r in cur_recs if r["relation"] == "right_of"]
        right_sat = sum(1 for r in right_recs if r["satisfied"])
        right_n = len(right_recs)
        right_rate = right_sat / right_n if right_n > 0 else 0.0

        omissions = sum(1 for r in cur_recs if "omission" in r["failure_category"])
        misplacements = sum(1 for r in cur_recs if r["failure_category"] == "misplacement")

        mean_clip = float(np.mean([r["clip_similarity"] for r in cur_recs]))
        mean_aes = float(np.mean([r["laion_aesthetic_score"] for r in cur_recs]))

        mcnemar_data = None
        if str_val > 0.0:
            both_pass_a = sum(1 for r_off, r_on in zip(off_recs, cur_recs, strict=True) if r_off["satisfied"] and r_on["satisfied"])
            gain_b = sum(1 for r_off, r_on in zip(off_recs, cur_recs, strict=True) if (not r_off["satisfied"]) and r_on["satisfied"])
            loss_c = sum(1 for r_off, r_on in zip(off_recs, cur_recs, strict=True) if r_off["satisfied"] and (not r_on["satisfied"]))
            both_fail_d = sum(1 for r_off, r_on in zip(off_recs, cur_recs, strict=True) if (not r_off["satisfied"]) and (not r_on["satisfied"]))
            p_val = exact_mcnemar_p_value(gain_b, loss_c)
            mcnemar_data = {
                "contingency_table": {"both_pass_a": both_pass_a, "gain_b": gain_b, "loss_c": loss_c, "both_fail_d": both_fail_d},
                "net_gain": gain_b - loss_c,
                "exact_mcnemar_p_value": p_val,
                "statistically_significant": p_val < 0.05,
            }

        cond_summaries[str(str_val)] = {
            "strength": str_val,
            "total_runs": n,
            "satisfied_count": n_sat,
            "satisfaction_rate": round(sat_rate, 4),
            "wilson_ci_95": [round(ci_sat[0], 2), round(ci_sat[1], 2)],
            "presence_count": n_pres,
            "presence_rate": round(pres_rate, 4),
            "presence_ci_95": [round(ci_pres[0], 2), round(ci_pres[1], 2)],
            "directional": {
                "total": dir_n,
                "satisfied": dir_sat,
                "rate": round(dir_rate, 4),
                "wilson_ci_95": [round(dir_ci[0], 2), round(dir_ci[1], 2)],
                "left_of": {"total": left_n, "satisfied": left_sat, "rate": round(left_rate, 4)},
                "right_of": {"total": right_n, "satisfied": right_sat, "rate": round(right_rate, 4)},
            },
            "symmetric": {
                "total": sym_n,
                "satisfied": sym_sat,
                "rate": round(sym_rate, 4),
                "wilson_ci_95": [round(sym_ci[0], 2), round(sym_ci[1], 2)],
            },
            "failure_modes": {
                "omissions": omissions,
                "misplacements": misplacements,
            },
            "mean_clip_similarity": round(mean_clip, 4),
            "mean_laion_aesthetic_score": round(mean_aes, 4),
            "mcnemar_vs_off": mcnemar_data,
        }

    log("\n" + "=" * 80)
    log(f"SUMMARY FOR [{benchmark_name.upper()}]:")
    log("=" * 80)
    log(f"{'Condition':<12} | {'Satisfaction':<16} | {'Directional':<14} | {'Symmetric':<12} | {'Presence':<12} | {'LAION AES':<10} | {'CLIP Cos':<10} | {'McNemar p':<10}")
    log("-" * 110)
    for str_val in strengths:
        s = cond_summaries[str(str_val)]
        p_str = f"p={s['mcnemar_vs_off']['exact_mcnemar_p_value']:.4f}" if s['mcnemar_vs_off'] else "N/A (Ref)"
        sym_str = f"{s['symmetric']['rate']*100:.1f}%" if s['symmetric']['total'] > 0 else "N/A"
        log(f"str={s['strength']:<6.2f} | {s['satisfaction_rate']*100:>5.1f}% ({s['satisfied_count']}/{s['total_runs']}) | {s['directional']['rate']*100:>5.1f}% ({s['directional']['satisfied']}/{s['directional']['total']}) | {sym_str:<12} | {s['presence_rate']*100:>5.1f}% | {s['mean_laion_aesthetic_score']:<10.3f} | {s['mean_clip_similarity']:<10.3f} | {p_str:<10}")

    out_json = ROOT_DIR / "benchmarks" / f"phase5_sd35m_{benchmark_name}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": "stabilityai/stable-diffusion-3.5-medium",
            "benchmark": benchmark_name,
            "resolution": "512x512",
            "steps": 20,
            "conditions": cond_summaries,
            "details": eval_by_cond,
        }, f, indent=2)
    log(f"\n[+] Saved {benchmark_name} results to: {out_json}")

    return cond_summaries


def main():
    log("================================================================================")
    log("PHASE 5: COMPREHENSIVE POWERED STUDY (STANDARD 24 + HARD 24 BENCHMARKS)")
    log("================================================================================")

    # 1. Load Pre-Encoded Embeddings Bank
    emb_path = ROOT_DIR / "benchmarks" / "sd35m_prompt_embeddings.pt"
    log(f"Loading Prompt Embeddings Bank from {emb_path} ...")
    prompt_bank = torch.load(emb_path, map_location="cpu")
    log(f"[+] Loaded {len(prompt_bank)} pre-encoded prompt embeddings.")

    # 2. Tokenizer for semantic layout planning
    tok_t5 = AutoTokenizer.from_pretrained("models/sd35_medium/tokenizer_3")

    # 3. Benchmark A: Standard 24 Benchmark (N=192/condition, exact match to SD v1.5 dataset)
    run_benchmark_eval(
        benchmark_name="standard_24",
        specs=LATERAL_24_SPECS,
        prompt_bank=prompt_bank,
        tokenizer_t5=tok_t5,
        strengths=[0.0, 3.0, 6.0],
    )

    # 4. Benchmark B: Hard 24 Directional Benchmark (N=192/condition, 12 Left + 12 Right)
    run_benchmark_eval(
        benchmark_name="hard_24",
        specs=LATERAL_HARD_24_SPECS,
        prompt_bank=prompt_bank,
        tokenizer_t5=tok_t5,
        strengths=[0.0, 3.0, 6.0],
    )

    log("\n" + "=" * 80)
    log("ALL PHASE 5 EVALUATIONS COMPLETED SUCCESSFULLY!")
    log("=" * 80)


if __name__ == "__main__":
    main()
