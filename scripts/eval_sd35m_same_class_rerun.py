from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from transformers import AutoTokenizer

from app.services.editing.layout_guidance import (
    MMDiTJointAttentionHook,
    TwoPhaseSchedule,
    build_layout_guidance_bias,
)
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout
from scripts.eval_spatial_lateral_dedicated import (
    SEEDS_192,
)
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator
from scripts.hard_lateral_specs import LATERAL_HARD_24_SPECS

LOG_FILE = ROOT_DIR / "same_class_rerun.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


import torch.nn.functional as F


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



def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 80)
    log("RE-EVALUATING SAME-CLASS SUB-GROUP (PAIRS 01-06) ON SD 3.5 MEDIUM")
    log("Testing with Head-Noun De-duplication Fix in Semantic Planner")
    log("=" * 80)

    # 1. Load Pre-Encoded Embeddings Bank
    emb_path = ROOT_DIR / "benchmarks" / "sd35m_prompt_embeddings.pt"
    log(f"Loading Prompt Embeddings Bank from {emb_path} ...")
    prompt_bank = torch.load(emb_path, map_location="cpu")

    # 2. Tokenizer for semantic layout planning
    tok_t5 = AutoTokenizer.from_pretrained("models/sd35_medium/tokenizer_3")

    # 3. Select Same-Class Specs (12 prompts: 6 left_of + 6 right_of)
    same_class_specs = LATERAL_HARD_24_SPECS[:6] + LATERAL_HARD_24_SPECS[12:18]
    log(f"[+] Selected {len(same_class_specs)} same-class prompts (Pairs 01-06)")

    out_img_dir = ROOT_DIR / "benchmarks" / "images" / "sd35m_same_class_fixed"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    strengths = [0.0, 3.0, 6.0]
    seeds = SEEDS_192

    # 4. Pipeline Generation (Transformer & VAE on CUDA)
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

    total_images = len(strengths) * len(same_class_specs) * len(seeds)
    run_idx = 0
    t_gen_start = time.time()

    for str_val in strengths:
        log(f"\n>>> Condition: Guidance Strength = {str_val:.2f} <<<")
        for h in hooks:
            h.guidance_strength = str_val

        for spec in same_class_specs:
            pid = spec["id"]
            prompt = spec["prompt"]

            intent = analyze_prompt(prompt, mode="generate")
            plan = plan_semantic_layout(intent, tokenizer=tok_t5)
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

    log(f"\n[+] All {total_images} same-class images generated in {time.time() - t_gen_start:.1f}s! Freeing generation pipeline...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Strict Evaluation Loop
    log("\n2. Running Strict Evaluation (OWL-ViT + CLIP-ViT-L/14)...")
    evaluator = StrictSpatialEvaluator(device="cuda")

    all_details: dict[str, list[dict[str, Any]]] = {}

    for str_val in strengths:
        log(f"  Evaluating condition str={str_val:.2f} ...")
        records = []
        for spec in same_class_specs:
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

                records.append({
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
                })
        all_details[str(str_val)] = records


    # 6. Print Sub-Group Results & McNemar Tests
    log("\n" + "=" * 90)
    log("SAME-CLASS SUB-GROUP RESULTS AFTER PLANNER FIX (N=48 per direction, N=96 total):")
    log("=" * 90)

    for str_val in strengths:
        cur_recs = all_details[str(str_val)]
        n_tot = len(cur_recs)
        n_sat = sum(1 for r in cur_recs if r["satisfied"])
        n_pres = sum(1 for r in cur_recs if r["both_present"])
        sat_rate = n_sat / n_tot

        left_recs = [r for r in cur_recs if r["relation"] == "left_of"]
        left_sat = sum(1 for r in left_recs if r["satisfied"])
        left_pres = sum(1 for r in left_recs if r["both_present"])
        left_rate = left_sat / len(left_recs)

        right_recs = [r for r in cur_recs if r["relation"] == "right_of"]
        right_sat = sum(1 for r in right_recs if r["satisfied"])
        right_pres = sum(1 for r in right_recs if r["both_present"])
        right_rate = right_sat / len(right_recs)

        log(f"Strength {str_val:.2f}:")
        log(f"  Overall Same-Class: {sat_rate*100:5.1f}% ({n_sat:02d}/{n_tot}) | Presence: {n_pres/n_tot*100:5.1f}% ({n_pres}/{n_tot})")
        log(f"  - Left_Of  (N=48): {left_rate*100:5.1f}% ({left_sat:02d}/48) | Presence: {left_pres/48*100:5.1f}% ({left_pres}/48)")
        log(f"  - Right_Of (N=48): {right_rate*100:5.1f}% ({right_sat:02d}/48) | Presence: {right_pres/48*100:5.1f}% ({right_pres}/48)")

    # Save same-class rerun data
    out_json = ROOT_DIR / "benchmarks" / "same_class_fixed_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_details, f, indent=2)
    log(f"\n[+] Saved same-class rerun results to: {out_json}")


if __name__ == "__main__":
    main()
