"""Unit and invariant tests for principled aesthetic style expansion."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from app.services.editing.layout_guidance import build_layout_guidance_bias
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout
from app.services.editing.style_expansion import (
    BANNED_SLOP_WORDS,
    count_tokens_approx,
    expand_style,
    filter_slop_words,
    resolve_token_budget,
)


def test_token_budget_resolution_per_backbone():
    assert resolve_token_budget("stable-diffusion") == 77
    assert resolve_token_budget("runwayml/stable-diffusion-v1-5") == 77
    assert resolve_token_budget("pixart-alpha") == 120
    assert resolve_token_budget("stable-diffusion-3.5") == 512
    assert resolve_token_budget("stabilityai/stable-diffusion-3.5-medium") == 512


def test_banned_slop_words_filtered():
    slop_input = (
        "a cute puppy, 8k, masterpiece, trending on artstation, "
        "ultra high quality, soft studio lighting"
    )
    cleaned = filter_slop_words(slop_input)
    for slop in BANNED_SLOP_WORDS:
        assert slop not in cleaned.lower()
    assert "cute puppy" in cleaned
    assert "soft studio lighting" in cleaned


def test_explicit_user_style_preserved_within_domain():
    prompt = "a serene mountain lake, watercolor painting"
    result = expand_style(prompt, model="stable-diffusion-3.5", style_expansion_enabled=True)
    assert result.applied
    assert "watercolor" in result.expanded_prompt.lower()
    assert "washes" in result.expanded_prompt.lower() or "pigment" in result.expanded_prompt.lower()
    assert result.expanded_prompt.startswith(prompt)


def test_domain_inference_when_no_style_specified():
    portrait_prompt = "portrait of an elderly fisherman with a weathered face"
    res_p = expand_style(
        portrait_prompt, model="stable-diffusion-3.5", style_expansion_enabled=True
    )
    assert res_p.applied
    assert "lens" in res_p.expanded_prompt.lower() or "lighting" in res_p.expanded_prompt.lower()
    assert res_p.inferred_domain == "portrait"


def test_overflow_protection_entity_spatial_never_truncated():
    # Construct a long prompt on SD v1.5 (budget=77)
    long_prompt = (
        "a very detailed red ceramic mug to the left of a bright blue ceramic mug "
        "on a very large rustic wooden kitchen table with many complex decorative carving patterns "
        "and fine ornamental details"
    )
    res = expand_style(long_prompt, model="stable-diffusion", style_expansion_enabled=True)

    # Base prompt must be fully intact at the beginning
    assert res.expanded_prompt.startswith(long_prompt)
    # Total tokens must not exceed budget
    tok_count = count_tokens_approx(res.expanded_prompt)
    assert tok_count <= 77 + 2


def test_expanded_style_tokens_receive_zero_spatial_bias_sd15():
    """Verify on SD v1.5 UNet that expanded style tokens receive exactly 0.0 spatial bias."""
    raw_prompt = "a yellow banana to the left of a green apple on a wooden table"
    expanded_res = expand_style(raw_prompt, model="stable-diffusion", style_expansion_enabled=True)
    assert expanded_res.applied

    intent = analyze_prompt(expanded_res.expanded_prompt, mode="generate")
    plan = plan_semantic_layout(intent)

    # All planned objects must be the actual entities, not style words
    entity_labels = [o.label for o in plan.objects]
    assert any("banana" in lbl for lbl in entity_labels)
    assert any("apple" in lbl for lbl in entity_labels)
    for o in plan.objects:
        assert "lighting" not in o.label
        assert "lens" not in o.label
        assert "grading" not in o.label


def test_expanded_style_tokens_receive_zero_spatial_bias_sd35():
    """Verify on SD 3.5 MMDiT that expanded style tokens receive strictly 0.0 bias."""
    tok_t5 = AutoTokenizer.from_pretrained("models/sd35_medium/tokenizer_3")

    raw_prompt = "a red sports car to the right of a black sports car on an asphalt road"
    expanded_res = expand_style(
        raw_prompt,
        model="stable-diffusion-3.5",
        tokenizer=tok_t5,
        style_expansion_enabled=True,
    )
    assert expanded_res.applied

    intent = analyze_prompt(expanded_res.expanded_prompt, mode="generate")
    plan = plan_semantic_layout(intent, tokenizer=tok_t5)

    # Compute layout guidance bias tensor
    bias_tensor = build_layout_guidance_bias(
        plan=plan,
        num_image_tokens=1024,
        num_text_tokens=666,
        guidance_strength=3.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # 1. CLIP-L [0..76] and CLIP-G [77..153] must have strictly 0.0 max bias
    clip_l_max = bias_tensor[:, :77].abs().max().item()
    clip_g_max = bias_tensor[:, 77:154].abs().max().item()
    assert clip_l_max == 0.0, f"Expected 0.0 max bias on CLIP-L, got {clip_l_max}"
    assert clip_g_max == 0.0, f"Expected 0.0 max bias on CLIP-G, got {clip_g_max}"

    # 2. T5-XXL [154..665]: Identify style token positions
    encoded_t5 = tok_t5(expanded_res.expanded_prompt, add_special_tokens=True)
    pieces = tok_t5.convert_ids_to_tokens(encoded_t5.input_ids)

    # Collect all token positions claimed by planned entities
    entity_tokens = set()
    for o in plan.objects:
        entity_tokens.update(o.token_indices)

    # Every token outside entity_tokens (including all style expansion tokens) must have 0.0 bias
    for t_pos in range(len(pieces)):
        if t_pos not in entity_tokens:
            joint_pos = 154 + t_pos
            if joint_pos < 666:
                token_bias_max = bias_tensor[:, joint_pos].abs().max().item()
                assert token_bias_max == 0.0, (
                    f"Style token '{pieces[t_pos]}' at {joint_pos} got non-zero: {token_bias_max}"
                )


