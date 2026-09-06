"""Principled, token-budgeted style and aesthetic expansion module.

Expands aesthetic descriptors (medium, lighting, palette, mood, camera, texture, era)
while strictly preserving all entities, quantities, and spatial relationships verbatim.
All added style tokens receive 0.0 spatial attention bias by design.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.services.editing.semantic_planner import extract_words

logger = logging.getLogger(__name__)

BANNED_SLOP_WORDS = frozenset([
    "8k",
    "8k resolution",
    "masterpiece",
    "trending on artstation",
    "trending on art station",
    "ultra high quality",
    "ultra detailed",
    "best quality",
    "photorealistic 8k",
    "award winning",
    "hyperrealistic 8k",
])

BACKBONE_TOKEN_BUDGETS: dict[str, int] = {
    "stable-diffusion": 77,
    "sd15": 77,
    "runwayml/stable-diffusion-v1-5": 77,
    "pixart-alpha": 120,
    "PixArt-alpha/PixArt-XL-2-512x512": 120,
    "stable-diffusion-3.5": 512,
    "sd35": 512,
    "sd35_medium": 512,
    "sd35_large": 512,
    "stabilityai/stable-diffusion-3.5-medium": 512,
    "stabilityai/stable-diffusion-3.5-large": 512,
    "flux-dev": 512,
    "flux": 512,
}

DOMAIN_STYLE_PRESETS: dict[str, str] = {
    "watercolor": (
        "delicate watercolor washes, cold-press paper texture, "
        "translucent color pooling, soft pigment granulation"
    ),
    "watercolour": (
        "delicate watercolor washes, cold-press paper texture, "
        "translucent color pooling, soft pigment granulation"
    ),
    "oil painting": (
        "rich impasto brushwork, layered glazing, "
        "classical chiaroscuro lighting, textured linen canvas"
    ),
    "cyberpunk": (
        "volumetric neon lighting, cinematic chiaroscuro, "
        "reflective damp surfaces, 35mm atmospheric lens"
    ),
    "photorealistic": (
        "natural soft lighting, shallow depth of field, "
        "35mm prime lens, authentic color grading, fine grain"
    ),
    "photography": (
        "natural soft lighting, shallow depth of field, "
        "35mm prime lens, authentic color grading, fine grain"
    ),
    "portrait": (
        "soft studio key lighting, subtle rim light, "
        "shallow depth of field, 85mm portrait lens, natural texture"
    ),
    "landscape": (
        "golden hour lighting, atmospheric perspective, "
        "wide-angle cinematic composition, rich natural tones"
    ),
    "concept art": (
        "digital painting, dynamic atmospheric perspective, "
        "cinematic matte finish, dramatic mood"
    ),
    "pixel art": (
        "crisp 16-bit pixel cluster hierarchy, "
        "authentic CRT color palette, clean dithering"
    ),
    "claymation": (
        "tactile polymer clay texture, subtle fingerprint impressions, "
        "stop-motion studio lighting"
    ),
    "ukiyo-e": (
        "traditional Japanese woodblock print, washi paper texture, "
        "bold ink outlines, flat mineral pigments"
    ),
    "anime": (
        "clean cel-shaded line art, soft ambient bokeh, "
        "vibrant atmospheric grading, studio keyframe aesthetic"
    ),
    "architecture": (
        "clean rectilinear perspective, natural window light, "
        "authentic material textures, architectural photography"
    ),
    "vintage": (
        "authentic analog film grain, warm nostalgic color palette, "
        "subtle optical vignetting, 1970s print tone"
    ),
    "stained glass": (
        "luminous translucent glass segments, dark lead caming outlines, "
        "rich jewel-tone light refraction"
    ),
    "charcoal": (
        "expressive high-contrast charcoal smudges, "
        "textured laid paper, dynamic tonal shading"
    ),
    "ink drawing": (
        "fine cross-hatching linework, deep black archival ink, "
        "subtle ink wash gradients"
    ),
    "gouache": (
        "opaque matte gouache layers, velvet surface finish, "
        "vibrant saturated color blocking"
    ),
}



@dataclass(frozen=True)
class StyleExpansionResult:
    """Structured result of prompt aesthetic expansion."""

    original_prompt: str
    expanded_prompt: str
    style_clause: str
    applied: bool
    token_budget: int
    estimated_tokens: int
    inferred_domain: str | None = None
    provenance: str = "raw_prompt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "expanded_prompt": self.expanded_prompt,
            "style_clause": self.style_clause,
            "applied": self.applied,
            "token_budget": self.token_budget,
            "estimated_tokens": self.estimated_tokens,
            "inferred_domain": self.inferred_domain,
            "provenance": self.provenance,
        }


def resolve_token_budget(model: str) -> int:
    """Resolve conservative token budget for the given model backbone."""
    norm = model.lower().strip()
    return BACKBONE_TOKEN_BUDGETS.get(norm, 77)


def filter_slop_words(text: str) -> str:
    """Remove generic AI slop words and buzzwords."""
    cleaned = text
    for slop in BANNED_SLOP_WORDS:
        pattern = rf"\b{re.escape(slop)}\b"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,.")


def count_tokens_approx(text: str, tokenizer: Any = None) -> int:
    """Accurately count tokens using tokenizer if provided, else conservative piece estimation."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            encoded = tokenizer(text, add_special_tokens=False)
            ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            if ids and isinstance(ids[0], list):
                return len(ids[0])
            return len(ids)
        except Exception:
            pass
    # Conservative BPE approximation: words * 1.35 + punctuation
    words = extract_words(text)
    return max(1, int(len(words) * 1.35) + text.count(",") + text.count("."))


def infer_aesthetic_domain(prompt: str, style_hints: str | None = None) -> tuple[str | None, str]:
    """Infer the primary aesthetic domain from prompt or Pass 1 style_hints."""
    lowered = f"{prompt} {style_hints or ''}".lower()

    for domain, preset in DOMAIN_STYLE_PRESETS.items():
        if re.search(rf"\b{re.escape(domain)}\b", lowered):
            return domain, preset

    # Secondary heuristic inference from subjects
    portrait_cues = ("portrait", "woman", "man", "person", "face", "eyes", "warrior", "astronaut")
    if any(w in lowered for w in portrait_cues):
        return "portrait", DOMAIN_STYLE_PRESETS["portrait"]

    landscape_cues = (
        "mountain", "forest", "cliff", "fjord", "lake", "landscape", "sky", "sea", "ocean"
    )
    if any(w in lowered for w in landscape_cues):
        return "landscape", DOMAIN_STYLE_PRESETS["landscape"]

    arch_cues = ("building", "villa", "interior", "house", "room", "hall", "palace")
    if any(w in lowered for w in arch_cues):
        return "architecture", DOMAIN_STYLE_PRESETS["architecture"]

    scifi_cues = ("robot", "spaceship", "sci-fi", "space", "mech", "cyber")
    if any(w in lowered for w in scifi_cues):
        return "concept art", DOMAIN_STYLE_PRESETS["concept art"]

    # Coherent neutral fine-art photography default
    return "photography", DOMAIN_STYLE_PRESETS["photography"]


def expand_style(
    prompt: str,
    model: str = "stable-diffusion-3.5",
    tokenizer: Any = None,
    style_hints: str | None = None,
    custom_style_clause: str | None = None,
    style_expansion_enabled: bool | None = None,
    max_expansion_tokens: int = 40,
) -> StyleExpansionResult:
    """Enrich the aesthetic descriptors of a prompt within a strict per-backbone token budget.

    Guarantees:
    1. Base prompt, entities, counts, and spatial relations are never modified or truncated.
    2. Explicit user styles are preserved and elaborated within domain.
    3. Expanded tokens are strictly aesthetic and receive 0.0 spatial attention bias.
    """
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return StyleExpansionResult(
            original_prompt="",
            expanded_prompt="",
            style_clause="",
            applied=False,
            token_budget=resolve_token_budget(model),
            estimated_tokens=0,
            provenance="empty_prompt",
        )

    is_enabled = (
        style_expansion_enabled
        if style_expansion_enabled is not None
        else settings.STYLE_EXPANSION_ENABLED
    )
    token_budget = resolve_token_budget(model)

    base_tokens = count_tokens_approx(cleaned_prompt, tokenizer=tokenizer)
    available_budget = token_budget - base_tokens - 2  # 2 token safety buffer

    # If base prompt already consumes the entire budget or expansion is disabled, return untouched
    if not is_enabled and not custom_style_clause:
        return StyleExpansionResult(
            original_prompt=cleaned_prompt,
            expanded_prompt=cleaned_prompt,
            style_clause="",
            applied=False,
            token_budget=token_budget,
            estimated_tokens=base_tokens,
            provenance="raw_prompt",
        )

    if available_budget <= 4:
        logger.info(
            f"Style expansion skipped: prompt ({base_tokens} tok) exhausts budget ({token_budget})."
        )
        return StyleExpansionResult(
            original_prompt=cleaned_prompt,
            expanded_prompt=cleaned_prompt,
            style_clause="",
            applied=False,
            token_budget=token_budget,
            estimated_tokens=base_tokens,
            provenance="budget_exhausted",
        )


    # 1. Synthesize candidate style clause
    inferred_domain = None
    if custom_style_clause:
        candidate_clause = filter_slop_words(custom_style_clause)
        provenance = "user_override"
    elif style_hints and style_hints.lower().strip() != "none":
        candidate_clause = filter_slop_words(style_hints)
        provenance = "pass1_style_hints"
    else:
        domain, preset = infer_aesthetic_domain(cleaned_prompt, style_hints=style_hints)
        inferred_domain = domain
        candidate_clause = preset
        provenance = "domain_preset"

    # 2. Token Budget Enforcement & Non-Destructive Truncation
    clause_tokens = count_tokens_approx(candidate_clause, tokenizer=tokenizer)
    effective_max_tokens = min(available_budget, max_expansion_tokens)

    if clause_tokens > effective_max_tokens:
        # Gracefully trim trailing descriptors by comma boundaries
        parts = [p.strip() for p in candidate_clause.split(",") if p.strip()]
        trimmed_parts = []
        for part in parts:
            test_clause = ", ".join(trimmed_parts + [part])
            if count_tokens_approx(test_clause, tokenizer=tokenizer) <= effective_max_tokens:
                trimmed_parts.append(part)
            else:
                break
        final_clause = ", ".join(trimmed_parts)
    else:
        final_clause = candidate_clause

    if not final_clause:
        return StyleExpansionResult(
            original_prompt=cleaned_prompt,
            expanded_prompt=cleaned_prompt,
            style_clause="",
            applied=False,
            token_budget=token_budget,
            estimated_tokens=base_tokens,
            inferred_domain=inferred_domain,
            provenance="clause_empty_after_budget",
        )

    # Combine: base prompt is leading and untouched
    expanded_prompt = f"{cleaned_prompt}, {final_clause}"
    total_tokens = count_tokens_approx(expanded_prompt, tokenizer=tokenizer)

    return StyleExpansionResult(
        original_prompt=cleaned_prompt,
        expanded_prompt=expanded_prompt,
        style_clause=final_clause,
        applied=True,
        token_budget=token_budget,
        estimated_tokens=total_tokens,
        inferred_domain=inferred_domain,
        provenance=provenance,
    )
