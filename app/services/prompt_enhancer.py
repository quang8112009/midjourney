"""Small, dependency-free prompt expansion for image generation requests."""

from __future__ import annotations

import random

STYLE_MODIFIERS = (
    "cinematic photography",
    "editorial illustration",
    "painterly concept art",
    "fine-art realism",
    "stylized 3D render",
    "analog film aesthetic",
)

LIGHTING_MODIFIERS = (
    "soft golden-hour lighting",
    "dramatic volumetric lighting",
    "diffused studio lighting",
    "moody rim lighting",
    "natural window light",
    "high-contrast chiaroscuro lighting",
)

QUALITY_MODIFIERS = (
    "intricate details",
    "sharp focus",
    "professional color grading",
    "highly detailed textures",
    "refined composition",
    "polished high-resolution finish",
)


def enhance_prompt(prompt: str, *, seed: int | None = None) -> str:
    """Append one randomized style, lighting, and quality modifier.

    A local seeded generator makes the expansion reproducible when a request
    supplies a seed. Unseeded calls use the operating system's random source
    and do not mutate Python's module-level random state.
    """
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("Prompt must not be blank.")

    chooser = random.Random(seed) if seed is not None else random.SystemRandom()
    modifiers = (
        chooser.choice(STYLE_MODIFIERS),
        chooser.choice(LIGHTING_MODIFIERS),
        chooser.choice(QUALITY_MODIFIERS),
    )
    return ", ".join((normalized_prompt, *modifiers))
