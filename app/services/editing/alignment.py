"""Pre-generation prompt/image alignment check.

Runs before the denoise loop, so a prompt that conflicts with the source image is
caught while it is still cheap - no GPU time spent producing an image the user
will reject. Mirrors the chat pipeline's policy: proceed with a *stated*
assumption when the gap is recoverable, ask exactly one question when it is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import torch

from app.services.editing.adaptive_reference import (
    CoefficientConfig,
    calibrate_similarity,
    cosine_similarity,
    extract_region_embedding,
)

AlignmentStatus = Literal["aligned", "assumed", "clarify"]

# Cardinals answer "how many", ordinals answer "which one". Only an ordinal can
# contradict the scene: "add a second person" to a 3-person photo is incoherent,
# but "add a person" or "add another person" is a perfectly ordinary request.
_CARDINALS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_ORDINALS = {
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
}
_ADDITIVE = re.compile(r"\b(add|insert|place|put|include)\b", re.IGNORECASE)
_REMOVAL = re.compile(r"\b(remove|delete|erase|take out|get rid of)\b", re.IGNORECASE)


@dataclass(frozen=True)
class AlignmentReport:
    """Outcome of the pre-denoise check."""

    status: AlignmentStatus
    similarity: float
    calibrated_similarity: float
    reason: str
    assumption: str | None = None
    clarifying_question: str | None = None

    @property
    def should_generate(self) -> bool:
        """`clarify` is the only status that stops generation."""
        return self.status != "clarify"

    def as_log_dict(self) -> dict:
        return {
            "status": self.status,
            "similarity": round(self.similarity, 4),
            "calibrated_similarity": round(self.calibrated_similarity, 4),
            "reason": self.reason,
            "has_assumption": bool(self.assumption),
            "has_question": bool(self.clarifying_question),
        }


def _quantifier(
    prompt: str,
    subject: str,
    vocabulary: dict[str, int],
    *,
    allow_digits: bool = True,
) -> int | None:
    """Nearest quantifier to the left of `subject`, drawn from `vocabulary`.

    Scanning leftwards and taking the *nearest* match means the ordinal in
    "a second person" wins over the article that precedes it.
    """
    words = re.findall(r"[\w']+", prompt.lower())
    subject = subject.lower()
    for index, word in enumerate(words):
        if word != subject and word != f"{subject}s":
            continue
        for offset in range(1, 4):  # look back up to three words
            position = index - offset
            if position < 0:
                break
            candidate = words[position]
            if allow_digits and candidate.isdigit():
                return int(candidate)
            if candidate in vocabulary:
                return vocabulary[candidate]
    return None


def check_scene_conflict(prompt: str, scene_facts: dict[str, int] | None) -> str | None:
    """Detect a countable contradiction against known scene contents.

    `scene_facts` maps a subject to how many are already present, as produced by a
    detector or captioner. Without it this check is skipped rather than guessed.
    """
    if not scene_facts:
        return None
    for subject, present in scene_facts.items():
        # Additive: only an ordinal contradicts. "add a person" to a 3-person photo
        # is not a conflict, so indefinite articles must never reach this branch.
        if _ADDITIVE.search(prompt):
            ordinal = _quantifier(prompt, subject, _ORDINALS, allow_digits=False)
            if ordinal is not None and present >= ordinal:
                return (
                    f"the prompt asks for {subject} number {ordinal}, "
                    f"but the image already contains {present}"
                )
        # Removal: you cannot remove more than exist, whatever the numeral form.
        if _REMOVAL.search(prompt):
            count = _quantifier(prompt, subject, _CARDINALS)
            if count is not None and present < count:
                return (
                    f"the prompt asks to remove {count} {subject}(s), "
                    f"but the image only contains {present}"
                )
    return None


def check_prompt_image_alignment(
    *,
    prompt: str,
    prompt_embedding: torch.Tensor,
    source_image_embedding: torch.Tensor,
    edit_region_mask: torch.Tensor | None = None,
    scene_facts: dict[str, int] | None = None,
    config: CoefficientConfig | None = None,
    clarify_below: float = 0.05,
    assume_below: float = 0.25,
    allow_clarification: bool = True,
) -> AlignmentReport:
    """Assess whether the prompt can be applied to this image before denoising.

    `allow_clarification=False` suits a real-time product: every turn still
    generates, but the assumption is recorded instead of blocking on a question.
    """
    config = config or CoefficientConfig()
    if edit_region_mask is not None:
        region = extract_region_embedding(source_image_embedding, edit_region_mask)
    else:
        region = source_image_embedding
    similarity = cosine_similarity(prompt_embedding, region)
    calibrated = calibrate_similarity(similarity, config)

    conflict = check_scene_conflict(prompt, scene_facts)
    if conflict:
        if allow_clarification:
            return AlignmentReport(
                status="clarify",
                similarity=similarity,
                calibrated_similarity=calibrated,
                reason=f"scene_conflict: {conflict}",
                clarifying_question=(
                    "That doesn't match what I see in the image - "
                    f"{conflict}. Should I go ahead anyway?"
                ),
            )
        return AlignmentReport(
            status="assumed",
            similarity=similarity,
            calibrated_similarity=calibrated,
            reason=f"scene_conflict: {conflict}",
            assumption="Applying the edit as literally described despite the count mismatch.",
        )

    if calibrated < clarify_below:
        if allow_clarification:
            return AlignmentReport(
                status="clarify",
                similarity=similarity,
                calibrated_similarity=calibrated,
                reason="prompt is unrelated to the source region",
                clarifying_question=(
                    "I can't tell which part of this image that refers to - "
                    "which region should I edit?"
                ),
            )
        return AlignmentReport(
            status="assumed",
            similarity=similarity,
            calibrated_similarity=calibrated,
            reason="prompt is unrelated to the source region",
            assumption="Treating the request as a global edit of the whole image.",
        )

    if calibrated < assume_below:
        return AlignmentReport(
            status="assumed",
            similarity=similarity,
            calibrated_similarity=calibrated,
            reason="weak prompt/region match",
            assumption="Applying the edit to the most likely region for this prompt.",
        )

    return AlignmentReport(
        status="aligned",
        similarity=similarity,
        calibrated_similarity=calibrated,
        reason="prompt matches the target region",
    )
