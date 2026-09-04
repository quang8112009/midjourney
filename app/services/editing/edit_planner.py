"""Lightweight pre-denoise planning step.

Reads the prompt plus the source image *once*, before the denoise loop, and emits
the parameters the loop needs: which region to edit, how big the change is, and
what conditioning strengths to use. Deliberately cheap - one embedding comparison
and some string work, no extra diffusion passes - so it does not move latency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

import torch

from app.services.editing.adaptive_reference import (
    CoefficientConfig,
    ReferenceCoefficients,
    compute_adaptive_reference_coefficient,
)
from app.services.editing.alignment import AlignmentReport, check_prompt_image_alignment
from app.services.editing.masks import (
    area_ratio,
    as_soft_mask,
    bounding_box,
    dilate,
    feather,
)
from app.services.editing.prompt_intent import EditInstruction, PromptIntent
from app.services.editing.region_attention import AttentionCapture
from app.services.editing.semantic_planner import (
    SemanticLayoutPlan,
    clean_token_piece,
    extract_words,
    is_special_token,
    map_pieces_to_words,
)

logger = logging.getLogger(__name__)

EditScope = Literal["local", "regional", "global"]

__all__ = [
    "EditPlan",
    "EditScope",
    "align_token_roles",
    "classify_scope",
    "clean_token_piece",
    "extract_words",
    "is_special_token",
    "locate_edit_tokens",
    "map_pieces_to_words",
    "piece_matches_term",
    "plan_edit",
    "select_edit_terms",
]

# Structured fields occasionally retain connective/cue nouns for diagnostics.
# They are not edit targets and must never become attention terms.
_STOPWORDS = frozenset(
    """a an the this that these those and or but of to in on at by for with from into
    make made change turn set please can you it its is are be been being do does did
    my our your their his her image picture photo photograph now more less very really
    keep leave same while but without than then so as also just only up down""".split()
)
_NON_EDIT_TERMS = frozenset(
    """color colour tone shade keep keeping preserve preserving maintain maintaining
    retained retain leave leaving except without while context unchanged""".split()
)
_GLOBAL_HINTS = re.compile(
    r"\b(whole|entire|everything|all|overall|globally|background and|"
    r"style|photorealistic|watercolor|oil painting|anime|sketch|black and white|"
    r"sepia|grayscale|monochrome)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EditPlan:
    """Everything the denoise loop needs, decided up front."""

    prompt: str
    scope: EditScope
    mask: torch.Tensor
    mask_source: Literal["user", "attention", "semantic_plan", "global_fallback"]
    edit_terms: tuple[str, ...]
    edit_token_indices: tuple[int, ...]
    token_roles: tuple[str, ...]
    coefficients: ReferenceCoefficients
    alignment: AlignmentReport
    instruction: EditInstruction
    instruction_index: int
    attention_strength: float
    bounding_box: tuple[int, int, int, int] | None
    semantic_plan: SemanticLayoutPlan | None = None
    aspect: float = 1.0

    @property
    def should_generate(self) -> bool:
        return self.alignment.should_generate

    def as_log_dict(self) -> dict:
        return {
            "scope": self.scope,
            "mask_source": self.mask_source,
            "mask_area_ratio": round(area_ratio(self.mask), 4),
            "instruction": self.instruction.to_dict(),
            "instruction_index": self.instruction_index,
            "has_semantic_plan": self.semantic_plan is not None,
            "edit_terms": list(self.edit_terms),
            "token_roles": list(self.token_roles),
            "attention_strength": round(self.attention_strength, 4),
            "bounding_box": self.bounding_box,
            "aspect": round(self.aspect, 4),
            **self.coefficients.as_log_dict(),
            "alignment": self.alignment.as_log_dict(),
        }


def _structured_term_candidates(instruction: EditInstruction) -> set[str]:
    """Words explicitly carried by an instruction's structured fields."""
    values = [instruction.target, instruction.attribute, instruction.intensity]
    values.extend(instruction.nouns)
    if instruction.resolution is not None:
        values.extend((instruction.resolution.label, instruction.resolution.matched_on))

    candidates = {
        word
        for value in values
        if value
        for word in extract_words(value)
        if len(word) >= 3 and word not in _STOPWORDS and word not in _NON_EDIT_TERMS
    }
    constraint_words = {
        word for clause in instruction.constraints for word in extract_words(clause)
    }
    return candidates - constraint_words


def select_edit_terms(
    instruction: EditInstruction | str,
    *,
    limit: int = 6,
) -> tuple[str, ...]:
    """Structured edit terms in prompt order, without re-parsing intent."""
    if isinstance(instruction, str):
        seen: list[str] = []
        for word in re.findall(r"[a-zA-Z][a-zA-Z'-]+", instruction.lower()):
            if word in _STOPWORDS or len(word) < 3 or word in seen:
                continue
            seen.append(word)
        return tuple(seen[:limit])

    candidates = _structured_term_candidates(instruction)
    seen = []
    for word in extract_words(instruction.raw_text):
        if word not in candidates or word in seen:
            continue
        seen.append(word)
    # Detector-normalised labels need not occur verbatim in the instruction.
    for word in sorted(candidates):
        if word not in seen:
            seen.append(word)
    return tuple(seen[:limit])


def piece_matches_term(cleaned: str, terms) -> bool:
    """Exact match, or a sub-word piece of >=3 chars that *prefixes* a term.

    Deliberately not bidirectional substring matching: `"red" in "reduce"` and
    `"car" in "carpet"` are true, which tags unrelated words as edit targets and
    then region-masks them.
    """
    if not cleaned:
        return False
    if cleaned in terms:
        return True
    return len(cleaned) >= 3 and any(term.startswith(cleaned) for term in terms)


def locate_edit_tokens(
    prompt: str,
    terms: tuple[str, ...],
    tokenizer=None,
    *,
    max_length: int | None = None,
) -> tuple[int, ...]:
    """Map edit terms onto text-encoder token positions.

    With a tokenizer this is exact. Without one it falls back to word
    positions, which is only correct for word-level encoders - callers that care
    should pass the pipeline's own tokenizer.
    """
    words = extract_words(prompt)
    if tokenizer is not None:
        encoded = tokenizer(prompt, add_special_tokens=True)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        ids = list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)
        pieces = tokenizer.convert_ids_to_tokens(ids)
        word_index = map_pieces_to_words(pieces)
        indices = []
        for position, index in enumerate(word_index):
            if max_length is not None and position >= max_length:
                break
            if index is None or index >= len(words):
                continue
            if words[index] in terms:
                indices.append(position)
        return tuple(indices)
    return tuple(index for index, word in enumerate(words) if word in terms)


def align_token_roles(
    instruction: EditInstruction | str,
    tokenizer=None,
    *,
    max_length: int | None = None,
) -> tuple[str, ...]:
    """Map structured instruction roles into tokenizer sub-word positions.

    When a tokenizer is present, special tokens (BOS/EOS/PAD) and punctuation are
    assigned 'neutral', while sub-word pieces inherit their parent word's role.
    Without a tokenizer, returns word-level roles directly.
    """
    if isinstance(instruction, str):
        prompt = instruction
        from app.services.editing.region_attention import classify_token_roles

        word_roles = classify_token_roles(prompt)
    else:
        prompt = instruction.raw_text
        edit_terms = set(select_edit_terms(instruction))
        context_terms = {
            word for clause in instruction.constraints for word in extract_words(clause)
        }
        word_roles = tuple(
            (
                word,
                "context"
                if word in context_terms
                else "edit_target"
                if word in edit_terms
                else "neutral",
            )
            for word in extract_words(prompt)
        )

    if tokenizer is None:
        return tuple(role for _, role in word_roles)

    encoded = tokenizer(prompt, add_special_tokens=True)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    ids = list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)
    pieces = tokenizer.convert_ids_to_tokens(ids)

    word_index = map_pieces_to_words(pieces)
    token_roles: list[str] = []
    for position, index in enumerate(word_index):
        if max_length is not None and position >= max_length:
            break
        if index is None or index >= len(word_roles):
            token_roles.append("neutral")
            continue
        token_roles.append(word_roles[index][1])

    return tuple(token_roles)


def classify_scope(
    instruction: EditInstruction | str,
    mask: torch.Tensor,
    *,
    mask_is_explicit: bool = False,
) -> EditScope:
    """Resolve effective scope from structured intent and mask coverage.

    An explicit user mask wins over a global instruction because it is the more
    precise location signal. Without one, a structured global scope wins. Mask
    coverage can always promote a local instruction to regional/global.
    """
    ratio = area_ratio(mask)
    if isinstance(instruction, str):
        if ratio >= 0.6 or (bool(_GLOBAL_HINTS.search(instruction)) and not mask_is_explicit):
            return "global"
        if ratio <= 0.15:
            return "local"
        return "regional"

    if ratio >= 0.6 or (instruction.scope == "global" and not mask_is_explicit):
        return "global"
    if ratio <= 0.15:
        return "local"
    return "regional"


def _semantic_target_labels(instruction: EditInstruction) -> tuple[str, ...]:
    """Ordered labels that may ground an instruction in a layout object."""
    values: list[str | None] = [instruction.target]
    if instruction.resolution is not None:
        values.extend((instruction.resolution.label, instruction.resolution.matched_on))
    values.extend(instruction.nouns)
    constraint_words = {
        word for clause in instruction.constraints for word in extract_words(clause)
    }

    labels: list[str] = []
    for value in values:
        if not value:
            continue
        label = " ".join(extract_words(value))
        if (
            not label
            or label in labels
            or label in constraint_words
            or label in _STOPWORDS
            or label in _NON_EDIT_TERMS
        ):
            continue
        labels.append(label)
    return tuple(labels)


def _singular_label(label: str) -> str:
    if label.endswith("ies") and len(label) > 3:
        return label[:-3] + "y"
    if label.endswith("es") and len(label) > 3:
        return label[:-2]
    if label.endswith("s") and len(label) > 2:
        return label[:-1]
    return label


def _semantic_target_box(
    semantic_plan: SemanticLayoutPlan | None,
    instruction: EditInstruction,
):
    """Find a layout box using only PromptIntent-derived target labels."""
    if semantic_plan is None:
        return None
    labels = _semantic_target_labels(instruction)
    for label in labels:
        for planned_object in semantic_plan.objects:
            planned_label = " ".join(extract_words(planned_object.label))
            if planned_label == label:
                return planned_object.box
        for df in getattr(semantic_plan, "density_fields", ()):
            planned_label = " ".join(extract_words(df.label))
            if planned_label == label:
                return df.region
    for label in labels:
        singular = _singular_label(label)
        for planned_object in semantic_plan.objects:
            planned_label = " ".join(extract_words(planned_object.label))
            if _singular_label(planned_label) == singular:
                return planned_object.box
        for df in getattr(semantic_plan, "density_fields", ()):
            planned_label = " ".join(extract_words(df.label))
            if _singular_label(planned_label) == singular:
                return df.region
    return None


def plan_edit(
    *,
    intent: PromptIntent | None = None,
    prompt: str | None = None,
    instruction_index: int = 0,
    prompt_embedding: torch.Tensor,
    source_image_embedding: torch.Tensor,
    user_mask: torch.Tensor | None = None,
    attention_capture: AttentionCapture | None = None,
    scene_facts: dict[str, int] | None = None,
    tokenizer=None,
    base_guidance_scale: float = 7.5,
    config: CoefficientConfig | None = None,
    allow_clarification: bool = True,
    feather_radius: int = 2,
    dilate_radius: int = 1,
    latent_size: tuple[int, int] = (64, 64),
    aspect: float | None = None,
    instruction: EditInstruction | None = None,
    semantic_plan: SemanticLayoutPlan | None = None,
) -> EditPlan:
    """Resolve one PromptIntent instruction into denoising parameters."""
    if intent is None:
        if prompt is None:
            raise ValueError("Pass either intent= or prompt=")
        from app.services.editing.prompt_intent import analyze_prompt

        intent = analyze_prompt(prompt, mode="edit")

    if not isinstance(intent, PromptIntent):
        raise TypeError("plan_edit requires a PromptIntent")
    if intent.mode != "edit":
        raise ValueError("plan_edit requires an edit-mode PromptIntent")
    if intent.status == "clarify" and allow_clarification:
        raise ValueError("cannot plan an edit intent that requires clarification")
    if not intent.instructions:
        raise ValueError("edit intent contains no instructions")
    if (
        not isinstance(instruction_index, int)
        or instruction_index < 0
        or instruction_index >= len(intent.instructions)
    ):
        raise ValueError(
            f"instruction_index {instruction_index!r} is out of bounds for "
            f"{len(intent.instructions)} instruction(s)"
        )

    instruction = intent.instructions[instruction_index]
    prompt_str = instruction.raw_text
    config = config or CoefficientConfig()
    terms = select_edit_terms(instruction)
    token_indices = locate_edit_tokens(prompt_str, terms, tokenizer)
    token_roles = align_token_roles(instruction, tokenizer=tokenizer)

    resolved_aspect = (
        float(aspect)
        if aspect is not None
        else (latent_size[1] / max(latent_size[0], 1) if latent_size else 1.0)
    )

    # 1. Where to edit: explicit user mask wins; then a layout object matching
    #    the structured target/resolution/nouns;
    #    otherwise ask captured cross-attention; fallback to global.
    mask_source: Literal["user", "attention", "semantic_plan", "global_fallback"]
    resolved_user_mask = as_soft_mask(user_mask) if user_mask is not None else None
    if resolved_user_mask is not None and area_ratio(resolved_user_mask) == 0.0:
        logger.warning(
            "plan_edit received an empty user mask for %r; falling back to a global "
            "edit instead of denoising into a fully-masked region.",
            prompt_str[:80],
        )
        resolved_user_mask = None

    semantic_box = (
        _semantic_target_box(semantic_plan, instruction) if semantic_plan is not None else None
    )

    if resolved_user_mask is not None:
        from app.services.editing.masks import resize_mask

        mask = resize_mask(resolved_user_mask, *latent_size)
        mask_source = "user"
    elif semantic_box is not None:
        mask = semantic_box.to_mask(*latent_size)
        mask_source = "semantic_plan"
    else:
        inferred = (
            attention_capture.token_mask(list(token_indices), aspect=resolved_aspect)
            if attention_capture is not None and token_indices
            else None
        )
        if inferred is not None and area_ratio(inferred) > 0.0:
            mask = inferred
            mask_source = "attention"
        else:
            # No region evidence: edit everything rather than silently editing nothing.
            mask = torch.ones(1, 1, *latent_size)
            mask_source = "global_fallback"

    if mask_source != "global_fallback":
        mask = feather(dilate(mask, dilate_radius), feather_radius)

    scope = classify_scope(
        instruction,
        mask,
        mask_is_explicit=mask_source in ("user", "semantic_plan"),
    )

    # 2. How much to change, and how hard to hold the rest of the frame.
    coefficients = compute_adaptive_reference_coefficient(
        prompt_embedding=prompt_embedding,
        source_image_embedding=source_image_embedding,
        edit_region_mask=mask,
        base_guidance_scale=base_guidance_scale,
        config=config,
    )

    # 3. Is this prompt even applicable to this image?
    alignment = check_prompt_image_alignment(
        prompt=prompt_str,
        prompt_embedding=prompt_embedding,
        source_image_embedding=source_image_embedding,
        edit_region_mask=mask,
        scene_facts=scene_facts,
        config=config,
        allow_clarification=allow_clarification,
    )

    # A global edit must not clamp the prompt out of the frame it is meant to restyle.
    attention_strength = {"local": 1.0, "regional": 0.6, "global": 0.0}[scope]

    return EditPlan(
        prompt=prompt_str,
        scope=scope,
        mask=mask,
        mask_source=mask_source,
        edit_terms=terms,
        edit_token_indices=token_indices,
        token_roles=token_roles,
        coefficients=coefficients,
        alignment=alignment,
        instruction=instruction,
        instruction_index=instruction_index,
        attention_strength=attention_strength,
        bounding_box=bounding_box(mask),
        semantic_plan=semantic_plan,
        aspect=resolved_aspect,
    )
