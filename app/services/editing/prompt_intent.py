"""Structured prompt understanding, run before any denoising.

Turns a free-text edit request into a JSON-serialisable object the pipeline
consumes directly - target, action, attribute, scope, and a resolved region per
sub-instruction. Nothing downstream re-parses the raw prompt.

Deliberately lexical (regex + lexicons), not a model call: the whole stage runs in
well under a millisecond, so it cannot move end-to-end latency. That is also its
main limitation - see the docs.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

Action = Literal[
    "recolor", "remove", "add", "blur", "sharpen", "restyle",
    "lighten", "darken", "replace", "enhance", "resize", "unknown",
]
Scope = Literal["local", "global"]
IntentStatus = Literal["ok", "assumed", "clarify"]
PromptMode = Literal["generate", "edit"]

# Each action maps to the verbs that signal it. Order matters only for reporting;
# matching is longest-phrase-first so "take out" beats "take".
_ACTION_VERBS: dict[Action, tuple[str, ...]] = {
    "recolor": (
        "recolor", "recolour", "change the color", "change the colour",
        "make", "turn", "paint", "dye", "tint",
    ),
    "remove": ("remove", "delete", "erase", "get rid of", "take out", "clean up"),
    "add": ("add", "insert", "place", "put", "include"),
    "blur": ("blur", "defocus", "soften", "bokeh"),
    "sharpen": ("sharpen", "crisp", "focus"),
    "restyle": (
        "restyle", "convert", "render", "stylize", "stylise",
        "turn into", "make it look like",
    ),
    "lighten": ("lighten", "brighten", "light up"),
    "darken": ("darken", "dim"),
    "replace": ("replace", "swap", "substitute"),
    "enhance": ("enhance", "improve", "fix", "clean", "retouch", "better"),
    "resize": ("resize", "enlarge", "shrink", "scale"),
}

# "change"/"adjust" name no specific operation on their own; the attribute decides.
_GENERIC_VERBS = ("change", "alter", "modify", "adjust", "update", "edit")

_COLORS = frozenset(
    """red orange yellow green blue purple violet pink brown black white grey gray
    beige teal cyan magenta gold silver navy maroon turquoise crimson amber""".split()
)
_STYLES = frozenset(
    """watercolor watercolour oil sketch anime cartoon photorealistic realistic
    monochrome sepia grayscale greyscale vintage retro cyberpunk impressionist""".split()
)
_INTENSITIES = frozenset(
    "slightly subtly somewhat very much heavily strongly extremely".split()
)

# Targets that mean "the entire frame", not a region inside it.
_GLOBAL_TARGETS = frozenset(
    "image picture photo photograph everything all scene whole frame it this".split()
)
_POSITIONS = {
    "left": "left", "right": "right", "top": "top", "upper": "top",
    "bottom": "bottom", "lower": "bottom", "middle": "center", "center": "center",
    "centre": "center", "foreground": "foreground", "background": "background",
    "front": "foreground", "back": "background",
}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
}
# Clauses that constrain an edit rather than requesting a new one.
_CONSTRAINT_CUES = re.compile(
    r"\b(but|while|without|except|keep|keeping|preserve|preserving|maintain|leave|retain)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    """the a an of to in on at by for with from into its his her their my our your
    is are be been being do does did please can you it this that and or""".split()
)
# Prompts that request *something* but name neither target nor attribute.
_VAGUE = re.compile(
    r"^\s*(make|fix|improve|enhance|clean)\b.{0,30}$|^\s*(better|nicer|prettier)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SceneObject:
    """A candidate region from a detector/segmenter, used for disambiguation."""

    label: str
    center_x: float = 0.5
    center_y: float = 0.5
    area: float = 0.1
    salience: float | None = None

    @property
    def effective_salience(self) -> float:
        """Area stands in for salience when a detector supplies no score."""
        return self.area if self.salience is None else self.salience


@dataclass(frozen=True)
class TargetResolution:
    """Which concrete object an instruction refers to, and how that was decided."""

    label: str | None
    method: Literal["explicit", "position", "ordinal", "salience", "only_candidate", "unresolved"]
    index: int | None = None
    confidence: float = 1.0
    alternatives: tuple[str, ...] = ()
    matched_on: str | None = None
    """Which noun was matched against the scene - the target, or an owner noun."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EditInstruction:
    """One atomic edit. A multi-instruction prompt yields several of these."""

    raw_text: str
    action: Action
    target: str | None
    attribute: str | None
    scope: Scope
    position: str | None = None
    ordinal: int | None = None
    intensity: str | None = None
    constraints: tuple[str, ...] = ()
    nouns: tuple[str, ...] = ()
    confidence: float = 1.0
    resolution: TargetResolution | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["resolution"] = self.resolution.to_dict() if self.resolution else None
        return data


@dataclass(frozen=True)
class PromptIntent:
    """The structured object the denoising pipeline consumes."""

    prompt: str
    mode: PromptMode
    instructions: tuple[EditInstruction, ...]
    status: IntentStatus
    assumption: str | None = None
    clarifying_question: str | None = None
    trace: tuple[str, ...] = field(default=())

    @property
    def should_generate(self) -> bool:
        return self.status != "clarify"

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "mode": self.mode,
            "status": self.status,
            "assumption": self.assumption,
            "clarifying_question": self.clarifying_question,
            "instructions": [instruction.to_dict() for instruction in self.instructions],
            "trace": list(self.trace),
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def _normalise_word(word: str) -> str:
    """Strip possessives so "person's" matches a detector label of "person"."""
    return re.sub(r"'s$", "", word.strip("-'"))


def _match_action(text: str) -> tuple[Action, str | None]:
    """Longest verb phrase wins, so 'change the color' beats bare 'change'."""
    lowered = text.lower()
    best: tuple[Action, str] | None = None
    for action, verbs in _ACTION_VERBS.items():
        for verb in verbs:
            if re.search(rf"\b{re.escape(verb)}\b", lowered):
                if best is None or len(verb) > len(best[1]):
                    best = (action, verb)
    if best is None:
        for verb in _GENERIC_VERBS:
            if re.search(rf"\b{verb}\b", lowered):
                return "unknown", verb
        return "unknown", None
    return best[0], best[1]


def _resolve_generic_action(text: str, attribute: str | None) -> Action:
    """A bare 'change X to Y' becomes recolor/restyle/replace based on Y."""
    if attribute is not None:
        if attribute in _COLORS:
            return "recolor"
        if attribute in _STYLES:
            return "restyle"
        return "replace"
    if re.search(r"\b(colour|color|tone|shade)\b", text, re.IGNORECASE):
        return "recolor"
    return "unknown"


def split_instructions(prompt: str) -> list[str]:
    """Split a multi-instruction prompt into atomic requests.

    Splits on coordinators only when the following clause actually contains an
    action verb - so "red and blue stripes" stays one instruction while "recolor
    the shirt and blur the background" becomes two. Constraint clauses ("but keep
    the background") are never split off as edits.
    """
    parts = re.split(
        r"\s*(?:;|,\s*then\s+|\s+then\s+|\s+and\s+also\s+|\s+and\s+|,)\s*",
        prompt.strip(),
    )
    parts = [part for part in parts if part.strip()]
    if len(parts) <= 1:
        return [prompt.strip()] if prompt.strip() else []

    merged: list[str] = []
    for part in parts:
        action, _ = _match_action(part)
        is_constraint = bool(_CONSTRAINT_CUES.search(part)) and action == "unknown"
        # A fragment with no verb of its own continues the previous instruction.
        if merged and (action == "unknown" or is_constraint):
            merged[-1] = f"{merged[-1]} {part}".strip()
        else:
            merged.append(part.strip())
    return merged or [prompt.strip()]


def _extract_target(text: str, verb: str | None) -> str | None:
    """The noun the action applies to: the first content word after the verb."""
    lowered = text.lower()
    if verb:
        position = lowered.find(verb)
        if position >= 0:
            lowered = lowered[position + len(verb):]
    words = [_normalise_word(word) for word in re.findall(r"[a-z][a-z'-]+", lowered)]

    def candidates(allow_positions: bool):
        for word in words:
            if word in _STOPWORDS or word in _COLORS or word in _STYLES:
                continue
            if word in _ORDINALS or word in _INTENSITIES or word in _GENERIC_VERBS:
                continue
            if word in ("color", "colour", "tone", "shade"):
                continue
            if not allow_positions and word in _POSITIONS:
                continue
            yield word

    # A possessive names the owner first and the edited thing second: in
    # "the person's shirt" the mask belongs on the shirt, not the person.
    possessive = re.search(r"\b[a-z][a-z-]*'s\s+([a-z][a-z-]+)", lowered)
    if possessive and possessive.group(1) not in _STOPWORDS:
        return possessive.group(1)

    # "the person on the left" -> person; but "blur the background" -> background,
    # so position words are only skipped while a better noun is still available.
    return next(candidates(False), None) or next(candidates(True), None)


def _extract_nouns(text: str) -> tuple[str, ...]:
    """Every plausible noun, so disambiguation can match an owner ('person') that
    differs from the edit target ('shirt')."""
    words = [_normalise_word(word) for word in re.findall(r"[a-z][a-z'-]+", text.lower())]
    return tuple(
        word
        for word in words
        if word not in _STOPWORDS
        and word not in _COLORS
        and word not in _STYLES
        and word not in _ORDINALS
        and word not in _INTENSITIES
        and word not in _GENERIC_VERBS
        and not any(word in verbs for verbs in _ACTION_VERBS.values())
    )


def _extract_attribute(text: str, action: Action) -> str | None:
    """The specific detail: a colour, a style, or the word after 'to'/'into'."""
    lowered = text.lower()
    for color in _COLORS:
        if re.search(rf"\b{color}\b", lowered):
            return color
    for style in _STYLES:
        if re.search(rf"\b{style}\b", lowered):
            return style
    match = re.search(r"\b(?:to|into)\s+(?:a\s+|an\s+|the\s+)?([a-z][a-z'-]+)", lowered)
    if match and match.group(1) not in _STOPWORDS:
        return match.group(1)
    return None


def _extract_scope(target: str | None, text: str, action: Action) -> Scope:
    if target is None:
        return "global"
    if target in _GLOBAL_TARGETS:
        return "global"
    # A whole-frame restyle is global even when a subject is named.
    if action == "restyle" and re.search(r"\b(whole|entire|everything)\b", text, re.IGNORECASE):
        return "global"
    return "local"


def parse_instruction(text: str) -> EditInstruction:
    """Decompose one atomic request into target / action / attribute / scope."""
    action, verb = _match_action(text)
    lowered = text.lower()

    position = None
    for word, canonical in _POSITIONS.items():
        if re.search(rf"\b{word}\b", lowered):
            position = canonical
            break
    ordinal = None
    for word, value in _ORDINALS.items():
        if re.search(rf"\b{word}\b", lowered):
            ordinal = value
            break
    intensity = next((word for word in _INTENSITIES if re.search(rf"\b{word}\b", lowered)), None)

    constraints = tuple(
        clause.strip()
        for clause in re.split(_CONSTRAINT_CUES, text)[1:]
        if clause and clause.strip() and not _CONSTRAINT_CUES.fullmatch(clause.strip())
    )

    target = _extract_target(text, verb)
    attribute = _extract_attribute(text, action)
    if action == "unknown":
        action = _resolve_generic_action(text, attribute)
    scope = _extract_scope(target, text, action)
    nouns = _extract_nouns(text)

    # Confidence is a reportable signal, not a threshold to hide behind.
    confidence = 1.0
    if action == "unknown":
        confidence -= 0.4
    if target is None:
        confidence -= 0.3
    if attribute is None and action in ("recolor", "replace", "restyle"):
        confidence -= 0.2
    return EditInstruction(
        raw_text=text.strip(),
        action=action,
        target=target,
        attribute=attribute,
        scope=scope,
        position=position,
        ordinal=ordinal,
        intensity=intensity,
        constraints=constraints,
        nouns=nouns,
        confidence=max(confidence, 0.0),
    )


def resolve_target(
    instruction: EditInstruction,
    candidates: list[SceneObject] | None,
    *,
    salience_margin: float = 0.15,
) -> TargetResolution:
    """Pick which concrete object an instruction means.

    Order: an explicit position or ordinal in the prompt wins; otherwise the most
    salient candidate, but only when it wins clearly. A close call returns
    `unresolved` so the caller can ask instead of guessing.
    """
    if instruction.target is None:
        return TargetResolution(label=None, method="unresolved", confidence=0.0)
    if not candidates:
        # No detector output: trust the prompt's own noun.
        return TargetResolution(label=instruction.target, method="explicit")

    def matches(noun: str):
        return [
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if candidate.label.lower() == noun.lower()
        ]

    matched_on = instruction.target
    matching = matches(instruction.target)
    if not matching:
        # The edit target may be a part ("shirt") of an owner the detector knows
        # ("person"); disambiguate on whichever noun the scene actually contains.
        for noun in instruction.nouns:
            matching = matches(noun)
            if matching:
                matched_on = noun
                break
    if not matching:
        return TargetResolution(label=instruction.target, method="explicit", confidence=0.6)
    if len(matching) == 1:
        index, candidate = matching[0]
        return TargetResolution(
            label=candidate.label, method="only_candidate", index=index, matched_on=matched_on
        )

    if instruction.position:
        chooser = {
            "left": lambda item: item[1].center_x,
            "right": lambda item: -item[1].center_x,
            "top": lambda item: item[1].center_y,
            "bottom": lambda item: -item[1].center_y,
            "center": lambda item: abs(item[1].center_x - 0.5) + abs(item[1].center_y - 0.5),
            "foreground": lambda item: -item[1].area,
            "background": lambda item: item[1].area,
        }.get(instruction.position)
        if chooser is not None:
            index, candidate = min(matching, key=chooser)
            return TargetResolution(
                label=candidate.label, method="position", index=index, matched_on=matched_on
            )

    if instruction.ordinal is not None:
        ordered = sorted(matching, key=lambda item: item[1].center_x)
        if 1 <= instruction.ordinal <= len(ordered):
            index, candidate = ordered[instruction.ordinal - 1]
            return TargetResolution(
                label=candidate.label, method="ordinal", index=index, matched_on=matched_on
            )

    ranked = sorted(matching, key=lambda item: -item[1].effective_salience)
    top, runner_up = ranked[0], ranked[1]
    margin = top[1].effective_salience - runner_up[1].effective_salience
    if margin >= salience_margin:
        return TargetResolution(
            label=top[1].label, method="salience", index=top[0],
            confidence=min(1.0, 0.5 + margin), matched_on=matched_on,
            alternatives=tuple(candidate.label for _, candidate in ranked[1:]),
        )
    return TargetResolution(
        label=instruction.target, method="unresolved", confidence=0.0,
        alternatives=tuple(candidate.label for _, candidate in ranked),
        matched_on=matched_on,
    )


def analyze_prompt(
    prompt: str,
    *,
    mode: PromptMode = "edit",
    candidates: list[SceneObject] | None = None,
    image_type: str | None = None,
    main_subject: str | None = None,
    allow_clarification: bool = True,
) -> PromptIntent:
    """Build the single prompt contract used by generation and editing planners.

    Generation intents deliberately carry no edit instructions: semantic layout
    planning reads their prompt, while only edit mode runs action/target parsing.
    """
    trace: list[str] = []
    cleaned = prompt.strip()
    if mode not in ("generate", "edit"):
        raise ValueError(f"Unsupported prompt mode: {mode!r}")

    if mode == "generate":
        if not cleaned:
            return PromptIntent(
                prompt=prompt,
                mode=mode,
                instructions=(),
                status="clarify",
                clarifying_question="What would you like me to create?",
                trace=("empty generation prompt",),
            )
        return PromptIntent(
            prompt=prompt,
            mode=mode,
            instructions=(),
            status="ok",
            trace=("generation prompt; edit decomposition skipped",),
        )

    if not cleaned:
        return PromptIntent(
            prompt=prompt, mode=mode, instructions=(), status="clarify",
            clarifying_question="What would you like me to change?",
            trace=("empty prompt",),
        )

    texts = split_instructions(cleaned)
    trace.append(f"split into {len(texts)} instruction(s): {texts}")
    instructions = [parse_instruction(text) for text in texts]

    resolved: list[EditInstruction] = []
    unresolved: list[EditInstruction] = []
    for instruction in instructions:
        resolution = resolve_target(instruction, candidates)
        instruction = replace(instruction, resolution=resolution)
        trace.append(
            f"{instruction.raw_text!r} -> action={instruction.action} "
            f"target={instruction.target} attribute={instruction.attribute} "
            f"scope={instruction.scope} resolved_by={resolution.method}"
        )
        (unresolved if resolution.method == "unresolved" else resolved).append(instruction)

    instructions_tuple = tuple(resolved + unresolved)

    # Several equally-plausible objects: ask rather than pick one at random.
    ambiguous = [
        item for item in instructions_tuple
        if item.resolution and item.resolution.method == "unresolved"
    ]
    if ambiguous and candidates:
        resolution = ambiguous[0].resolution
        # Name the noun that is actually ambiguous (the owner), not the edit target.
        target = resolution.matched_on or ambiguous[0].target
        options = resolution.alternatives
        if allow_clarification:
            return PromptIntent(
                prompt=prompt, mode=mode, instructions=instructions_tuple, status="clarify",
                clarifying_question=(
                    f"There are {len(options)} matches for '{target}' - "
                    "which one should I edit?"
                ),
                trace=tuple(trace + [f"ambiguous target {target!r} among {list(options)}"]),
            )
        return PromptIntent(
            prompt=prompt, mode=mode, instructions=instructions_tuple, status="assumed",
            assumption=(
                f"Editing the {ambiguous[0].target} of the most prominent '{target}'."
            ),
            trace=tuple(trace + ["ambiguous target; defaulted to most salient"]),
        )

    # "change the shirt" names a target but no change. Proceeding would invent an
    # edit the user never asked for, so treat it like any other severe ambiguity.
    undefined = [
        item
        for item in instructions_tuple
        if item.action == "unknown" and item.attribute is None and item.target is not None
    ]
    if undefined:
        first = undefined[0]
        trace.append(f"no operation specified for target {first.target!r}")
        if allow_clarification:
            return PromptIntent(
                prompt=prompt, mode=mode, instructions=instructions_tuple, status="clarify",
                clarifying_question=(
                    f"What should I change about the {first.target}?"
                ),
                trace=tuple(trace),
            )
        return PromptIntent(
            prompt=prompt, mode=mode, instructions=instructions_tuple, status="assumed",
            assumption=(
                f"No specific change was given for the {first.target}; "
                "applying a general enhancement to it."
            ),
            trace=tuple(trace),
        )

    # Underspecified: reason from what the image is rather than guessing blindly.
    vague = _VAGUE.match(cleaned) and all(
        i.target is None or i.target in _GLOBAL_TARGETS for i in instructions_tuple
    )
    if vague:
        subject = main_subject or "the main subject"
        kind = image_type or "photo"
        assumption = (
            f"'{cleaned}' is underspecified; treating it as a global enhancement of "
            f"this {kind}, focused on {subject}."
        )
        trace.append(f"vague prompt; assumption from image context ({kind}, {subject})")
        if image_type is None and main_subject is None and allow_clarification:
            return PromptIntent(
                prompt=prompt, mode=mode, instructions=instructions_tuple, status="clarify",
                clarifying_question=(
                    "What would you like improved - lighting, colour, or sharpness?"
                ),
                trace=tuple(trace + ["no image context available to ground an assumption"]),
            )
        return PromptIntent(
            prompt=prompt, mode=mode, instructions=instructions_tuple, status="assumed",
            assumption=assumption, trace=tuple(trace),
        )

    return PromptIntent(
        prompt=prompt,
        mode=mode,
        instructions=instructions_tuple,
        status="ok",
        trace=tuple(trace),
    )
