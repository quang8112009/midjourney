"""Two-pass conversational reasoning pipeline and orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.chat_prompts import (
    REASONING_PROMPT_VERSION,
    REASONING_SYSTEM_PROMPT,
    build_response_system_prompt,
)
from app.services.chat_provider import ChatProvider, ModelMessage

trace_logger = logging.getLogger("uvicorn.error")

IntentKind = Literal[
    "generate_image",
    "refine_image",
    "question",
    "capability",
    "other",
]
AmbiguityHandling = Literal["clear", "assume", "clarify"]
ConstraintStatus = Literal["allowed", "unsupported", "refuse"]
ReasoningAction = Literal["generate_image", "respond", "clarify", "refuse"]


class AmbiguityAnalysis(BaseModel):
    """Structured ambiguity assessment."""

    model_config = ConfigDict(extra="forbid")

    is_ambiguous: bool = False
    assumption: str | None = None
    clarifying_question: str | None = None

    @field_validator("assumption", "clarifying_question", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ("none", "n/a", "null", "no"):
            return None
        return cleaned


class ReasoningAnalysis(BaseModel):
    """Structured output from Pass 1 (Reasoning Analysis)."""

    model_config = ConfigDict(extra="forbid")

    intent: str = Field(min_length=1, max_length=1_000)
    context_notes: str = "none"
    ambiguity: AmbiguityAnalysis = Field(default_factory=AmbiguityAnalysis)
    constraints: str = "none"
    response_plan: str = Field(min_length=1, max_length=2_000)
    action: ReasoningAction = "respond"
    generation_prompt: str | None = Field(default=None, max_length=2_000)
    raw_xml: str | None = None

    @field_validator("intent", "context_notes", "constraints", "response_plan", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> str:
        return value.strip() if isinstance(value, str) else str(value)

    @field_validator("generation_prompt", mode="before")
    @classmethod
    def normalize_prompt(cls, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ("none", "n/a", "null"):
            return None
        return cleaned


class PipelineTiming(BaseModel):
    """Timing metadata for observability across both pipeline passes."""

    reasoning_ms: float | None = None
    response_ms: float = 0.0
    total_ms: float = 0.0


class ReasoningOutputError(RuntimeError):
    """Raised when provider output cannot safely satisfy the contract."""


class ReasoningParseError(ReasoningOutputError):
    """Raised when structured reasoning tags cannot be parsed."""


class _SinglePassConfigured(BaseException):
    """Internal signal that the reasoning pass is disabled by configuration."""


def _extract_tag_content(xml_text: str, tag_name: str) -> str | None:
    """Extract inner content of an XML tag, case-insensitively and handling whitespace."""
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, xml_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def parse_reasoning_xml(raw_text: str) -> ReasoningAnalysis:
    """Parse Pass 1 XML output into a typed ReasoningAnalysis model."""
    text = raw_text.strip()
    # Strip markdown code fencing if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:xml|json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # If the text is JSON (e.g. from structured json response), parse as JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if "intent" in data or "intent_kind" in data or "user_goal" in data:
                intent = data.get("intent") or data.get("user_goal") or "User request"
                context_notes = data.get("context_notes") or "none"
                if isinstance(data.get("relevant_context"), list):
                    context_notes = "; ".join(data["relevant_context"]) or "none"

                ambiguity_val = data.get("ambiguity")
                if isinstance(ambiguity_val, dict):
                    ambiguity = AmbiguityAnalysis(**ambiguity_val)
                else:
                    is_ambig = data.get("ambiguity_handling") in ("assume", "clarify") or bool(
                        data.get("is_ambiguous")
                    )
                    ambiguity = AmbiguityAnalysis(
                        is_ambiguous=is_ambig,
                        assumption=data.get("assumption"),
                        clarifying_question=data.get("clarifying_question"),
                    )

                response_plan = data.get("response_plan")
                if isinstance(response_plan, list):
                    response_plan = "; ".join(response_plan)
                elif not response_plan:
                    response_plan = "Respond clearly"

                action = data.get("action", "respond")
                if action not in ("respond", "generate_image", "clarify", "refuse"):
                    action = "respond"

                return ReasoningAnalysis(
                    intent=intent,
                    context_notes=context_notes,
                    ambiguity=ambiguity,
                    constraints=data.get("constraints") or data.get("constraint_summary") or "none",
                    response_plan=str(response_plan),
                    action=action,
                    generation_prompt=data.get("generation_prompt"),
                    raw_xml=raw_text,
                )
        except Exception:
            pass

    # Find reasoning block
    reasoning_match = re.search(
        r"<reasoning>(.*?)</reasoning>", text, flags=re.DOTALL | re.IGNORECASE
    )
    block = reasoning_match.group(1).strip() if reasoning_match else text

    intent = _extract_tag_content(block, "intent")
    if not intent:
        raise ReasoningParseError("Could not extract <intent> from reasoning output.")

    context_notes = _extract_tag_content(block, "context_notes") or "none"
    constraints = _extract_tag_content(block, "constraints") or "none"
    response_plan = _extract_tag_content(block, "response_plan") or "Respond clearly to the user."

    # Parse ambiguity block or inner tags
    ambiguity_block = _extract_tag_content(block, "ambiguity")
    search_scope = ambiguity_block if ambiguity_block is not None else block

    is_ambig_text = _extract_tag_content(search_scope, "is_ambiguous")
    is_ambiguous = False
    if is_ambig_text:
        is_ambiguous = is_ambig_text.lower() in ("true", "yes", "1")

    assumption = _extract_tag_content(search_scope, "assumption")
    if assumption and assumption.strip().lower() in ("none", "n/a", "null", "no"):
        assumption = None

    clarifying_question = _extract_tag_content(search_scope, "clarifying_question")
    if clarifying_question and clarifying_question.strip().lower() in ("none", "n/a", "null", "no"):
        clarifying_question = None

    ambiguity = AmbiguityAnalysis(
        is_ambiguous=is_ambiguous or bool(clarifying_question or assumption),
        assumption=assumption,
        clarifying_question=clarifying_question,
    )

    action_text = _extract_tag_content(block, "action")
    action: ReasoningAction = "respond"
    if action_text:
        cleaned_action = action_text.lower().strip()
        if cleaned_action in ("respond", "generate_image", "clarify", "refuse"):
            action = cleaned_action  # type: ignore[assignment]
    elif clarifying_question:
        action = "clarify"

    generation_prompt = _extract_tag_content(block, "generation_prompt")
    if generation_prompt and generation_prompt.strip().lower() in ("none", "n/a", "null"):
        generation_prompt = None
    if action == "generate_image" and not generation_prompt:
        generation_prompt = intent

    return ReasoningAnalysis(
        intent=intent,
        context_notes=context_notes,
        ambiguity=ambiguity,
        constraints=constraints,
        response_plan=response_plan,
        action=action,
        generation_prompt=generation_prompt,
        raw_xml=raw_text,
    )


@dataclass(frozen=True, slots=True)
class ReasonedTurn:
    """Outcome of the two-pass reasoning pipeline."""

    trace_id: str
    action: ReasoningAction
    public_response: str
    generation_prompt: str | None
    fallback_used: bool
    fallback_reason: str | None
    reasoning_analysis: ReasoningAnalysis | None = None
    timing: PipelineTiming = field(default_factory=PipelineTiming)


class ReasoningService:
    """Two-pass reasoning pipeline: Pass 1 (Analysis) -> Pass 2 (Response Generation)."""

    _PRIVATE_OUTPUT_MARKERS = (
        "<reasoning",
        "</reasoning>",
        "<hidden_reasoning_analysis>",
        "</hidden_reasoning_analysis>",
        "<hidden_guidance>",
        "</hidden_guidance>",
        "<analysis>",
        "</analysis>",
    )

    def __init__(
        self,
        provider: ChatProvider,
        *,
        two_pass_enabled: bool = True,
        reasoning_timeout_seconds: float = 4.0,
        response_timeout_seconds: float = 12.0,
        max_reasoning_tokens: int = 500,
        max_response_tokens: int = 1_200,
        max_history_turns: int = 8,
        max_generation_prompt_length: int = 2_000,
        reasoning_temperature: float | None = 0.1,
        response_temperature: float | None = 0.7,
        log_decision_content: bool = False,
        # Backward compatibility alias
        timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
    ):
        self.provider = provider
        self.two_pass_enabled = two_pass_enabled
        self.reasoning_timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else reasoning_timeout_seconds
        )
        self.response_timeout_seconds = response_timeout_seconds
        self.max_reasoning_tokens = max_reasoning_tokens
        self.max_response_tokens = (
            max_output_tokens if max_output_tokens is not None else max_response_tokens
        )
        self.max_history_turns = max_history_turns
        self.max_generation_prompt_length = max_generation_prompt_length
        self.reasoning_temperature = reasoning_temperature
        self.response_temperature = response_temperature
        self.log_decision_content = log_decision_content

    async def reason(
        self,
        *,
        session_id: str,
        messages: Sequence[ModelMessage],
    ) -> ReasonedTurn:
        """Run the two-pass pipeline: Pass 1 analysis with timeout, then Pass 2 response."""
        if not messages or messages[-1].role != "user":
            raise ValueError("messages must end with the current user turn")

        trace_id = uuid.uuid4().hex
        total_started = time.perf_counter()
        analysis: ReasoningAnalysis | None = None
        fallback_used = False
        fallback_reason: str | None = None
        reasoning_ms: float | None = None

        # -------------------------------------------------------------
        # PASS 1: Fast Reasoning Pass
        # -------------------------------------------------------------
        reasoning_history = list(messages[-self.max_history_turns :])
        reasoning_start = time.perf_counter()
        try:
            if not self.two_pass_enabled:
                # Configured single-pass mode: no analysis is produced, so Pass 2
                # runs with the plain direct-reply prompt. This is a deliberate
                # setting, not a failure, so fallback_used stays False.
                raise _SinglePassConfigured
            raw_reasoning = await asyncio.wait_for(
                self.provider.complete(
                    instructions=REASONING_SYSTEM_PROMPT,
                    messages=reasoning_history,
                    output_schema=None,
                    timeout_seconds=self.reasoning_timeout_seconds,
                    max_output_tokens=self.max_reasoning_tokens,
                    temperature=self.reasoning_temperature,
                ),
                timeout=self.reasoning_timeout_seconds,
            )
            reasoning_ms = round((time.perf_counter() - reasoning_start) * 1_000, 2)
            analysis = parse_reasoning_xml(raw_reasoning)
        except _SinglePassConfigured:
            analysis = None
            reasoning_ms = None
        except (Exception, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            reasoning_ms = round((time.perf_counter() - reasoning_start) * 1_000, 2)
            fallback_used = True
            fallback_reason = type(exc).__name__
            trace_logger.warning(
                "Reasoning pass failed/timed out (reason=%s, trace_id=%s). "
                "Falling back to direct response pass.",
                fallback_reason,
                trace_id,
            )
            analysis = None

        # -------------------------------------------------------------
        # PASS 2: Response Pass (Guided by hidden reasoning context)
        # -------------------------------------------------------------
        response_start = time.perf_counter()
        response_instructions = build_response_system_prompt(analysis)
        public_response = ""
        response_ms = 0.0
        used_local_fallback = False

        try:
            raw_response = await asyncio.wait_for(
                self.provider.complete(
                    instructions=response_instructions,
                    messages=messages,
                    output_schema=None,
                    timeout_seconds=self.response_timeout_seconds,
                    max_output_tokens=self.max_response_tokens,
                    temperature=self.response_temperature,
                ),
                timeout=self.response_timeout_seconds,
            )
            response_ms = round((time.perf_counter() - response_start) * 1_000, 2)
            public_response = raw_response.strip()
            if not public_response:
                raise ReasoningOutputError("Response pass returned empty text.")
            self._ensure_public_only(public_response)
        except (Exception, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            response_ms = round((time.perf_counter() - response_start) * 1_000, 2)
            local_reason = f"{fallback_reason or 'ok'}+{type(exc).__name__}"
            fallback_used = True
            fallback_reason = local_reason
            used_local_fallback = True
            trace_logger.warning(
                "Response pass failed (reason=%s, trace_id=%s). Using stable local fallback.",
                local_reason,
                trace_id,
            )
            # Safe local fallback ensuring the user never receives a 500
            public_response = (
                "I couldn't process the conversational context right now. "
                "What image or topic would you like me to assist you with?"
            )

        # A clarification must be exactly one targeted question. Pass 2 phrases it so
        # it fits the conversation, but a chatty model can stack several; falling back
        # to the analysis's own question keeps the guarantee structural rather than
        # dependent on the response prompt being obeyed.
        awaiting_clarification = analysis is not None and analysis.action == "clarify"
        analysis_question = analysis.ambiguity.clarifying_question if analysis else None
        if (
            awaiting_clarification
            and analysis_question
            and not used_local_fallback
            and public_response.count("?") > 1
        ):
            public_response = analysis_question

        # The application states the assumption so the wording stays consistent across
        # turns. Only a turn that is actually asking for clarification suppresses it:
        # an analysis that assumed *and* noted a question still owes the user the
        # assumption, otherwise it becomes a silent one.
        assumption = analysis.ambiguity.assumption if analysis is not None else None
        if assumption and not used_local_fallback and not awaiting_clarification:
            if not public_response.lower().startswith("assumption:"):
                public_response = f"Assumption: {assumption}\n\n{public_response}"

        # Resolve action and generation prompt
        if analysis is not None:
            action = analysis.action
            generation_prompt = (
                analysis.generation_prompt if analysis.action == "generate_image" else None
            )
            if generation_prompt and len(generation_prompt) > self.max_generation_prompt_length:
                truncated = generation_prompt[: self.max_generation_prompt_length]
                head, _, _ = truncated.rpartition(" ")
                generation_prompt = head or truncated
            if used_local_fallback:
                # The safety net apologises, so do not also generate an image.
                action = "clarify"
                generation_prompt = None
        else:
            action = "clarify" if used_local_fallback else "respond"
            generation_prompt = None

        total_ms = round((time.perf_counter() - total_started) * 1_000, 2)
        timing = PipelineTiming(
            reasoning_ms=reasoning_ms,
            response_ms=response_ms,
            total_ms=total_ms,
        )

        self._log_trace(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=messages[-1].turn_id,
            analysis=analysis,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            timing=timing,
        )

        return ReasonedTurn(
            trace_id=trace_id,
            action=action,
            public_response=public_response,
            generation_prompt=generation_prompt,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            reasoning_analysis=analysis,
            timing=timing,
        )

    def _ensure_public_only(self, response: str) -> None:
        """Ensure that private tags or hidden reasoning markers never reach public output."""
        normalized = response.lower()
        for marker in self._PRIVATE_OUTPUT_MARKERS:
            if marker in normalized:
                raise ReasoningOutputError(
                    "Potential private reasoning content in public response."
                )

    def _log_trace(
        self,
        *,
        trace_id: str,
        session_id: str,
        turn_id: str,
        analysis: ReasoningAnalysis | None,
        fallback_used: bool,
        fallback_reason: str | None,
        timing: PipelineTiming,
    ) -> None:
        trace = {
            "event": "chat_reasoning_trace",
            "trace_id": trace_id,
            "session_hash": hashlib.sha256(session_id.encode()).hexdigest()[:12],
            "turn_id": turn_id,
            "prompt_version": REASONING_PROMPT_VERSION,
            "two_pass_enabled": self.two_pass_enabled,
            "provider": self.provider.provider_name,
            "model": self.provider.model_name,
            "reasoning_latency_ms": timing.reasoning_ms,
            "response_latency_ms": timing.response_ms,
            "latency_ms": timing.total_ms,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        }
        if analysis is not None:
            trace.update(
                {
                    "is_ambiguous": analysis.ambiguity.is_ambiguous,
                    "has_context_notes": analysis.context_notes.strip().lower() != "none",
                    "has_assumption": bool(analysis.ambiguity.assumption),
                    "has_clarifying_question": bool(analysis.ambiguity.clarifying_question),
                    "action": analysis.action,
                }
            )
            if self.log_decision_content:
                trace["analysis_summary"] = {
                    "intent": analysis.intent,
                    "context_notes": analysis.context_notes,
                    "assumption": analysis.ambiguity.assumption,
                    "clarifying_question": analysis.ambiguity.clarifying_question,
                    "constraints": analysis.constraints,
                    "response_plan": analysis.response_plan,
                }
        trace_logger.info("%s", json.dumps(trace, sort_keys=True))

    async def aclose(self) -> None:
        await self.provider.aclose()
