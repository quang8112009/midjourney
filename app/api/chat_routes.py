"""Public conversational endpoint with hidden structured reasoning."""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.api.routes import (
    GenerateRequest,
    GenerateResponse,
    _client_key,
    _enforce_generation_rate_limit,
    execute_generation,
)
from app.core.config import settings
from app.core.dependencies import chat_service
from app.core.rate_limit import SlidingWindowRateLimiter
from app.services.model_service import MAX_SEED, STABLE_DIFFUSION

logger = logging.getLogger(__name__)
router = APIRouter()
chat_rate_limiter = SlidingWindowRateLimiter(settings.CHAT_RATE_LIMIT_PER_MINUTE)


class ChatGenerationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["stable-diffusion", "pixart-alpha"] = STABLE_DIFFUSION
    negative_prompt: str | None = Field(
        None,
        max_length=settings.MAX_NEGATIVE_PROMPT_LENGTH,
    )
    width: int | None = Field(None, ge=256, le=2048, multiple_of=8)
    height: int | None = Field(None, ge=256, le=2048, multiple_of=8)
    num_inference_steps: int | None = Field(None, ge=1, le=100)
    guidance_scale: float | None = Field(None, ge=1.0, le=20.0)
    num_images: int = Field(1, ge=1, le=4)
    seed: int | None = Field(None, ge=0, le=MAX_SEED)
    enhance_prompt: bool = False

    @field_validator("negative_prompt", mode="before")
    @classmethod
    def normalize_negative_prompt(cls, value):
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_generation_budget(self):
        try:
            GenerateRequest(prompt="chat option validation", **self.model_dump())
        except ValidationError as exc:
            message = exc.errors(include_url=False)[0]["msg"]
            raise ValueError(message) from exc
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: uuid.UUID | None = None
    message: str = Field(min_length=1, max_length=settings.CHAT_MAX_MESSAGE_LENGTH)
    generation: ChatGenerationOptions = Field(default_factory=ChatGenerationOptions)
    debug: bool = False

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value):
        return value.strip() if isinstance(value, str) else value


ChatStatus = Literal[
    "generated",
    "needs_clarification",
    "responded",
    "refused",
    "fallback_completed",
]


class TimingMetadata(BaseModel):
    reasoning_ms: float | None = None
    response_ms: float = 0.0
    total_ms: float = 0.0


class AmbiguityDebugInfo(BaseModel):
    is_ambiguous: bool
    assumption: str | None = None
    clarifying_question: str | None = None


class ReasoningDebugInfo(BaseModel):
    intent: str
    context_notes: str
    ambiguity: AmbiguityDebugInfo
    constraints: str
    response_plan: str
    action: str
    generation_prompt: str | None = None
    raw_xml: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    turn_id: str
    trace_id: str
    status: ChatStatus
    message: str
    generation: GenerateResponse | None = None
    timing: TimingMetadata | None = None
    reasoning_debug: ReasoningDebugInfo | None = None


def _enforce_chat_rate_limit(request: Request) -> None:
    retry_after = chat_rate_limiter.check(_client_key(request))
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Chat rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )


def _chat_status(action: str, fallback_used: bool) -> ChatStatus:
    if action == "generate_image":
        return "generated"
    if action == "clarify":
        return "needs_clarification"
    if action == "refuse":
        return "refused"
    return "fallback_completed" if fallback_used else "responded"


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    debug: bool | None = Query(None, description="Include raw reasoning output for debugging"),
):
    _enforce_chat_rate_limit(request)
    request_id = uuid.uuid4().hex
    # Reasoning output is internal, so the request flag is only honored when the
    # deployment itself enables debugging. In production it is silently ignored.
    is_debug = (payload.debug if debug is None else debug) and settings.DEBUG

    async def generate_image(generation_prompt: str):
        _enforce_generation_rate_limit(request)
        try:
            generation_payload = GenerateRequest(
                prompt=generation_prompt,
                **payload.generation.model_dump(),
            )
        except ValidationError as exc:
            logger.warning(
                "Reasoned prompt failed generation validation (request_id=%s)",
                request_id,
            )
            raise HTTPException(
                status_code=422,
                detail="The resolved image request is outside generation limits.",
            ) from exc
        generated = await execute_generation(generation_payload)
        return generated, generated.effective_prompt

    try:
        outcome = await chat_service.handle_turn(
            session_id=str(payload.session_id) if payload.session_id else None,
            message=payload.message,
            generate=generate_image,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected chat failure (request_id=%s)", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected chat failure. Request ID: {request_id}",
        ) from exc

    reasoning_turn = outcome.reasoning
    analysis = reasoning_turn.reasoning_analysis

    timing_meta: TimingMetadata | None = None
    reasoning_debug: ReasoningDebugInfo | None = None

    if is_debug:
        timing_meta = TimingMetadata(
            reasoning_ms=reasoning_turn.timing.reasoning_ms,
            response_ms=reasoning_turn.timing.response_ms,
            total_ms=reasoning_turn.timing.total_ms,
        )
        if analysis is not None:
            reasoning_debug = ReasoningDebugInfo(
                intent=analysis.intent,
                context_notes=analysis.context_notes,
                ambiguity=AmbiguityDebugInfo(
                    is_ambiguous=analysis.ambiguity.is_ambiguous,
                    assumption=analysis.ambiguity.assumption,
                    clarifying_question=analysis.ambiguity.clarifying_question,
                ),
                constraints=analysis.constraints,
                response_plan=analysis.response_plan,
                action=analysis.action,
                generation_prompt=analysis.generation_prompt,
                raw_xml=analysis.raw_xml,
                fallback_used=reasoning_turn.fallback_used,
                fallback_reason=reasoning_turn.fallback_reason,
            )
        else:
            reasoning_debug = ReasoningDebugInfo(
                intent="none",
                context_notes="none",
                ambiguity=AmbiguityDebugInfo(is_ambiguous=False),
                constraints="none",
                response_plan="Direct response fallback",
                action=reasoning_turn.action,
                generation_prompt=None,
                raw_xml=None,
                fallback_used=True,
                fallback_reason=reasoning_turn.fallback_reason,
            )

    return ChatResponse(
        session_id=outcome.session_id,
        turn_id=outcome.assistant_turn.turn_id,
        trace_id=outcome.reasoning.trace_id,
        status=_chat_status(
            outcome.reasoning.action,
            outcome.reasoning.fallback_used,
        ),
        message=outcome.reasoning.public_response,
        generation=outcome.generation,
        timing=timing_meta,
        reasoning_debug=reasoning_debug,
    )


@router.delete("/chat/sessions/{session_id}", status_code=204)
async def delete_chat_session(session_id: uuid.UUID):
    chat_service.delete_session(str(session_id))
    return Response(status_code=204)
