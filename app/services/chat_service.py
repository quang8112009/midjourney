"""Session-aware orchestration for conversational image requests."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.services.chat_provider import ModelMessage
from app.services.conversation_store import (
    ConversationStore,
    ConversationTurn,
)
from app.services.reasoning_service import ReasonedTurn, ReasoningService

GenerationCallback = Callable[[str], Awaitable[tuple[Any, str]]]


@dataclass(frozen=True, slots=True)
class ChatTurnOutcome:
    session_id: str
    user_turn: ConversationTurn
    assistant_turn: ConversationTurn
    reasoning: ReasonedTurn
    generation: Any | None


class ChatService:
    """Serialize each session's read/reason/generate/append transaction."""

    def __init__(
        self,
        *,
        store: ConversationStore,
        reasoning_service: ReasoningService,
    ):
        self.store = store
        self.reasoning_service = reasoning_service
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = threading.Lock()

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        with self._locks_guard:
            lock = self._session_locks.setdefault(session_id, asyncio.Lock())
            if len(self._session_locks) > self.store.max_sessions * 2:
                removable = [
                    key
                    for key, candidate in self._session_locks.items()
                    if key != session_id and not candidate.locked()
                ]
                for key in removable[: self.store.max_sessions]:
                    self._session_locks.pop(key, None)
            return lock

    async def handle_turn(
        self,
        *,
        message: str,
        session_id: str | None = None,
        generate: GenerationCallback | None = None,
    ) -> ChatTurnOutcome:
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must not be blank")

        resolved_session_id = self.store.create_session(session_id)
        async with self._session_lock(resolved_session_id):
            snapshot = self.store.get_snapshot(resolved_session_id)
            history = snapshot.turns if snapshot is not None else ()
            user_turn_id = uuid.uuid4().hex
            messages = [
                ModelMessage(
                    role=turn.role,
                    content=turn.content,
                    turn_id=turn.turn_id,
                )
                for turn in history
            ]
            if snapshot is not None and snapshot.last_generation_prompt:
                context_source_id = history[-1].turn_id if history else "session"
                messages.append(
                    ModelMessage(
                        role="assistant",
                        content=snapshot.last_generation_prompt,
                        turn_id=f"{context_source_id}:generation-prompt",
                        kind="generation_context",
                    )
                )
            messages.append(
                ModelMessage(
                    role="user",
                    content=normalized_message,
                    turn_id=user_turn_id,
                )
            )

            reasoned = await self.reasoning_service.reason(
                session_id=resolved_session_id,
                messages=messages,
            )
            generation = None
            effective_generation_prompt = None
            if reasoned.action == "generate_image":
                if generate is None or reasoned.generation_prompt is None:
                    raise RuntimeError("Image generation callback is unavailable.")
                generation, effective_generation_prompt = await generate(reasoned.generation_prompt)

            assistant_turn_id = uuid.uuid4().hex
            user_turn, assistant_turn = self.store.append_exchange(
                resolved_session_id,
                user_content=normalized_message,
                assistant_content=reasoned.public_response,
                user_turn_id=user_turn_id,
                assistant_turn_id=assistant_turn_id,
                last_generation_prompt=effective_generation_prompt,
            )
            return ChatTurnOutcome(
                session_id=resolved_session_id,
                user_turn=user_turn,
                assistant_turn=assistant_turn,
                reasoning=reasoned,
                generation=generation,
            )

    def delete_session(self, session_id: str) -> bool:
        deleted = self.store.delete_session(session_id)
        with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is not None and not lock.locked():
                self._session_locks.pop(session_id, None)
        return deleted

    async def aclose(self) -> None:
        await self.reasoning_service.aclose()
