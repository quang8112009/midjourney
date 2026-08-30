"""Bounded, thread-safe, in-process storage for public conversation state."""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

ConversationRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One immutable, user-visible turn.

    The intentionally small shape prevents reasoning traces or other hidden
    model output from being mixed into conversational memory.
    """

    turn_id: str
    role: ConversationRole
    content: str


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    """An immutable point-in-time view of one conversation."""

    session_id: str
    turns: tuple[ConversationTurn, ...]
    last_generation_prompt: str | None


@dataclass(slots=True)
class _Conversation:
    turns: deque[ConversationTurn]
    history_chars: int
    last_generation_prompt: str | None
    touched_at: float


class ConversationStore:
    """Keep a bounded least-recently-used set of in-memory conversations.

    Sessions expire after ``ttl_seconds`` without a read or write. Public
    methods are safe to call from multiple worker threads, and snapshots never
    expose the mutable containers used internally.
    """

    def __init__(
        self,
        *,
        max_sessions: int = 1_000,
        max_messages_per_session: int = 12,
        max_history_chars: int = 12_000,
        ttl_seconds: float = 3_600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if max_messages_per_session < 2:
            raise ValueError("max_messages_per_session must be at least 2")
        if max_history_chars < 1:
            raise ValueError("max_history_chars must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        self.max_history_chars = max_history_chars
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._sessions: OrderedDict[str, _Conversation] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_identifier(value: str | uuid.UUID, field: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{field} must not be blank")
        if field == "session_id":
            try:
                return str(uuid.UUID(normalized))
            except ValueError:
                pass
        return normalized

    @staticmethod
    def _normalize_content(content: str, field: str) -> str:
        if not isinstance(content, str):
            raise TypeError(f"{field} must be a string")
        normalized = content.strip()
        if not normalized:
            raise ValueError(f"{field} must not be blank")
        return normalized

    def _expire_locked(self, now: float) -> int:
        expired = [
            session_id
            for session_id, conversation in self._sessions.items()
            if now - conversation.touched_at >= self.ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]
        return len(expired)

    def _touch_locked(
        self,
        session_id: str,
        conversation: _Conversation,
        now: float,
    ) -> None:
        conversation.touched_at = now
        self._sessions.move_to_end(session_id)

    def _get_or_create_locked(self, session_id: str, now: float) -> _Conversation:
        conversation = self._sessions.get(session_id)
        if conversation is None:
            conversation = _Conversation(
                turns=deque(),
                history_chars=0,
                last_generation_prompt=None,
                touched_at=now,
            )
            self._sessions[session_id] = conversation
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        else:
            self._touch_locked(session_id, conversation, now)
        return conversation

    def _existing_locked(
        self,
        session_id: str,
        now: float,
        *,
        touch: bool,
    ) -> _Conversation | None:
        conversation = self._sessions.get(session_id)
        if conversation is not None and touch:
            self._touch_locked(session_id, conversation, now)
        return conversation

    @staticmethod
    def _remove_left_locked(conversation: _Conversation) -> ConversationTurn:
        removed = conversation.turns.popleft()
        conversation.history_chars -= len(removed.content)
        return removed

    def _trim_history_locked(self, conversation: _Conversation) -> None:
        while (
            len(conversation.turns) > self.max_messages_per_session
            or conversation.history_chars > self.max_history_chars
        ):
            removed = self._remove_left_locked(conversation)
            if (
                removed.role == "user"
                and conversation.turns
                and conversation.turns[0].role == "assistant"
            ):
                self._remove_left_locked(conversation)

        while conversation.turns and conversation.turns[0].role == "assistant":
            self._remove_left_locked(conversation)

    def create_session(self, session_id: str | uuid.UUID | None = None) -> str:
        """Create or touch a session and return its opaque identifier."""
        normalized_id = (
            str(uuid.uuid4())
            if session_id is None
            else self._normalize_identifier(session_id, "session_id")
        )
        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            self._get_or_create_locked(normalized_id, now)
        return normalized_id

    def append_turn(
        self,
        session_id: str | uuid.UUID,
        *,
        role: ConversationRole,
        content: str,
        turn_id: str | uuid.UUID | None = None,
    ) -> ConversationTurn:
        """Append one public turn, evicting the oldest history as needed."""
        normalized_session_id = self._normalize_identifier(session_id, "session_id")
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        normalized_content = self._normalize_content(content, "content")
        if len(normalized_content) > self.max_history_chars:
            raise ValueError("content exceeds max_history_chars")
        normalized_turn_id = (
            uuid.uuid4().hex if turn_id is None else self._normalize_identifier(turn_id, "turn_id")
        )
        turn = ConversationTurn(
            turn_id=normalized_turn_id,
            role=role,
            content=normalized_content,
        )

        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            conversation = self._get_or_create_locked(normalized_session_id, now)
            if any(existing.turn_id == turn.turn_id for existing in conversation.turns):
                raise ValueError(f"turn_id '{turn.turn_id}' already exists in this session")

            conversation.turns.append(turn)
            conversation.history_chars += len(turn.content)
            self._trim_history_locked(conversation)
        return turn

    def append_exchange(
        self,
        session_id: str | uuid.UUID,
        *,
        user_content: str,
        assistant_content: str,
        user_turn_id: str | uuid.UUID | None = None,
        assistant_turn_id: str | uuid.UUID | None = None,
        last_generation_prompt: str | None = None,
    ) -> tuple[ConversationTurn, ConversationTurn]:
        """Atomically append one public user/assistant exchange."""
        normalized_session_id = self._normalize_identifier(session_id, "session_id")
        normalized_user = self._normalize_content(user_content, "user_content")
        normalized_assistant = self._normalize_content(
            assistant_content,
            "assistant_content",
        )
        if len(normalized_user) + len(normalized_assistant) > self.max_history_chars:
            raise ValueError("exchange exceeds max_history_chars")
        normalized_prompt = (
            None
            if last_generation_prompt is None
            else self._normalize_content(last_generation_prompt, "last_generation_prompt")
        )
        if normalized_prompt is not None and len(normalized_prompt) > self.max_history_chars:
            raise ValueError("last_generation_prompt exceeds max_history_chars")

        user_turn = ConversationTurn(
            turn_id=(
                uuid.uuid4().hex
                if user_turn_id is None
                else self._normalize_identifier(user_turn_id, "user_turn_id")
            ),
            role="user",
            content=normalized_user,
        )
        assistant_turn = ConversationTurn(
            turn_id=(
                uuid.uuid4().hex
                if assistant_turn_id is None
                else self._normalize_identifier(assistant_turn_id, "assistant_turn_id")
            ),
            role="assistant",
            content=normalized_assistant,
        )
        if user_turn.turn_id == assistant_turn.turn_id:
            raise ValueError("exchange turn IDs must be unique")

        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            conversation = self._get_or_create_locked(normalized_session_id, now)
            existing_ids = {turn.turn_id for turn in conversation.turns}
            if user_turn.turn_id in existing_ids or assistant_turn.turn_id in existing_ids:
                raise ValueError("exchange turn ID already exists in this session")
            conversation.turns.extend((user_turn, assistant_turn))
            conversation.history_chars += len(user_turn.content) + len(assistant_turn.content)
            if normalized_prompt is not None:
                conversation.last_generation_prompt = normalized_prompt
            self._trim_history_locked(conversation)
        return user_turn, assistant_turn

    def get_history(
        self,
        session_id: str | uuid.UUID,
    ) -> tuple[ConversationTurn, ...]:
        """Return immutable public history, or an empty tuple for a missing session."""
        normalized_session_id = self._normalize_identifier(session_id, "session_id")
        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            conversation = self._existing_locked(
                normalized_session_id,
                now,
                touch=True,
            )
            return tuple(conversation.turns) if conversation is not None else ()

    def get_snapshot(
        self,
        session_id: str | uuid.UUID,
    ) -> ConversationSnapshot | None:
        """Return the session's public state without exposing internal containers."""
        normalized_session_id = self._normalize_identifier(session_id, "session_id")
        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            conversation = self._existing_locked(
                normalized_session_id,
                now,
                touch=True,
            )
            if conversation is None:
                return None
            return ConversationSnapshot(
                session_id=normalized_session_id,
                turns=tuple(conversation.turns),
                last_generation_prompt=conversation.last_generation_prompt,
            )

    def set_last_generation_prompt(
        self,
        session_id: str | uuid.UUID,
        prompt: str | None,
    ) -> None:
        """Record the last effective image prompt, or clear it with ``None``."""
        normalized_session_id = self._normalize_identifier(session_id, "session_id")
        normalized_prompt = None if prompt is None else self._normalize_content(prompt, "prompt")
        if normalized_prompt is not None and len(normalized_prompt) > self.max_history_chars:
            raise ValueError("prompt exceeds max_history_chars")

        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            if normalized_prompt is None:
                conversation = self._existing_locked(
                    normalized_session_id,
                    now,
                    touch=True,
                )
                if conversation is None:
                    return
            else:
                conversation = self._get_or_create_locked(normalized_session_id, now)
            conversation.last_generation_prompt = normalized_prompt

    def get_last_generation_prompt(
        self,
        session_id: str | uuid.UUID,
    ) -> str | None:
        """Return the last effective image prompt for a live session."""
        snapshot = self.get_snapshot(session_id)
        return snapshot.last_generation_prompt if snapshot is not None else None

    def delete_session(self, session_id: str | uuid.UUID) -> bool:
        """Delete one session and report whether it existed."""
        normalized_session_id = self._normalize_identifier(session_id, "session_id")
        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            return self._sessions.pop(normalized_session_id, None) is not None

    def purge_expired(self) -> int:
        """Remove expired sessions and return the number removed."""
        now = self._clock()
        with self._lock:
            return self._expire_locked(now)

    @property
    def session_count(self) -> int:
        """Return the current number of non-expired sessions."""
        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            return len(self._sessions)

    def clear(self) -> None:
        """Remove all sessions."""
        with self._lock:
            self._sessions.clear()
