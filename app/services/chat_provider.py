"""Provider boundary for structured conversational model calls."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx


class ChatProviderError(RuntimeError):
    """Base class for sanitized provider failures."""


class ChatProviderTimeout(ChatProviderError):
    """Raised when the configured provider deadline is exceeded."""


class ChatProviderUnavailable(ChatProviderError):
    """Raised when no provider is configured or reachable."""


class ChatProviderRefusal(ChatProviderError):
    """Raised when a provider returns a refusal instead of structured output."""


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: Literal["user", "assistant", "system"]
    content: str
    turn_id: str
    kind: Literal["public", "generation_context"] = "public"

    def provider_content(self) -> str:
        return f"[turn_id={self.turn_id} kind={self.kind}] {self.content}"


class ChatProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def complete(
        self,
        *,
        instructions: str,
        messages: Sequence[ModelMessage],
        output_schema: dict | None = None,
        timeout_seconds: float = 10.0,
        max_output_tokens: int = 1_000,
        temperature: float | None = None,
    ) -> str: ...

    async def aclose(self) -> None: ...


class UnconfiguredChatProvider:
    """Explicit unavailable provider used when chat settings are incomplete."""

    @property
    def provider_name(self) -> str:
        return "unconfigured"

    @property
    def model_name(self) -> str:
        return "unconfigured"

    async def complete(self, **_kwargs) -> str:
        raise ChatProviderUnavailable("Conversational model is not configured.")

    async def aclose(self) -> None:
        return None


class OpenAIResponsesProvider:
    """OpenAI API adapter supporting Responses and Chat endpoints."""

    def __init__(
        self,
        *,
        api_base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
    ):
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        instructions: str,
        messages: Sequence[ModelMessage],
        output_schema: dict | None = None,
        timeout_seconds: float = 10.0,
        max_output_tokens: int = 1_000,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "instructions": instructions,
            "input": [
                {"role": message.role, "content": message.provider_content()}
                for message in messages
            ],
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if output_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "reasoned_chat_turn",
                    "strict": True,
                    "schema": output_schema,
                }
            }

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._api_base_url}/responses"
        try:
            response = await self._client.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ChatProviderTimeout("Conversational model request timed out.") from exc
        except httpx.HTTPError as exc:
            raise ChatProviderUnavailable("Conversational model request failed.") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise ChatProviderError("Conversational model returned invalid JSON.") from exc

        text = self._extract_output_text(body)
        if text is None:
            raise ChatProviderError("Conversational model returned no text output.")
        return text

    @staticmethod
    def _extract_output_text(body: object) -> str | None:
        if not isinstance(body, dict):
            return None
        top_level = body.get("output_text")
        if isinstance(top_level, str) and top_level.strip():
            return top_level.strip()

        # Chat completions choice format
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message", {})
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg["content"]).strip()

        output = body.get("output", [])
        if not isinstance(output, list):
            return None
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            contents = item.get("content", [])
            if not isinstance(contents, list):
                continue
            for content in contents:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "refusal":
                    raise ChatProviderRefusal("Conversational model refused the request.")
                text = content.get("text")
                if content.get("type") == "output_text" and isinstance(text, str):
                    if text.strip():
                        return text.strip()
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class AnthropicChatProvider:
    """Anthropic Messages API adapter."""

    def __init__(
        self,
        *,
        api_base_url: str = "https://api.anthropic.com/v1",
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        instructions: str,
        messages: Sequence[ModelMessage],
        output_schema: dict | None = None,
        timeout_seconds: float = 10.0,
        max_output_tokens: int = 1_000,
        temperature: float | None = None,
    ) -> str:
        formatted_messages = [
            {
                "role": message.role if message.role in ("user", "assistant") else "user",
                "content": message.provider_content(),
            }
            for message in messages
        ]

        payload: dict[str, Any] = {
            "model": self._model,
            "system": instructions,
            "messages": formatted_messages,
            "max_tokens": max_output_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        try:
            response = await self._client.post(
                f"{self._api_base_url}/messages",
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ChatProviderTimeout("Anthropic model request timed out.") from exc
        except httpx.HTTPError as exc:
            raise ChatProviderUnavailable("Anthropic model request failed.") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise ChatProviderError("Anthropic returned invalid JSON.") from exc

        content_blocks = body.get("content", [])
        if isinstance(content_blocks, list):
            text_pieces = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_pieces.append(block.get("text", ""))
            result = "".join(text_pieces).strip()
            if result:
                return result

        raise ChatProviderError("Anthropic returned no text content.")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class ScriptedProvider:
    """Predictable scripted provider for unit tests and local simulations."""

    def __init__(
        self,
        *steps: Any,
        provider_name: str = "scripted",
        model_name: str = "scripted-model",
    ):
        from collections import deque

        self._steps = deque(steps)
        self._provider_name = provider_name
        self._model_name = model_name
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if not self._steps:
            raise AssertionError("Unexpected provider call: no more scripted steps available")
        step = self._steps.popleft()
        if isinstance(step, BaseException):
            raise step
        result = step(**kwargs) if callable(step) else step
        if inspect.isawaitable(result):
            result = await result
        return str(result)

    async def aclose(self) -> None:
        self.closed = True
