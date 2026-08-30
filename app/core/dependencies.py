"""Shared application dependencies."""

from app.core.config import settings
from app.services.chat_provider import (
    AnthropicChatProvider,
    ChatProvider,
    OpenAIResponsesProvider,
    UnconfiguredChatProvider,
)
from app.services.chat_service import ChatService
from app.services.conversation_store import ConversationStore
from app.services.model_service import ModelService
from app.services.reasoning_service import ReasoningService

# Global model service instance - shared across the application
model_service = ModelService()

chat_provider: ChatProvider
if settings.CHAT_MODEL.strip():
    if settings.CHAT_PROVIDER_TYPE == "anthropic":
        chat_provider = AnthropicChatProvider(
            api_base_url=settings.CHAT_API_BASE_URL,
            api_key=settings.CHAT_API_KEY.get_secret_value(),
            model=settings.CHAT_MODEL,
        )
    else:
        chat_provider = OpenAIResponsesProvider(
            api_base_url=settings.CHAT_API_BASE_URL,
            api_key=settings.CHAT_API_KEY.get_secret_value(),
            model=settings.CHAT_MODEL,
        )
else:
    chat_provider = UnconfiguredChatProvider()

conversation_store = ConversationStore(
    max_sessions=settings.CHAT_MAX_SESSIONS,
    max_messages_per_session=settings.CHAT_MAX_HISTORY_MESSAGES,
    max_history_chars=settings.CHAT_MAX_HISTORY_CHARS,
    ttl_seconds=settings.CHAT_SESSION_TTL_SECONDS,
)

reasoning_service = ReasoningService(
    chat_provider,
    two_pass_enabled=settings.CHAT_TWO_PASS_ENABLED,
    reasoning_timeout_seconds=settings.CHAT_REASONING_TIMEOUT_SECONDS,
    response_timeout_seconds=settings.CHAT_RESPONSE_TIMEOUT_SECONDS,
    max_reasoning_tokens=settings.CHAT_MAX_REASONING_TOKENS,
    max_response_tokens=settings.CHAT_MAX_OUTPUT_TOKENS,
    max_history_turns=settings.CHAT_REASONING_MAX_TURNS,
    max_generation_prompt_length=settings.MAX_PROMPT_LENGTH,
    log_decision_content=settings.CHAT_LOG_DECISION_CONTENT,
)

chat_service = ChatService(
    store=conversation_store,
    reasoning_service=reasoning_service,
)
