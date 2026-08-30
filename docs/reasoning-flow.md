# Conversational reasoning flow

## Current gaps and chosen approach

The original application accepted one image prompt at a time. It had no chat
endpoint, session history, language-model provider, ambiguity policy, or
reasoning trace. The random prompt enhancer improves visual wording, but it
cannot infer intent or resolve references such as “make it warmer.”

The chat path uses **two model calls per turn**: a hidden reasoning pass that
analyzes the turn, then a response pass that writes the public reply with that
analysis as explicit input. This is not chain-of-thought: the analysis contains
only bounded conclusions about intent, relevant context, ambiguity, constraints,
and response shape. The API returns the answer and optional image result, never
the analysis, unless debugging is explicitly enabled on the deployment.

Splitting the passes means the reply is written against a short, clean brief
rather than the raw conversation, which is what improves ambiguous and
context-dependent turns. The cost is one extra round trip; the reasoning pass is
therefore given a small token cap and a short timeout, and it degrades to a
direct reply instead of stalling the turn. See
[the two-pass guide](two-pass-reasoning.md) for the trade-off in detail.

It follows official OpenAI guidance to
preserve [conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
and avoid prompting for visible step-by-step reasoning in favor of concise
outcome-focused instructions described in the
[reasoning best-practices guide](https://developers.openai.com/api/docs/guides/reasoning-best-practices).

## Request flow

1. `POST /api/v1/chat` validates the message, session UUID, rate limit, and all
   image-generation controls before calling a provider.
2. The bounded in-memory store supplies public user/assistant turns in order.
   Hidden decisions are never stored in conversation memory.
3. **Pass 1 (reasoning).** Skipped entirely when `CHAT_TWO_PASS_ENABLED=false`,
   which answers the turn in one call. Otherwise the provider receives the
   versioned reasoning prompt and the last `CHAT_REASONING_MAX_TURNS` turns,
   and returns a `<reasoning>`
   XML block: `intent`, `context_notes`, ambiguity handling, `constraints`,
   `response_plan`, action, and an optional standalone generation prompt.
   `parse_reasoning_xml` and Pydantic validate it into `ReasoningAnalysis`;
   extra fields are rejected and `"none"` placeholders normalize to `None`.
4. **Pass 2 (response).** The analysis is rendered into the `instructions` as a
   `<hidden_reasoning_analysis>` block — never appended to `messages` — and the
   provider writes the public reply. If Pass 1 produced no analysis, the same
   call runs with the plain direct-reply prompt instead.
5. For `clarify`, Pass 2 phrases the question the analysis called for; if the
   reply contains more than one question the server substitutes the analysis's
   single question, so a clarification is always exactly one. For an assumption
   on any acting turn, the server prepends `Assumption:` consistently unless
   Pass 2 already stated it.
   For `generate_image`, the server passes only the standalone prompt into the
   existing resource-bounded generation path; the model cannot select unchecked
   GPU parameters.
6. The completed public exchange and last effective image prompt are committed
   atomically. Concurrent turns for the same session are serialized, while
   different sessions can reason concurrently.

### Failure behavior

If the reasoning pass times out, fails, or returns unparseable output, the
response pass still runs with the same public history and no analysis, so the
user gets a normal reply. If the response pass also fails — or returns empty
text, or leaks reasoning tags — the service returns one stable clarification
question with HTTP 200 and suppresses any image generation. Provider errors and
raw outputs never enter the response or memory. Caller cancellation still
propagates and does not commit a misleading turn.

## API and configuration

Example request:

```json
{
  "session_id": null,
  "message": "Create a white sneaker product shot",
  "generation": {
    "model": "stable-diffusion",
    "width": 512,
    "height": 512,
    "enhance_prompt": false
  }
}
```

The response contains `session_id`, public `turn_id`, support-safe `trace_id`,
status, public `message`, and an optional normal generation response. It carries
no reasoning field unless the deployment sets `DEBUG=true` *and* the request
opts in with `"debug": true`, which adds `reasoning_debug` with the analysis and
per-pass timings.

Configure a model through:

```dotenv
CHAT_API_BASE_URL=https://api.openai.com/v1
CHAT_API_KEY=...
CHAT_MODEL=your-chat-capable-model
CHAT_TWO_PASS_ENABLED=true
CHAT_REASONING_TIMEOUT_SECONDS=4
CHAT_RESPONSE_TIMEOUT_SECONDS=12
CHAT_MAX_REASONING_TOKENS=500
CHAT_MAX_OUTPUT_TOKENS=1200
```

Leaving `CHAT_MODEL` empty uses the graceful local clarification path and never
attempts a network request. The included adapter targets the OpenAI Responses
API. To support another vendor, implement the `ChatProvider` protocol and wire
it in `app/core/dependencies.py`; the analysis schema, memory, API, and tests do
not need to change.

Memory is intentionally in-process, TTL-bound, LRU-bounded, and limited by both
message count and characters. It does not survive restarts and must be replaced
with a protocol-compatible Redis or database store before running multiple API
workers or requiring durable conversation history.

## Tracing and privacy

Every completed turn logs one JSON `chat_reasoning_trace` with a trace ID,
hashed session ID, turn ID, prompt version, provider/model, intent, context
count, ambiguity, constraint, action, latency, and sanitized fallback class.
Raw messages, answers, prompts, provider output, and credentials are omitted by
default. `CHAT_LOG_DECISION_CONTENT=true` adds the validated decision summary
for controlled debugging; it still does not log chain-of-thought or raw model
output. Treat those logs as potentially sensitive and apply normal access and
retention controls.

## Before and after examples

These “before” responses illustrate the literal one-shot behavior that the new
contract is designed to prevent.

### Ambiguous first turn

```text
User: Make it more dramatic.
Before: Generating “Make it more dramatic.”
After: Which image or prompt should I make more dramatic?
```

### Context-dependent refinement

```text
User: Create a product shot of a white sneaker on a neutral background.
Assistant: Creating the product shot.
User: Make it warmer but keep the background neutral.
Before: Generates an unrelated warm scene from only the latest sentence.
After: Preserves the white sneaker and neutral background, adds warm studio
       lighting, and generates from a complete standalone prompt.
```

### Reasonable assumption

```text
User: Make a launch graphic for tomorrow.
Before: Silently picks a format.
After: Assumption: Use a square social-media composition.

       I’ll create an editable launch graphic for tomorrow.
```

### Capability constraint

```text
User: Email the finished image to Maya.
Before: Claims success or treats the sentence as an image prompt.
After: I can’t send email, but I can help draft the message and prepare the
       image for you to attach.
```

The deterministic tests use scripted providers and mocked diffusion pipelines,
so normal CI never calls an external model or downloads a checkpoint.
