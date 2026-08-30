# Two-pass reasoning

The chat path splits one turn into two provider calls: a hidden **reasoning pass** that
analyzes the turn, and a **response pass** that writes the user-facing reply with that
analysis as explicit input. The user only ever sees the output of the second pass.

Implementation: `app/services/reasoning_service.py` (`ReasoningService.reason`),
`app/services/chat_prompts.py` (both prompts), `app/services/chat_service.py` (session
locking and generation).

## Architecture

```
POST /api/v1/chat
      |
      v
  ChatService.reason()            per-session lock, bounded public history
      |
      v
+-------------------------------------------------------------+
| PASS 1 - reasoning         REASONING_SYSTEM_PROMPT           |
|   last N turns  ->  <reasoning> XML  ->  ReasoningAnalysis   |
|   short timeout, small token cap, low temperature (0.1)      |
+-------------------------------------------------------------+
      |                                    |
      | analysis                           | timeout / parse error / provider error
      |                                    v
      |                               analysis = None
      |                               fallback_used = True
      |                                    |
      v                                    v
+-------------------------------------------------------+
| PASS 2 - response   build_response_system_prompt(...)  |
|   analysis rendered into `instructions`, never into    |
|   `messages`; longer timeout, temperature 0.7          |
|   analysis is None -> plain direct-reply prompt        |
+-------------------------------------------------------+
                                          |
                                          | on failure: stable local
                                          | clarification, still HTTP 200
                                          v
                            assumption prefix, action resolution,
                            optional image generation, atomic commit
```

Key properties:

- **The analysis never enters conversation memory.** It is rendered into the `instructions`
  (system) field as a `<hidden_reasoning_analysis>` block, so it cannot be mistaken for user
  content or persisted by `ConversationStore`.
- **The reply is scrubbed.** `_ensure_public_only` rejects any response containing reasoning
  tags (`<reasoning`, `<hidden_reasoning_analysis>`, …) and the turn degrades to the local
  fallback rather than leaking.
- **`clarify` still runs both passes, but stays one question.** The analysis decides *that* a
  question is needed and drafts it; Pass 2 phrases it in the conversation's voice. If that
  reply stacks more than one question, the service substitutes the analysis's own question, so
  the one-question guarantee is structural rather than dependent on the prompt being obeyed.
  `action` stays `clarify`, so no image is generated.
- **An assumption is never silent.** Whenever the turn acts instead of clarifying, an
  assumption from the analysis is prefixed to the reply unless Pass 2 already stated it.
- **Image generation runs after both passes** in `ChatService`, using only the validated
  standalone `generation_prompt`. The model never selects GPU parameters.

## When two passes beat one

Worth the extra round trip when:

- **The turn is ambiguous.** "Make it more dramatic." with no history. A single call tends to
  commit to an interpretation while it is still drafting prose; a dedicated pass decides
  *clarify vs. assume* before any wording exists.
- **The turn references earlier context.** "Make it warmer but keep the background neutral"
  requires resolving the sneaker and the neutral background into a standalone prompt. The
  reasoning pass produces that prompt as a discrete artifact instead of burying it in prose.
- **You want the plan as an auditable artifact.** `ReasoningAnalysis` is typed, logged, and
  can be returned under `debug`, so you can inspect *why* a turn was answered a given way.
- **Reply quality matters more than latency**, because the response pass gets a short, clean
  brief instead of the raw conversation.

Not worth it when:

- **The turn is short and unambiguous** ("Create a red fox in snow") — the analysis mostly
  restates the request.
- **The path is latency-critical.** Two sequential calls roughly double time-to-first-token
  versus one. Tighten `CHAT_REASONING_TIMEOUT_SECONDS` so a slow reasoning pass degrades to
  a direct reply quickly, rather than stalling the turn, or set
  `CHAT_TWO_PASS_ENABLED=false` to drop back to a single call entirely.

### Turning it off

`CHAT_TWO_PASS_ENABLED=false` skips the reasoning pass and answers with one provider call
using the plain direct-reply prompt — the same prompt the pipeline already falls back to when
Pass 1 fails. This is the rollout kill switch: it needs no code change, restores the
one-round-trip latency profile, and gives up analysis-guided replies, cross-turn prompt
resolution, and the auditable analysis artifact. Because it is configuration rather than
degradation, `fallback_used` stays `False` and `reasoning_ms` is `None`; the trace log
records `two_pass_enabled` on every turn so both modes are distinguishable in aggregate.

## Degradation

Failure of the reasoning pass is never surfaced to the user:

| Failure | Behavior |
|---|---|
| Pass 1 times out / errors / returns unparseable XML | `analysis = None`, Pass 2 runs with the plain direct-reply prompt, `fallback_used=True` |
| Pass 2 fails, or leaks reasoning tags, or returns empty | Stable local clarification, HTTP 200, `action` forced to `clarify` so no image is generated |
| `CHAT_MODEL` empty | No network call at all; local clarification path |

`fallback_reason` records the sanitized exception class (e.g. `TimeoutError`,
`ReasoningParseError`, or `ok+ChatProviderUnavailable` when only Pass 2 failed). Raw provider
output never reaches the response or the logs.

## Configuration

```dotenv
CHAT_TWO_PASS_ENABLED=true          # false -> single direct call, no reasoning pass
CHAT_REASONING_TIMEOUT_SECONDS=4    # Pass 1 budget - keep short, it degrades gracefully
CHAT_MAX_REASONING_TOKENS=500
CHAT_REASONING_TEMPERATURE=0.1      # analysis should be stable
CHAT_REASONING_MAX_TURNS=8          # history turns fed to Pass 1

CHAT_RESPONSE_TIMEOUT_SECONDS=12    # Pass 2 budget
CHAT_MAX_OUTPUT_TOKENS=1200
CHAT_RESPONSE_TEMPERATURE=0.7       # reply should sound natural
```

`temperature` is only included in the provider payload when it is not `None`, since some
reasoning models reject the parameter.

## Inspecting a turn

With `DEBUG=true`, `POST /api/v1/chat` with `"debug": true` (or `?debug=true`) returns a
`reasoning_debug` object containing the parsed analysis and per-pass timings. When `DEBUG` is
false the flag is silently ignored and the public response contract is unchanged. Timings are
written to the `chat_reasoning_trace` JSON log on every turn regardless.

For an offline walkthrough of all four paths — ambiguous, context-dependent, clear, and
reasoning-pass failure — run:

```bash
python scripts/two_pass_demo.py
```

It uses a scripted in-process provider: no API key, no network, no model download.

## Extending the analysis

To add a field to `ReasoningAnalysis`:

1. Add the field to `ReasoningAnalysis` in `app/services/reasoning_service.py` with a default,
   so older/degraded outputs still validate.
2. Parse it in `parse_reasoning_xml` via `_extract_tag_content`.
3. Describe the new tag in `REASONING_SYSTEM_PROMPT` inside the `<reasoning>` block.
4. Render it in the `<hidden_reasoning_analysis>` block in `build_response_system_prompt` so
   Pass 2 can actually use it.
5. Bump `REASONING_PROMPT_VERSION` so traces attribute behavior to the right prompt.
6. If it is private, add its tag to `_PRIVATE_OUTPUT_MARKERS` so a leaked reply is caught.
