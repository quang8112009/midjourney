"""Tests for the two-pass reasoning pipeline, parser, orchestrator, and providers."""

import asyncio
import hashlib
import json
import unittest

import httpx

from app.services.chat_prompts import (
    REASONING_PROMPT_VERSION,
    REASONING_SYSTEM_PROMPT,
    build_response_system_prompt,
)
from app.services.chat_provider import (
    AnthropicChatProvider,
    ChatProviderUnavailable,
    ModelMessage,
    OpenAIResponsesProvider,
    ScriptedProvider,
)
from app.services.reasoning_service import (
    ReasoningParseError,
    ReasoningService,
    parse_reasoning_xml,
)


def sample_reasoning_xml(
    *,
    intent="Answer the user's question",
    context_notes="none",
    is_ambiguous=False,
    assumption="none",
    clarifying_question="none",
    constraints="none",
    response_plan="Provide a concise and helpful response.",
    action="respond",
    generation_prompt="none",
) -> str:
    return f"""
<reasoning>
  <intent>{intent}</intent>
  <context_notes>{context_notes}</context_notes>
  <ambiguity>
    <is_ambiguous>{str(is_ambiguous).lower()}</is_ambiguous>
    <assumption>{assumption}</assumption>
    <clarifying_question>{clarifying_question}</clarifying_question>
  </ambiguity>
  <constraints>{constraints}</constraints>
  <response_plan>{response_plan}</response_plan>
  <action>{action}</action>
  <generation_prompt>{generation_prompt}</generation_prompt>
</reasoning>
""".strip()


def current_message(content="Create a red fox", turn_id="current-turn"):
    return ModelMessage(role="user", content=content, turn_id=turn_id)


class ReasoningXmlParserTests(unittest.TestCase):
    def test_parse_valid_xml(self):
        xml = sample_reasoning_xml(
            intent="Generate a sunset painting",
            context_notes="User prefers watercolor",
            is_ambiguous=True,
            assumption="Use warm colors",
            clarifying_question="none",
            constraints="Still images only",
            response_plan="Confirm creation with warm palette",
            action="generate_image",
            generation_prompt="A watercolor painting of a vibrant sunset over mountains",
        )
        parsed = parse_reasoning_xml(xml)
        self.assertEqual(parsed.intent, "Generate a sunset painting")
        self.assertEqual(parsed.context_notes, "User prefers watercolor")
        self.assertTrue(parsed.ambiguity.is_ambiguous)
        self.assertEqual(parsed.ambiguity.assumption, "Use warm colors")
        self.assertIsNone(parsed.ambiguity.clarifying_question)
        self.assertEqual(parsed.constraints, "Still images only")
        self.assertEqual(parsed.response_plan, "Confirm creation with warm palette")
        self.assertEqual(parsed.action, "generate_image")
        self.assertEqual(
            parsed.generation_prompt,
            "A watercolor painting of a vibrant sunset over mountains",
        )

    def test_parse_xml_with_markdown_fences(self):
        raw = f"```xml\n{sample_reasoning_xml(intent='Markdown wrapped test')}\n```"
        parsed = parse_reasoning_xml(raw)
        self.assertEqual(parsed.intent, "Markdown wrapped test")
        self.assertEqual(parsed.action, "respond")

    def test_parse_json_fallback(self):
        json_data = json.dumps({
            "intent": "JSON test",
            "context_notes": "from prior turn",
            "ambiguity": {
                "is_ambiguous": False,
                "assumption": None,
                "clarifying_question": None,
            },
            "constraints": "none",
            "response_plan": "Answer clearly",
            "action": "respond",
        })
        parsed = parse_reasoning_xml(json_data)
        self.assertEqual(parsed.intent, "JSON test")
        self.assertEqual(parsed.context_notes, "from prior turn")
        self.assertFalse(parsed.ambiguity.is_ambiguous)

    def test_parse_invalid_raises_error(self):
        with self.assertRaises(ReasoningParseError):
            parse_reasoning_xml("This has no XML tags at all.")


class ReasoningServiceTwoPassTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_pass_success_runs_reasoning_then_response(self):
        reasoning_xml = sample_reasoning_xml(
            intent="User wants explanation of diffusion steps",
            response_plan="Explain steps clearly and concisely",
            action="respond",
        )
        response_text = "Diffusion steps determine how many denoising iterations are performed."
        provider = ScriptedProvider(reasoning_xml, response_text)
        service = ReasoningService(
            provider,
            reasoning_timeout_seconds=2.0,
            response_timeout_seconds=5.0,
            max_reasoning_tokens=300,
            max_response_tokens=600,
        )
        messages = [current_message("What do inference steps do?")]

        turn = await service.reason(session_id="session-1", messages=messages)

        self.assertEqual(turn.action, "respond")
        self.assertEqual(turn.public_response, response_text)
        self.assertFalse(turn.fallback_used)
        self.assertIsNotNone(turn.reasoning_analysis)
        self.assertEqual(
            turn.reasoning_analysis.intent, "User wants explanation of diffusion steps"
        )
        self.assertIsNotNone(turn.timing)
        self.assertGreaterEqual(turn.timing.total_ms, 0)
        self.assertIsNotNone(turn.timing.reasoning_ms)
        self.assertGreaterEqual(turn.timing.response_ms, 0)

        # Check that exactly 2 provider calls were made
        self.assertEqual(len(provider.calls), 2)
        # Pass 1
        self.assertEqual(provider.calls[0]["instructions"], REASONING_SYSTEM_PROMPT)
        self.assertEqual(provider.calls[0]["max_output_tokens"], 300)
        self.assertEqual(provider.calls[0]["temperature"], 0.1)
        # Pass 2
        self.assertIn("<hidden_reasoning_analysis>", provider.calls[1]["instructions"])
        self.assertEqual(provider.calls[1]["max_output_tokens"], 600)
        self.assertEqual(provider.calls[1]["temperature"], 0.7)

    async def test_ambiguity_clarify_action_flows_to_response_pass(self):
        reasoning_xml = sample_reasoning_xml(
            intent="Improve unspecified image",
            is_ambiguous=True,
            clarifying_question="Which image or prompt would you like me to improve?",
            action="clarify",
            response_plan="Ask the user for the target image",
        )
        clarifying_reply = "Which image or prompt would you like me to improve?"
        provider = ScriptedProvider(reasoning_xml, clarifying_reply)
        service = ReasoningService(provider)

        turn = await service.reason(
            session_id="ambiguous-session",
            messages=[current_message("Make it better")],
        )

        self.assertEqual(turn.action, "clarify")
        self.assertEqual(turn.public_response, clarifying_reply)
        self.assertTrue(turn.reasoning_analysis.ambiguity.is_ambiguous)
        self.assertEqual(
            turn.reasoning_analysis.ambiguity.clarifying_question,
            "Which image or prompt would you like me to improve?",
        )
        self.assertFalse(turn.fallback_used)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(turn.public_response.count("?"), 1)

    async def test_context_dependent_refinement_preserves_details(self):
        messages = [
            ModelMessage(role="user", content="A white sneaker", turn_id="t1"),
            ModelMessage(role="assistant", content="Created the white sneaker.", turn_id="t2"),
            current_message("Make it warmer", turn_id="t3"),
        ]
        reasoning_xml = sample_reasoning_xml(
            intent="Add warm lighting to the white sneaker from turn 1",
            context_notes="Subject is white sneaker from t1",
            action="generate_image",
            generation_prompt="A product shot of a white sneaker under warm studio lighting",
            response_plan="Confirm the updated warm sneaker generation",
        )
        response_text = "I'm generating the white sneaker with warmer studio lighting for you."
        provider = ScriptedProvider(reasoning_xml, response_text)
        service = ReasoningService(provider)

        turn = await service.reason(session_id="ctx-session", messages=messages)

        self.assertEqual(turn.action, "generate_image")
        self.assertEqual(
            turn.generation_prompt,
            "A product shot of a white sneaker under warm studio lighting",
        )
        self.assertEqual(turn.public_response, response_text)
        self.assertFalse(turn.fallback_used)
        self.assertEqual(len(provider.calls), 2)

    async def test_reasoning_pass_failure_falls_back_gracefully_to_response_pass(self):
        # Pass 1 throws an error, Pass 2 generates a direct response
        provider = ScriptedProvider(
            RuntimeError("Reasoning API temporarily failed"),
            "I'm here to help you create images or answer questions.",
        )
        service = ReasoningService(provider)

        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            turn = await service.reason(
                session_id="fallback-session",
                messages=[current_message("Hello")],
            )

        self.assertTrue(turn.fallback_used)
        self.assertEqual(turn.fallback_reason, "RuntimeError")
        self.assertIsNone(turn.reasoning_analysis)
        self.assertEqual(
            turn.public_response, "I'm here to help you create images or answer questions."
        )
        self.assertEqual(len(provider.calls), 2)
        # Pass 2 received fallback instructions without hidden reasoning
        self.assertNotIn("<hidden_reasoning_analysis>", provider.calls[1]["instructions"])
        self.assertIn("RuntimeError", "\n".join(logs.output))

    async def test_reasoning_pass_timeout_cancels_and_falls_back(self):
        cancelled = asyncio.Event()

        async def slow_reasoning(**_kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        provider = ScriptedProvider(
            slow_reasoning,
            "Direct response after reasoning timeout.",
        )
        service = ReasoningService(provider, reasoning_timeout_seconds=0.02)

        turn = await service.reason(
            session_id="timeout-session",
            messages=[current_message("Generate something")],
        )

        self.assertTrue(cancelled.is_set())
        self.assertTrue(turn.fallback_used)
        self.assertEqual(turn.public_response, "Direct response after reasoning timeout.")
        self.assertEqual(len(provider.calls), 2)

    async def test_caller_cancellation_propagates_immediately(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow_call(**_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        provider = ScriptedProvider(slow_call)
        service = ReasoningService(provider, reasoning_timeout_seconds=10.0)
        task = asyncio.create_task(
            service.reason(session_id="cancel-sess", messages=[current_message("Wait")])
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(provider.calls), 1)

    async def test_double_failure_degrades_to_stable_local_reply(self):
        # Both Pass 1 and Pass 2 fail
        provider = ScriptedProvider(
            ChatProviderUnavailable("pass 1 fail"),
            ChatProviderUnavailable("pass 2 fail"),
        )
        service = ReasoningService(provider)

        turn = await service.reason(
            session_id="double-fail-session",
            messages=[current_message("Hello")],
        )

        self.assertTrue(turn.fallback_used)
        self.assertEqual(turn.action, "clarify")
        self.assertIn("couldn't process", turn.public_response.lower())
        self.assertNotIn("secret", turn.public_response.lower())
        self.assertEqual(len(provider.calls), 2)

    async def test_private_canaries_and_hidden_tags_are_filtered_from_public_output(self):
        canary = "SECRET_REASONING_CANARY"
        reasoning_xml = sample_reasoning_xml(intent=canary)
        leaked_reply = (
            f"<hidden_reasoning_analysis>{canary}</hidden_reasoning_analysis> Here is the answer."
        )
        provider = ScriptedProvider(reasoning_xml, leaked_reply)
        service = ReasoningService(provider)

        turn = await service.reason(
            session_id="canary-session",
            messages=[current_message("Tell me private info")],
        )

        self.assertTrue(turn.fallback_used)
        self.assertNotIn(canary, turn.public_response)
        self.assertNotIn("<hidden_reasoning_analysis>", turn.public_response)

    async def test_trace_logging_includes_timing_and_privacy(self):
        session_id = "PRIVATE_SESSION_KEY"
        reasoning_xml = sample_reasoning_xml(intent="Draw a tree")
        response_text = "Here is a description of the tree."
        provider = ScriptedProvider(reasoning_xml, response_text)
        service = ReasoningService(provider)

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            turn = await service.reason(
                session_id=session_id,
                messages=[current_message("Draw a tree", turn_id="t-tree")],
            )

        self.assertEqual(len(logs.output), 1)
        trace_str = logs.output[0].split(":", 2)[2]
        trace = json.loads(trace_str)
        self.assertEqual(trace["event"], "chat_reasoning_trace")
        self.assertEqual(trace["trace_id"], turn.trace_id)
        self.assertEqual(trace["turn_id"], "t-tree")
        self.assertEqual(trace["prompt_version"], REASONING_PROMPT_VERSION)
        self.assertFalse(trace["fallback_used"])
        self.assertGreaterEqual(trace["latency_ms"], 0)
        self.assertIsNotNone(trace["reasoning_latency_ms"])
        self.assertIsNotNone(trace["response_latency_ms"])
        # Privacy check: raw session_id is not in log, only hash
        self.assertNotIn(session_id, json.dumps(trace))
        self.assertEqual(
            trace["session_hash"], hashlib.sha256(session_id.encode()).hexdigest()[:12]
        )


class ClarificationAndAssumptionInvariantTests(unittest.IsolatedAsyncioTestCase):
    """Guarantees that must hold however Pass 2 chooses to phrase the reply."""

    async def test_clarify_reply_is_reduced_to_one_question(self):
        reasoning_xml = sample_reasoning_xml(
            is_ambiguous=True,
            clarifying_question="Which image should I edit?",
            action="clarify",
        )
        chatty = "Which image do you mean? Do you want it warmer? Keep the background?"
        service = ReasoningService(ScriptedProvider(reasoning_xml, chatty))

        turn = await service.reason(
            session_id="s", messages=[current_message("Make it better")]
        )

        self.assertEqual(turn.action, "clarify")
        self.assertEqual(turn.public_response.count("?"), 1)
        self.assertEqual(turn.public_response, "Which image should I edit?")

    async def test_single_question_clarify_reply_keeps_its_natural_phrasing(self):
        reasoning_xml = sample_reasoning_xml(
            is_ambiguous=True,
            clarifying_question="Which image should I edit?",
            action="clarify",
        )
        natural = "Happy to help - which of your images should I make more dramatic?"
        service = ReasoningService(ScriptedProvider(reasoning_xml, natural))

        turn = await service.reason(
            session_id="s", messages=[current_message("Make it better")]
        )

        # The reduction must not fire on a well-formed reply.
        self.assertEqual(turn.public_response, natural)

    async def test_assumption_is_stated_even_when_analysis_also_noted_a_question(self):
        reasoning_xml = sample_reasoning_xml(
            is_ambiguous=True,
            assumption="Use a square social-media composition",
            clarifying_question="Should it be square or wide?",
            action="generate_image",
            generation_prompt="a launch graphic",
        )
        service = ReasoningService(
            ScriptedProvider(reasoning_xml, "Creating your launch graphic now.")
        )

        turn = await service.reason(
            session_id="s", messages=[current_message("Launch graphic for tomorrow")]
        )

        # The turn is acting, not clarifying, so the assumption must not go unstated.
        self.assertEqual(turn.action, "generate_image")
        self.assertTrue(turn.public_response.startswith("Assumption: "))
        self.assertIn("Use a square social-media composition", turn.public_response)

    async def test_assumption_is_suppressed_on_a_clarify_turn(self):
        reasoning_xml = sample_reasoning_xml(
            is_ambiguous=True,
            assumption="Assume a square composition",
            clarifying_question="Square or wide?",
            action="clarify",
        )
        service = ReasoningService(ScriptedProvider(reasoning_xml, "Square or wide?"))

        turn = await service.reason(
            session_id="s", messages=[current_message("Launch graphic")]
        )

        self.assertEqual(turn.action, "clarify")
        self.assertNotIn("Assumption:", turn.public_response)


class SinglePassFlagTests(unittest.IsolatedAsyncioTestCase):
    """CHAT_TWO_PASS_ENABLED=false must answer in one provider call."""

    async def test_disabled_flag_skips_reasoning_pass(self):
        response_text = "Stable Diffusion denoises an image over several steps."
        provider = ScriptedProvider(response_text)
        service = ReasoningService(provider, two_pass_enabled=False)

        turn = await service.reason(
            session_id="session-single",
            messages=[current_message("What do inference steps do?")],
        )

        # Exactly one round trip, using the plain direct-reply prompt.
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["instructions"], build_response_system_prompt(None))
        self.assertNotIn("<hidden_reasoning_analysis>", provider.calls[0]["instructions"])

        self.assertEqual(turn.public_response, response_text)
        self.assertEqual(turn.action, "respond")
        self.assertIsNone(turn.reasoning_analysis)
        self.assertIsNone(turn.generation_prompt)
        # Skipping the pass is configuration, not degradation.
        self.assertFalse(turn.fallback_used)
        self.assertIsNone(turn.fallback_reason)
        self.assertIsNone(turn.timing.reasoning_ms)
        self.assertGreaterEqual(turn.timing.response_ms, 0)

    async def test_disabled_flag_still_uses_local_fallback_on_response_failure(self):
        provider = ScriptedProvider(ChatProviderUnavailable("boom"))
        service = ReasoningService(provider, two_pass_enabled=False)

        turn = await service.reason(
            session_id="session-single",
            messages=[current_message("Draw a fox")],
        )

        self.assertTrue(turn.fallback_used)
        self.assertEqual(turn.fallback_reason, "ok+ChatProviderUnavailable")
        self.assertEqual(turn.action, "clarify")
        self.assertIn("What image or topic", turn.public_response)

    async def test_trace_records_single_pass_mode(self):
        provider = ScriptedProvider("A direct reply.")
        service = ReasoningService(provider, two_pass_enabled=False)

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            await service.reason(
                session_id="session-single",
                messages=[current_message("Hello", turn_id="t-1")],
            )

        trace = json.loads(logs.output[0].split(":", 2)[2])
        self.assertFalse(trace["two_pass_enabled"])
        self.assertIsNone(trace["reasoning_latency_ms"])
        self.assertFalse(trace["fallback_used"])

    async def test_enabled_by_default(self):
        self.assertTrue(ReasoningService(ScriptedProvider()).two_pass_enabled)


class ProviderAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_responses_provider(self):
        def handler(request):
            return httpx.Response(200, json={"output_text": "  openai output  "})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OpenAIResponsesProvider(
                api_base_url="https://api.openai.com/v1",
                api_key="test-key",
                model="gpt-4o",
                client=client,
            )
            res = await provider.complete(
                instructions="system instructions",
                messages=[current_message("hi")],
                timeout_seconds=2.0,
                max_output_tokens=100,
                temperature=0.2,
            )
            self.assertEqual(res, "openai output")
            self.assertEqual(provider.provider_name, "openai")

    async def test_anthropic_chat_provider(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "claude-3-5-sonnet")
            self.assertEqual(payload["system"], "system prompt")
            self.assertEqual(payload["max_tokens"], 200)
            self.assertEqual(payload["temperature"], 0.1)
            self.assertEqual(request.headers["x-api-key"], "anthropic-key")
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "  anthropic output  "}]},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = AnthropicChatProvider(
                api_base_url="https://api.anthropic.com/v1",
                api_key="anthropic-key",
                model="claude-3-5-sonnet",
                client=client,
            )
            res = await provider.complete(
                instructions="system prompt",
                messages=[current_message("hello claude")],
                timeout_seconds=3.0,
                max_output_tokens=200,
                temperature=0.1,
            )
            self.assertEqual(res, "anthropic output")
            self.assertEqual(provider.provider_name, "anthropic")


if __name__ == "__main__":
    unittest.main()
