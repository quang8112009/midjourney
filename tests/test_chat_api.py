"""API integration tests for the two-pass conversational chat endpoint."""

import unittest
import warnings
from unittest.mock import AsyncMock, patch

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)

from fastapi.testclient import TestClient

from app.api.chat_routes import chat_rate_limiter
from app.api.routes import (
    GenerateResponse,
    GenerationParameters,
    generation_rate_limiter,
)
from app.core.config import settings
from app.core.dependencies import chat_service
from main import app


def analysis_xml(
    *,
    intent="Answer the user",
    context_notes="none",
    is_ambiguous=False,
    assumption=None,
    clarifying_question=None,
    constraints="none",
    response_plan="Give a concise answer",
    action="respond",
    generation_prompt=None,
):
    """Build a Pass 1 analysis payload in the XML contract."""
    parts = [
        f"<intent>{intent}</intent>",
        f"<context_notes>{context_notes}</context_notes>",
        "<ambiguity>",
        f"  <is_ambiguous>{str(is_ambiguous).lower()}</is_ambiguous>",
    ]
    if assumption:
        parts.append(f"  <assumption>{assumption}</assumption>")
    if clarifying_question:
        parts.append(f"  <clarifying_question>{clarifying_question}</clarifying_question>")
    parts.append("</ambiguity>")
    parts.append(f"<constraints>{constraints}</constraints>")
    parts.append(f"<response_plan>{response_plan}</response_plan>")
    parts.append(f"<action>{action}</action>")
    if generation_prompt:
        parts.append(f"<generation_prompt>{generation_prompt}</generation_prompt>")
    return "<reasoning>\n" + "\n".join(parts) + "\n</reasoning>"


class ScriptedProvider:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("Unexpected call: no scripted outputs left")
        output = self.outputs.pop(0)
        if callable(output):
            output = output(kwargs)
        if isinstance(output, BaseException):
            raise output
        return str(output)

    async def aclose(self):
        return None


def fake_generation(prompt="a red fox", effective_prompt=None):
    effective_prompt = effective_prompt or prompt
    return GenerateResponse(
        id="generation-1",
        images=["/outputs/generated.png"],
        prompt=prompt,
        effective_prompt=effective_prompt,
        prompt_enhanced=effective_prompt != prompt,
        model="stable-diffusion",
        parameters=GenerationParameters(
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            num_images=1,
            seed=42,
            seeds=[42],
            elapsed_seconds=0.25,
        ),
    )


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        chat_rate_limiter.reset()
        generation_rate_limiter.reset()
        chat_service.store.clear()
        self.original_provider = chat_service.reasoning_service.provider
        # These cases assert two-pass semantics (analysis-derived action, clarify
        # short-circuit, reasoning_debug), so pin the mode instead of inheriting
        # whatever CHAT_TWO_PASS_ENABLED the environment happens to set.
        # Single-pass behaviour is covered by SinglePassFlagTests.
        self.original_two_pass = chat_service.reasoning_service.two_pass_enabled
        chat_service.reasoning_service.two_pass_enabled = True
        self.client = TestClient(app)

    def tearDown(self):
        chat_service.reasoning_service.provider = self.original_provider
        chat_service.reasoning_service.two_pass_enabled = self.original_two_pass
        chat_service.store.clear()

    def test_ambiguous_message_returns_one_question_without_hidden_record(self):
        question = "Which image or prompt should I make more dramatic?"
        provider = ScriptedProvider(
            analysis_xml(
                intent="Refine an unspecified image",
                is_ambiguous=True,
                clarifying_question=question,
                action="clarify",
            ),
            question,
        )
        chat_service.reasoning_service.provider = provider

        response = self.client.post("/api/v1/chat", json={"message": "Make it dramatic"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "needs_clarification")
        self.assertEqual(body["message"].count("?"), 1)
        self.assertIsNone(body["generation"])
        self.assertEqual(body["message"], question)
        self.assertIsNone(body["reasoning_debug"])
        self.assertNotIn("response_plan", response.text)
        self.assertNotIn("<reasoning>", response.text)
        # Two-pass: 1 analysis call + 1 response call
        self.assertEqual(len(provider.calls), 2)

    @patch("app.api.chat_routes.execute_generation", new_callable=AsyncMock)
    def test_image_intent_uses_validated_generation_options(self, execute_generation):
        prompt = "a red fox in a snowy forest"
        provider = ScriptedProvider(
            analysis_xml(
                intent="Create a fox image",
                response_plan="Resolve the scene, then confirm the generation",
                action="generate_image",
                generation_prompt=prompt,
            ),
            "Creating the fox image.",
        )
        chat_service.reasoning_service.provider = provider
        execute_generation.return_value = fake_generation(prompt)

        response = self.client.post(
            "/api/v1/chat",
            json={
                "message": "Create a red fox in snow",
                "generation": {"seed": 42, "enhance_prompt": False},
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "generated")
        self.assertEqual(body["generation"]["prompt"], prompt)
        generation_payload = execute_generation.call_args.args[0]
        self.assertEqual(generation_payload.prompt, prompt)
        self.assertEqual(generation_payload.seed, 42)
        self.assertEqual(chat_service.store.get_last_generation_prompt(body["session_id"]), prompt)
        self.assertEqual(body["message"], "Creating the fox image.")
        self.assertIsNone(body["reasoning_debug"])
        # Two-pass: one analysis call and one response call
        self.assertEqual(len(provider.calls), 2)

    @patch.object(settings, "DEBUG", True)
    @patch("app.api.chat_routes.execute_generation", new_callable=AsyncMock)
    def test_debug_flag_exposes_reasoning_and_timing_metadata(self, execute_generation):
        execute_generation.return_value = fake_generation(
            "A minimalist vector logo of a hummingbird"
        )
        provider = ScriptedProvider(
            analysis_xml(
                intent="Generate a hummingbird icon",
                context_notes="User requested minimalist vector style",
                is_ambiguous=False,
                constraints="none",
                response_plan="Confirm vector image generation",
                action="generate_image",
                generation_prompt="A minimalist vector logo of a hummingbird",
            ),
            "I'm creating your hummingbird logo now.",
        )
        chat_service.reasoning_service.provider = provider

        response = self.client.post(
            "/api/v1/chat?debug=true",
            json={"message": "Create a minimalist hummingbird logo"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNotNone(body["timing"])
        self.assertIsNotNone(body["timing"]["reasoning_ms"])
        self.assertGreaterEqual(body["timing"]["total_ms"], 0)

        self.assertIsNotNone(body["reasoning_debug"])
        debug_info = body["reasoning_debug"]
        self.assertEqual(debug_info["intent"], "Generate a hummingbird icon")
        self.assertEqual(debug_info["context_notes"], "User requested minimalist vector style")
        self.assertFalse(debug_info["ambiguity"]["is_ambiguous"])
        self.assertEqual(debug_info["action"], "generate_image")
        self.assertEqual(
            debug_info["generation_prompt"],
            "A minimalist vector logo of a hummingbird",
        )
        self.assertFalse(debug_info["fallback_used"])

    def test_multi_turn_request_sends_ordered_public_history(self):
        provider = ScriptedProvider(
            # Turn 1: pass 1 + pass 2
            analysis_xml(intent="Explain CPU support"),
            "Stable Diffusion can run on CPU.",
            # Turn 2: pass 1 + pass 2
            analysis_xml(
                intent="Keep the prior answer concise",
                context_notes="The user asked about CPU support",
            ),
            "It will be slower on CPU.",
        )
        chat_service.reasoning_service.provider = provider
        first = self.client.post(
            "/api/v1/chat",
            json={"message": "Can Stable Diffusion run on CPU?"},
        ).json()

        second = self.client.post(
            "/api/v1/chat",
            json={"session_id": first["session_id"], "message": "Keep it concise"},
        )

        self.assertEqual(second.status_code, 200)
        # calls: [0]=turn-1 pass1, [1]=turn-1 pass2, [2]=turn-2 pass1, [3]=turn-2 pass2
        second_messages = provider.calls[2]["messages"]
        self.assertEqual(
            [(item.role, item.content) for item in second_messages],
            [
                ("user", "Can Stable Diffusion run on CPU?"),
                ("assistant", "Stable Diffusion can run on CPU."),
                ("user", "Keep it concise"),
            ],
        )
        self.assertEqual(second.json()["status"], "responded")
        self.assertEqual(len(provider.calls), 4)

    def test_invalid_structured_output_uses_direct_fallback_without_leaking_it(self):
        provider = ScriptedProvider(
            '{"invalid_json_missing_keys": true}',
            "A concise direct response.",
        )
        chat_service.reasoning_service.provider = provider

        response = self.client.post("/api/v1/chat", json={"message": "Help me"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "fallback_completed")
        self.assertEqual(response.json()["message"], "A concise direct response.")
        self.assertEqual(len(provider.calls), 2)

    def test_double_provider_failure_degrades_to_stable_clarification(self):
        provider = ScriptedProvider(TimeoutError("secret"), RuntimeError("also secret"))
        chat_service.reasoning_service.provider = provider

        response = self.client.post("/api/v1/chat", json={"message": "Make it better"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "needs_clarification")
        self.assertEqual(response.json()["message"].count("?"), 1)
        self.assertNotIn("secret", response.text)
        self.assertEqual(len(provider.calls), 2)

    def test_refusal_maps_to_public_status_without_internal_constraint(self):
        provider = ScriptedProvider(
            analysis_xml(
                intent="Handle an unsafe request",
                constraints="Request violates the configured safety boundary.",
                response_plan="Decline, then offer a safe alternative",
                action="refuse",
            ),
            "I can't help create that image, but I can help with a safe alternative.",
        )
        chat_service.reasoning_service.provider = provider

        response = self.client.post("/api/v1/chat", json={"message": "unsafe request"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "refused")
        self.assertIsNone(response.json()["generation"])
        self.assertNotIn("constraints", response.text)
        self.assertNotIn("configured safety boundary", response.text)

    @patch.object(settings, "DEBUG", False)
    def test_debug_flag_is_ignored_when_deployment_debug_is_off(self):
        provider = ScriptedProvider(
            analysis_xml(intent="Answer plainly", response_plan="Answer briefly"),
            "A plain public answer.",
        )
        chat_service.reasoning_service.provider = provider

        response = self.client.post(
            "/api/v1/chat?debug=true",
            json={"message": "hello", "debug": True},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["message"], "A plain public answer.")
        self.assertIsNone(body["reasoning_debug"])
        self.assertIsNone(body["timing"])
        self.assertNotIn("Answer plainly", response.text)
        self.assertNotIn("response_plan", response.text)

    def test_request_validation_and_session_deletion(self):
        invalid = [
            {"message": "   "},
            {"message": "test", "session_id": "not-a-uuid"},
            {"message": "test", "unknown": True},
            {"message": "test", "generation": {"width": 257}},
            {
                "message": "test",
                "generation": {"width": 2048, "height": 2048},
            },
            {
                "message": "test",
                "generation": {
                    "model": "pixart-alpha",
                    "width": 1024,
                    "height": 1024,
                },
            },
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/v1/chat", json=payload).status_code, 422)

        provider = ScriptedProvider(analysis_xml(), "Hello back.")
        chat_service.reasoning_service.provider = provider
        session_id = self.client.post("/api/v1/chat", json={"message": "hello"}).json()[
            "session_id"
        ]
        deleted = self.client.delete(f"/api/v1/chat/sessions/{session_id}")
        self.assertEqual(deleted.status_code, 204)
        self.assertIsNone(chat_service.store.get_snapshot(session_id))

    def test_chat_rate_limit_returns_retry_after(self):
        original_limit = chat_rate_limiter.limit
        chat_rate_limiter.limit = 2
        provider = ScriptedProvider(
            analysis_xml(), "One",
            analysis_xml(), "Two",
        )
        chat_service.reasoning_service.provider = provider
        try:
            self.assertEqual(
                self.client.post("/api/v1/chat", json={"message": "one"}).status_code,
                200,
            )
            self.assertEqual(
                self.client.post("/api/v1/chat", json={"message": "two"}).status_code,
                200,
            )
            response = self.client.post("/api/v1/chat", json={"message": "three"})
        finally:
            chat_rate_limiter.limit = original_limit
            chat_rate_limiter.reset()
        self.assertEqual(response.status_code, 429)
        self.assertIn("Retry-After", response.headers)

    @patch.object(settings, "DEBUG", True)
    def test_chat_debug_fallback_when_analysis_is_none(self):
        provider = ScriptedProvider(
            '{"invalid_json": true}',
            "Direct fallback message",
        )
        chat_service.reasoning_service.provider = provider
        response = self.client.post(
            "/api/v1/chat?debug=true",
            json={"message": "Hello"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["reasoning_debug"]["fallback_used"])
        self.assertEqual(body["reasoning_debug"]["intent"], "none")


if __name__ == "__main__":
    unittest.main()
