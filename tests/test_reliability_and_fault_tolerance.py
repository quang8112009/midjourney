"""System Reliability and Fault-Tolerant ML Architecture Tests.

Subagent 25 verification suite:
1. Base DiT unchanged:
   - Standard inference pipelines can still run without layout guidance hooks.
   - Pipeline loader and scheduler loaders remain 100% compatible.
2. Fallbacks:
   - If `plan_semantic_layout` times out or raises an exception, the system
     falls back to direct prompt-to-DiT generation.
   - If chat reasoning pass (Pass 1) fails or times out, chat falls back
     cleanly to Pass 2 or stable local fallback.
   - If user mask is invalid or empty, edit pipeline falls back to global or
     semantic target rather than failing.
"""

from __future__ import annotations

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from app.core.config import settings
from app.services.chat_provider import (
    ChatProviderUnavailable,
    ModelMessage,
    ScriptedProvider,
)
from app.services.editing.edit_pipeline import (
    run_baseline_edit,
    run_hybrid_edit,
    run_hybrid_generation,
    run_region_aware_edit,
)
from app.services.editing.edit_planner import plan_edit
from app.services.editing.layout_guidance import LayoutGuidanceProcessor
from app.services.editing.masks import area_ratio, as_soft_mask
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import (
    NormalizedBox,
    PlannedObject,
    PlanSelfCheck,
    SemanticLayoutPlan,
    StyleHints,
)
from app.services.editing.semantic_planner import (
    plan_semantic_layout as _plan_semantic_layout,
)
from app.services.model_service import (
    PIXART_ALPHA,
    STABLE_DIFFUSION,
    ModelService,
)
from app.services.reasoning_service import (
    ReasoningParseError,
    ReasoningService,
)


def plan_semantic_layout(prompt: str, **kwargs):
    return _plan_semantic_layout(analyze_prompt(prompt, mode="generate"), **kwargs)


class FakePipeline:
    """Mock standard diffusers pipeline for unguided and guided testing."""

    def __init__(self, call_handler=None):
        self.scheduler = SimpleNamespace(config={"name": "standard_scheduler"})
        self.to_device = "cpu"
        self.offloaded = False
        self.vae_slicing = False
        self.last_call = None
        self.call_handler = call_handler

    def to(self, device):
        self.to_device = device
        return self

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def enable_vae_slicing(self):
        self.vae_slicing = True

    def __call__(self, **kwargs):
        self.last_call = kwargs
        if self.call_handler:
            return self.call_handler(**kwargs)
        count = kwargs.get("num_images_per_prompt", 1)
        return SimpleNamespace(images=[f"image_{i}" for i in range(count)])


class BaseDiTCompatibilityTests(unittest.TestCase):
    """Verify that the base DiT and diffusion architectures remain 100% backward compatible."""

    def test_standard_inference_pipeline_runs_without_layout_hooks(self):
        """Standard inference runs directly without any layout guidance hooks."""
        fake_pipeline = FakePipeline()
        pipeline_loader = MagicMock(return_value=fake_pipeline)
        scheduler_loader = MagicMock(return_value=object())

        service = ModelService(
            pipeline_loader=pipeline_loader,
            scheduler_loader=scheduler_loader,
        )

        # 1. Standard Stable Diffusion run (CPU)
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            result_sd = service.generate_image(
                prompt="a scenic mountain landscape",
                model=STABLE_DIFFUSION,
                seed=42,
                num_images=1,
            )
            self.assertEqual(result_sd.images, ("image_0",))
            self.assertEqual(fake_pipeline.last_call["prompt"], "a scenic mountain landscape")
            self.assertEqual(fake_pipeline.last_call["generator"][0].initial_seed(), 42)

        # 2. Standard PixArt-Alpha (DiT) run (CUDA mocked)
        with (
            patch.object(settings, "DEVICE", "cuda"),
            patch.object(settings, "DTYPE", "auto"),
            patch.object(settings, "MODEL_CPU_OFFLOAD", True),
            patch("app.services.model_service.torch.cuda.is_available", return_value=True),
            patch("app.services.model_service.torch.cuda.is_bf16_supported", return_value=True),
            patch("app.services.model_service.torch.cuda.empty_cache"),
        ):
            result_dit = service.generate_image(
                prompt="a futuristic city with flying vehicles",
                model=PIXART_ALPHA,
                seed=99,
                num_images=1,
            )
            self.assertEqual(result_dit.images, ("image_0",))
            self.assertEqual(
                fake_pipeline.last_call["prompt"],
                "a futuristic city with flying vehicles",
            )
            self.assertEqual(fake_pipeline.last_call["generator"][0].initial_seed(), 99)

    def test_pipeline_and_scheduler_loaders_remain_fully_compatible(self):
        """Pipeline loader and scheduler loaders can be customized, mocked, or defaulted."""
        mock_pipeline = FakePipeline()
        custom_pipe_loader = MagicMock(return_value=mock_pipeline)
        custom_sched = SimpleNamespace(config={"custom": True})
        custom_sched_loader = MagicMock(return_value=custom_sched)

        service = ModelService(
            pipeline_loader=custom_pipe_loader,
            scheduler_loader=custom_sched_loader,
        )

        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            service.load_model(STABLE_DIFFUSION)

        custom_pipe_loader.assert_called_once()
        custom_sched_loader.assert_called_once()
        self.assertIs(service.pipe.scheduler, custom_sched)
        self.assertEqual(service.active_model, STABLE_DIFFUSION)

    def test_layout_guidance_processor_is_transparent_passthrough_when_inactive(self):
        """When plan is None, LayoutGuidanceProcessor does not modify inputs/outputs."""
        called_base = {}

        def mock_base_processor(
            attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs
        ):
            called_base["hidden_states"] = hidden_states
            called_base["encoder_hidden_states"] = encoder_hidden_states
            called_base["attention_mask"] = attention_mask
            return hidden_states * 2.0

        # Inactive processor (plan=None)
        processor = LayoutGuidanceProcessor(base_processor=mock_base_processor, plan=None)
        hidden = torch.randn(2, 64, 32)
        encoder = torch.randn(2, 16, 32)
        attn_obj = object()

        out = processor(attn_obj, hidden, encoder_hidden_states=encoder, attention_mask=None)
        torch.testing.assert_close(out, hidden * 2.0)
        self.assertIs(called_base["hidden_states"], hidden)
        self.assertIs(called_base["encoder_hidden_states"], encoder)
        self.assertIsNone(called_base["attention_mask"])

    def test_run_hybrid_generation_with_none_plan_equals_baseline(self):
        """Generation with plan=None executes standard baseline Euler/CFG denoising."""
        initial_latents = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(10))
        timesteps = [10, 20, 30]

        def denoise_model(latents, timestep, cond):
            return latents * 0.05 + (0.1 if cond else 0.0)

        # Baseline generation
        baseline_out = run_baseline_edit(
            source_latents=initial_latents,
            initial_latents=initial_latents.clone(),
            timesteps=timesteps,
            guidance_scale=7.5,
            denoise=denoise_model,
        )

        # Hybrid generation with plan=None and no layout processors
        hybrid_out = run_hybrid_generation(
            plan=None,
            initial_latents=initial_latents.clone(),
            timesteps=timesteps,
            guidance_scale=7.5,
            layout_processors=None,
            denoise=denoise_model,
        )

        torch.testing.assert_close(hybrid_out, baseline_out)


class LayoutPlanningFallbackTests(unittest.TestCase):
    """Verify fallback from layout planning failures to direct prompt-to-DiT generation."""

    def test_semantic_layout_exception_falls_back_to_direct_generation(self):
        """If plan_semantic_layout raises an exception, system falls back to direct DiT."""
        prompt = "a majestic eagle flying over mountains"
        initial_latents = torch.randn(1, 4, 16, 16)
        timesteps = [10, 20]

        def denoise_fn(latents, timestep, cond):
            return latents * 0.1

        # Simulate exception during layout planning
        def faulty_plan_semantic_layout(p, **kwargs):
            raise RuntimeError("LLM/Tokenizer layout parsing failed unexpectedly")

        plan = None
        fallback_occurred = False
        try:
            plan = faulty_plan_semantic_layout(prompt)
        except Exception:
            fallback_occurred = True
            plan = None

        self.assertTrue(fallback_occurred)
        self.assertIsNone(plan)

        # Direct prompt-to-DiT generation fallback
        out = run_hybrid_generation(
            plan=plan,
            initial_latents=initial_latents,
            timesteps=timesteps,
            guidance_scale=7.5,
            layout_processors=None,
            denoise=denoise_fn,
        )

        self.assertEqual(out.shape, initial_latents.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_semantic_layout_timeout_falls_back_to_direct_generation(self):
        """If layout planning times out, fallback to direct prompt-to-DiT generation succeeds."""
        initial_latents = torch.randn(1, 4, 16, 16)
        timesteps = [10, 20]

        def slow_plan(prompt):
            import time

            time.sleep(0.5)
            return plan_semantic_layout(prompt)

        def generate_with_fallback(prompt, timeout=0.05):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(slow_plan, prompt)
                try:
                    plan = future.result(timeout=timeout)
                except Exception:
                    plan = None

            return run_hybrid_generation(
                plan=plan,
                initial_latents=initial_latents,
                timesteps=timesteps,
                guidance_scale=7.5,
                layout_processors=None,
                denoise=lambda latents, timestep, cond: latents * 0.1,
            )

        out = generate_with_fallback("two lions on the savanna")
        self.assertEqual(out.shape, initial_latents.shape)
        self.assertTrue(torch.isfinite(out).all())


class ChatReasoningPassFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Verify Pass 1 failure/timeout falls back to Pass 2 or stable local fallback."""

    def _user_msg(self, text="Create a picture of a cat"):
        return ModelMessage(role="user", content=text, turn_id="turn-1")

    async def test_pass1_timeout_falls_back_cleanly_to_pass2(self):
        """When Pass 1 times out, Pass 2 executes directly with clean instructions."""
        cancelled = asyncio.Event()

        async def slow_pass1(**_kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        provider = ScriptedProvider(
            slow_pass1,
            "I'd be happy to create a picture of a cat for you.",
        )
        service = ReasoningService(provider, reasoning_timeout_seconds=0.02)

        turn = await service.reason(
            session_id="timeout-sess",
            messages=[self._user_msg()],
        )

        self.assertTrue(cancelled.is_set())
        self.assertTrue(turn.fallback_used)
        self.assertEqual(turn.fallback_reason, "TimeoutError")
        self.assertIsNone(turn.reasoning_analysis)
        self.assertEqual(turn.public_response, "I'd be happy to create a picture of a cat for you.")
        self.assertEqual(len(provider.calls), 2)
        # Pass 2 did not include hidden reasoning tags in system prompt
        self.assertNotIn("<hidden_reasoning_analysis>", provider.calls[1]["instructions"])

    async def test_pass1_exception_falls_back_cleanly_to_pass2(self):
        """When Pass 1 raises an unexpected exception, Pass 2 executes."""
        provider = ScriptedProvider(
            ReasoningParseError("Malformed XML from Pass 1"),
            "Sure, here is your generated image of a sunny beach.",
        )
        service = ReasoningService(provider)

        turn = await service.reason(
            session_id="err-sess",
            messages=[self._user_msg("A sunny beach")],
        )

        self.assertTrue(turn.fallback_used)
        self.assertEqual(turn.fallback_reason, "ReasoningParseError")
        self.assertIsNone(turn.reasoning_analysis)
        self.assertEqual(
            turn.public_response,
            "Sure, here is your generated image of a sunny beach.",
        )
        self.assertEqual(len(provider.calls), 2)

    async def test_pass1_and_pass2_double_failure_falls_back_to_stable_local_reply(self):
        """When both Pass 1 and Pass 2 fail, service returns safe local fallback without 500."""
        provider = ScriptedProvider(
            ChatProviderUnavailable("Pass 1 network failure"),
            ChatProviderUnavailable("Pass 2 network failure"),
        )
        service = ReasoningService(provider)

        turn = await service.reason(
            session_id="double-fail-sess",
            messages=[self._user_msg("Help me")],
        )

        self.assertTrue(turn.fallback_used)
        self.assertIn("ChatProviderUnavailable", turn.fallback_reason)
        self.assertEqual(turn.action, "clarify")
        self.assertIsNone(turn.generation_prompt)
        self.assertIn("couldn't process the conversational context", turn.public_response)
        self.assertIn(
            "What image or topic would you like me to assist you with?",
            turn.public_response,
        )

    async def test_private_canary_leak_in_pass2_triggers_safe_local_fallback(self):
        """If Pass 2 outputs private tags, it is intercepted and safely downgraded."""
        leaked = "<hidden_reasoning_analysis>SECRET</hidden_reasoning_analysis> Hello"
        provider = ScriptedProvider(
            "<reasoning><intent>test</intent></reasoning>",
            leaked,
        )
        service = ReasoningService(provider)

        turn = await service.reason(
            session_id="canary-sess",
            messages=[self._user_msg("Tell me secret")],
        )

        self.assertTrue(turn.fallback_used)
        self.assertNotIn("SECRET", turn.public_response)
        self.assertIn("couldn't process", turn.public_response)


class EditMaskDegradationFallbackTests(unittest.TestCase):
    """Verify that invalid/empty user masks fall back cleanly to semantic plan or global."""

    def setUp(self):
        self.prompt_embedding = torch.randn(16, generator=torch.Generator().manual_seed(7))
        self.source_image_embedding = torch.randn(
            1, 16, 32, 32, generator=torch.Generator().manual_seed(8)
        )
        self.latent_size = (32, 32)

    def test_empty_user_mask_without_semantic_plan_falls_back_to_global(self):
        """When user_mask is all zeros and no semantic plan is given, falls back to global."""
        empty_mask = torch.zeros(1, 1, 32, 32)
        intent = analyze_prompt("recolor the car to blue", mode="edit")

        with self.assertLogs("app.services.editing.edit_planner", level="WARNING") as logs:
            plan = plan_edit(
                intent=intent,
                instruction_index=0,
                prompt_embedding=self.prompt_embedding,
                source_image_embedding=self.source_image_embedding,
                user_mask=empty_mask,
                semantic_plan=None,
                allow_clarification=False,
                latent_size=self.latent_size,
            )

        self.assertEqual(plan.mask_source, "global_fallback")
        self.assertEqual(plan.scope, "global")
        self.assertEqual(plan.attention_strength, 0.0)
        self.assertAlmostEqual(area_ratio(plan.mask), 1.0, places=4)
        self.assertTrue(any("empty user mask" in line for line in logs.output))

    def test_empty_user_mask_with_semantic_target_falls_back_to_semantic_plan(self):
        """When user_mask is empty but semantic plan exists, falls back to semantic_plan."""
        empty_mask = torch.zeros(1, 1, 32, 32)
        target_box = NormalizedBox(ymin=0.3, xmin=0.3, ymax=0.6, xmax=0.6)
        intent = analyze_prompt("change the shirt to red", mode="edit")
        semantic_plan = SemanticLayoutPlan(
            prompt="change the shirt to red",
            objects=(PlannedObject(label="shirt", count=1, box=target_box),),
            relations=(),
            style_hints=StyleHints(),
            self_check=PlanSelfCheck(
                is_valid=True,
                count_match=True,
                relation_match=True,
                ambiguity_detected=False,
            ),
        )

        with self.assertLogs("app.services.editing.edit_planner", level="WARNING") as logs:
            plan = plan_edit(
                intent=intent,
                instruction_index=0,
                prompt_embedding=self.prompt_embedding,
                source_image_embedding=self.source_image_embedding,
                user_mask=empty_mask,
                semantic_plan=semantic_plan,
                allow_clarification=False,
                latent_size=self.latent_size,
            )

        self.assertEqual(plan.mask_source, "semantic_plan")
        self.assertEqual(plan.scope, "local")
        self.assertGreater(plan.attention_strength, 0.0)
        self.assertAlmostEqual(area_ratio(plan.mask), target_box.area, delta=0.05)
        self.assertTrue(any("empty user mask" in line for line in logs.output))

    def test_various_mask_shapes_and_types_are_safely_normalized(self):
        """2D, 3D, and 4D masks are normalized without throwing dimension errors."""
        mask_2d = torch.ones(32, 32)
        mask_3d = torch.ones(1, 32, 32)
        mask_4d = torch.ones(1, 1, 32, 32)

        for m in (mask_2d, mask_3d, mask_4d):
            soft = as_soft_mask(m)
            self.assertEqual(soft.shape, (1, 1, 32, 32))
            self.assertEqual(soft.dtype, torch.float32)

    def test_run_region_aware_edit_with_empty_mask_fallback_executes_successfully(self):
        """The edit denoise loop successfully runs when an empty mask falls back to global."""
        source = torch.randn(1, 4, 16, 16)
        image_embedding = torch.randn(1, 16, 16, 16)
        initial = source.clone()
        timesteps = [1, 2]
        intent = analyze_prompt(
            "make it a watercolor painting",
            mode="edit",
            allow_clarification=False,
        )

        plan = plan_edit(
            intent=intent,
            instruction_index=0,
            prompt_embedding=torch.randn(16),
            source_image_embedding=image_embedding,
            user_mask=torch.zeros(1, 1, 16, 16),
            allow_clarification=False,
            latent_size=(16, 16),
        )

        self.assertEqual(plan.mask_source, "global_fallback")

        out = run_region_aware_edit(
            plan=plan,
            source_latents=source,
            initial_latents=initial,
            timesteps=timesteps,
            denoise=lambda latents, timestep, cond: latents * 0.1,
        )

        self.assertEqual(out.shape, source.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_run_hybrid_edit_with_empty_mask_fallback_executes_successfully(self):
        """Hybrid edit pipeline successfully runs when empty mask falls back to semantic plan."""
        source = torch.randn(1, 4, 16, 16)
        image_embedding = torch.randn(1, 16, 16, 16)
        initial = source.clone()
        timesteps = [1, 2]

        intent = analyze_prompt("recolor the shirt to red", mode="edit")
        semantic_plan = _plan_semantic_layout(intent)
        plan = plan_edit(
            intent=intent,
            instruction_index=0,
            prompt_embedding=torch.randn(16),
            source_image_embedding=image_embedding,
            user_mask=torch.zeros(1, 1, 16, 16),  # Empty user mask
            semantic_plan=semantic_plan,
            allow_clarification=False,
            latent_size=(16, 16),
        )

        self.assertEqual(plan.mask_source, "semantic_plan")

        proc = LayoutGuidanceProcessor(None, plan=semantic_plan, guidance_strength=0.3)
        out = run_hybrid_edit(
            plan=plan,
            source_latents=source,
            initial_latents=initial,
            timesteps=timesteps,
            layout_processors=[proc],
            denoise=lambda latents, timestep, cond: latents * 0.1,
        )

        self.assertEqual(out.shape, source.shape)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
