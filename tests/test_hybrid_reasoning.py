"""Evaluation suite for the Hybrid Reasoning & Soft Layout Guidance DiT pipeline.

Targets known diffusion failure modes:
1. Object count accuracy (e.g., "three red apples and two green pears")
2. Relation correctness (e.g., "monkey riding giraffe" vs "giraffe riding monkey")
3. Edit-region accuracy (precise edit targeting without leaking into context)
4. Aesthetic-quality control set (unconstrained creative freedom on style/lighting tokens)
5. Soft vs hard guidance ablation (soft bias preserves stylistic diversity)
"""

import unittest

import torch

from app.services.editing.edit_pipeline import (
    EditTrace,
    run_hybrid_edit,
    run_hybrid_generation,
    set_layout_guidance,
)
from app.services.editing.edit_planner import (
    align_token_roles,
    locate_edit_tokens,
    map_pieces_to_words,
    plan_edit,
)
from app.services.editing.layout_guidance import (
    CosineSchedule,
    DepthAwareSchedule,
    LayoutGuidanceProcessor,
    LinearSchedule,
    TwoPhaseSchedule,
    ablation_soft_vs_hard,
    apply_layout_guidance,
    build_layout_guidance_bias,
    compute_attention_entropy,
    compute_gradient_flow,
)
from app.services.editing.metrics import evaluate_aesthetic_freedom
from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import (
    AdaptiveGuidanceConfig,
    DensityField,
    GaussianSpatialPrior,
    NormalizedBox,
    PlanSelfCheck,
    SemanticLayoutPlan,
    VisualContext,
    VisualEntity,
    _compute_layout_boxes,
    _extract_quantified_nouns,
    compute_adaptive_guidance_strength,
    extract_style_hints,
)
from app.services.editing.semantic_planner import (
    plan_semantic_layout as _plan_semantic_layout,
)
from app.services.editing.vision_backbone import (
    MockVisionBackbone,
    VisionFeatureProjector,
    VisualFeatureMap,
)


def plan_semantic_layout(prompt: str, **kwargs):
    """Test convenience that still exercises the strict PromptIntent boundary."""
    return _plan_semantic_layout(
        analyze_prompt(prompt, mode="generate"),
        **kwargs,
    )


class MockTokenizer:
    """Mock tokenizer returning token IDs and readable sub-word pieces."""

    def __init__(self, pieces: list[str] | None = None):
        self.pieces = pieces

    def __call__(self, prompt: str, add_special_tokens: bool = True):
        if self.pieces is not None:
            return {"input_ids": list(range(len(self.pieces)))}
        words = prompt.split()
        tokens = ["<s>"] + words + ["</s>"]
        return {"input_ids": list(range(len(tokens)))}

    def convert_ids_to_tokens(self, ids: list[int]):
        if self.pieces is not None:
            return self.pieces
        return ["<s>", "three", "red", "apples", "and", "two", "green", "pears", "</s>"]


class ObjectCountAccuracyTests(unittest.TestCase):
    def test_layout_planner_rejects_raw_strings(self):
        with self.assertRaises(TypeError):
            _plan_semantic_layout("two cats")

    def test_multi_object_counts_are_correctly_planned(self):
        prompt = "three red apples and two green pears on a rustic wooden table"
        plan = plan_semantic_layout(prompt)

        labels = {obj.label: obj.count for obj in plan.objects}
        self.assertIn("apples", labels)
        self.assertEqual(labels["apples"], 3)
        self.assertIn("pears", labels)
        self.assertEqual(labels["pears"], 2)

        self.assertTrue(plan.self_check.is_valid)
        self.assertTrue(plan.self_check.count_match)

        # Ensure spatial boxes for apples and pears do not fully overlap
        apple_box = next(obj.box for obj in plan.objects if obj.label == "apples")
        pear_box = next(obj.box for obj in plan.objects if obj.label == "pears")
        self.assertNotEqual(apple_box.center, pear_box.center)

    def test_single_object_default_count(self):
        prompt = "a majestic red fox standing in a snowy meadow"
        plan = plan_semantic_layout(prompt)

        self.assertTrue(any(obj.label == "fox" and obj.count == 1 for obj in plan.objects))
        self.assertTrue(plan.self_check.is_valid)

    def test_digit_counts_and_number_words(self):
        prompt = "4 white swans and 5 black ducks on a crystal lake"
        plan = plan_semantic_layout(prompt)

        labels = {obj.label: obj.count for obj in plan.objects}
        self.assertIn("swans", labels)
        self.assertEqual(labels["swans"], 4)
        self.assertIn("ducks", labels)
        self.assertEqual(labels["ducks"], 5)
        self.assertTrue(plan.self_check.is_valid)

    def test_empty_prompt_handling(self):
        plan = plan_semantic_layout("")
        self.assertFalse(plan.self_check.is_valid)
        self.assertTrue(plan.self_check.ambiguity_detected)
        self.assertEqual(len(plan.objects), 0)


class RelationCorrectnessTests(unittest.TestCase):
    def test_monkey_riding_giraffe_directionality(self):
        prompt = "a cheerful monkey riding a tall giraffe"
        plan = plan_semantic_layout(prompt)

        self.assertEqual(len(plan.relations), 1)
        rel = plan.relations[0]
        self.assertEqual(rel.subject, "monkey")
        self.assertEqual(rel.relation_type, "riding")
        self.assertEqual(rel.object, "giraffe")

        # Monkey (rider) must be positioned physically ABOVE the giraffe (mount)
        boxes = {obj.label: obj.box for obj in plan.objects}
        self.assertIn("monkey", boxes)
        self.assertIn("giraffe", boxes)
        self.assertLess(boxes["monkey"].ymin, boxes["giraffe"].ymin)
        self.assertLess(boxes["monkey"].center[0], boxes["giraffe"].center[0])

    def test_reversed_relation_reverses_spatial_layout(self):
        # Reversed prompt: "a giraffe riding a monkey"
        prompt = "a giraffe riding a small monkey"
        plan = plan_semantic_layout(prompt)

        rel = plan.relations[0]
        self.assertEqual(rel.subject, "giraffe")
        self.assertEqual(rel.relation_type, "riding")
        self.assertEqual(rel.object, "monkey")

        boxes = {obj.label: obj.box for obj in plan.objects}
        # Giraffe (now rider) must be placed on top of monkey
        self.assertLess(boxes["giraffe"].ymin, boxes["monkey"].ymin)

    def test_side_by_side_relation(self):
        prompt = "a golden retriever sitting next to a fluffy cat"
        plan = plan_semantic_layout(prompt)

        rel = plan.relations[0]
        self.assertEqual(rel.relation_type, "next_to")
        boxes = {obj.label: obj.box for obj in plan.objects}
        self.assertIn("retriever", boxes)
        self.assertIn("cat", boxes)
        # Horizontal separation
        self.assertNotEqual(boxes["retriever"].xmin, boxes["cat"].xmin)

    def test_nested_inside_relation(self):
        prompt = "a golden key inside a crystal box"
        plan = plan_semantic_layout(prompt)

        self.assertEqual(len(plan.relations), 1)
        rel = plan.relations[0]
        self.assertEqual(rel.subject, "key")
        self.assertEqual(rel.relation_type, "inside")
        self.assertEqual(rel.object, "box")

        boxes = {obj.label: obj.box for obj in plan.objects}
        self.assertIn("key", boxes)
        self.assertIn("box", boxes)
        self.assertTrue(boxes["box"].contains(boxes["key"]))
        self.assertLess(boxes["key"].area, boxes["box"].area)

    def test_under_relation_directionality(self):
        prompt = "a sleeping cat under a wooden table"
        plan = plan_semantic_layout(prompt)

        self.assertEqual(len(plan.relations), 1)
        rel = plan.relations[0]
        self.assertEqual(rel.subject, "cat")
        self.assertEqual(rel.relation_type, "under")
        self.assertEqual(rel.object, "table")

        boxes = {obj.label: obj.box for obj in plan.objects}
        self.assertIn("cat", boxes)
        self.assertIn("table", boxes)
        # Cat (subject) positioned lower vertically than table (object)
        self.assertGreater(boxes["cat"].ymin, boxes["table"].ymin)

    def test_in_front_of_and_behind_relations(self):
        prompt_front = "a warrior in front of a giant castle"
        plan_front = plan_semantic_layout(prompt_front)
        self.assertEqual(plan_front.relations[0].relation_type, "in_front_of")
        boxes_f = {obj.label: obj.box for obj in plan_front.objects}
        self.assertGreater(boxes_f["warrior"].ymin, boxes_f["castle"].ymin)

        prompt_behind = "a glowing moon behind dark clouds"
        plan_behind = plan_semantic_layout(prompt_behind)
        self.assertEqual(plan_behind.relations[0].relation_type, "behind")
        boxes_b = {obj.label: obj.box for obj in plan_behind.objects}
        self.assertLess(boxes_b["moon"].ymin, boxes_b["clouds"].ymin)

    def test_multiple_unlinked_objects_allocation(self):
        prompt = "a cat, a dog, and a rabbit"
        plan = plan_semantic_layout(prompt)

        self.assertEqual(len(plan.objects), 3)
        labels = [obj.label for obj in plan.objects]
        self.assertEqual(labels, ["cat", "dog", "rabbit"])

        # Disjoint horizontal bands
        cat_box = plan.objects[0].box
        dog_box = plan.objects[1].box
        rab_box = plan.objects[2].box
        self.assertLess(cat_box.xmax, dog_box.xmin + 0.05)
        self.assertLess(dog_box.xmax, rab_box.xmin + 0.05)

        # Ambiguity flag and assumption logging
        self.assertTrue(plan.self_check.ambiguity_detected)
        self.assertTrue(any("unlinked" in a for a in plan.self_check.assumptions))


class EditTargetAccuracyTests(unittest.TestCase):
    def test_semantic_layout_is_layout_only(self):
        prompt = "change the shirt color to vibrant crimson"
        intent = analyze_prompt(prompt, mode="edit")
        plan = _plan_semantic_layout(intent)

        self.assertTrue(any(obj.label == "shirt" for obj in plan.objects))
        self.assertFalse(hasattr(plan, "edit_target"))
        self.assertNotIn("edit_target", plan.to_dict())
        self.assertEqual(intent.instructions[0].target, "shirt")
        self.assertEqual(intent.instructions[0].action, "recolor")

    def test_edit_planner_incorporates_semantic_plan(self):
        prompt = "recolor the jacket to dark blue"
        intent = analyze_prompt(prompt, mode="edit")
        semantic_plan = _plan_semantic_layout(intent)

        dim = 16
        prompt_emb = torch.randn(dim)
        image_emb = torch.randn(1, dim, 32, 32)

        edit_plan = plan_edit(
            intent=intent,
            instruction_index=0,
            prompt_embedding=prompt_emb,
            source_image_embedding=image_emb,
            semantic_plan=semantic_plan,
            allow_clarification=False,
            latent_size=(32, 32),
        )

        self.assertEqual(edit_plan.mask_source, "semantic_plan")
        self.assertIsNotNone(edit_plan.semantic_plan)
        self.assertEqual(edit_plan.scope, "local")

    def test_edit_planner_handles_semantic_plan_with_explicit_mask(self):
        prompt = "recolor the shirt to red"
        intent = analyze_prompt(prompt, mode="edit")
        custom_mask = torch.zeros(1, 1, 16, 16)
        custom_mask[..., 4:12, 4:12] = 1.0
        plan = _plan_semantic_layout(intent)

        dim = 16
        prompt_emb = torch.randn(dim)
        image_emb = torch.randn(1, dim, 32, 32)

        edit_plan = plan_edit(
            intent=intent,
            instruction_index=0,
            prompt_embedding=prompt_emb,
            source_image_embedding=image_emb,
            user_mask=custom_mask,
            semantic_plan=plan,
            allow_clarification=False,
            latent_size=(32, 32),
        )

        self.assertEqual(edit_plan.mask_source, "user")
        self.assertEqual(edit_plan.mask.shape, (1, 1, 32, 32))
        self.assertGreater(float(edit_plan.mask[0, 0, 16, 16]), 0.5)

    def test_edit_planner_preserves_aspect_ratio(self):
        prompt = "recolor the car to blue"
        intent = analyze_prompt(prompt, mode="edit")
        dim = 16
        prompt_emb = torch.randn(dim)
        image_emb = torch.randn(1, dim, 32, 64)

        edit_plan = plan_edit(
            intent=intent,
            instruction_index=0,
            prompt_embedding=prompt_emb,
            source_image_embedding=image_emb,
            allow_clarification=False,
            latent_size=(32, 64),
            aspect=2.0,
        )
        self.assertEqual(edit_plan.aspect, 2.0)
        self.assertIn("aspect", edit_plan.as_log_dict())


class AestheticControlSetTests(unittest.TestCase):
    """Confirm that purely aesthetic prompts receive ZERO layout bias on style tokens,
    preserving full artistic interpretation, lighting, and texture freedom."""

    @staticmethod
    def _make_tokenizer_for_prompt(prompt: str):
        words = prompt.split()
        pieces = ["<start_of_text>"] + [w + "</w>" for w in words] + ["<end_of_text>"]
        return MockTokenizer(pieces=pieces)

    def test_style_tokens_are_unconstrained(self):
        prompt = "a dreamy ethereal cyberpunk street in watercolor style with volumetric lighting"
        plan = plan_semantic_layout(prompt)

        # Style hints are extracted
        self.assertIn("watercolor", plan.style_hints.medium)
        self.assertIn("cyberpunk", plan.style_hints.mood)
        self.assertIn("volumetric lighting", plan.style_hints.lighting)
        self.assertTrue(plan.style_hints.is_unconstrained)

        # Bias matrix for cross-attention
        bias = build_layout_guidance_bias(
            plan,
            num_image_tokens=256,
            num_text_tokens=16,
            guidance_strength=0.3,
        )

        # For pure style prompt where street is unindexed or style tokens are mapped,
        # style tokens must receive 0.0 bias across all spatial positions
        for st_idx in plan.style_hints.style_tokens:
            self.assertAlmostEqual(float(bias[:, st_idx].abs().max()), 0.0, places=6)

    def test_control_set_prompts_comprehensive_extraction(self):
        """Verify the full aesthetic control set covering medium, lighting, mood."""
        control_cases = [
            {
                "prompt": (
                    "a dreamy ethereal cyberpunk street in watercolor style "
                    "with volumetric lighting"
                ),
                "medium": ["watercolor"],
                "mood": ["dreamy", "ethereal", "cyberpunk"],
                "lighting": ["volumetric lighting"],
                "composition": [],
            },
            {
                "prompt": (
                    "cinematic portrait of a warrior at golden hour "
                    "with volumetric lighting and god rays"
                ),
                "medium": [],
                "mood": [],
                "lighting": ["volumetric lighting", "golden hour", "god rays"],
                "composition": ["portrait"],
            },
            {
                "prompt": (
                    "photorealistic close-up macro of dew drops on a blooming rose, 8k resolution"
                ),
                "medium": ["photorealistic"],
                "mood": [],
                "lighting": [],
                "composition": ["close-up", "macro", "8k resolution"],
            },
            {
                "prompt": "whimsical anime landscape with pastel colors and bokeh background",
                "medium": ["anime", "pastel"],
                "mood": ["whimsical"],
                "lighting": ["bokeh"],
                "composition": ["bokeh background"],
            },
        ]

        for case in control_cases:
            prompt = case["prompt"]
            with self.subTest(prompt=prompt):
                tok = self._make_tokenizer_for_prompt(prompt)
                plan = plan_semantic_layout(prompt, tokenizer=tok)

                for med in case["medium"]:
                    self.assertIn(med, plan.style_hints.medium)
                for mood in case["mood"]:
                    self.assertIn(mood, plan.style_hints.mood)
                for light in case["lighting"]:
                    self.assertIn(light, plan.style_hints.lighting)
                for comp in case["composition"]:
                    self.assertIn(comp, plan.style_hints.composition)

                self.assertTrue(plan.style_hints.is_unconstrained)
                self.assertGreater(len(plan.style_hints.style_tokens), 0)

    def test_zero_spatial_bias_verification_across_control_set(self):
        """Verify style tokens receive 0.0 bias across all spatial positions in control set."""
        prompts = [
            "a dreamy ethereal cyberpunk street in watercolor style with volumetric lighting",
            "cinematic portrait of a warrior at golden hour with volumetric lighting and god rays",
            "photorealistic close-up macro of dew drops on a blooming rose, 8k resolution",
            "whimsical anime landscape with pastel colors and bokeh background",
        ]

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                tok = self._make_tokenizer_for_prompt(prompt)
                num_tokens = len(tok.pieces)
                plan = plan_semantic_layout(prompt, tokenizer=tok)

                bias = build_layout_guidance_bias(
                    plan,
                    num_image_tokens=256,
                    num_text_tokens=num_tokens,
                    guidance_strength=0.3,
                )

                # Style tokens receive strictly 0.0 bias everywhere
                for st_idx in plan.style_hints.style_tokens:
                    self.assertAlmostEqual(float(bias[:, st_idx].abs().max()), 0.0, places=6)

                # Aesthetic freedom evaluation score is 1.0 (100% unconstrained)
                freedom = evaluate_aesthetic_freedom(plan, bias)
                self.assertTrue(freedom["zero_bias_verified"])
                self.assertEqual(freedom["aesthetic_freedom_score"], 1.0)
                for _st_idx, max_bias in freedom["style_token_max_bias"].items():
                    self.assertAlmostEqual(max_bias, 0.0, places=6)

    def test_soft_layout_guidance_adds_positive_bias(self):
        # 1 image token, 2 text tokens
        cross_attn = torch.zeros(1, 256, 4)
        box = NormalizedBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5)

        # Token 0 mapped to box, Token 1 is a style token (None)
        token_map = {0: box, 1: None}
        guided = apply_layout_guidance(
            cross_attn,
            token_to_region_map=token_map,
            guidance_strength=0.3,
        )

        # Token 0 receives +0.3 in its region
        self.assertGreater(float(guided[0, :, 0].max()), 0.25)
        # Token 1 (style) receives strictly 0.0 bias
        self.assertAlmostEqual(float(guided[0, :, 1].abs().max()), 0.0, places=6)

    def test_visual_freedom_and_prompt_diversity_metrics(self):
        """Confirms that guidance_strength=0.3 preserves style/texture on aesthetic tokens."""
        prompt = (
            "cinematic portrait of a warrior at golden hour with volumetric lighting and god rays"
        )
        tok = self._make_tokenizer_for_prompt(prompt)
        plan = plan_semantic_layout(prompt, tokenizer=tok)

        # Create cross attention logits simulating DiT cross-attention (1, 256, num_tokens)
        num_tokens = len(tok.pieces)
        logits = torch.randn(1, 256, num_tokens)

        # Apply guidance with strength 0.3
        guided_logits = apply_layout_guidance(logits.clone(), plan=plan, guidance_strength=0.3)

        # Cross attention on style tokens must be identical to unguided logits (0 bias added)
        for st_idx in plan.style_hints.style_tokens:
            diff = (guided_logits[..., st_idx] - logits[..., st_idx]).abs().max()
            self.assertAlmostEqual(float(diff), 0.0, places=6)

        # Soft guidance retains full attention entropy and gradient flow on unconstrained positions
        warrior_box = plan.objects[0].box
        warrior_token_idx = (
            plan.objects[0].token_indices[0] if plan.objects[0].token_indices else 5
        )
        ablation = ablation_soft_vs_hard(
            logits,
            warrior_box,
            target_token_idx=warrior_token_idx,
            soft_strength=0.3,
            hard_penalty=-12.0,
        )
        self.assertAlmostEqual(ablation["soft_entropy_retention"], 1.0, places=3)
        self.assertAlmostEqual(ablation["soft_gradient_retention"], 1.0, places=3)


class SoftVsHardAblationTests(unittest.TestCase):
    def test_soft_bias_does_not_mask_out_unmapped_regions(self):
        """Unlike hard masking (-12 or -inf), soft guidance (+0.3) allows tokens
        to attend everywhere, gently encouraging planned spatial grounding."""
        logits = torch.randn(1, 64, 4)
        box = NormalizedBox(ymin=0.0, xmin=0.0, ymax=0.5, xmax=0.5)
        token_map = {0: box}

        soft_guided = apply_layout_guidance(
            logits.clone(), token_to_region_map=token_map, guidance_strength=0.3
        )

        # Attention probabilities after softmax
        soft_probs = soft_guided.softmax(dim=-1)
        # Token 0 still has non-zero probability outside the box
        outside_prob = float(soft_probs[0, 48:, 0].mean())
        self.assertGreater(outside_prob, 0.01)

    def test_apply_layout_guidance_supports_image_first_and_text_first(self):
        box = NormalizedBox(ymin=0.0, xmin=0.0, ymax=0.5, xmax=0.5)
        token_map = {0: box}

        # Image-first: (1, 64, 4)
        logits_img = torch.zeros(1, 64, 4)
        guided_img = apply_layout_guidance(
            logits_img, token_to_region_map=token_map, guidance_strength=0.3
        )
        self.assertEqual(guided_img.shape, (1, 64, 4))
        self.assertGreater(float(guided_img[0, 0, 0]), 0.25)  # inside box
        self.assertEqual(float(guided_img[0, 63, 0]), 0.0)   # outside box

        # Text-first: (1, 4, 64)
        logits_txt = torch.zeros(1, 4, 64)
        guided_txt = apply_layout_guidance(
            logits_txt, token_to_region_map=token_map, guidance_strength=0.3
        )
        self.assertEqual(guided_txt.shape, (1, 4, 64))
        self.assertGreater(float(guided_txt[0, 0, 0]), 0.25)  # inside box
        self.assertEqual(float(guided_txt[0, 0, 63]), 0.0)   # outside box

    def test_entropy_and_gradient_flow_math(self):
        # Uniform logits
        uniform_logits = torch.zeros(1, 10, 4)
        ent = compute_attention_entropy(uniform_logits)
        # Expected entropy for 4 uniform outcomes is ln(4) ~ 1.3863
        import math
        self.assertAlmostEqual(float(ent.mean()), math.log(4), places=4)

        # Gradient flow for p = 0.25 -> 0.25 * 0.75 = 0.1875
        grad = compute_gradient_flow(uniform_logits, target_token_idx=0)
        self.assertAlmostEqual(float(grad.mean()), 0.1875, places=4)

    def test_ablation_soft_vs_hard_metrics(self):
        logits = torch.randn(1, 64, 8)
        box = NormalizedBox(ymin=0.2, xmin=0.2, ymax=0.8, xmax=0.8)
        results = ablation_soft_vs_hard(
            logits, box, target_token_idx=1, soft_strength=0.3, hard_penalty=-12.0
        )

        self.assertIn("soft_entropy_retention", results)
        self.assertIn("hard_entropy_retention", results)
        self.assertIn("soft_gradient_retention", results)
        self.assertIn("hard_gradient_retention", results)

        # Soft guidance retains 100% entropy and gradient outside the box
        self.assertAlmostEqual(results["soft_entropy_retention"], 1.0, places=3)
        self.assertAlmostEqual(results["soft_gradient_retention"], 1.0, places=3)

        # Hard masking severely harms gradient retention outside the box
        self.assertLess(results["hard_gradient_retention"], 0.01)

    def test_numerical_stability_across_dtypes(self):
        """Verify numerical stability and finiteness across float32, float16, and bfloat16."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                # Standard range
                logits = torch.randn(2, 32, 8, dtype=dtype)
                ent = compute_attention_entropy(logits)
                grad = compute_gradient_flow(logits, target_token_idx=1)
                self.assertEqual(ent.dtype, dtype)
                self.assertEqual(grad.dtype, dtype)
                self.assertTrue(torch.isfinite(ent).all())
                self.assertTrue(torch.isfinite(grad).all())
                self.assertTrue((ent >= 0.0).all())
                self.assertTrue(((grad >= 0.0) & (grad <= 0.25)).all())

                # Extreme logit ranges (testing underflow/overflow resistance)
                extreme = torch.tensor([[-50.0, 0.0, 50.0, -100.0]], dtype=dtype)
                ent_ext = compute_attention_entropy(extreme)
                grad_ext = compute_gradient_flow(extreme, target_token_idx=2)
                self.assertTrue(torch.isfinite(ent_ext).all())
                self.assertTrue(torch.isfinite(grad_ext).all())


class LayoutGuidanceProcessorTests(unittest.TestCase):
    def setUp(self):
        try:
            from diffusers.models.attention_processor import Attention, AttnProcessor2_0
        except ImportError:
            self.skipTest("diffusers not installed")
        self.dim = 32
        self.heads = 4
        self.attn = Attention(
            query_dim=self.dim,
            cross_attention_dim=self.dim,
            heads=self.heads,
            dim_head=self.dim // self.heads,
        )
        self.plan = plan_semantic_layout("a cute puppy on a green lawn")
        self.processor = LayoutGuidanceProcessor(
            AttnProcessor2_0(), plan=self.plan, guidance_strength=0.3, schedule_cutoff=0.8
        )
        self.attn.set_processor(self.processor)

    def test_diffusers_compatibility_across_batch_sizes(self):
        for batch in (1, 2, 4):
            with self.subTest(batch=batch):
                hidden = torch.randn(batch, 64, self.dim)
                encoder = torch.randn(batch, 8, self.dim)
                out = self.attn(hidden, encoder_hidden_states=encoder)
                self.assertEqual(tuple(out.shape), (batch, 64, self.dim))

    def test_step_progress_scheduling_cutoff(self):
        self.processor.set_step_progress(0.5)
        self.assertTrue(self.processor._active)

        # At progress >= cutoff (0.8), guidance fades out to 100% unconstrained
        self.processor.set_step_progress(0.85)
        self.assertFalse(self.processor._active)

    def test_two_phase_scheduling_boundary_and_dynamics(self):
        # Phase 1: 0.0, 0.2, 0.5, 0.79 -> active
        for prog in [0.0, 0.2, 0.5, 0.79]:
            with self.subTest(progress=prog):
                self.processor.set_step_progress(prog)
                self.assertTrue(self.processor._active)

        # Phase 2: 0.8, 0.81, 0.95, 1.0 -> deactivated
        for prog in [0.8, 0.81, 0.95, 1.0]:
            with self.subTest(progress=prog):
                self.processor.set_step_progress(prog)
                self.assertFalse(self.processor._active)

    def test_set_layout_guidance_utility(self):
        proc = LayoutGuidanceProcessor(
            None, plan=None, guidance_strength=0.0, schedule_cutoff=0.5, aspect=1.0
        )
        set_layout_guidance(
            [proc],
            self.plan,
            guidance_strength=0.4,
            schedule_cutoff=0.75,
            aspect=1.5,
        )
        self.assertEqual(proc.plan, self.plan)
        self.assertEqual(proc.guidance_strength, 0.4)
        self.assertEqual(proc.schedule_cutoff, 0.75)
        self.assertEqual(proc.aspect, 1.5)


class HybridPipelineExecutionTests(unittest.TestCase):
    def setUp(self):
        self.initial_latents = torch.randn(1, 4, 16, 16)
        self.timesteps = list(range(4))

        def denoise(latents, timestep, cond):
            return latents * 0.1

        self.denoise = denoise

    def test_run_hybrid_generation_executes_smoothly(self):
        prompt = "two cats sitting on a red rug"
        plan = plan_semantic_layout(prompt)

        out = run_hybrid_generation(
            plan=plan,
            initial_latents=self.initial_latents,
            timesteps=self.timesteps,
            denoise=self.denoise,
            guidance_scale=7.5,
        )
        self.assertEqual(out.shape, self.initial_latents.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_run_hybrid_generation_handles_none_plan_gracefully(self):
        out = run_hybrid_generation(
            plan=None,
            initial_latents=self.initial_latents,
            timesteps=self.timesteps,
            denoise=self.denoise,
            guidance_scale=7.5,
        )
        self.assertEqual(out.shape, self.initial_latents.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_run_hybrid_generation_progress_tracking_and_cutoff_transitions(self):
        plan = plan_semantic_layout("a red car on a highway")
        proc = LayoutGuidanceProcessor(
            None, plan=plan, guidance_strength=0.3, schedule_cutoff=0.8
        )
        timesteps = list(range(10))  # 10 steps -> progress values: 0.1, 0.2, ..., 1.0
        trace = EditTrace()

        recorded_states = []

        def denoise_pair_with_tracking(latents, timestep):
            recorded_states.append((proc._active, proc.schedule_cutoff))
            return latents * 0.1, latents * 0.1

        run_hybrid_generation(
            plan=plan,
            initial_latents=self.initial_latents,
            timesteps=timesteps,
            layout_processors=[proc],
            denoise_pair=denoise_pair_with_tracking,
            trace=trace,
        )

        self.assertEqual(len(trace.steps), 10)
        # Check progress values in trace
        for i, step_info in enumerate(trace.steps):
            expected_prog = (i + 1) / 10.0
            self.assertAlmostEqual(step_info["progress"], expected_prog, places=4)

        # Check processor states during execution:
        # Steps 0..7 (progress 0.1 to 0.8): Phase 1
        # progress 0.1..0.7 active=True, at 0.8 active=False since 0.8 >= 0.8 (cutoff)
        # Specifically:
        # step 0: progress=0.1 -> active=True
        # step 1: progress=0.2 -> active=True
        # ...
        # step 6: progress=0.7 -> active=True
        # step 7: progress=0.8 -> active=False (progress >= schedule_cutoff)
        # step 8: progress=0.9 -> active=False
        # step 9: progress=1.0 -> active=False
        for step_idx in range(7):  # progress 0.1 .. 0.7
            self.assertTrue(recorded_states[step_idx][0], f"Step {step_idx} should be active")
        for step_idx in range(7, 10):  # progress 0.8 .. 1.0
            self.assertFalse(recorded_states[step_idx][0], f"Step {step_idx} should be inactive")

    def test_run_hybrid_edit_progress_tracking_and_cutoff_transitions(self):
        prompt = "recolor the shirt to red"
        intent = analyze_prompt(prompt, mode="edit")
        semantic_plan = _plan_semantic_layout(intent)
        edit_plan = plan_edit(
            intent=intent,
            instruction_index=0,
            prompt_embedding=torch.randn(16),
            source_image_embedding=torch.randn(1, 16, 16, 16),
            semantic_plan=semantic_plan,
            allow_clarification=False,
            latent_size=(16, 16),
        )
        proc = LayoutGuidanceProcessor(
            None, plan=semantic_plan, guidance_strength=0.3, schedule_cutoff=0.8
        )
        timesteps = list(range(10))
        trace = EditTrace()
        recorded_states = []

        def denoise_pair_with_tracking(latents, timestep):
            recorded_states.append(proc._active)
            return latents * 0.1, latents * 0.1

        source = torch.randn(1, 4, 16, 16)
        run_hybrid_edit(
            plan=edit_plan,
            source_latents=source,
            initial_latents=self.initial_latents,
            timesteps=timesteps,
            layout_processors=[proc],
            denoise_pair=denoise_pair_with_tracking,
            trace=trace,
        )

        self.assertEqual(len(trace.steps), 10)
        for i, step_info in enumerate(trace.steps):
            expected_prog = (i + 1) / 10.0
            self.assertAlmostEqual(step_info["progress"], expected_prog, places=4)

        for step_idx in range(7):  # progress 0.1 .. 0.7
            self.assertTrue(recorded_states[step_idx], f"Step {step_idx} should be active")
        for step_idx in range(7, 10):  # progress 0.8 .. 1.0
            self.assertFalse(recorded_states[step_idx], f"Step {step_idx} should be inactive")


class QuantifierExtractionAndColumnPartitioningTests(unittest.TestCase):
    """Subagent 4 test suite: Quantifier Extraction and Spatial Column Partitioning."""

    def test_word_and_digit_numeral_parsing(self):
        cases = [
            ("one tiger", "tiger", 1),
            ("two lions", "lions", 2),
            ("three wolves", "wolves", 3),
            ("four bears", "bears", 4),
            ("five eagles", "eagles", 5),
            ("six hawks", "hawks", 6),
            ("seven dolphins", "dolphins", 7),
            ("eight whales", "whales", 8),
            ("nine foxes", "foxes", 9),
            ("ten panthers", "panthers", 10),
            ("pair of swans", "swans", 2),
            ("a pair of swans", "swans", 2),
            ("couple of pigeons", "pigeons", 2),
            ("a couple of pigeons", "pigeons", 2),
            ("trio of musicians", "musicians", 3),
            ("a trio of musicians", "musicians", 3),
            ("quad bikes", "bikes", 4),
            ("a quad of bikes", "bikes", 4),
            ("1 elephant", "elephant", 1),
            ("2 giraffes", "giraffes", 2),
            ("3 monkeys", "monkeys", 3),
            ("12 robots", "robots", 12),
        ]
        for prompt, expected_noun, expected_count in cases:
            extracted = _extract_quantified_nouns(prompt)
            self.assertTrue(len(extracted) >= 1, f"Failed for {prompt}")
            matched = [q for q in extracted if q[0] == expected_noun]
            self.assertEqual(
                len(matched), 1, f"Expected {expected_noun} in {extracted} for {prompt}"
            )
            self.assertEqual(
                matched[0][1],
                expected_count,
                f"Count mismatch for {prompt}: {matched[0][1]} != {expected_count}",
            )

    def test_leading_adjectives_and_attributes_binding(self):
        prompt = "three red apples and two green pears"
        extracted = _extract_quantified_nouns(prompt)
        expected = [("apples", 3, ["red"]), ("pears", 2, ["green"])]
        self.assertEqual(extracted, expected)

        plan = plan_semantic_layout(prompt)
        self.assertEqual(len(plan.objects), 2)
        obj_map = {o.label: o for o in plan.objects}
        self.assertIn("apples", obj_map)
        self.assertEqual(obj_map["apples"].count, 3)
        self.assertEqual(obj_map["apples"].attributes, ("red",))
        self.assertIn("pears", obj_map)
        self.assertEqual(obj_map["pears"].count, 2)
        self.assertEqual(obj_map["pears"].attributes, ("green",))

        # Complex multi-adjective prompt
        prompt_complex = "a cute fluffy small dog and one shiny golden crown"
        extracted_complex = _extract_quantified_nouns(prompt_complex)
        self.assertEqual(
            extracted_complex,
            [("dog", 1, ["cute", "fluffy", "small"]), ("crown", 1, ["shiny", "golden"])],
        )

    def test_spatial_column_partitioning_for_n_unlinked_entities(self):
        for n in range(1, 8):
            entities = [(f"item_{i}", 1, []) for i in range(n)]
            boxes_map = _compute_layout_boxes(entities, [])
            boxes = [boxes_map[f"item_{i}"] for i in range(n)]

            self.assertEqual(len(boxes), n)
            # Verify coordinates are in valid [0, 1] range
            for b in boxes:
                self.assertGreaterEqual(b.xmin, 0.0)
                self.assertLessEqual(b.xmax, 1.0)
                self.assertGreaterEqual(b.ymin, 0.0)
                self.assertLessEqual(b.ymax, 1.0)
                self.assertGreater(b.width, 0.0)
                self.assertGreater(b.height, 0.0)

            # For N > 1, verify strictly non-overlapping horizontal order and safety margin
            if n > 1:
                for i in range(n - 1):
                    b1 = boxes[i]
                    b2 = boxes[i + 1]
                    self.assertFalse(b1.overlaps(b2), f"Box {i} and {i+1} overlap for N={n}")
                    self.assertEqual(b1.iou(b2), 0.0)
                    self.assertLess(b1.xmax, b2.xmin, f"Box {i} right >= Box {i+1} left for N={n}")
                    # Positive safety margin
                    margin = b2.xmin - b1.xmax
                    self.assertGreater(margin, 0.005, f"Insufficient margin for N={n}: {margin}")

    def test_edge_cases_handling(self):
        # 1. Empty prompt
        plan_empty = plan_semantic_layout("")
        self.assertFalse(plan_empty.self_check.is_valid)
        self.assertTrue(plan_empty.self_check.ambiguity_detected)
        self.assertEqual(len(plan_empty.objects), 0)

        # 2. Single entity
        plan_single = plan_semantic_layout("a majestic dragon")
        self.assertTrue(plan_single.self_check.is_valid)
        self.assertEqual(len(plan_single.objects), 1)
        self.assertEqual(plan_single.objects[0].box, NormalizedBox(0.15, 0.15, 0.85, 0.85))

        # 3. Duplicate labels
        plan_dup = plan_semantic_layout("a cat and a cat")
        self.assertEqual(len(plan_dup.objects), 1)
        self.assertEqual(plan_dup.objects[0].label, "cat")

        # 4. Mixed quantifiers in single prompt
        plan_mixed = plan_semantic_layout("a pair of cats and 3 golden fish and a single bird")
        self.assertTrue(plan_mixed.self_check.is_valid)
        labels_counts = [(o.label, o.count, o.attributes) for o in plan_mixed.objects]
        self.assertIn(("cats", 2, ()), labels_counts)
        self.assertIn(("fish", 3, ("golden",)), labels_counts)
        self.assertIn(("bird", 1, ()), labels_counts)


class TokenizerAlignmentAndSubwordTests(unittest.TestCase):
    """Subagent 2 test suite: NLP Tokenizers and Cross-Attention Token Alignment.

    Verifies BPE (CLIP), SentencePiece (T5), WordPiece (BERT), and Byte-Level BPE (GPT-2/RoBERTa),
    ensuring exact token index alignment without drifting or substring false positives.
    """

    def test_clip_bpe_alignment_and_substring_isolation(self):
        prompt = "change the red car, but keep the carpet"
        # CLIP uses </w> suffix on word boundaries
        clip_pieces = [
            "<|startoftext|>", "change</w>", "the</w>", "red</w>", "car</w>", ",</w>",
            "but</w>", "keep</w>", "the</w>", "car", "pet</w>", "<|endoftext|>"
        ]
        tok = MockTokenizer(clip_pieces)

        # 'car' should match ONLY token 4, not the 'car' subword of 'carpet' (tokens 9, 10)
        car_tokens = locate_edit_tokens(prompt, ("car",), tok)
        self.assertEqual(car_tokens, (4,))

        # 'carpet' should match tokens (9, 10)
        carpet_tokens = locate_edit_tokens(prompt, ("carpet",), tok)
        self.assertEqual(carpet_tokens, (9, 10))

        # align_token_roles: car is edit_target, carpet is context
        instruction = analyze_prompt(prompt, mode="edit").instructions[0]
        roles = align_token_roles(instruction, tok)
        self.assertEqual(roles[0], "neutral")   # <|startoftext|>
        self.assertEqual(roles[4], "edit_target") # car</w>
        self.assertEqual(roles[5], "neutral")   # ,</w>
        self.assertEqual(roles[9], "context")   # car (subword of carpet)
        self.assertEqual(roles[10], "context")  # pet</w> (subword of carpet)
        self.assertEqual(roles[11], "neutral")  # <|endoftext|>

    def test_t5_sentencepiece_multi_token_subwords(self):
        prompt = "three red apples and two green pears on a rustic wooden table"
        # T5 uses \u2581 prefix on word starts
        t5_pieces = [
            "\u2581three", "\u2581red", "\u2581app", "les", "\u2581and",
            "\u2581two", "\u2581green", "\u2581pear", "s", "\u2581on",
            "\u2581a", "\u2581rustic", "\u2581wooden", "\u2581table", "</s>"
        ]
        tok = MockTokenizer(t5_pieces)

        apples_tokens = locate_edit_tokens(prompt, ("apples",), tok)
        pears_tokens = locate_edit_tokens(prompt, ("pears",), tok)
        self.assertEqual(apples_tokens, (2, 3))
        self.assertEqual(pears_tokens, (7, 8))

        # SemanticLayoutPlan object token mapping
        plan = plan_semantic_layout(prompt, tokenizer=tok)
        obj_apples = next(o for o in plan.objects if o.label == "apples")
        obj_pears = next(o for o in plan.objects if o.label == "pears")
        obj_table = next(o for o in plan.objects if o.label == "table")

        self.assertEqual(obj_apples.token_indices, (2, 3))
        self.assertEqual(obj_pears.token_indices, (7, 8))
        self.assertEqual(obj_table.token_indices, (13,))

        # Regions correctly registered in token_to_region_map
        self.assertEqual(plan.token_to_region_map[2], obj_apples.box)
        self.assertEqual(plan.token_to_region_map[3], obj_apples.box)
        self.assertEqual(plan.token_to_region_map[7], obj_pears.box)
        self.assertEqual(plan.token_to_region_map[8], obj_pears.box)
        self.assertEqual(plan.token_to_region_map[13], obj_table.box)

    def test_bert_wordpiece_continuation(self):
        prompt = "change photorealistic jacket crimson"
        bert_pieces = ["[CLS]", "change", "photo", "##realistic", "jacket", "crimson", "[SEP]"]
        tok = MockTokenizer(bert_pieces)

        mapping = map_pieces_to_words(bert_pieces)
        self.assertEqual(mapping, [None, 0, 1, 1, 2, 3, None])

        instruction = analyze_prompt(prompt, mode="edit").instructions[0]
        roles = align_token_roles(instruction, tok)
        self.assertEqual(roles[0], "neutral")
        self.assertEqual(roles[1], "neutral")     # change (verb)
        self.assertEqual(roles[2], "neutral")     # style cue, not the parsed target
        self.assertEqual(roles[3], "neutral")     # ##realistic
        self.assertEqual(roles[4], "edit_target") # jacket
        self.assertEqual(roles[5], "edit_target") # crimson
        self.assertEqual(roles[6], "neutral")

    def test_byte_level_bpe_gpt2_roberta(self):
        prompt = "a red car and a carpet"
        # Byte-level BPE uses Ġ (\u0120) for whitespace prefix
        gpt_pieces = ["<s>", "a", "Ġred", "Ġcar", "Ġand", "Ġa", "Ġcar", "pet", "</s>"]
        tok = MockTokenizer(gpt_pieces)

        mapping = map_pieces_to_words(gpt_pieces)
        self.assertEqual(mapping, [None, 0, 1, 2, 3, 4, 5, 5, None])

        car_tokens = locate_edit_tokens(prompt, ("car",), tok)
        self.assertEqual(car_tokens, (3,))

    def test_single_character_words_and_punctuation_do_not_drift(self):
        prompt = "a cat, a dog, and a rabbit"
        clip_pieces = [
            "<|startoftext|>", "a</w>", "cat", ",</w>", "a</w>", "dog", ",</w>",
            "and</w>", "a</w>", "rabbit</w>", "<|endoftext|>"
        ]
        tok = MockTokenizer(clip_pieces)

        mapping = map_pieces_to_words(clip_pieces)
        # Verify punctuation (index 3 and 6) is None and does not shift subsequent words
        self.assertEqual(mapping, [None, 0, 1, None, 2, 3, None, 4, 5, 6, None])

        dog_tokens = locate_edit_tokens(prompt, ("dog",), tok)
        self.assertEqual(dog_tokens, (5,))

        rabbit_tokens = locate_edit_tokens(prompt, ("rabbit",), tok)
        self.assertEqual(rabbit_tokens, (9,))

    def test_style_hints_token_alignment_prevents_substring_false_positives(self):
        prompt = "a smart foil in digital art and oil painting style"
        pieces = [
            "<|startoftext|>", "a</w>", "sm", "art</w>", "f", "oil</w>", "in</w>",
            "digital</w>", "art</w>", "and</w>", "oil</w>", "paint", "ing</w>",
            "style</w>", "<|endoftext|>"
        ]
        tok = MockTokenizer(pieces)

        hints = extract_style_hints(prompt, tok)
        # 'smart' (pieces 2, 3) must NOT match 'art'
        # 'foil' (pieces 4, 5) must NOT match 'oil'
        # 'digital art' (pieces 7, 8) MUST match
        # 'oil painting' (pieces 10, 11, 12) MUST match
        self.assertNotIn(2, hints.style_tokens)
        self.assertNotIn(3, hints.style_tokens)
        self.assertNotIn(4, hints.style_tokens)
        self.assertNotIn(5, hints.style_tokens)
        self.assertIn(7, hints.style_tokens)
        self.assertIn(8, hints.style_tokens)
        self.assertIn(10, hints.style_tokens)
        self.assertIn(11, hints.style_tokens)
        self.assertIn(12, hints.style_tokens)
        self.assertEqual(hints.style_tokens, (7, 8, 10, 11, 12))


class PreDenoiseValidationAndNonBlockingAssertionTests(unittest.TestCase):
    """Subagent 6 test suite: Pre-denoise Validation and Non-blocking Assertion Systems.

    Verifies:
    1. PlanSelfCheck fields: is_valid, count_match, relation_match, ambiguity_detected,
       assumptions, notes.
    2. Non-blocking behavior: When ambiguity or non-standard relations are found, the
       system logs clear assumptions into `assumptions` and proceeds without throwing
       exceptions or blocking generation.
    3. Count and relation match verification logic:
       - Quantified counts > 1 log explicit slotting assumption.
       - Multiple unlinked objects flag ambiguity and log left-to-right balanced layout assumption.
       - Abstract / non-object prompts log full-frame style priors assumption.
    4. Integration with hybrid generation pipeline.
    """

    def test_plan_self_check_fields_and_dict_serialization(self):
        check = PlanSelfCheck(
            is_valid=True,
            count_match=True,
            relation_match=True,
            ambiguity_detected=False,
            assumptions=("Test assumption 1", "Test assumption 2"),
            notes="Pre-denoise semantic plan self-check passed.",
        )
        self.assertTrue(check.is_valid)
        self.assertTrue(check.count_match)
        self.assertTrue(check.relation_match)
        self.assertFalse(check.ambiguity_detected)
        self.assertEqual(check.assumptions, ("Test assumption 1", "Test assumption 2"))
        self.assertEqual(check.notes, "Pre-denoise semantic plan self-check passed.")

        # Serialization to dictionary
        d = check.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["is_valid"], True)
        self.assertEqual(d["count_match"], True)
        self.assertEqual(d["relation_match"], True)
        self.assertEqual(d["ambiguity_detected"], False)
        self.assertEqual(d["assumptions"], ["Test assumption 1", "Test assumption 2"])
        self.assertEqual(d["notes"], "Pre-denoise semantic plan self-check passed.")

    def test_quantified_counts_greater_than_one_log_slotting_assumption(self):
        prompt = "three red apples and two green pears on a wooden table"
        plan = plan_semantic_layout(prompt)

        self.assertTrue(plan.self_check.is_valid)
        self.assertTrue(plan.self_check.count_match)
        self.assertTrue(plan.self_check.relation_match)

        # Assumptions must explicitly note slotting for counts > 1
        assumptions = plan.self_check.assumptions
        self.assertTrue(
            any("Planning 3 distinct spatial slots for 'apples'" in a for a in assumptions),
            f"Missing 3 slots assumption for apples: {assumptions}",
        )
        self.assertTrue(
            any("Planning 2 distinct spatial slots for 'pears'" in a for a in assumptions),
            f"Missing 2 slots assumption for pears: {assumptions}",
        )

    def test_multiple_unlinked_objects_flag_ambiguity_and_log_balanced_layout(self):
        prompt = "a cat, a dog, a parrot, and a rabbit"
        plan = plan_semantic_layout(prompt)

        self.assertTrue(plan.self_check.is_valid)
        self.assertTrue(plan.self_check.ambiguity_detected)
        self.assertFalse(plan.self_check.relation_match)  # No explicit relations, >1 objects

        assumptions = plan.self_check.assumptions
        expected_substr = "Assumed left-to-right balanced layout for multiple unlinked entities."
        self.assertTrue(
            any(expected_substr in a for a in assumptions),
            f"Missing unlinked entities layout assumption: {assumptions}",
        )

    def test_abstract_and_non_object_prompts_log_full_frame_style_priors(self):
        abstract_prompts = [
            "dreamy ethereal cyberpunk digital art with volumetric lighting",
            "watercolor and oil painting with cinematic lighting",
            "photorealistic 3d render in anime concept art style",
            "hyperrealistic vibrant glowing neon cyberpunk digital art",
        ]
        for prompt in abstract_prompts:
            with self.subTest(prompt=prompt):
                plan = plan_semantic_layout(prompt)
                self.assertTrue(plan.self_check.is_valid)
                self.assertTrue(plan.self_check.ambiguity_detected)
                self.assertEqual(len(plan.objects), 0)
                self.assertEqual(len(plan.relations), 0)

                assumptions = plan.self_check.assumptions
                expected_substr = (
                    "No distinct physical objects identified; applying full-frame style priors."
                )
                self.assertTrue(
                    any(expected_substr in a for a in assumptions),
                    f"Missing style priors assumption for '{prompt}': {assumptions}",
                )

    def test_non_blocking_behavior_proceeds_without_raising_exceptions(self):
        ambiguous_and_edge_prompts = [
            "",  # Empty
            "   ",  # Whitespace only
            # Many unlinked:
            "a cat and a dog and a horse and a mouse and an elephant and a tiger and a lion",
            "something weird floating mysteriously inside nothingness",  # Ambiguous
            "a triangle above a square under a circle next to a pentagon",  # Complex relations
            "recolor the unseen background to vibrant turquoise",  # Edit with regional scope
        ]
        for prompt in ambiguous_and_edge_prompts:
            with self.subTest(prompt=prompt):
                # Must return a valid SemanticLayoutPlan object without raising
                plan = plan_semantic_layout(prompt)
                self.assertIsInstance(plan, SemanticLayoutPlan)
                self.assertIsInstance(plan.self_check, PlanSelfCheck)
                self.assertIsInstance(plan.self_check.assumptions, tuple)
                self.assertIsInstance(plan.self_check.notes, str)

                # Dictionaries and JSON conversion must never fail
                plan_dict = plan.to_dict()
                self.assertIn("self_check", plan_dict)
                plan_json = plan.to_json()
                self.assertIsInstance(plan_json, str)

    def test_non_blocking_execution_in_hybrid_generation(self):
        """Verify that plans with ambiguity execute seamlessly in generation."""
        test_prompts = [
            "a cat and a dog",  # Unlinked entities ambiguity
            "ethereal dreamy watercolor mood",  # No objects / full-frame priors
            "",  # Empty prompt
        ]
        initial_latents = torch.randn(1, 4, 8, 8)
        timesteps = [0, 1]

        def dummy_denoise(latents, timestep, cond):
            return latents * 0.5

        for prompt in test_prompts:
            with self.subTest(prompt=prompt):
                plan = plan_semantic_layout(prompt)
                output = run_hybrid_generation(
                    plan=plan,
                    initial_latents=initial_latents,
                    timesteps=timesteps,
                    denoise=dummy_denoise,
                )
                self.assertEqual(output.shape, initial_latents.shape)
                self.assertTrue(torch.isfinite(output).all())


class GaussianSpatialGuidanceTests(unittest.TestCase):
    """Unit tests for 2D anisotropic Gaussian spatial priors and heatmaps."""

    def test_box_to_gaussian_conversion(self):
        box = NormalizedBox(ymin=0.2, xmin=0.1, ymax=0.6, xmax=0.5)
        g = box.to_gaussian(rotation=0.0, coverage_sigma=2.0)

        self.assertAlmostEqual(g.mu_y, 0.4, places=3)
        self.assertAlmostEqual(g.mu_x, 0.3, places=3)
        self.assertAlmostEqual(g.sigma_y, 0.1, places=3)
        self.assertAlmostEqual(g.sigma_x, 0.1, places=3)
        self.assertAlmostEqual(g.theta, 0.0, places=3)

    def test_gaussian_center_and_anisotropic_scale(self):
        g = GaussianSpatialPrior(mu_y=0.5, mu_x=0.25, sigma_y=0.2, sigma_x=0.08, theta=0.0)
        self.assertEqual(g.center, (0.5, 0.25))
        self.assertEqual(g.scale, (0.2, 0.08))
        self.assertEqual(g.center_y, 0.5)
        self.assertEqual(g.center_x, 0.25)
        self.assertEqual(g.scale_y, 0.2)
        self.assertEqual(g.scale_x, 0.08)

    def test_gaussian_rotation_and_covariance(self):
        # 45-degree rotation
        theta = 3.14159265 / 4.0
        g = GaussianSpatialPrior(mu_y=0.5, mu_x=0.5, sigma_y=0.2, sigma_x=0.1, theta=theta)
        cov = g.covariance
        self.assertIsInstance(cov, tuple)
        self.assertEqual(len(cov), 2)
        # Covariance off-diagonal should be non-zero when rotated and anisotropic
        self.assertNotEqual(cov[0][1], 0.0)
        self.assertAlmostEqual(cov[0][1], cov[1][0], places=4)

    def test_gaussian_to_box_inversion(self):
        orig_box = NormalizedBox(ymin=0.2, xmin=0.2, ymax=0.8, xmax=0.8)
        g = orig_box.to_gaussian(coverage_sigma=2.0)
        rec_box = g.to_box(confidence_sigma=2.0)

        self.assertAlmostEqual(rec_box.center[0], orig_box.center[0], places=3)
        self.assertAlmostEqual(rec_box.center[1], orig_box.center[1], places=3)
        self.assertAlmostEqual(rec_box.height, orig_box.height, places=3)
        self.assertAlmostEqual(rec_box.width, orig_box.width, places=3)

    def test_gaussian_rasterization_heatmap_dimensions_and_properties(self):
        g = GaussianSpatialPrior(mu_y=0.5, mu_x=0.5, sigma_y=0.15, sigma_x=0.15)
        heatmap = g.to_heatmap(height=32, width=32)

        self.assertEqual(heatmap.shape, (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(heatmap).all())
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        # Center peak
        self.assertGreater(float(heatmap[0, 0, 16, 16]), 0.8)
        # Outer boundary should decay towards 0
        self.assertLess(float(heatmap[0, 0, 0, 0]), 0.05)

    def test_numerical_stability_with_small_sigma(self):
        # Very small sigma must not produce NaNs or Inf
        g_small = GaussianSpatialPrior(mu_y=0.5, mu_x=0.5, sigma_y=1e-5, sigma_x=1e-5)
        heatmap = g_small.to_heatmap(height=16, width=16)
        self.assertTrue(torch.isfinite(heatmap).all())
        self.assertFalse(torch.isnan(heatmap).any())

    def test_overlapping_gaussian_entities(self):
        prompt = "a red apple on a wooden table"
        tok = MockTokenizer(["<s>", "a", "red", "apple", "on", "a", "wooden", "table", "</s>"])
        plan = plan_semantic_layout(prompt, tokenizer=tok, guidance_mode="gaussian")
        self.assertEqual(plan.guidance_mode, "gaussian")
        self.assertTrue(all(obj.gaussian is not None for obj in plan.objects))

        bias_gaussian = build_layout_guidance_bias(
            plan,
            num_image_tokens=256,
            num_text_tokens=16,
            guidance_strength=0.3,
            guidance_mode="gaussian",
        )
        self.assertEqual(bias_gaussian.shape, (256, 16))
        self.assertTrue(torch.isfinite(bias_gaussian).all())
        self.assertGreater(float(bias_gaussian.max()), 0.1)


class AdaptiveGuidanceStrengthTests(unittest.TestCase):
    """Unit tests for dynamic adaptive guidance strength."""

    def test_adaptive_gamma_heuristic_and_clamping(self):
        cfg = AdaptiveGuidanceConfig(
            base_gamma=0.2, entity_scale=0.05, min_gamma=0.2, max_gamma=0.5
        )

        # 0 entities -> base_gamma (0.2)
        plan_0 = plan_semantic_layout("")
        gamma_0 = compute_adaptive_guidance_strength(plan_0, config=cfg)
        self.assertAlmostEqual(gamma_0, 0.2, places=3)

        # 1 entity -> 0.2 + 0.05 * 1 = 0.25 (plus complexity)
        plan_1 = plan_semantic_layout("a lone wolf in a snowy mountain")
        gamma_1 = compute_adaptive_guidance_strength(plan_1, config=cfg)
        self.assertGreaterEqual(gamma_1, 0.25)
        self.assertLessEqual(gamma_1, 0.50)

        # Many entities -> clamped to max_gamma (0.5)
        prompt_many = "a cat, a dog, a lion, a tiger, an elephant, a bear, a wolf, and a fox"
        plan_many = plan_semantic_layout(prompt_many)
        gamma_many = compute_adaptive_guidance_strength(plan_many, config=cfg)
        self.assertEqual(gamma_many, 0.5)

    def test_manual_guidance_strength_override(self):
        plan = plan_semantic_layout("a cat and a dog")
        # Manual strength 0.42 strictly overrides adaptive calculation
        gamma = compute_adaptive_guidance_strength(plan, manual_strength=0.42)
        self.assertEqual(gamma, 0.42)

    def test_disabled_adaptive_guidance_uses_base_gamma(self):
        cfg = AdaptiveGuidanceConfig(base_gamma=0.25, enabled=False)
        plan = plan_semantic_layout("a cat, a dog, and a rabbit")
        gamma = compute_adaptive_guidance_strength(plan, config=cfg)
        self.assertEqual(gamma, 0.25)


class MultiModalVisualGroundingTests(unittest.TestCase):
    """Unit tests for multi-modal visual context grounding and co-reference."""

    def test_no_reference_fallback_is_pure_text_planning(self):
        plan = plan_semantic_layout("a majestic red fox in a forest", visual_context=None)
        self.assertIsNone(plan.visual_context)
        self.assertEqual(len(plan.objects), 2)

    def test_visual_context_entity_matching_and_id_preservation(self):
        ref_box = NormalizedBox(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4)
        vis_entity = VisualEntity(entity_id="char_hero_01", label="warrior", box=ref_box)
        v_ctx = VisualContext(entities=(vis_entity,))

        plan_in_place = plan_semantic_layout(
            "enhance this warrior with golden armor",
            visual_context=v_ctx,
        )
        self.assertIsNotNone(plan_in_place.visual_context)
        warrior_in_place = next((o for o in plan_in_place.objects if o.label == "warrior"), None)
        self.assertIsNotNone(warrior_in_place)
        self.assertEqual(warrior_in_place.entity_id, "char_hero_01")
        self.assertEqual(warrior_in_place.box, ref_box)

        plan_rel = plan_semantic_layout(
            "move this warrior next to the castle",
            visual_context=v_ctx,
        )
        warrior_rel = next((o for o in plan_rel.objects if o.label == "warrior"), None)
        self.assertIsNotNone(warrior_rel)
        self.assertEqual(warrior_rel.entity_id, "char_hero_01")

    def test_ambiguous_demonstrative_with_single_visual_reference(self):
        ref_box = NormalizedBox(ymin=0.3, xmin=0.3, ymax=0.7, xmax=0.7)
        vis_entity = VisualEntity(entity_id="car_vintage_99", label="vehicle", box=ref_box)
        v_ctx = VisualContext(entities=(vis_entity,))

        plan = plan_semantic_layout("make that object larger", visual_context=v_ctx)
        self.assertTrue(len(plan.objects) >= 1)
        self.assertEqual(plan.objects[0].entity_id, "car_vintage_99")


class InteractiveLayoutCanvasIntegrationTests(unittest.TestCase):
    """Unit tests for interactive layout canvas override schema."""

    def test_layout_override_replaces_automatic_boxes_and_gaussians(self):
        custom_layout = [
            {
                "label": "custom_dragon",
                "count": 1,
                "box": {"ymin": 0.05, "xmin": 0.05, "ymax": 0.45, "xmax": 0.45},
                "entity_id": "drag_01",
            },
            {
                "label": "custom_knight",
                "count": 1,
                "ymin": 0.50,
                "xmin": 0.50,
                "ymax": 0.90,
                "xmax": 0.90,
                "entity_id": "kni_02",
            }
        ]

        plan = plan_semantic_layout(
            "a battle scene",
            layout_override=custom_layout,
            guidance_mode="gaussian",
        )

        self.assertEqual(len(plan.objects), 2)
        obj0 = plan.objects[0]
        self.assertEqual(obj0.label, "custom_dragon")
        self.assertEqual(obj0.entity_id, "drag_01")
        self.assertEqual(obj0.box.ymin, 0.05)
        self.assertIsNotNone(obj0.gaussian)

        obj1 = plan.objects[1]
        self.assertEqual(obj1.label, "custom_knight")
        self.assertEqual(obj1.entity_id, "kni_02")
        self.assertEqual(obj1.box.ymin, 0.50)


class EvalHybridReasoningBenchmarkTests(unittest.TestCase):
    """Test suite verifying scripts/eval_hybrid_reasoning.py runner functionality."""

    def test_benchmark_runner_executes_and_returns_valid_report(self):
        import argparse

        from scripts.eval_hybrid_reasoning import (
            evaluate_aesthetic_control_set,
            evaluate_guidance_ablation,
            evaluate_object_counts,
            evaluate_spatial_relationships,
            run_all_benchmarks,
        )

        count_res = evaluate_object_counts()
        self.assertEqual(count_res["entity_count_accuracy"], 1.0)

        rel_res = evaluate_spatial_relationships()
        self.assertEqual(rel_res["relation_accuracy"], 1.0)

        aes_res = evaluate_aesthetic_control_set()
        self.assertEqual(aes_res["aesthetic_freedom_score"], 1.0)
        self.assertTrue(aes_res["zero_bias_verified_overall"])

        abl_res = evaluate_guidance_ablation(sweep=True)
        self.assertAlmostEqual(abl_res["soft_entropy_retention"], 1.0, places=3)
        self.assertIsNotNone(abl_res["sweep"])

        args = argparse.Namespace(
            json=False,
            seeds=1,
            sweep=False,
            category="all",
            verbose=False,
            output=None,
        )
        report = run_all_benchmarks(args)
        self.assertIn("summary", report)
        self.assertIn("categories", report)
        self.assertGreater(report["summary"]["leakage_reduction_pct"], 90.0)


class ContinuousDensityFieldAndCrowdDynamicsTests(unittest.TestCase):
    """Unit test suite for Subagent 2: Continuous Density Field Representations & Crowd Dynamics."""

    def test_density_field_dataclass_initialization_and_properties(self):
        box = NormalizedBox(ymin=0.2, xmin=0.3, ymax=0.8, xmax=0.9)
        df = DensityField(
            entity_id="swarm_01",
            label="bees",
            expected_count=50,
            density=1.5,
            center=(0.5, 0.6),
            scale=(0.3, 0.3),
            region=box,
            distribution_type="radial",
            falloff=1.5,
            seed=42,
            token_indices=(2, 3),
            mu_z=0.5,
        )
        self.assertEqual(df.entity_id, "swarm_01")
        self.assertEqual(df.label, "bees")
        self.assertEqual(df.expected_count, 50)
        self.assertAlmostEqual(df.density, 1.5)
        self.assertEqual(df.center, (0.5, 0.6))
        self.assertEqual(df.center_y, 0.5)
        self.assertEqual(df.center_x, 0.6)
        self.assertEqual(df.scale, (0.3, 0.3))
        self.assertEqual(df.scale_y, 0.3)
        self.assertEqual(df.scale_x, 0.3)
        self.assertEqual(df.distribution_type, "radial")
        self.assertAlmostEqual(df.falloff, 1.5)
        self.assertEqual(df.seed, 42)
        self.assertEqual(df.token_indices, (2, 3))
        self.assertAlmostEqual(df.mu_z, 0.5)

        # from_region constructor
        df_from_region = DensityField.from_region(
            region=box,
            label="lanterns",
            expected_count=100,
            density=1.2,
            distribution_type="uniform",
            falloff=2.5,
        )
        self.assertEqual(df_from_region.label, "lanterns")
        self.assertEqual(df_from_region.expected_count, 100)
        self.assertEqual(df_from_region.center, box.center)
        self.assertTrue(df_from_region.entity_id.startswith("lanterns_density_"))

        # to_dict serialization
        d = df.to_dict()
        self.assertEqual(d["entity_id"], "swarm_01")
        self.assertEqual(d["label"], "bees")
        self.assertEqual(d["expected_count"], 50)
        self.assertEqual(d["distribution_type"], "radial")
        self.assertEqual(d["token_indices"], [2, 3])

    def test_density_field_heatmap_rasterization_distributions(self):
        box = NormalizedBox(ymin=0.2, xmin=0.2, ymax=0.8, xmax=0.8)
        H, W = 32, 32

        for dist_type in ("gaussian", "uniform", "radial", "elongated"):
            df = DensityField(
                entity_id=f"test_{dist_type}",
                label="particles",
                expected_count=40,
                density=1.0,
                center=(0.5, 0.5),
                scale=(0.2, 0.2),
                region=box,
                distribution_type=dist_type,
                falloff=2.0,
                seed=123,
            )
            heatmap = df.to_heatmap(height=H, width=W)
            self.assertEqual(heatmap.shape, (1, 1, H, W))
            self.assertTrue(torch.isfinite(heatmap).all())
            self.assertTrue((heatmap >= 0.0).all())
            self.assertGreater(float(heatmap.max()), 0.5)

            # Center should have peak density
            center_val = float(heatmap[0, 0, H // 2, W // 2])
            corner_val = float(heatmap[0, 0, 0, 0])
            self.assertGreater(center_val, corner_val)

    def test_density_field_uniform_plateau_characteristics(self):
        box = NormalizedBox(ymin=0.3, xmin=0.3, ymax=0.7, xmax=0.7)
        df = DensityField(
            entity_id="uniform_field",
            label="flowers",
            expected_count=100,
            density=1.0,
            center=(0.5, 0.5),
            scale=(0.1, 0.1),
            region=box,
            distribution_type="uniform",
            falloff=3.0,
            seed=None,
        )
        heatmap = df.to_heatmap(height=64, width=64)
        # Inside plateau (center and nearby), density is ~1.0
        self.assertAlmostEqual(float(heatmap[0, 0, 32, 32]), 1.0, places=2)
        self.assertAlmostEqual(float(heatmap[0, 0, 30, 30]), 1.0, places=2)
        # Far outside box, density decays close to 0
        self.assertLess(float(heatmap[0, 0, 5, 5]), 0.1)

    def test_planner_detection_large_counts(self):
        prompts = [
            ("50 bees buzzing in a sunny meadow", "bees", 50, "radial"),
            ("hundreds of stars shining in the night sky", "stars", 100, "radial"),
            ("12 robots marching in an arena", "robots", 12, "gaussian"),
            ("many stars in the cosmos", "stars", 20, "radial"),
        ]
        for prompt, exp_label, exp_count, exp_dist in prompts:
            plan = plan_semantic_layout(prompt)
            self.assertEqual(len(plan.density_fields), 1, f"Failed for {prompt}")
            df = plan.density_fields[0]
            self.assertEqual(df.label, exp_label)
            self.assertEqual(df.expected_count, exp_count)
            self.assertEqual(df.distribution_type, exp_dist)

    def test_planner_detection_collective_crowd_nouns(self):
        prompts = [
            ("a swarm of angry bees", "bees", 50, "radial"),
            ("a flock of migrating birds over the ocean", "birds", 25, "elongated"),
            ("a crowd of cheering spectators in a stadium", "spectators", 40, "uniform"),
            ("a sea of glowing lanterns on water", "lanterns", 100, "uniform"),
            ("a field of blooming sunflowers under sunlight", "sunflowers", 100, "uniform"),
        ]
        for prompt, exp_label, exp_count, exp_dist in prompts:
            plan = plan_semantic_layout(prompt)
            self.assertTrue(
                len(plan.density_fields) >= 1,
                f"Expected density field for '{prompt}', got objects={plan.objects}",
            )
            df = next(d for d in plan.density_fields if d.label == exp_label)
            self.assertEqual(df.expected_count, exp_count)
            self.assertEqual(df.distribution_type, exp_dist)

    def test_hybrid_discrete_object_and_density_field_coexistence(self):
        prompt = "a warrior in armor surrounded by a swarm of bees"
        words = ["<s>"] + prompt.split() + ["</s>"]
        tok = MockTokenizer(pieces=words)
        plan = plan_semantic_layout(prompt, tokenizer=tok)
        self.assertTrue(any(obj.label == "warrior" for obj in plan.objects))
        self.assertTrue(any(df.label == "bees" for df in plan.density_fields))

        # Soft layout guidance bias integration
        bias = build_layout_guidance_bias(
            plan,
            num_image_tokens=256,
            num_text_tokens=len(words),
            guidance_strength=0.3,
        )
        self.assertEqual(bias.shape, (256, len(words)))
        self.assertTrue(torch.isfinite(bias).all())
        self.assertTrue((bias >= 0.0).all())
        self.assertGreater(float(bias.max()), 0.0)


class DepthAwareGaussianSpatialGuidanceTests(unittest.TestCase):
    """Unit tests for 3D depth-aware Gaussian priors, volumetric rasterization, and occlusion."""

    def test_default_mu_z_and_depth_parameters(self):
        g = GaussianSpatialPrior(mu_y=0.5, mu_x=0.5, sigma_y=0.2, sigma_x=0.2)
        self.assertEqual(g.mu_z, 0.5)
        self.assertEqual(g.sigma_z, 0.2)
        self.assertEqual(g.depth_confidence, 1.0)
        self.assertEqual(g.center_3d, (0.5, 0.5, 0.5))
        self.assertEqual(g.scale_3d, (0.2, 0.2, 0.2))

        # Covariance 3D
        cov_3d = g.covariance_3d
        self.assertEqual(len(cov_3d), 3)
        self.assertAlmostEqual(cov_3d[2][2], 0.04, places=4)
        self.assertEqual(cov_3d[0][2], 0.0)

    def test_depth_normalization_and_clamping(self):
        g_clamped = GaussianSpatialPrior(
            mu_y=0.5,
            mu_x=0.5,
            sigma_y=0.2,
            sigma_x=0.2,
            mu_z=1.8,
            sigma_z=-0.5,
            depth_confidence=2.5,
        )
        self.assertEqual(g_clamped.mu_z, 1.0)
        self.assertEqual(g_clamped.sigma_z, 1e-4)
        self.assertEqual(g_clamped.depth_confidence, 1.0)

    def test_3d_volume_rasterization_dimensions_and_properties(self):
        g = GaussianSpatialPrior(
            mu_y=0.5, mu_x=0.5, sigma_y=0.15, sigma_x=0.15, mu_z=0.25, sigma_z=0.10
        )
        vol = g.to_volume(depth_bins=16, height=32, width=32)
        self.assertEqual(vol.shape, (1, 1, 16, 32, 32))
        self.assertTrue(torch.isfinite(vol).all())
        self.assertTrue((vol >= 0.0).all())

        # Peak should be centered around depth bin ~ 4 (0.25 * 16) and spatial center (16, 16)
        peak_z = int(0.25 * 16)
        self.assertGreater(float(vol[0, 0, peak_z, 16, 16]), float(vol[0, 0, 14, 16, 16]))

    def test_relative_depth_inference_in_front_of_and_behind(self):
        # "in front of" -> subject closer (smaller mu_z) than object
        plan_front = plan_semantic_layout("a red cube in front of a blue sphere")
        cube_front = next(o for o in plan_front.objects if o.label == "cube")
        sphere_front = next(o for o in plan_front.objects if o.label == "sphere")
        self.assertLess(cube_front.gaussian.mu_z, sphere_front.gaussian.mu_z)
        self.assertAlmostEqual(cube_front.gaussian.mu_z, 0.25, places=2)
        self.assertAlmostEqual(sphere_front.gaussian.mu_z, 0.70, places=2)

        # "behind" -> subject deeper (larger mu_z) than object
        plan_behind = plan_semantic_layout("a cat behind a wooden chair")
        cat_behind = next(o for o in plan_behind.objects if o.label == "cat")
        chair_behind = next(o for o in plan_behind.objects if o.label == "chair")
        self.assertGreater(cat_behind.gaussian.mu_z, chair_behind.gaussian.mu_z)
        self.assertAlmostEqual(cat_behind.gaussian.mu_z, 0.75, places=2)
        self.assertAlmostEqual(chair_behind.gaussian.mu_z, 0.25, places=2)

    def test_relative_depth_inference_behind_translucent_glass_window(self):
        plan = plan_semantic_layout("a person partially hidden behind a glass window")
        person = next(o for o in plan.objects if o.label == "person")
        window = next(o for o in plan.objects if o.label == "window")
        self.assertGreater(person.gaussian.mu_z, window.gaussian.mu_z)
        self.assertAlmostEqual(person.gaussian.mu_z, 0.75, places=2)
        self.assertAlmostEqual(window.gaussian.mu_z, 0.25, places=2)

    def test_relative_depth_inference_far_in_front_and_far_behind(self):
        plan_far_front = plan_semantic_layout("a flower far in front of a distant mountain")
        flower = next(o for o in plan_far_front.objects if o.label == "flower")
        mountain = next(o for o in plan_far_front.objects if o.label == "mountain")
        self.assertLess(flower.gaussian.mu_z, 0.20)
        self.assertGreater(mountain.gaussian.mu_z, 0.75)

    def test_entity_overlaps_and_soft_occlusion_modulation(self):
        prompt = "a small kitten in front of a big dog"
        words = ["<s>"] + prompt.split() + ["</s>"]
        tok = MockTokenizer(pieces=words)
        plan = plan_semantic_layout(prompt, tokenizer=tok)

        kitten = next(o for o in plan.objects if o.label == "kitten")
        dog = next(o for o in plan.objects if o.label == "dog")
        self.assertLess(kitten.gaussian.mu_z, dog.gaussian.mu_z)

        # Depth-aware guidance bias computation
        bias = build_layout_guidance_bias(
            plan,
            num_image_tokens=256,
            num_text_tokens=len(words),
            guidance_strength=0.3,
            depth_guidance_enabled=True,
        )
        self.assertEqual(bias.shape, (256, len(words)))
        self.assertTrue(torch.isfinite(bias).all())
        self.assertGreater(float(bias.max()), 0.0)


class VisualFeatureMapCrossAttentionTests(unittest.TestCase):
    """Unit tests for spatial visual feature map extraction, projection, and cross-attention."""

    def test_mock_vision_backbone_spatial_features(self):
        backbone = MockVisionBackbone(output_dim=768, spatial_resolution=(16, 16))
        self.assertEqual(backbone.output_dim, 768)
        self.assertEqual(backbone.spatial_resolution, (16, 16))

        feat_map = backbone.encode_image()
        self.assertIsInstance(feat_map, VisualFeatureMap)
        self.assertEqual(feat_map.spatial_features.shape, (1, 256, 768))
        self.assertEqual(feat_map.num_tokens, 256)
        self.assertEqual(feat_map.feature_dim, 768)
        self.assertEqual(feat_map.batch_size, 1)

    def test_vision_feature_projector_dimension_projection(self):
        projector = VisionFeatureProjector(vision_dim=768, cross_attention_dim=1024)
        x = torch.randn(2, 64, 768)
        out = projector(x)
        self.assertEqual(out.shape, (2, 64, 1024))
        self.assertTrue(torch.isfinite(out).all())

    def test_layout_guidance_processor_with_spatial_visual_features(self):
        backbone = MockVisionBackbone(output_dim=512, spatial_resolution=(8, 8))
        feat_map = backbone.encode_image()
        v_ctx = VisualContext(feature_map=feat_map, spatial_features=feat_map.to_flattened())

        plan = plan_semantic_layout("a hero in futuristic armor", visual_context=v_ctx)
        self.assertIsNotNone(plan.visual_context)
        self.assertIsNotNone(plan.visual_context.spatial_features)

        def mock_base_processor(attn, hidden_states, encoder_hidden_states=None, **kwargs):
            # Verify encoder_hidden_states received concatenated visual features
            return hidden_states + encoder_hidden_states.mean()

        processor = LayoutGuidanceProcessor(
            base_processor=mock_base_processor,
            plan=plan,
            guidance_strength=0.3,
            visual_cross_attn_enabled=True,
            visual_feature_strength=0.25,
        )

        hidden_states = torch.randn(1, 64, 128)
        text_tokens = torch.randn(1, 16, 128)

        out = processor(None, hidden_states, encoder_hidden_states=text_tokens)
        self.assertEqual(out.shape, hidden_states.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_visual_features_disabled_fallback(self):
        plan = plan_semantic_layout("a simple prompt without visual context")
        self.assertIsNone(plan.visual_context)

        def mock_base_processor(attn, hidden_states, encoder_hidden_states=None, **kwargs):
            return hidden_states

        processor = LayoutGuidanceProcessor(
            base_processor=mock_base_processor,
            plan=plan,
            guidance_strength=0.3,
            visual_cross_attn_enabled=False,
        )
        hidden_states = torch.randn(1, 64, 128)
        text_tokens = torch.randn(1, 16, 128)
        out = processor(None, hidden_states, encoder_hidden_states=text_tokens)
        self.assertEqual(out.shape, hidden_states.shape)


class GuidanceSchedulingTests(unittest.TestCase):
    """Unit tests for reverse-time guidance schedules."""

    def test_two_phase_schedule(self):
        sched = TwoPhaseSchedule(schedule_cutoff=0.8)
        self.assertEqual(sched.weight(0.0), 1.0)
        self.assertEqual(sched.weight(0.5), 1.0)
        self.assertEqual(sched.weight(0.79), 1.0)
        self.assertEqual(sched.weight(0.80), 0.0)
        self.assertEqual(sched.weight(1.0), 0.0)

    def test_depth_aware_schedule_foreground_vs_background(self):
        sched = DepthAwareSchedule(schedule_cutoff=0.8, depth_decay=0.5)

        # Foreground entity (mu_z = 0.1) gets boosted guidance in mid steps
        w_fg = sched.weight(progress=0.4, mu_z=0.1)
        self.assertGreater(w_fg, 1.0)

        # Background entity (mu_z = 0.9) decays smoothly
        w_bg = sched.weight(progress=0.6, mu_z=0.9)
        self.assertLess(w_bg, 1.0)
        self.assertGreater(w_bg, 0.0)

        # Past cutoff -> 0.0
        self.assertEqual(sched.weight(progress=0.85, mu_z=0.1), 0.0)

    def test_linear_and_cosine_schedules(self):
        lin = LinearSchedule(start_weight=1.0, end_weight=0.0)
        self.assertAlmostEqual(lin.weight(0.0), 1.0)
        self.assertAlmostEqual(lin.weight(0.5), 0.5)
        self.assertAlmostEqual(lin.weight(1.0), 0.0)

        cos_sched = CosineSchedule(schedule_cutoff=0.8)
        self.assertAlmostEqual(cos_sched.weight(0.0), 1.0)
        self.assertAlmostEqual(cos_sched.weight(0.4), 0.5)
        self.assertAlmostEqual(cos_sched.weight(0.8), 0.0)


class InteractiveRotationTests(unittest.TestCase):
    """Unit tests for interactive rotation controls and theta synchronization."""

    def test_rotation_preserves_center_and_scale(self):
        g_orig = GaussianSpatialPrior(mu_y=0.4, mu_x=0.6, sigma_y=0.2, sigma_x=0.1, theta=0.0)
        g_rot = GaussianSpatialPrior(
            mu_y=g_orig.mu_y,
            mu_x=g_orig.mu_x,
            sigma_y=g_orig.sigma_y,
            sigma_x=g_orig.sigma_x,
            theta=1.5708,
        )
        self.assertEqual(g_rot.center, g_orig.center)
        self.assertEqual(g_rot.scale, g_orig.scale)
        self.assertAlmostEqual(g_rot.theta, 1.5708, places=3)

    def test_90_degree_rotation_inverts_anisotropy(self):
        # 90 degrees rotation around optical axis
        g_horiz = GaussianSpatialPrior(mu_y=0.5, mu_x=0.5, sigma_y=0.1, sigma_x=0.3, theta=0.0)
        g_vert = GaussianSpatialPrior(mu_y=0.5, mu_x=0.5, sigma_y=0.1, sigma_x=0.3, theta=1.5707963)

        hm_horiz = g_horiz.to_heatmap(height=32, width=32).squeeze()
        hm_vert = g_vert.to_heatmap(height=32, width=32).squeeze()

        # Horizontal gaussian has wider x profile than y profile
        self.assertGreater(float(hm_horiz[16, 24]), float(hm_horiz[24, 16]))
        # 90-deg rotated gaussian has taller y profile than x profile
        self.assertGreater(float(hm_vert[24, 16]), float(hm_vert[16, 24]))

    def test_layout_override_with_theta_and_mu_z(self):
        custom_layout = [
            {
                "label": "rotated_sword",
                "count": 1,
                "box": {"ymin": 0.2, "xmin": 0.2, "ymax": 0.8, "xmax": 0.4},
                "theta": 0.7854,
                "mu_z": 0.25,
                "entity_id": "sword_01",
            }
        ]

        plan = plan_semantic_layout(
            "a fantasy weapon on display",
            layout_override=custom_layout,
            guidance_mode="gaussian",
        )
        self.assertEqual(len(plan.objects), 1)
        obj = plan.objects[0]
        self.assertEqual(obj.label, "rotated_sword")
        self.assertEqual(obj.entity_id, "sword_01")
        self.assertAlmostEqual(obj.gaussian.theta, 0.7854, places=3)
        self.assertAlmostEqual(obj.gaussian.mu_z, 0.25, places=2)


if __name__ == "__main__":
    unittest.main()
