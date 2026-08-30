"""Tests for the region-aware editing modules.

All CPU, no checkpoint, no network - the modules take injected embeddings and an
injected denoiser precisely so they stay testable.
"""

import unittest

import torch

from app.services.editing.adaptive_reference import (
    CoefficientConfig,
    apply_edge_blending,
    apply_region_guidance,
    blend_latents,
    calibrate_similarity,
    compute_adaptive_reference_coefficient,
    legacy_reference_coefficient,
    preservation_at_step,
)
from app.services.editing.alignment import check_prompt_image_alignment, check_scene_conflict
from app.services.editing.edit_pipeline import (
    batched_cfg_denoiser,
    run_baseline_edit,
    run_region_aware_edit,
    set_region_bias,
)
from app.services.editing.edit_planner import (
    align_token_roles,
    classify_scope,
    locate_edit_tokens,
    map_pieces_to_words,
    plan_edit,
    select_edit_terms,
)
from app.services.editing.masks import area_ratio, bounding_box, feather, iou, resize_mask
from app.services.editing.metrics import (
    evaluate_edit,
    inside_alignment,
    ssim,
    unintended_change_ratio,
)
from app.services.editing.region_attention import (
    AttentionCapture,
    RegionAwareAttnProcessor,
    build_attention_bias,
    classify_token_roles,
    extract_edit_mask,
    latent_grid,
    mask_to_token_weights,
    masked_cross_attention,
    region_attention_bias,
)

DIM = 16


def box_mask(top, left, bottom, right, size=64):
    mask = torch.zeros(1, 1, size, size)
    mask[..., top:bottom, left:right] = 1.0
    return mask


def embedding_at(similarity, reference, seed=0):
    """A vector at a known cosine similarity to `reference`."""
    unit = reference / reference.norm()
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(reference.shape, generator=generator)
    orthogonal = noise - unit * torch.dot(noise, unit)
    orthogonal = orthogonal / orthogonal.norm()
    return similarity * unit + (1 - similarity**2) ** 0.5 * orthogonal


class MaskUtilityTests(unittest.TestCase):
    def test_area_ratio_matches_pixel_fraction(self):
        self.assertAlmostEqual(area_ratio(box_mask(0, 0, 32, 32)), 0.25, places=4)

    def test_feather_softens_edges_without_leaving_range(self):
        softened = feather(box_mask(20, 20, 40, 40), radius=3)
        self.assertGreaterEqual(float(softened.min()), 0.0)
        self.assertLessEqual(float(softened.max()), 1.0)
        # A feathered edge introduces intermediate values a binary mask lacks.
        self.assertTrue(bool(((softened > 0.01) & (softened < 0.99)).any()))

    def test_bounding_box_is_tight(self):
        self.assertEqual(bounding_box(box_mask(10, 12, 30, 40)), (10, 12, 30, 40))

    def test_empty_mask_has_no_box(self):
        self.assertIsNone(bounding_box(torch.zeros(1, 1, 8, 8)))

    def test_iou_of_disjoint_masks_is_zero(self):
        self.assertEqual(iou(box_mask(0, 0, 8, 8), box_mask(40, 40, 50, 50)), 0.0)


class AdaptiveCoefficientTests(unittest.TestCase):
    def setUp(self):
        self.prompt = torch.randn(DIM, generator=torch.Generator().manual_seed(1))
        self.config = CoefficientConfig()

    def _coefficients(self, mask, similarity):
        image = embedding_at(similarity, self.prompt).view(1, DIM, 1, 1).expand(1, DIM, 64, 64)
        return compute_adaptive_reference_coefficient(
            prompt_embedding=self.prompt,
            source_image_embedding=image.contiguous(),
            edit_region_mask=mask,
            base_guidance_scale=7.5,
            config=self.config,
        )

    def test_smaller_edit_region_preserves_more(self):
        small = self._coefficients(box_mask(0, 0, 8, 8), 0.25)
        large = self._coefficients(box_mask(0, 0, 56, 56), 0.25)
        self.assertGreater(small.ref_weight, large.ref_weight)

    def test_higher_conflict_raises_edit_strength_and_inside_guidance(self):
        mask = box_mask(0, 0, 8, 8)
        conflicting = self._coefficients(mask, 0.10)
        agreeing = self._coefficients(mask, 0.40)
        self.assertGreater(conflicting.edit_strength, agreeing.edit_strength)
        self.assertGreater(conflicting.inside_scale, agreeing.inside_scale)

    def test_local_high_conflict_edit_keeps_strong_inside_guidance(self):
        """The bug in the reference pseudo-code: a local edit must not be damped."""
        local = self._coefficients(box_mask(0, 0, 8, 8), 0.10)
        self.assertGreater(local.inside_scale, 7.5)
        self.assertGreater(local.ref_weight, 0.5)  # and still preserves outside

    def test_legacy_formula_damps_local_edits(self):
        """Documents why the reference formula was replaced, so the fix is not undone."""
        image = embedding_at(0.10, self.prompt).view(1, DIM, 1, 1).expand(1, DIM, 64, 64)
        local = legacy_reference_coefficient(
            prompt_embedding=self.prompt,
            source_image_embedding=image.contiguous(),
            edit_region_mask=box_mask(0, 0, 8, 8),
        )
        globalish = legacy_reference_coefficient(
            prompt_embedding=self.prompt,
            source_image_embedding=image.contiguous(),
            edit_region_mask=box_mask(0, 0, 62, 62),
        )
        # inside_scale under the legacy rule is base * (1 - w) * 2
        self.assertLess((1 - local) * 2, (1 - globalish) * 2)

    def test_ref_weight_stays_within_configured_bounds(self):
        for similarity in (-0.5, 0.0, 0.25, 0.9):
            for mask in (box_mask(0, 0, 2, 2), torch.ones(1, 1, 64, 64)):
                coefficients = self._coefficients(mask, similarity)
                self.assertGreaterEqual(coefficients.ref_weight, self.config.min_ref_weight)
                self.assertLessEqual(coefficients.ref_weight, self.config.max_ref_weight)

    def test_calibration_expands_the_narrow_similarity_band(self):
        config = CoefficientConfig(similarity_floor=0.1, similarity_ceiling=0.4)
        self.assertEqual(calibrate_similarity(0.05, config), 0.0)
        self.assertEqual(calibrate_similarity(0.55, config), 1.0)
        self.assertAlmostEqual(calibrate_similarity(0.25, config), 0.5, places=5)

    def test_guidance_map_is_higher_inside_than_outside(self):
        mask = box_mask(20, 20, 40, 40)
        coefficients = self._coefficients(mask, 0.10)
        scale_map = apply_region_guidance(mask, coefficients)
        self.assertGreater(float(scale_map[..., 30, 30]), float(scale_map[..., 0, 0]))

    def test_preservation_ramps_in_over_the_schedule(self):
        self.assertLess(preservation_at_step(0.8, 0.0), preservation_at_step(0.8, 0.5))
        self.assertAlmostEqual(preservation_at_step(0.8, 1.0), 0.8, places=6)

    def test_blend_restores_source_outside_the_mask(self):
        source = torch.zeros(1, 4, 8, 8)
        edited = torch.ones(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :4, :4] = 1.0
        blended = blend_latents(edited, source, mask, strength=1.0)
        self.assertAlmostEqual(float(blended[0, 0, 6, 6]), 0.0, places=6)  # outside -> source
        self.assertAlmostEqual(float(blended[0, 0, 1, 1]), 1.0, places=6)  # inside -> edit

    def test_blend_latents_matches_device_and_dtype(self):
        source = torch.zeros(1, 4, 8, 8, dtype=torch.float32)
        edited = torch.ones(1, 4, 8, 8, dtype=torch.float64)
        mask = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
        mask[..., :4, :4] = 1.0
        blended = blend_latents(edited, source, mask, strength=1.0)
        self.assertEqual(blended.dtype, torch.float64)
        self.assertEqual(blended.device, edited.device)


class RegionAttentionTests(unittest.TestCase):
    def test_latent_grid_factorises_token_count(self):
        self.assertEqual(latent_grid(256), (16, 16))
        self.assertEqual(latent_grid(64), (8, 8))

    def test_bias_suppresses_edit_tokens_only_outside_the_region(self):
        mask = box_mask(4, 4, 8, 8, size=16)
        bias = region_attention_bias(
            mask, num_image_tokens=256, num_text_tokens=8,
            edit_token_indices=[3], strength=1.0,
        )
        weights = mask_to_token_weights(mask, 256)
        inside = weights >= 0.5
        self.assertAlmostEqual(float(bias[inside][:, 3].max()), 0.0, places=4)
        self.assertLess(float(bias[~inside][:, 3].max()), -1.0)
        # Untargeted tokens are never penalised.
        self.assertAlmostEqual(float(bias[:, 0].abs().max()), 0.0, places=6)

    def test_bias_strength_is_graded_not_saturated(self):
        mask = box_mask(4, 4, 8, 8, size=16)
        magnitudes = []
        for strength in (0.25, 0.5, 1.0):
            bias = region_attention_bias(
                mask, num_image_tokens=256, num_text_tokens=4,
                edit_token_indices=[1], strength=strength,
            )
            magnitudes.append(abs(float(bias.min())))
        self.assertLess(magnitudes[0], magnitudes[1])
        self.assertLess(magnitudes[1], magnitudes[2])

    def test_zero_strength_is_a_no_op(self):
        bias = region_attention_bias(
            box_mask(0, 0, 4, 4, size=16), num_image_tokens=256,
            num_text_tokens=4, edit_token_indices=[1], strength=0.0,
        )
        self.assertAlmostEqual(float(bias.abs().max()), 0.0, places=6)

    def test_attention_capture_recovers_the_attended_region(self):
        mask = box_mask(4, 4, 8, 8, size=16)
        weights = mask_to_token_weights(mask, 256) >= 0.5
        logits = torch.zeros(256, 6)
        logits[weights, 2] = 6.0
        capture = AttentionCapture()
        capture.add(logits.softmax(dim=-1)[None])
        inferred = capture.token_mask([2])
        self.assertIsNotNone(inferred)
        self.assertGreater(iou(inferred, resize_mask(mask, 16, 16)), 0.9)

    def test_capture_ignores_mismatched_resolutions(self):
        capture = AttentionCapture()
        capture.add(torch.rand(1, 64, 4))
        capture.add(torch.rand(1, 256, 4))  # different block resolution
        self.assertEqual(capture.mean_map.shape, (64, 4))


class TokenRoleTests(unittest.TestCase):
    def test_edit_targets_are_tagged(self):
        roles = dict(classify_token_roles("change the shirt color to red"))
        self.assertEqual(roles["shirt"], "edit_target")
        self.assertEqual(roles["red"], "edit_target")
        self.assertEqual(roles["the"], "neutral")

    def test_preservation_clause_becomes_context(self):
        roles = dict(classify_token_roles("make it warmer but keep the background neutral"))
        self.assertEqual(roles["background"], "context")
        self.assertEqual(roles["warmer"], "edit_target")

    def test_roles_cover_every_word(self):
        prompt = "change the jacket to red but keep the background neutral"
        self.assertEqual(len(classify_token_roles(prompt)), len(prompt.split()))

    def test_align_token_roles_with_subword_tokenizer(self):
        class MockSubwordTokenizer:
            def __call__(self, prompt, add_special_tokens=True):
                return {"input_ids": [101, 1, 2, 3, 4, 5, 102]}

            def convert_ids_to_tokens(self, ids):
                # Simulated BPE: "photorealistic" split into "photo", "##realistic"
                return ["<s>", "change", "photo", "##realistic", "jacket", "red", "</s>"]

        roles = align_token_roles(
            "change photorealistic jacket red",
            tokenizer=MockSubwordTokenizer(),
        )
        self.assertEqual(
            roles,
            (
                "neutral",
                "neutral",
                "edit_target",
                "edit_target",
                "edit_target",
                "edit_target",
                "neutral",
            ),
        )


class TokenSegmentationTests(unittest.TestCase):
    """Sub-word alignment must group pieces into words, not match them by substring."""

    class Tok:
        def __init__(self, pieces):
            self.pieces = pieces

        def __call__(self, prompt, add_special_tokens=True):
            return {"input_ids": list(range(len(self.pieces)))}

        def convert_ids_to_tokens(self, ids):
            return self.pieces

    def test_wordpiece_continuation_inherits_its_parent_role(self):
        pieces = ["<s>", "change", "photo", "##realistic", "jacket", "red", "</s>"]
        roles = align_token_roles(
            "change photorealistic jacket red", tokenizer=self.Tok(pieces)
        )
        self.assertEqual(roles[2], "edit_target")
        self.assertEqual(roles[3], "edit_target")  # the ## continuation
        self.assertEqual(roles[0], "neutral")
        self.assertEqual(roles[6], "neutral")

    def test_subword_piece_binds_to_its_own_word_not_a_lookalike(self):
        """The 'car' piece of 'car'+'##pet' belongs to carpet, not to the word 'car'.

        Matching pieces by string made that piece resolve to the earlier standalone
        'car' (edit_target) instead of the carpet it is part of (context).
        """
        prompt = "change the red car but preserve the carpet"
        pieces = ["change", "the", "red", "car", "but", "preserve", "the", "car", "##pet"]
        roles = align_token_roles(prompt, tokenizer=self.Tok(pieces))
        self.assertEqual(roles[3], "edit_target")  # the real 'car'
        self.assertEqual(roles[7], "context")  # 'car' piece of 'carpet'
        self.assertEqual(roles[8], "context")  # '##pet' continuation

    def test_sentencepiece_scheme(self):
        pieces = ["\u2581change", "\u2581the", "\u2581red", "\u2581car", "pet"]
        roles = align_token_roles("change the red carpet", tokenizer=self.Tok(pieces))
        self.assertEqual(roles[2], "edit_target")
        self.assertEqual(roles[4], "edit_target")  # continuation of "carpet"

    def test_end_of_word_bpe_scheme(self):
        pieces = ["change</w>", "the</w>", "red</w>", "car", "pet</w>"]
        roles = align_token_roles("change the red carpet", tokenizer=self.Tok(pieces))
        self.assertEqual(roles[3], "edit_target")
        self.assertEqual(roles[4], "edit_target")

    def test_locate_and_align_agree(self):
        pieces = ["<s>", "change", "the", "red", "car", "reduce", "carpet", "</s>"]
        prompt = "change the red car"
        roles = align_token_roles(prompt, tokenizer=self.Tok(pieces))
        located = locate_edit_tokens(
            prompt, select_edit_terms(prompt), self.Tok(pieces)
        )
        by_role = tuple(i for i, r in enumerate(roles) if r == "edit_target")
        self.assertEqual(located, by_role)

    def test_special_tokens_map_to_no_word(self):
        mapping = map_pieces_to_words(["<s>", "red", "</s>", "<pad>"])
        self.assertIsNone(mapping[0])
        self.assertIsNone(mapping[2])
        self.assertEqual(mapping[1], 0)

    def test_more_pieces_than_words_stays_neutral(self):
        """Padding past the prompt must not index off the end of the role list."""
        pieces = ["red", "car", "<pad>", "<pad>", "extra"]
        roles = align_token_roles("red car", tokenizer=self.Tok(pieces))
        self.assertEqual(len(roles), len(pieces))
        self.assertEqual(roles[4], "neutral")


class DiffusersProcessorTests(unittest.TestCase):
    """The processor must survive real diffusers attention at CFG batch sizes."""

    def setUp(self):
        try:
            from diffusers.models.attention_processor import Attention, AttnProcessor2_0
        except ImportError:  # pragma: no cover - diffusers is a hard dep here
            self.skipTest("diffusers not installed")
        self.q, self.k, self.dim, self.heads = 64, 6, 32, 4
        self.attn = Attention(
            query_dim=self.dim, cross_attention_dim=self.dim,
            heads=self.heads, dim_head=self.dim // self.heads,
        )
        mask = box_mask(2, 2, 6, 6, size=8)
        roles = ["edit_target", "context", "neutral", "neutral", "edit_target", "neutral"]
        self.bias = build_attention_bias(
            mask, roles, num_image_tokens=self.q
        ).transpose(0, 1)
        self.processor = RegionAwareAttnProcessor(AttnProcessor2_0())
        self.attn.set_processor(self.processor)

    def test_runs_at_classifier_free_guidance_batch_sizes(self):
        self.processor.set_bias(self.bias)
        for batch in (1, 2, 4):
            with self.subTest(batch=batch):
                out = self.attn(
                    torch.randn(batch, self.q, self.dim),
                    encoder_hidden_states=torch.randn(batch, self.k, self.dim),
                )
                self.assertEqual(tuple(out.shape), (batch, self.q, self.dim))

    def test_bias_actually_changes_the_output(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, self.q, self.dim)
        encoder = torch.randn(2, self.k, self.dim)
        self.processor.set_bias(None)
        plain = self.attn(hidden, encoder_hidden_states=encoder)
        self.processor.set_bias(self.bias)
        masked = self.attn(hidden, encoder_hidden_states=encoder)
        self.assertFalse(torch.allclose(plain, masked))

    def test_a_batched_bias_is_passed_through_unexpanded(self):
        batched = self.bias.unsqueeze(0).repeat(2, 1, 1)
        self.processor.set_bias(batched)
        out = self.attn(
            torch.randn(2, self.q, self.dim),
            encoder_hidden_states=torch.randn(2, self.k, self.dim),
        )
        self.assertEqual(tuple(out.shape), (2, self.q, self.dim))


class RoleAwareBiasTests(unittest.TestCase):
    def setUp(self):
        self.mask = box_mask(4, 8, 8, 12, size=16)
        self.roles = ["edit_target", "context", "neutral"]
        self.bias = build_attention_bias(
            self.mask, self.roles, leak_penalty=-12.0, context_boost=0.5,
            num_image_tokens=256,
        )
        self.inside = mask_to_token_weights(self.mask, 256) >= 0.5

    def test_bias_shape_is_text_by_image(self):
        self.assertEqual(tuple(self.bias.shape), (3, 256))

    def test_edit_target_is_penalised_outside_only(self):
        self.assertAlmostEqual(float(self.bias[0][self.inside].max()), 0.0, places=4)
        self.assertLess(float(self.bias[0][~self.inside].max()), -1.0)

    def test_context_is_boosted_inside_only(self):
        self.assertGreater(float(self.bias[1][self.inside].min()), 0.0)
        self.assertAlmostEqual(float(self.bias[1][~self.inside].max()), 0.0, places=4)

    def test_neutral_tokens_are_untouched(self):
        self.assertAlmostEqual(float(self.bias[2].abs().max()), 0.0, places=6)

    def test_penalty_is_finite_so_fp16_does_not_overflow(self):
        """-1e4 from the reference sketch becomes -inf in fp16 and yields NaNs."""
        half = self.bias.to(torch.float16)
        self.assertTrue(bool(torch.isfinite(half).all()))

    def test_masked_cross_attention_confines_edit_token_influence(self):
        generator = torch.Generator().manual_seed(9)
        query = torch.randn(1, 256, 8, generator=generator)
        key = torch.randn(1, 3, 8, generator=generator)
        value = torch.zeros(1, 3, 8)
        value[0, 0] = 1.0  # only the edit_target token carries signal
        plain = masked_cross_attention(query, key, value)
        masked = masked_cross_attention(query, key, value, self.bias)
        # Outside the region the edit token's contribution must collapse.
        self.assertLess(
            float(masked[0][~self.inside].abs().mean()),
            float(plain[0][~self.inside].abs().mean()),
        )

    def test_masked_cross_attention_matches_manual_softmax(self):
        generator = torch.Generator().manual_seed(11)
        query = torch.randn(1, 4, 8, generator=generator)
        key = torch.randn(1, 3, 8, generator=generator)
        value = torch.randn(1, 3, 8, generator=generator)
        bias = torch.randn(3, 4, generator=generator)
        expected = (
            ((query @ key.transpose(-1, -2)) * (8**-0.5) + bias.transpose(-1, -2))
            .softmax(dim=-1)
            @ value
        )
        torch.testing.assert_close(
            masked_cross_attention(query, key, value, bias), expected
        )


class ExtractEditMaskTests(unittest.TestCase):
    def test_extraction_recovers_the_attended_region(self):
        mask = box_mask(4, 4, 8, 8, size=16)
        weights = mask_to_token_weights(mask, 256) >= 0.5
        logits = torch.zeros(256, 5)
        logits[weights, 1] = 6.0
        soft, binary = extract_edit_mask(logits.softmax(dim=-1)[None], [1], threshold=0.35)
        self.assertGreater(iou(binary, resize_mask(mask, 16, 16)), 0.9)
        self.assertGreaterEqual(float(soft.min()), 0.0)
        self.assertLessEqual(float(soft.max()), 1.0)

    def test_threshold_controls_binary_area(self):
        logits = torch.rand(256, 4, generator=torch.Generator().manual_seed(4))
        _, loose = extract_edit_mask(logits[None], [0], threshold=0.2)
        _, tight = extract_edit_mask(logits[None], [0], threshold=0.8)
        self.assertGreaterEqual(area_ratio(loose), area_ratio(tight))

    def test_no_valid_keywords_yields_an_empty_mask(self):
        _, binary = extract_edit_mask(torch.rand(1, 64, 4), [99])
        self.assertEqual(area_ratio(binary), 0.0)


class EdgeBlendingTests(unittest.TestCase):
    def test_feathering_removes_the_hard_seam(self):
        source = torch.zeros(1, 4, 16, 16)
        denoised = torch.ones(1, 4, 16, 16)
        mask = box_mask(4, 4, 12, 12, size=16)
        hard = denoised * mask + source * (1 - mask)
        soft = apply_edge_blending(denoised, source, mask, blend_width=3)

        def max_step(x):
            return float((x[..., 1:, :] - x[..., :-1, :]).abs().max())

        self.assertLess(max_step(soft), max_step(hard))

    def test_blend_width_zero_is_a_hard_composite(self):
        source = torch.zeros(1, 4, 16, 16)
        denoised = torch.ones(1, 4, 16, 16)
        mask = box_mask(4, 4, 12, 12, size=16)
        blended = apply_edge_blending(denoised, source, mask, blend_width=0)
        self.assertAlmostEqual(float(blended[0, 0, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(blended[0, 0, 8, 8]), 1.0, places=6)


class SharedMaskTests(unittest.TestCase):
    """The goal requires attention masking and guidance to use the same edit_mask."""

    def test_plan_mask_drives_both_bias_and_guidance_map(self):
        prompt_embedding = torch.randn(DIM, generator=torch.Generator().manual_seed(6))
        image = (
            embedding_at(0.20, prompt_embedding)
            .view(1, DIM, 1, 1).expand(1, DIM, 64, 64).contiguous()
        )
        plan = plan_edit(
            prompt="change the jacket to red but keep the background neutral",
            prompt_embedding=prompt_embedding,
            source_image_embedding=image,
            user_mask=box_mask(20, 20, 38, 38),
            allow_clarification=False,
        )
        scale_map = apply_region_guidance(plan.mask, plan.coefficients)
        bias = build_attention_bias(plan.mask, list(plan.token_roles), num_image_tokens=4096)
        inside = mask_to_token_weights(plan.mask, 4096) >= 0.5
        flat_scale = scale_map.reshape(-1)
        # Both derive from plan.mask, so both must agree on which cells are inside.
        self.assertGreater(float(flat_scale[inside].mean()), float(flat_scale[~inside].mean()))
        targets = [i for i, r in enumerate(plan.token_roles) if r == "edit_target"]
        self.assertTrue(targets)
        self.assertLess(float(bias[targets][:, ~inside].max()), 0.0)
        self.assertAlmostEqual(float(bias[targets][:, inside].max()), 0.0, places=4)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.prompt = torch.randn(DIM, generator=torch.Generator().manual_seed(2))

    def _report(self, similarity, **kwargs):
        image = embedding_at(similarity, self.prompt).view(1, DIM, 1, 1).expand(1, DIM, 32, 32)
        return check_prompt_image_alignment(
            prompt=kwargs.pop("prompt", "make the shirt red"),
            prompt_embedding=self.prompt,
            source_image_embedding=image.contiguous(),
            edit_region_mask=torch.ones(1, 1, 32, 32),
            **kwargs,
        )

    def test_matching_prompt_is_aligned_and_generates(self):
        report = self._report(0.40)
        self.assertEqual(report.status, "aligned")
        self.assertTrue(report.should_generate)

    def test_unrelated_prompt_asks_one_question(self):
        report = self._report(-0.5)
        self.assertEqual(report.status, "clarify")
        self.assertFalse(report.should_generate)
        self.assertEqual(report.clarifying_question.count("?"), 1)

    def test_realtime_mode_never_blocks_and_records_an_assumption(self):
        report = self._report(-0.5, allow_clarification=False)
        self.assertEqual(report.status, "assumed")
        self.assertTrue(report.should_generate)
        self.assertIsNotNone(report.assumption)

    def test_weak_match_proceeds_with_an_assumption(self):
        report = self._report(0.16)
        self.assertEqual(report.status, "assumed")
        self.assertTrue(report.should_generate)

    def test_count_conflict_is_detected(self):
        self.assertIsNotNone(
            check_scene_conflict("add a second person to the photo", {"person": 3})
        )

    def test_count_conflict_respects_the_nearest_quantifier(self):
        # "a second person" must read as 2, not as the article "a".
        self.assertIsNone(check_scene_conflict("add a second person", {"person": 1}))

    def test_no_scene_facts_means_no_guessing(self):
        self.assertIsNone(check_scene_conflict("add a second person", None))


class Tokenizer2:
    """Pieces for 'make the car blue'."""

    def __call__(self, prompt, add_special_tokens=True):
        return {"input_ids": list(range(4))}

    def convert_ids_to_tokens(self, ids):
        return ["make", "the", "car", "blue"]


class ReviewRegressionTests(unittest.TestCase):
    """Regressions for defects found in the post-build review."""

    def test_adding_an_object_is_not_a_conflict(self):
        """'add a person' to a 3-person photo is ordinary, not a contradiction."""
        for prompt in ("add a person on the left", "add another person to the photo",
                       "add a dog next to them"):
            self.assertIsNone(
                check_scene_conflict(prompt, {"person": 3, "dog": 0}), prompt
            )

    def test_ordinal_conflict_is_still_caught(self):
        self.assertIsNotNone(
            check_scene_conflict("add a second person to the photo", {"person": 3})
        )

    def test_removal_beyond_availability_is_still_caught(self):
        self.assertIsNotNone(check_scene_conflict("remove three cats", {"cat": 1}))

    def test_removal_within_availability_is_allowed(self):
        self.assertIsNone(check_scene_conflict("remove two cats", {"cat": 4}))

    def test_prompt_hint_cannot_override_an_explicit_user_mask(self):
        """A keyword must not silently widen a user's region to a global edit."""
        small = box_mask(20, 28, 28, 36)
        self.assertEqual(
            classify_scope("change the jacket to a watercolor style", small,
                           mask_is_explicit=True),
            "local",
        )
        # Without an explicit mask the hint still applies.
        self.assertEqual(
            classify_scope("change the jacket to a watercolor style", small), "global"
        )

    def test_large_explicit_mask_is_still_promoted_to_global(self):
        """Coverage-based promotion must survive the override fix."""
        self.assertEqual(
            classify_scope("recolor it", torch.ones(1, 1, 64, 64), mask_is_explicit=True),
            "global",
        )

    def test_planner_respects_an_explicit_mask_with_a_style_word(self):
        prompt_embedding = torch.randn(DIM, generator=torch.Generator().manual_seed(8))
        image = (
            embedding_at(0.25, prompt_embedding)
            .view(1, DIM, 1, 1).expand(1, DIM, 64, 64).contiguous()
        )
        plan = plan_edit(
            prompt="change the jacket to a watercolor style",
            prompt_embedding=prompt_embedding,
            source_image_embedding=image,
            user_mask=box_mask(20, 20, 30, 30),
            allow_clarification=False,
        )
        self.assertEqual(plan.scope, "local")
        self.assertGreater(plan.attention_strength, 0.0)  # masking stays ON

    def test_empty_user_mask_falls_back_instead_of_denoising_a_no_op(self):
        prompt_embedding = torch.randn(DIM, generator=torch.Generator().manual_seed(12))
        image = (
            embedding_at(0.25, prompt_embedding)
            .view(1, DIM, 1, 1).expand(1, DIM, 64, 64).contiguous()
        )
        with self.assertLogs("app.services.editing.edit_planner", level="WARNING"):
            plan = plan_edit(
                prompt="change the jacket to red",
                prompt_embedding=prompt_embedding,
                source_image_embedding=image,
                user_mask=torch.zeros(1, 1, 64, 64),
                allow_clarification=False,
            )
        self.assertEqual(plan.mask_source, "global_fallback")
        self.assertGreater(area_ratio(plan.mask), 0.0)

    def test_token_matching_is_not_substring_based(self):
        """'car' must not select the token for 'carpet'."""

        class Tokenizer:
            def __call__(self, prompt, add_special_tokens=True):
                return {"input_ids": list(range(4))}

            def convert_ids_to_tokens(self, ids):
                return ["make", "the", "carpet", "blue"]

        self.assertEqual(
            locate_edit_tokens("make the carpet blue", ("car",), Tokenizer()), ()
        )
        # ...but it does select the token for 'car' itself.
        self.assertEqual(
            locate_edit_tokens("make the car blue", ("car",), Tokenizer2()), (2,)
        )

    def test_degenerate_latent_grid_warns(self):
        with self.assertLogs("app.services.editing.region_attention", level="WARNING"):
            latent_grid(257)

    def test_processor_warns_when_the_bias_shape_is_unusable(self):
        """A dropped bias disables masking silently; it must be reported."""

        class Attn:
            pass

        def base(attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
            return hidden_states

        processor = RegionAwareAttnProcessor(base)
        processor.set_bias(torch.zeros(999, 7))  # wrong shape for the block below
        with self.assertLogs("app.services.editing.region_attention", level="WARNING"):
            processor(Attn(), torch.zeros(1, 64, 8),
                      encoder_hidden_states=torch.zeros(1, 4, 8))


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.prompt_embedding = torch.randn(DIM, generator=torch.Generator().manual_seed(3))
        self.image = (
            embedding_at(0.30, self.prompt_embedding)
            .view(1, DIM, 1, 1)
            .expand(1, DIM, 64, 64)
            .contiguous()
        )

    def test_edit_terms_drop_stopwords(self):
        self.assertEqual(select_edit_terms("change the shirt color to red"),
                         ("shirt", "color", "red"))

    def test_scope_classification(self):
        self.assertEqual(classify_scope("recolor the mug", box_mask(0, 0, 8, 8)), "local")
        self.assertEqual(classify_scope("recolor the mug", box_mask(0, 0, 40, 40)), "regional")
        self.assertEqual(classify_scope("recolor it", torch.ones(1, 1, 64, 64)), "global")

    def test_prompt_hint_forces_global_scope_despite_a_small_mask(self):
        self.assertEqual(
            classify_scope("make the whole image a watercolor painting", box_mask(0, 0, 8, 8)),
            "global",
        )

    def test_user_mask_wins_and_local_edits_get_full_attention_masking(self):
        plan = plan_edit(
            prompt="change the shirt color to red",
            prompt_embedding=self.prompt_embedding,
            source_image_embedding=self.image,
            user_mask=box_mask(24, 24, 40, 40),
            allow_clarification=False,
        )
        self.assertEqual(plan.mask_source, "user")
        self.assertEqual(plan.scope, "local")
        self.assertEqual(plan.attention_strength, 1.0)
        self.assertIsNotNone(plan.bounding_box)

    def test_global_edits_disable_attention_masking(self):
        plan = plan_edit(
            prompt="make the whole image a watercolor painting",
            prompt_embedding=self.prompt_embedding,
            source_image_embedding=self.image,
            allow_clarification=False,
        )
        self.assertEqual(plan.scope, "global")
        self.assertEqual(plan.attention_strength, 0.0)

    def test_missing_region_evidence_falls_back_to_a_global_edit(self):
        plan = plan_edit(
            prompt="make it better",
            prompt_embedding=self.prompt_embedding,
            source_image_embedding=self.image,
            allow_clarification=False,
        )
        self.assertEqual(plan.mask_source, "global_fallback")
        self.assertAlmostEqual(area_ratio(plan.mask), 1.0, places=5)

    def test_conflicting_prompt_stops_before_denoising(self):
        plan = plan_edit(
            prompt="add a second person to the photo",
            prompt_embedding=self.prompt_embedding,
            source_image_embedding=self.image,
            scene_facts={"person": 3},
            allow_clarification=True,
        )
        self.assertFalse(plan.should_generate)
        self.assertEqual(plan.alignment.status, "clarify")

    def test_plan_log_dict_is_json_safe(self):
        import json

        plan = plan_edit(
            prompt="change the shirt to red",
            prompt_embedding=self.prompt_embedding,
            source_image_embedding=self.image,
            user_mask=box_mask(24, 24, 40, 40),
            allow_clarification=False,
        )
        json.dumps(plan.as_log_dict())  # must not raise


class EditPipelineTests(unittest.TestCase):
    """The headline claim: the same denoiser leaks less under the region-aware loop."""

    def setUp(self):
        generator = torch.Generator().manual_seed(5)
        self.source = torch.randn(1, 4, 32, 32, generator=generator)
        self.direction = torch.randn(1, 4, 1, 1, generator=generator)
        self.direction = self.direction / self.direction.norm()
        self.mask = box_mask(12, 12, 20, 20, size=32)
        self.timesteps = list(range(8))
        target = self.source + self.direction

        def denoise(latents, timestep, conditional):
            return latents - (target if conditional else self.source)

        def step(latents, noise_pred, timestep):
            return latents - noise_pred / len(self.timesteps)

        self.denoise, self.step = denoise, step

    def _plan(self):
        return plan_edit(
            prompt="change the shirt color to red",
            prompt_embedding=self.direction.reshape(-1),
            source_image_embedding=self.source,
            user_mask=self.mask,
            allow_clarification=False,
            latent_size=(32, 32),
        )

    def test_region_aware_loop_reduces_leakage_versus_baseline(self):
        plan = self._plan()
        baseline = run_baseline_edit(
            source_latents=self.source, initial_latents=self.source.clone(),
            timesteps=self.timesteps, denoise=self.denoise,
            guidance_scale=1.0, step=self.step,
        )
        proposed = run_region_aware_edit(
            plan=plan, source_latents=self.source, initial_latents=self.source.clone(),
            timesteps=self.timesteps, denoise=self.denoise, step=self.step,
        )
        baseline_leak = unintended_change_ratio(self.source, baseline, plan.mask)
        proposed_leak = unintended_change_ratio(self.source, proposed, plan.mask)
        self.assertLess(proposed_leak, baseline_leak)

    def test_region_aware_loop_still_changes_the_region(self):
        """A clean edit that never happened is not an improvement."""
        plan = self._plan()
        proposed = run_region_aware_edit(
            plan=plan, source_latents=self.source, initial_latents=self.source.clone(),
            timesteps=self.timesteps, denoise=self.denoise, step=self.step,
        )
        inside_change = float(
            ((proposed - self.source).abs() * resize_mask(plan.mask, 32, 32)).sum()
        )
        self.assertGreater(inside_change, 0.0)
        self.assertGreater(inside_alignment(self.source, proposed, self.direction, plan.mask), 0.9)

    def test_preservation_outside_improves(self):
        plan = self._plan()
        outside = 1.0 - resize_mask(plan.mask, 32, 32)
        baseline = run_baseline_edit(
            source_latents=self.source, initial_latents=self.source.clone(),
            timesteps=self.timesteps, denoise=self.denoise,
            guidance_scale=1.0, step=self.step,
        )
        proposed = run_region_aware_edit(
            plan=plan, source_latents=self.source, initial_latents=self.source.clone(),
            timesteps=self.timesteps, denoise=self.denoise, step=self.step,
        )
        self.assertGreater(
            ssim(self.source, proposed, outside), ssim(self.source, baseline, outside)
        )

    def test_disabling_blending_is_a_supported_ablation(self):
        plan = self._plan()
        without = run_region_aware_edit(
            plan=plan, source_latents=self.source, initial_latents=self.source.clone(),
            timesteps=self.timesteps, denoise=self.denoise, step=self.step, blend=False,
        )
        with_blend = run_region_aware_edit(
            plan=plan, source_latents=self.source, initial_latents=self.source.clone(),
            timesteps=self.timesteps, denoise=self.denoise, step=self.step, blend=True,
        )
        self.assertLess(
            unintended_change_ratio(self.source, with_blend, plan.mask),
            unintended_change_ratio(self.source, without, plan.mask),
        )

    def test_run_region_aware_edit_preserves_device_and_dtype(self):
        plan = self._plan()
        source_f64 = self.source.to(torch.float64)
        initial_f64 = self.source.clone().to(torch.float64)

        def denoise_f64(latents, timestep, conditional):
            return latents * 0.1

        out = run_region_aware_edit(
            plan=plan,
            source_latents=source_f64,
            initial_latents=initial_f64,
            timesteps=self.timesteps,
            denoise=denoise_f64,
            step=self.step,
        )
        self.assertEqual(out.dtype, torch.float64)
        self.assertEqual(out.device, initial_f64.device)

    def test_set_region_bias_with_aspect(self):
        plan = self._plan()

        class DummyProcessor:
            def __init__(self):
                self.bias = None

            def set_bias(self, bias):
                self.bias = bias

        proc = DummyProcessor()
        set_region_bias(
            [proc],
            plan,
            num_image_tokens=1024,
            num_text_tokens=8,
            aspect=2.0,
        )
        self.assertIsNotNone(proc.bias)
        self.assertEqual(proc.bias.shape, (1024, 8))


class BatchedGuidanceTests(unittest.TestCase):
    """Classifier-free guidance in one forward instead of two."""

    def setUp(self):
        generator = torch.Generator().manual_seed(21)
        self.source = torch.randn(1, 4, 32, 32, generator=generator)
        self.direction = torch.randn(1, 4, 1, 1, generator=generator)
        self.target = self.source + self.direction
        self.timesteps = list(range(6))
        self.plan = plan_edit(
            prompt="change the jacket to red",
            prompt_embedding=self.direction.reshape(-1),
            source_image_embedding=self.source,
            user_mask=box_mask(10, 10, 20, 20, size=32),
            allow_clarification=False,
            latent_size=(32, 32),
        )

    def _step(self, latents, noise_pred, timestep):
        return latents - noise_pred / len(self.timesteps)

    def test_batched_and_two_call_paths_agree(self):
        calls = {"two": 0, "batched": 0}

        def two_call(latents, timestep, conditional):
            calls["two"] += 1
            return latents - (self.target if conditional else self.source)

        def model_call(batched, timestep, embeddings):
            calls["batched"] += 1
            half = batched.shape[0] // 2
            return torch.cat(
                [batched[:half] - self.source, batched[half:] - self.target], dim=0
            )

        pair = batched_cfg_denoiser(model_call, torch.zeros(1, 2, 4), torch.ones(1, 2, 4))
        kwargs = dict(
            plan=self.plan, source_latents=self.source,
            timesteps=self.timesteps, step=self._step,
        )
        a = run_region_aware_edit(
            initial_latents=self.source.clone(), denoise=two_call, **kwargs
        )
        b = run_region_aware_edit(
            initial_latents=self.source.clone(), denoise_pair=pair, **kwargs
        )
        torch.testing.assert_close(a, b)
        # Halved forward passes is the entire point.
        self.assertEqual(calls["two"], 2 * len(self.timesteps))
        self.assertEqual(calls["batched"], len(self.timesteps))

    def test_baseline_loop_also_accepts_the_batched_convention(self):
        def model_call(batched, timestep, embeddings):
            half = batched.shape[0] // 2
            return torch.cat(
                [batched[:half] - self.source, batched[half:] - self.target], dim=0
            )

        pair = batched_cfg_denoiser(model_call, torch.zeros(1, 2, 4), torch.ones(1, 2, 4))
        out = run_baseline_edit(
            source_latents=self.source, initial_latents=self.source.clone(),
            timesteps=self.timesteps, guidance_scale=1.0,
            denoise_pair=pair, step=self._step,
        )
        self.assertEqual(tuple(out.shape), tuple(self.source.shape))

    def test_missing_both_conventions_is_an_error(self):
        with self.assertRaises(ValueError):
            run_region_aware_edit(
                plan=self.plan, source_latents=self.source,
                initial_latents=self.source.clone(), timesteps=self.timesteps,
            )


class MetricTests(unittest.TestCase):
    def test_ssim_is_one_for_identical_images(self):
        image = torch.rand(1, 3, 32, 32)
        self.assertAlmostEqual(ssim(image, image), 1.0, places=4)

    def test_leakage_is_zero_when_change_is_confined(self):
        source = torch.rand(1, 3, 32, 32)
        mask = box_mask(0, 0, 8, 8, size=32)
        edited = source.clone()
        edited[..., :8, :8] = 0.0
        self.assertAlmostEqual(unintended_change_ratio(source, edited, mask), 0.0, places=5)

    def test_leakage_is_one_when_change_misses_entirely(self):
        source = torch.rand(1, 3, 32, 32)
        mask = box_mask(0, 0, 8, 8, size=32)
        edited = source.clone()
        edited[..., 16:, 16:] = 0.0
        self.assertAlmostEqual(unintended_change_ratio(source, edited, mask), 1.0, places=5)

    def test_preservation_metrics_are_nan_for_a_global_edit(self):
        """Undefined, not zero - a global edit has no outside region to preserve."""
        source = torch.rand(1, 3, 16, 16)
        metrics = evaluate_edit(
            source=source, edited=torch.rand(1, 3, 16, 16),
            edit_mask=torch.ones(1, 1, 16, 16), alignment=1.0,
        )
        self.assertNotEqual(metrics.preservation_ssim, metrics.preservation_ssim)


if __name__ == "__main__":
    unittest.main()
