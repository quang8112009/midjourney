"""Tests for the structured prompt-understanding stage.

Grouped by the three failure modes the stage exists to fix: ambiguous prompts,
multi-object images, and multi-instruction prompts.
"""

import json
import unittest

from app.services.editing.prompt_intent import (
    SceneObject,
    analyze_prompt,
    parse_instruction,
    split_instructions,
)


def people(*positions_and_areas):
    return [
        SceneObject("person", center_x=x, center_y=0.5, area=area)
        for x, area in positions_and_areas
    ]


class PromptModeTests(unittest.TestCase):
    def test_generation_mode_skips_edit_decomposition(self):
        intent = analyze_prompt(
            "three red apples and two green pears",
            mode="generate",
        )
        self.assertEqual(intent.mode, "generate")
        self.assertEqual(intent.status, "ok")
        self.assertEqual(intent.instructions, ())

    def test_edit_mode_is_the_default(self):
        intent = analyze_prompt("recolor the shirt to red")
        self.assertEqual(intent.mode, "edit")
        self.assertEqual(len(intent.instructions), 1)

    def test_empty_generation_prompt_asks_what_to_create(self):
        intent = analyze_prompt("  ", mode="generate")
        self.assertEqual(intent.status, "clarify")
        self.assertIn("create", intent.clarifying_question)


class SplittingTests(unittest.TestCase):
    """Multi-instruction prompts must not lose a sub-edit."""

    def test_and_splits_two_actions(self):
        self.assertEqual(
            split_instructions("change the shirt to red and blur the background"),
            ["change the shirt to red", "blur the background"],
        )

    def test_attribute_conjunction_does_not_split(self):
        """'red and blue' is one instruction, not two."""
        self.assertEqual(len(split_instructions("make the stripes red and blue")), 1)

    def test_constraint_clause_is_not_a_separate_edit(self):
        parts = split_instructions("change the jacket to red but keep the background neutral")
        self.assertEqual(len(parts), 1)

    def test_three_instructions(self):
        parts = split_instructions(
            "remove the logo, blur the background and brighten the face"
        )
        self.assertEqual(len(parts), 3)

    def test_then_sequences(self):
        parts = split_instructions("remove the logo then sharpen the face")
        self.assertEqual(len(parts), 2)

    def test_single_instruction_is_unchanged(self):
        self.assertEqual(split_instructions("blur the background"), ["blur the background"])

    def test_empty_prompt(self):
        self.assertEqual(split_instructions("   "), [])


class DecompositionTests(unittest.TestCase):
    """target / action / attribute / scope extraction."""

    def test_recolor_with_explicit_colour(self):
        instruction = parse_instruction("change the shirt color to red")
        self.assertEqual(instruction.action, "recolor")
        self.assertEqual(instruction.target, "shirt")
        self.assertEqual(instruction.attribute, "red")
        self.assertEqual(instruction.scope, "local")

    def test_blur_keeps_its_target(self):
        """'background' is a legitimate target, not only a position word."""
        instruction = parse_instruction("blur the background")
        self.assertEqual(instruction.action, "blur")
        self.assertEqual(instruction.target, "background")

    def test_removal(self):
        instruction = parse_instruction("remove the logo from the corner")
        self.assertEqual(instruction.action, "remove")
        self.assertEqual(instruction.target, "logo")

    def test_global_restyle(self):
        instruction = parse_instruction("make the whole image a watercolor painting")
        self.assertEqual(instruction.scope, "global")
        self.assertEqual(instruction.attribute, "watercolor")

    def test_possessive_target_is_the_owned_thing(self):
        """In "the person's shirt" the mask belongs on the shirt."""
        instruction = parse_instruction("change the person's shirt to blue")
        self.assertEqual(instruction.target, "shirt")
        self.assertIn("person", instruction.nouns)

    def test_position_and_ordinal_are_captured(self):
        left = parse_instruction("recolor the shirt of the person on the left")
        self.assertEqual(left.position, "left")
        second = parse_instruction("recolor the second person's shirt")
        self.assertEqual(second.ordinal, 2)

    def test_constraints_are_recorded_not_dropped(self):
        instruction = parse_instruction("change the jacket to red but keep the background neutral")
        self.assertTrue(instruction.constraints)

    def test_confidence_drops_when_underspecified(self):
        vague = parse_instruction("fix this")
        precise = parse_instruction("change the shirt to red")
        self.assertLess(vague.confidence, precise.confidence)


class DisambiguationTests(unittest.TestCase):
    """Multiple objects of the same type."""

    def test_position_selects_the_named_object(self):
        intent = analyze_prompt(
            "change the shirt of the person on the left", candidates=people((0.2, 0.2), (0.8, 0.2))
        )
        resolution = intent.instructions[0].resolution
        self.assertEqual(resolution.method, "position")
        self.assertEqual(resolution.index, 0)

    def test_right_selects_the_other_object(self):
        intent = analyze_prompt(
            "change the shirt of the person on the right", candidates=people((0.2, 0.2), (0.8, 0.2))
        )
        self.assertEqual(intent.instructions[0].resolution.index, 1)

    def test_ordinal_selects_left_to_right(self):
        intent = analyze_prompt(
            "change the second person's shirt", candidates=people((0.2, 0.2), (0.8, 0.2))
        )
        resolution = intent.instructions[0].resolution
        self.assertEqual(resolution.method, "ordinal")
        self.assertEqual(resolution.index, 1)

    def test_salience_decides_when_one_object_dominates(self):
        intent = analyze_prompt(
            "change the person's shirt", candidates=people((0.2, 0.55), (0.8, 0.05))
        )
        resolution = intent.instructions[0].resolution
        self.assertEqual(resolution.method, "salience")
        self.assertEqual(resolution.index, 0)

    def test_equal_objects_ask_instead_of_guessing(self):
        """The constraint: do not guess blindly when ambiguity is severe."""
        intent = analyze_prompt(
            "change the person's shirt", candidates=people((0.2, 0.2), (0.8, 0.2))
        )
        self.assertEqual(intent.status, "clarify")
        self.assertFalse(intent.should_generate)
        self.assertEqual(intent.clarifying_question.count("?"), 1)
        self.assertIn("person", intent.clarifying_question)

    def test_realtime_mode_assumes_instead_of_blocking(self):
        intent = analyze_prompt(
            "change the person's shirt",
            candidates=people((0.2, 0.2), (0.8, 0.2)),
            allow_clarification=False,
        )
        self.assertEqual(intent.status, "assumed")
        self.assertTrue(intent.should_generate)
        self.assertIn("shirt", intent.assumption)

    def test_single_candidate_needs_no_disambiguation(self):
        intent = analyze_prompt("change the person's shirt", candidates=people((0.5, 0.3)))
        self.assertEqual(intent.instructions[0].resolution.method, "only_candidate")

    def test_no_detector_output_trusts_the_prompt(self):
        intent = analyze_prompt("change the shirt to red")
        self.assertEqual(intent.instructions[0].resolution.method, "explicit")
        self.assertEqual(intent.status, "ok")


class AmbiguousPromptTests(unittest.TestCase):
    def test_vague_prompt_without_context_asks(self):
        intent = analyze_prompt("make it look better")
        self.assertEqual(intent.status, "clarify")
        self.assertIsNotNone(intent.clarifying_question)

    def test_vague_prompt_with_image_context_assumes_and_logs_it(self):
        intent = analyze_prompt(
            "make it look better", image_type="portrait", main_subject="a woman in a red coat"
        )
        self.assertEqual(intent.status, "assumed")
        self.assertIn("portrait", intent.assumption)
        self.assertIn("woman", intent.assumption)

    def test_specific_prompt_is_not_treated_as_vague(self):
        self.assertEqual(analyze_prompt("change the shirt to red").status, "ok")

    def test_empty_prompt_asks(self):
        intent = analyze_prompt("   ")
        self.assertEqual(intent.status, "clarify")
        self.assertEqual(intent.instructions, ())


class UnderspecifiedActionTests(unittest.TestCase):
    """A target with no stated change must not be invented."""

    def test_target_without_an_operation_asks(self):
        intent = analyze_prompt(
            "change the person's shirt", candidates=people((0.2, 0.55), (0.8, 0.05))
        )
        self.assertEqual(intent.status, "clarify")
        self.assertIn("shirt", intent.clarifying_question)

    def test_stating_the_change_proceeds(self):
        intent = analyze_prompt(
            "change the person's shirt to red", candidates=people((0.2, 0.55), (0.8, 0.05))
        )
        self.assertEqual(intent.status, "ok")

    def test_actions_that_need_no_attribute_proceed(self):
        for prompt in ("blur the background", "remove the logo", "sharpen the face"):
            self.assertEqual(analyze_prompt(prompt).status, "ok", prompt)

    def test_realtime_mode_records_an_assumption(self):
        intent = analyze_prompt(
            "change the person's shirt",
            candidates=people((0.2, 0.55), (0.8, 0.05)),
            allow_clarification=False,
        )
        self.assertEqual(intent.status, "assumed")
        self.assertIn("shirt", intent.assumption)


class StructuredOutputTests(unittest.TestCase):
    """The output must be directly consumable - no re-parsing downstream."""

    def test_intent_is_json_serialisable(self):
        intent = analyze_prompt(
            "change the shirt to red and blur the background",
            candidates=people((0.3, 0.4)),
        )
        payload = json.loads(intent.to_json())
        self.assertEqual(payload["mode"], "edit")
        self.assertEqual(len(payload["instructions"]), 2)
        first = payload["instructions"][0]
        for key in ("raw_text", "action", "target", "attribute", "scope", "resolution"):
            self.assertIn(key, first)

    def test_trace_is_populated_for_debugging(self):
        intent = analyze_prompt("change the shirt to red and blur the background")
        self.assertTrue(intent.trace)
        self.assertTrue(any("split into 2" in line for line in intent.trace))

    def test_multi_instruction_keeps_every_sub_edit(self):
        intent = analyze_prompt(
            "remove the logo, blur the background and brighten the face"
        )
        actions = [instruction.action for instruction in intent.instructions]
        self.assertEqual(len(actions), 3)
        self.assertIn("remove", actions)
        self.assertIn("blur", actions)
        self.assertIn("lighten", actions)

    def test_plan_edit_consumes_the_instruction_without_reparsing(self):
        import torch

        from app.services.editing.edit_planner import plan_edit

        intent = analyze_prompt("make the whole image a watercolor painting")
        plan = plan_edit(
            intent=intent,
            instruction_index=0,
            prompt_embedding=torch.randn(8),
            source_image_embedding=torch.randn(1, 8, 64, 64),
            allow_clarification=False,
        )
        self.assertEqual(plan.scope, "global")
        self.assertEqual(plan.instruction, intent.instructions[0])
        self.assertEqual(plan.instruction_index, 0)
        self.assertNotIn("make", plan.edit_terms)
        json.dumps(plan.as_log_dict())

    def test_plan_edit_rejects_non_actionable_contracts(self):
        import torch

        from app.services.editing.edit_planner import plan_edit

        kwargs = {
            "instruction_index": 0,
            "prompt_embedding": torch.randn(8),
            "source_image_embedding": torch.randn(1, 8, 64, 64),
        }
        with self.assertRaises(ValueError):
            plan_edit(
                intent=analyze_prompt("a red fox", mode="generate"),
                **kwargs,
            )
        with self.assertRaises(ValueError):
            plan_edit(intent=analyze_prompt("make it better"), **kwargs)
        with self.assertRaises(ValueError):
            plan_edit(
                intent=analyze_prompt("recolor the shirt red"),
                instruction_index=1,
                prompt_embedding=kwargs["prompt_embedding"],
                source_image_embedding=kwargs["source_image_embedding"],
            )


if __name__ == "__main__":
    unittest.main()
