"""Regression tests for defects found in the semantic-layout review.

Kept separate from test_hybrid_reasoning.py so the two suites can evolve
independently. Each test pins a specific defect that was reproduced before it
was fixed.
"""

import unittest

from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import (
    _RELATION_WORDS,
    _SPATIAL_WORDS,
    NormalizedBox,
    _allocate_free_slots,
    _free_x_intervals,
    drop_spurious_relations,
    extract_position_constraints,
)
from app.services.editing.semantic_planner import (
    plan_semantic_layout as _plan_semantic_layout,
)


def plan_semantic_layout(prompt: str, **kwargs):
    return _plan_semantic_layout(analyze_prompt(prompt, mode="generate"), **kwargs)


def worst_overlap(plan) -> float:
    boxes = [obj.box for obj in plan.objects]
    return max(
        (a.iou(b) for index, a in enumerate(boxes) for b in boxes[index + 1 :]),
        default=0.0,
    )


class PositionalLayoutTests(unittest.TestCase):
    """A stated position must actually move the object."""

    def test_left_and_right_are_separated(self):
        plan = plan_semantic_layout("two cats on the left and a dog on the right")
        boxes = {obj.label: obj.box for obj in plan.objects}
        self.assertIn("cats", boxes)
        self.assertIn("dog", boxes)
        self.assertLess(boxes["cats"].center[1], boxes["dog"].center[1])
        self.assertLess(worst_overlap(plan), 0.1)

    def test_single_object_honours_its_position(self):
        plan = plan_semantic_layout("a bird on the left")
        box = plan.objects[0].box
        self.assertLess(box.center[1], 0.5)

    def test_top_and_bottom_are_separated_vertically(self):
        plan = plan_semantic_layout("a lamp at the top and a rug at the bottom")
        boxes = {obj.label: obj.box for obj in plan.objects}
        if "lamp" in boxes and "rug" in boxes:
            self.assertLess(boxes["lamp"].center[0], boxes["rug"].center[0])

    def test_position_constraints_are_extracted(self):
        positions = extract_position_constraints(
            "two cats on the left and a dog on the right", {"cats", "dog"}
        )
        self.assertEqual(positions, {"cats": "left", "dog": "right"})

    def test_position_words_without_a_known_object_are_ignored(self):
        self.assertEqual(extract_position_constraints("on the left", set()), {})


class PhantomObjectTests(unittest.TestCase):
    """Prepositions and relation verbs are not renderable objects."""

    def test_no_function_words_are_planned(self):
        prompts = [
            "two cats on the left and a dog on the right",
            "a monkey riding a giraffe",
            "a cat above a dog",
            "a bird in front of a tree",
            "a retriever next to a cat",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                labels = {obj.label for obj in plan_semantic_layout(prompt).objects}
                self.assertFalse(labels & set(_SPATIAL_WORDS), labels)
                self.assertFalse(labels & set(_RELATION_WORDS), labels)

    def test_real_objects_survive(self):
        labels = {obj.label for obj in plan_semantic_layout("a monkey riding a giraffe").objects}
        self.assertEqual(labels, {"monkey", "giraffe"})


class SpuriousRelationTests(unittest.TestCase):
    def test_sentinel_relations_are_dropped(self):
        kept = drop_spurious_relations(
            [("dog", "on", "object"), ("subject", "on", "cat"), ("cat", "on", "mat")], {}
        )
        self.assertEqual(kept, [("cat", "on", "mat")])

    def test_relations_between_positioned_objects_are_dropped(self):
        kept = drop_spurious_relations([("cats", "on", "dog")], {"cats": "left", "dog": "right"})
        self.assertEqual(kept, [])


class FreeSlotAllocationTests(unittest.TestCase):
    """Unplaced objects must not be tiled over regions already taken."""

    def test_free_intervals_exclude_occupied_space(self):
        placed = [NormalizedBox(ymin=0.1, xmin=0.25, ymax=0.5, xmax=0.75)]
        free = _free_x_intervals(placed)
        for start, end in free:
            self.assertTrue(end <= 0.25 + 1e-6 or start >= 0.75 - 1e-6)

    def test_allocation_avoids_placed_boxes(self):
        placed = [NormalizedBox(ymin=0.1, xmin=0.25, ymax=0.5, xmax=0.75)]
        allocated = _allocate_free_slots([("apples", 3, [])], placed)
        self.assertEqual(len(allocated), 1)
        self.assertLess(allocated[0][1].iou(placed[0]), 0.1)

    def test_allocation_without_placed_boxes_tiles_evenly(self):
        allocated = _allocate_free_slots([("a", 1, []), ("b", 1, [])], [])
        self.assertEqual(len(allocated), 2)
        self.assertLess(allocated[0][1].iou(allocated[1][1]), 0.1)

    def test_objects_and_relations_do_not_collide(self):
        """The case the always-true self-check used to hide."""
        plan = plan_semantic_layout("three red apples and two green pears on a rustic wooden table")
        self.assertLess(worst_overlap(plan), 0.5)
        self.assertTrue(plan.self_check.is_valid)


class SelfCheckTests(unittest.TestCase):
    """The self-check must be able to fail."""

    def test_valid_plan_passes(self):
        plan = plan_semantic_layout("a monkey riding a giraffe")
        self.assertTrue(plan.self_check.is_valid)
        self.assertIn("passed", plan.self_check.notes)

    def test_empty_prompt_is_invalid(self):
        plan = plan_semantic_layout("   ")
        self.assertFalse(plan.self_check.is_valid)

    def test_notes_explain_a_failure(self):
        """A failing check must say why, not just report False."""
        plan = plan_semantic_layout("   ")
        self.assertTrue(plan.self_check.notes)


if __name__ == "__main__":
    unittest.main()
