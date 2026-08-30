import unittest
from unittest.mock import patch

from app.services.prompt_enhancer import (
    LIGHTING_MODIFIERS,
    QUALITY_MODIFIERS,
    STYLE_MODIFIERS,
    enhance_prompt,
)


class PromptEnhancerTests(unittest.TestCase):
    def test_appends_one_modifier_from_each_predefined_list(self):
        with patch("app.services.prompt_enhancer.random.SystemRandom") as system_random:
            system_random.return_value.choice.side_effect = lambda values: values[-1]

            result = enhance_prompt("  a cat  ")

        self.assertEqual(
            result,
            (
                "a cat, analog film aesthetic, high-contrast chiaroscuro lighting, "
                "polished high-resolution finish"
            ),
        )
        self.assertEqual(
            [call.args[0] for call in system_random.return_value.choice.call_args_list],
            [STYLE_MODIFIERS, LIGHTING_MODIFIERS, QUALITY_MODIFIERS],
        )

    def test_seed_makes_the_expansion_reproducible(self):
        first = enhance_prompt("a cat", seed=42)
        second = enhance_prompt("a cat", seed=42)

        self.assertEqual(first, second)
        prompt, style, lighting, quality = first.split(", ")
        self.assertEqual(prompt, "a cat")
        self.assertIn(style, STYLE_MODIFIERS)
        self.assertIn(lighting, LIGHTING_MODIFIERS)
        self.assertIn(quality, QUALITY_MODIFIERS)

    def test_blank_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            enhance_prompt("   ")


if __name__ == "__main__":
    unittest.main()
