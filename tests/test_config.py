import unittest

from pydantic import ValidationError

from app.core.config import Settings


class SettingsTests(unittest.TestCase):
    def test_clean_defaults_are_internally_consistent(self):
        configured = Settings(_env_file=None)
        self.assertEqual(configured.DEVICE, "auto")
        self.assertEqual(configured.DTYPE, "auto")
        self.assertEqual(configured.CHAT_MODEL, "")
        self.assertGreaterEqual(configured.CHAT_MAX_HISTORY_MESSAGES, 2)
        self.assertLessEqual(
            configured.DEFAULT_WIDTH * configured.DEFAULT_HEIGHT,
            configured.MAX_BATCH_PIXELS,
        )

    def test_invalid_device_and_default_budget_are_rejected(self):
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, DEVICE="tpu")
        with self.assertRaises(ValidationError):
            Settings(
                _env_file=None,
                DEFAULT_WIDTH=1024,
                DEFAULT_HEIGHT=1024,
                MAX_BATCH_PIXELS=262_144,
            )
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, CHAT_REASONING_TIMEOUT_SECONDS=0)
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, CHAT_MAX_HISTORY_MESSAGES=1)
        with self.assertRaises(ValidationError):
            Settings(
                _env_file=None,
                CHAT_MAX_MESSAGE_LENGTH=4_000,
                CHAT_MAX_HISTORY_CHARS=10_000,
            )


if __name__ == "__main__":
    unittest.main()
