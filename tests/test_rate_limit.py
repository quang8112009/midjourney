import unittest

from app.core.rate_limit import SlidingWindowRateLimiter


class RateLimiterTests(unittest.TestCase):
    def test_window_limit_and_expiry(self):
        limiter = SlidingWindowRateLimiter(limit=2, window_seconds=10)
        self.assertIsNone(limiter.check("client", now=0))
        self.assertIsNone(limiter.check("client", now=1))
        self.assertEqual(limiter.check("client", now=2), 8)
        self.assertIsNone(limiter.check("other", now=2))
        self.assertIsNone(limiter.check("client", now=11))

    def test_reset_clears_events(self):
        limiter = SlidingWindowRateLimiter(limit=1)
        self.assertIsNone(limiter.check("client", now=0))
        self.assertIsNotNone(limiter.check("client", now=1))
        limiter.reset()
        self.assertIsNone(limiter.check("client", now=1))


if __name__ == "__main__":
    unittest.main()
