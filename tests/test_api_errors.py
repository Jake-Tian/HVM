import unittest
from types import SimpleNamespace

from utils.api_errors import is_fatal_api_error


class ApiErrorClassificationTest(unittest.TestCase):
    def test_auth_and_billing_failures_are_fatal(self):
        for error in (
            SimpleNamespace(status_code=401),
            SimpleNamespace(response=SimpleNamespace(status_code=403)),
            RuntimeError("insufficient_quota"),
            RuntimeError("You exceeded your current quota; check billing"),
        ):
            with self.subTest(error=error):
                self.assertTrue(is_fatal_api_error(error))

    def test_rate_limits_and_temporary_capacity_are_retryable(self):
        for error in (
            RuntimeError("429 rate limit exceeded"),
            RuntimeError("request quota reached for tokens per minute"),
            RuntimeError("insufficient system capacity"),
            RuntimeError("temporary server error"),
        ):
            with self.subTest(error=error):
                self.assertFalse(is_fatal_api_error(error))


if __name__ == "__main__":
    unittest.main()
