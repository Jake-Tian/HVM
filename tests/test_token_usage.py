import unittest
from types import SimpleNamespace

from utils.token_usage import (
    TokenUsage,
    build_token_summary,
    merge_stage_usage,
    merge_usage,
    usage_from_response,
)


class TokenUsageTest(unittest.TestCase):
    def test_openai_compatible_usage_uses_total_only(self):
        response = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=15),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ))
        usage = usage_from_response(response, "gpt-test", "openai")
        self.assertEqual(usage, {"total": 120})
        self.assertEqual(int(usage), 120)

    def test_langchain_usage_metadata_falls_back_to_input_plus_output(self):
        response = SimpleNamespace(usage_metadata={
            "input_tokens": 80,
            "output_tokens": 12,
            "input_token_details": {"cache_read": 7},
            "output_token_details": {"reasoning": 3},
        })
        usage = usage_from_response(response, "qwen-test", "dashscope")
        self.assertEqual(usage, {"total": 92})

    def test_merge_and_summary_are_flat_totals(self):
        first = TokenUsage({"total": 12})
        second = TokenUsage({"total": 25})
        stages = merge_stage_usage({"planner": first}, {"planner": second})
        summary = build_token_summary(stages)
        self.assertEqual(summary, {"planner": 37, "total": 37})

    def test_legacy_usage_shapes_remain_compatible(self):
        usage = merge_usage(
            10,
            {"input": 20, "output": 5},
            {"total_tokens": 7},
        )
        self.assertEqual(usage, {"total": 42})
        self.assertEqual(10 + usage, 52)


if __name__ == "__main__":
    unittest.main()
