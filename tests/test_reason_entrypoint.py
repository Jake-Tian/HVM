import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import reason
from reasoning_variants.fixed_pipeline_backup import reason_pipeline
from reasoning.agent import DEFAULT_BUDGET, build_agent
from reasoning_variants.five_tool_helper import build_agent as legacy_build_agent


class _FakeAgent:
    def invoke(self, state):
        return {
            "messages": [*state["messages"], AIMessage(content="answer")],
            "total_tokens": 7,
        }


class ReasonEntrypointTest(unittest.TestCase):
    def test_default_reason_uses_five_tool_agent(self):
        graph = object()
        with patch.object(reason, "build_agent", return_value=_FakeAgent()) as build:
            result = reason.reason(graph, "video", "question")

        build.assert_called_once_with(graph, "video", budget=DEFAULT_BUDGET)
        self.assertEqual(result["final_answer"], "answer")
        self.assertEqual(result["rounds"], [])
        self.assertEqual(result["token_summaries"], {"total": 7})

    def test_fixed_pipeline_remains_reexported(self):
        self.assertIs(reason.reason_pipeline, reason_pipeline)

    def test_legacy_five_tool_import_remains_compatible(self):
        self.assertIs(legacy_build_agent, build_agent)


if __name__ == "__main__":
    unittest.main()
