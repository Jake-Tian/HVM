import unittest
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from reasoning.runtime import (
    build_graph_stats,
    execute_tool_calls,
    verifier_hints,
)


class FakeTool:
    def __init__(self, result):
        self.result = result

    def invoke(self, args):
        return self.result


class FiveToolRuntimeTest(unittest.TestCase):
    def test_graph_stats_keep_the_prompt_contract(self):
        edges = {
            1: SimpleNamespace(clip_id=0, scene=None),
            2: SimpleNamespace(clip_id=3, scene="office"),
        }
        graph = SimpleNamespace(
            characters={"<Alice>": object()},
            objects={"pen": object()},
            edges=edges,
            conversations={1: object()},
        )

        stats = build_graph_stats(graph)

        self.assertEqual(stats, """--- Graph Stats ---
Characters: <Alice>
Total Object Nodes: 1
Total Edges: 2 (High-level: 1, Low-level: 1)
Total Clips: 3
Total Conversations: 1
Scenes: office
Graph Construction Info:
- The video is divided into 30-second clips (Clip 1 = 0-30s, Clip 2 = 30-60s, etc.).
- Character nodes are enclosed in angle brackets (e.g., <robot>, <Alice>).
- Object nodes are plain text (e.g., coffee, table).
- High-level edges (clip_id=0) represent overall attributes and relationships.
- Low-level edges represent specific actions/states occurring in specific clips.
-------------------""")

    def test_executor_preserves_history_messages_and_tokens(self):
        result = execute_tool_calls(
            [{
                "name": "general_search",
                "args": {"query": "pen"},
                "id": "call-1",
            }],
            {"general_search": FakeTool(("evidence", {"total": 7}))},
            [],
        )

        self.assertEqual(result["total_tokens"], 7)
        self.assertEqual(result["tool_call_history"], [
            {"name": "general_search", "args": {"query": "pen"}}
        ])
        self.assertEqual(result["messages"][0].content, "evidence")

    def test_executor_rejects_duplicate_video_clip(self):
        result = execute_tool_calls(
            [{
                "name": "watch_video_clip",
                "args": {"clip_id": 4, "focus": "red pen"},
                "id": "call-2",
            }],
            {"watch_video_clip": FakeTool(("unused", 0))},
            [4],
        )

        self.assertEqual(result["clip_history"], [])
        self.assertIn("already watched clip 4", result["messages"][0].content)

    def test_verifier_keeps_repeated_call_and_temporal_hints(self):
        state = {
            "question": "What happened after Alice left?",
            "messages": [ToolMessage(content="evidence", tool_call_id="call-3")],
            "tool_call_history": [
                {"name": "general_search", "args": {"query": "Alice"}},
                {"name": "general_search", "args": {"query": "Alice"}},
            ],
        }

        hints = verifier_hints(state)

        self.assertEqual(len(hints), 2)
        self.assertIn("repeated the exact same tool call", hints[0].content)
        self.assertIn("temporal-order question", hints[1].content)


if __name__ == "__main__":
    unittest.main()
