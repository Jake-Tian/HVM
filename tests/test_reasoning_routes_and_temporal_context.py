import sys
import unittest
from types import ModuleType, SimpleNamespace

# An existing routing test installs lightweight cv2/numpy stubs globally.
# Remove those non-module stubs before importing the real reasoning stack.
for _module_name in ("cv2", "numpy"):
    _module = sys.modules.get(_module_name)
    if _module is not None and not isinstance(_module, ModuleType):
        sys.modules.pop(_module_name, None)

from reasoning_variants.three_route.tools import (
    execute_search_action_evidence,
    execute_search_temporal_context,
    get_tools,
)
from reasoning_variants.three_route.agent import (
    _extract_rewatch_clips,
    _focus_preserves_location_object,
    _has_consecutive_general_searches,
    _tools_for_route,
    classify_location_mode,
    classify_question_route,
)
from utils.prompts import (
    prompt_final_answer,
    prompt_parse_query,
    prompt_planner_strategy,
    prompt_planner_strategy_action_frequency,
    prompt_planner_strategy_location,
)


class ReasoningRouteTest(unittest.TestCase):
    def test_routes_only_location_and_action_frequency_questions(self):
        self.assertEqual(
            classify_question_route("Where is the air-conditioning remote now?"),
            "location",
        )
        self.assertEqual(
            classify_question_route("Which shelf should the book be placed on?"),
            "location",
        )
        for question in (
            "Which rack should the coat be placed on?",
            "Should the cup be placed on the table or handed over?",
            "Is the book on the upper shelf or the lower shelf?",
        ):
            self.assertEqual(classify_question_route(question), "location")
        self.assertEqual(
            classify_question_route("Which tier contains the paper?"),
            "general",
        )
        self.assertEqual(
            classify_question_route("How many times was the wardrobe opened?"),
            "action_frequency",
        )
        self.assertEqual(
            classify_question_route("How often did the robot visit the desk?"),
            "action_frequency",
        )
        self.assertEqual(
            classify_question_route("How many cups are on the table?"),
            "general",
        )
        self.assertEqual(
            classify_question_route("How many drawers did the robot open?"),
            "general",
        )
        self.assertEqual(
            classify_question_route("What is the number of people in the room?"),
            "general",
        )
        self.assertEqual(
            classify_question_route("Why did Alice ask the robot for help?"),
            "general",
        )

    def test_location_parser_prompt_requires_high_object_weight(self):
        self.assertIn("Object-first weighting is mandatory", prompt_parse_query)
        self.assertIn("0.9-1.0", prompt_parse_query)

    def test_route_specific_tools_are_isolated(self):
        graph = SimpleNamespace()
        tools = get_tools(graph, "video")
        tool_names = {tool.name for tool in tools}
        self.assertNotIn("get_frequency_stats", tool_names)
        self.assertEqual(
            tool_names,
            {
                "general_search",
                "search_temporal_context",
                "search_action_evidence",
                "search_object_events",
                "watch_video_clip",
                "complete_task",
            },
        )
        location_tools = {
            tool.name for tool in _tools_for_route(tools, "location")
        }
        self.assertIn("watch_video_clip", location_tools)
        self.assertIn("search_object_events", location_tools)
        self.assertNotIn("search_action_evidence", location_tools)
        frequency_tools = {
            tool.name for tool in _tools_for_route(tools, "action_frequency")
        }
        self.assertIn("search_action_evidence", frequency_tools)
        self.assertNotIn("watch_video_clip", frequency_tools)
        self.assertNotIn("search_object_events", frequency_tools)
        general_tools = {tool.name for tool in _tools_for_route(tools, "general")}
        self.assertNotIn("search_action_evidence", general_tools)
        self.assertNotIn("watch_video_clip", general_tools)
        self.assertNotIn("search_object_events", general_tools)

    def test_prompts_prioritize_evidence_escalation_and_complete_answers(self):
        self.assertIn("Never use more than two", prompt_planner_strategy)
        self.assertIn("fully addresses the question", prompt_planner_strategy)
        self.assertIn("only when", prompt_planner_strategy_location)
        self.assertIn("watch_video_clip", prompt_planner_strategy_location)
        self.assertIn("search_object_events", prompt_planner_strategy_location)
        self.assertIn("Never use a third general search", prompt_planner_strategy_action_frequency)
        self.assertNotIn("watch_video_clip", prompt_planner_strategy)
        self.assertNotIn("watch_video_clip", prompt_planner_strategy_action_frequency)
        self.assertIn("intermediate memory", prompt_planner_strategy_action_frequency)
        self.assertIn("working summary", prompt_planner_strategy_action_frequency)
        self.assertIn("Never infer zero", prompt_planner_strategy_action_frequency)
        self.assertIn("occurrence_count", prompt_planner_strategy_action_frequency)
        self.assertIn("search_action_evidence", prompt_planner_strategy_action_frequency)
        self.assertIn("include all of them", prompt_final_answer)

    def test_location_mode_contract(self):
        self.assertEqual(
            classify_location_mode("Where did Alice do her homework?"), "event"
        )
        self.assertEqual(
            classify_location_mode("Where did the towel originally hang?"), "object"
        )

    def test_generic_rewatch_helpers(self):
        self.assertTrue(_focus_preserves_location_object(
            "red pen", "Track the red pen between the desk and holder"
        ))
        self.assertFalse(_focus_preserves_location_object(
            "red pen", "Track the blue folder between two shelves"
        ))
        self.assertEqual(
            _extract_rewatch_clips("Suggested rewatch clips: 4, 9, 12"),
            [4, 9, 12],
        )

    def test_detects_consecutive_general_searches(self):
        self.assertTrue(_has_consecutive_general_searches([
            {"name": "general_search"},
            {"name": "general_search"},
        ]))
        self.assertFalse(_has_consecutive_general_searches([
            {"name": "general_search"},
            {"name": "search_temporal_context"},
        ]))

class GraphTemporalContextTest(unittest.TestCase):
    def test_returns_graph_edges_and_conversations(self):
        graph = SimpleNamespace(
            edges={
                1: SimpleNamespace(
                    id=1,
                    clip_id=7,
                    source="<robot>",
                    content="takes",
                    target="red pen",
                    scene="study",
                )
            },
            conversations={
                1: SimpleNamespace(
                    clips=[7],
                    format_messages=lambda: "Cloe: Please bring me the red pen.",
                )
            },
        )

        result, tokens = execute_search_temporal_context(
            graph,
            clip_id=7,
        )

        self.assertEqual(tokens, 0)
        self.assertIn("[7] robot takes red pen.", result)
        self.assertIn("Cloe: Please bring me the red pen.", result)

    def test_returns_empty_message_when_graph_has_no_temporal_evidence(self):
        graph = SimpleNamespace(
            edges={},
            conversations={},
        )

        result, tokens = execute_search_temporal_context(
            graph,
            clip_id=7,
        )

        self.assertEqual(tokens, 0)
        self.assertEqual(result, "No temporal information found around clip 7.")


class RawActionEvidenceTest(unittest.TestCase):
    def test_returns_pre_triple_behaviors_and_dialogue(self):
        memory = {
            "7": {
                "characters_behavior": [
                    "<Tom> presses the print button six times."
                ],
                "conversation": [["<Tom>", "I printed six posters."]],
            },
            "8": {
                "characters_behavior": ["<Tom> picks up the posters."],
                "conversation": [],
            },
        }

        result, tokens = execute_search_action_evidence(
            "video", clip_id=7, window=1, episodic_memory=memory
        )

        self.assertEqual(tokens, 0)
        self.assertIn("presses the print button six times", result)
        self.assertIn("I printed six posters", result)
        self.assertIn("picks up the posters", result)


if __name__ == "__main__":
    unittest.main()
