import unittest
from unittest.mock import patch

from classes.edge_class import Edge
from classes.hetero_graph import HeteroGraph
from utils.abstraction_config import AbstractionConfig


class IncrementalAbstractionTest(unittest.TestCase):
    def setUp(self):
        Edge._id_counter = 0

    @staticmethod
    def _add_edge(graph, clip_id, source, target, content):
        for node in (source, target):
            if node is not None and not node.startswith("<"):
                graph.objects.setdefault(node, object())
        edge = Edge(
            clip_id=clip_id,
            source=source,
            target=target,
            content=content,
            scene="test",
        )
        graph.add_edge(edge)
        return edge.id

    def _attribute_graph(self):
        graph = HeteroGraph()
        character = graph.add_character("alice")
        for clip_id in (1, 2, 3):
            for index in (1, 2):
                self._add_edge(
                    graph,
                    clip_id,
                    character,
                    f"object_{clip_id}_{index}",
                    f"action_{clip_id}_{index}",
                )
        return graph, character

    @staticmethod
    def _record_attribute_calls(graph):
        calls = []

        def record(character_name, incremental=False, through_clip=None):
            visible = graph._low_level_edges_of(character_name, through_clip)
            new_ids = visible - graph._char_consumed_edges[character_name]
            if not new_ids:
                return 0
            calls.append(
                (
                    through_clip,
                    [graph.edges[edge_id].clip_id for edge_id in sorted(new_ids)],
                )
            )
            graph._char_consumed_edges[character_name].update(new_ids)
            graph._char_last_degree[character_name] = len(visible)
            return 0

        graph.character_attributes = record
        graph.character_relationships = lambda *args, **kwargs: 0
        return calls

    def test_incremental_mode_cannot_see_future_clips(self):
        graph, _ = self._attribute_graph()
        calls = self._record_attribute_calls(graph)

        graph.run_abstraction(
            AbstractionConfig(
                incremental_enabled=True,
                interval_node=4,
                interval_pair=999,
                final_lower_bound_node=1,
                final_lower_bound_pair=999,
            )
        )

        self.assertEqual(calls, [(2, [1, 1, 2, 2]), (None, [3, 3])])

    def test_final_only_mode_summarizes_all_evidence_once(self):
        graph, _ = self._attribute_graph()
        calls = self._record_attribute_calls(graph)

        graph.run_abstraction(
            AbstractionConfig(
                incremental_enabled=False,
                interval_node=4,
                interval_pair=999,
                final_lower_bound_node=1,
                final_lower_bound_pair=999,
            )
        )

        self.assertEqual(calls, [(None, [1, 1, 2, 2, 3, 3])])

    def test_incremental_prompt_and_bookkeeping_use_only_visible_edges(self):
        graph, character = self._attribute_graph()
        graph._reset_abstraction_state()

        with patch(
            "classes.hetero_graph.generate_text_response",
            return_value=("{}", 0),
        ) as generate:
            graph.character_attributes(
                character,
                incremental=True,
                through_clip=2,
            )
            first_prompt = generate.call_args.args[0]

            graph.character_attributes(character, incremental=True)
            final_prompt = generate.call_args.args[0]

        self.assertIn("action_2_2", first_prompt)
        self.assertNotIn("action_3_1", first_prompt)
        self.assertNotIn("action_1_1", final_prompt)
        self.assertIn("action_3_1", final_prompt)
        self.assertEqual(
            graph._char_consumed_edges[character],
            graph._low_level_edges_of(character),
        )

    def test_object_mediated_relationship_respects_clip_cutoff(self):
        graph = HeteroGraph()
        alice = graph.add_character("alice")
        bob = graph.add_character("bob")
        first = self._add_edge(graph, 1, alice, "shared_object", "touches")
        second = self._add_edge(graph, 3, "shared_object", bob, "handed to")

        self.assertEqual(
            graph._shared_low_level_edge_ids(alice, bob, through_clip=1),
            set(),
        )
        self.assertEqual(
            graph._shared_low_level_edge_ids(alice, bob, through_clip=3),
            {first, second},
        )


if __name__ == "__main__":
    unittest.main()
