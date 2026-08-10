import unittest
from collections import defaultdict
from types import SimpleNamespace

from reasoning_variants.three_route.object_event_search import (
    build_object_event_report,
    search_object_events,
)


def edge(edge_id, clip_id, source, content, target, scene="room"):
    return SimpleNamespace(
        id=edge_id,
        clip_id=clip_id,
        source=source,
        content=content,
        target=target,
        scene=scene,
    )


def graph_with(objects, edges):
    graph = SimpleNamespace(
        objects={name: SimpleNamespace(embedding=None) for name in objects},
        edges={item.id: item for item in edges},
        adjacency_list_out=defaultdict(list),
        adjacency_list_in=defaultdict(list),
        conversations={},
    )
    for item in edges:
        graph.adjacency_list_out[item.source].append(item.id)
        graph.adjacency_list_in[item.target].append(item.id)
    return graph


class ObjectEventSearchTest(unittest.TestCase):
    def test_builds_source_destination_and_current_timeline(self):
        graph = graph_with(
            ["black bag", "blue stand", "bed", "coat rack", "drawer"],
            [
                edge(1, 31, "black bag", "hangs on", "blue stand"),
                edge(2, 32, "<Anna>", "takes", "black bag"),
                edge(3, 32, "<Anna>", "puts", "black bag"),
                edge(4, 32, "black bag", "is on", "bed"),
                edge(5, 32, "<robot>", "takes", "black bag"),
                edge(6, 32, "<robot>", "places", "black bag"),
                edge(7, 32, "black bag", "is on", "coat rack"),
                edge(8, 54, "black bag", "is in", "drawer"),
            ],
        )

        current = build_object_event_report(graph, "black bag", "current")
        source = build_object_event_report(graph, "black bag", "source")
        destination = build_object_event_report(
            graph, "black bag", "destination"
        )

        self.assertEqual(
            current["answer_candidates"][0]["destination_location"],
            "in drawer",
        )
        robot_pickup = [
            item for item in source["answer_candidates"]
            if item["actor"] == "<robot>"
        ][0]
        self.assertEqual(robot_pickup["source_location"], "on bed")
        self.assertEqual(
            destination["answer_candidates"][-1]["destination_location"],
            "on coat rack",
        )

    def test_expands_container_location_hierarchy(self):
        graph = graph_with(
            ["pen", "pen holder", "desk"],
            [
                edge(1, 6, "pen holder", "is on", "desk"),
                edge(2, 33, "pen", "is in", "pen holder"),
            ],
        )

        report = build_object_event_report(graph, "pen", "current")

        self.assertEqual(
            report["answer_candidates"][0]["destination_location"],
            "in pen holder on desk",
        )

    def test_uses_lexical_canonical_name_without_embedding_call(self):
        graph = graph_with(["glass cup", "cupboard"], [])

        report = build_object_event_report(
            graph,
            "Alan's glass cup",
            "history",
            embedding_fn=lambda _: self.fail("embedding should not be called"),
        )

        self.assertEqual(report["canonical_object"], "glass cup")
        self.assertEqual(report["match_type"], "lexical")

    def test_formats_compact_report(self):
        graph = graph_with(
            ["blue folder", "shelf"],
            [edge(1, 40, "blue folder", "is on", "shelf")],
        )
        graph.conversations = {
            1: SimpleNamespace(
                messages=[
                    [
                        "<Tom>",
                        "Put the blue folder back on the shelf.",
                        40,
                        None,
                    ]
                ]
            )
        }

        text = search_object_events(graph, "blue folder", "current")

        self.assertIn("Canonical object: blue folder (exact)", text)
        self.assertIn("[clip 40] on shelf", text)
        self.assertIn(
            "[clip 40] <Tom>: Put the blue folder back on the shelf.",
            text,
        )
        self.assertIn("Suggested rewatch clips: None", text)

    def test_rejects_unknown_intent(self):
        graph = graph_with(["pen"], [])
        with self.assertRaises(ValueError):
            build_object_event_report(graph, "pen", "latest")


if __name__ == "__main__":
    unittest.main()
