import unittest
from types import SimpleNamespace

from reasoning.tools import (
    _format_frequency_candidates,
    _format_frequency_report,
    _group_adjacent_clips,
)


class FiveToolFrequencyStatsTest(unittest.TestCase):
    def test_groups_only_adjacent_candidate_clips(self):
        self.assertEqual(
            _group_adjacent_clips([11, 3, 4, 8, 9, 9]),
            [[3, 4], [8, 9], [11]],
        )

    def test_candidate_evidence_includes_raw_pre_triple_memory(self):
        edge = SimpleNamespace(
            clip_id=4,
            source="<Mary>",
            content="uses",
            target="mop",
        )
        evidence = _format_frequency_candidates(
            [(0.9, edge)],
            {
                "4": {
                    "characters_behavior": ["<Mary> mops the floor twice."],
                    "conversation": [["<Mary>", "I cleaned it twice."]],
                }
            },
        )

        self.assertIn("Graph [4]: <Mary> uses mop", evidence)
        self.assertIn("Raw [4]: <Mary> mops the floor twice.", evidence)
        self.assertIn("Dialogue [4] <Mary>: I cleaned it twice.", evidence)

    def test_report_uses_confirmed_plus_probable_as_best_count(self):
        report = {
            "counting_unit": "one completed use of the mop",
            "events": [
                {
                    "clip_ids": [4],
                    "evidence": "Mary completes one mopping episode.",
                    "occurrence_count": 1,
                    "status": "confirmed",
                },
                {
                    "clip_ids": [27],
                    "evidence": "A later independent mop use is strongly implied.",
                    "occurrence_count": 1,
                    "status": "probable",
                },
                {
                    "clip_ids": [28],
                    "evidence": "Duplicate description of clip 27.",
                    "occurrence_count": 1,
                    "status": "merged",
                },
            ],
        }

        formatted = _format_frequency_report(report)

        self.assertIn("Confirmed count: 1", formatted)
        self.assertIn("Probable additional count: 1", formatted)
        self.assertIn("Best count: 2", formatted)


if __name__ == "__main__":
    unittest.main()
