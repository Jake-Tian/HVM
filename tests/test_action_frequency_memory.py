import unittest

from reasoning_variants.three_route.frequency_memory import (
    ActionEvent,
    ActionFrequencyMemory,
    format_action_frequency_memory,
    update_action_frequency_memory,
)


class ActionFrequencyMemoryTest(unittest.TestCase):
    def test_formats_compact_event_ledger(self):
        memory = {
            "counting_unit": "one completed weighing episode",
            "events": [
                {
                    "event_id": "E1",
                    "clip_ids": [47, 48],
                    "description": "Anna uses the scale",
                    "status": "confirmed",
                    "occurrence_count": 2,
                    "merged_into": None,
                },
                {
                    "event_id": "E2",
                    "clip_ids": [47],
                    "description": "The scale is placed and adjusted",
                    "status": "merged",
                    "merged_into": "E1",
                },
            ],
            "total_confirmed": 2,
        }

        formatted = format_action_frequency_memory(memory)

        self.assertIn("Counting unit: one completed weighing episode", formatted)
        self.assertIn("E1 | clips 47-48 | confirmed x2", formatted)
        self.assertIn("E2 | clip 47 | merged into E1", formatted)
        self.assertIn("Total confirmed: 2", formatted)

    def test_update_recomputes_total_from_confirmed_events(self):
        captured = {}

        def fake_generate(prompt, schema):
            captured["prompt"] = prompt
            self.assertIs(schema, ActionFrequencyMemory)
            return (
                ActionFrequencyMemory(
                    counting_unit="one completed weighing episode",
                    events=[
                        ActionEvent(
                            event_id="E1",
                            clip_ids=[47, 48],
                            description="Anna uses the scale",
                            status="confirmed",
                            occurrence_count=2,
                        ),
                        ActionEvent(
                            event_id="E2",
                            clip_ids=[47],
                            description="Scale setup is part of E1",
                            status="merged",
                            merged_into="E1",
                        ),
                    ],
                    total_confirmed=99,
                ),
                {"total": 123},
            )

        memory, usage = update_action_frequency_memory(
            question="How many times was the scale used?",
            current_memory={},
            new_evidence="[47] robot places scale. [48] Anna uses scale.",
            generate=fake_generate,
        )

        self.assertEqual(memory["total_confirmed"], 2)
        self.assertEqual(usage, {"total": 123})
        self.assertIn("preparatory sub-actions", captured["prompt"])
        self.assertIn("adjacent clips", captured["prompt"])
        self.assertIn("reset or a new completed episode", captured["prompt"])
        self.assertIn("occurrence_count", captured["prompt"])

    def test_legacy_events_default_to_one_occurrence(self):
        formatted = format_action_frequency_memory({
            "counting_unit": "one surprise reaction",
            "events": [{
                "event_id": "E1",
                "clip_ids": [25],
                "description": "Betty is surprised",
                "status": "confirmed",
            }],
        })

        self.assertIn("confirmed x1", formatted)
        self.assertIn("Total confirmed: 1", formatted)
