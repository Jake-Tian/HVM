import unittest
from unittest.mock import patch

from classes.output_structure import ExtractedTriple, TripleExtraction
from process_full_video import extract_behavior_triples


class TripleExtractionStructureTest(unittest.TestCase):
    def test_structured_response_is_converted_for_graph_insertion(self):
        parsed = TripleExtraction(triples=[
            ExtractedTriple(
                source="<robot>",
                content="puts",
                target="red pen",
            ),
            ExtractedTriple(
                source="red pen",
                content="is in",
                target="pen holder on the right side of the desk",
            ),
        ])

        with patch(
            "process_full_video.generate_text_response",
            return_value=(parsed, {"total": 12}),
        ) as generate:
            triples, tokens = extract_behavior_triples([
                "<robot> puts the red pen in the pen holder on the right side of the desk."
            ])

        self.assertEqual(triples, [
            ["<robot>", "puts", "red pen"],
            ["red pen", "is in", "pen holder on the right side of the desk"],
        ])
        self.assertEqual(tokens, {"total": 12})
        self.assertIs(generate.call_args.kwargs["text_format"], TripleExtraction)
        self.assertNotIn("reasoning_effort", generate.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
