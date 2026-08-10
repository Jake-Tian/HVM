import unittest

from scripts.graph.rebuild_graph_from_memorization import parse_appearance_records


class AppearanceParsingTest(unittest.TestCase):
    def test_parses_saved_pydantic_repr(self):
        raw = (
            "[Appearance(name='<Alice>', "
            "appearance=\"female, women's blue jacket\")]"
        )
        self.assertEqual(parse_appearance_records(raw), [{
            "name": "<Alice>",
            "appearance": "female, women's blue jacket",
        }])

    def test_rejects_executable_expression(self):
        with self.assertRaises(ValueError):
            parse_appearance_records("[run_command('unsafe')]")


if __name__ == "__main__":
    unittest.main()
