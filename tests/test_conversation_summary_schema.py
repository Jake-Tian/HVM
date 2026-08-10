import unittest

from classes.output_structure import ConversationSummary


class ConversationSummarySchemaTest(unittest.TestCase):
    def test_nested_arrays_have_items_schemas(self):
        schema = ConversationSummary.model_json_schema()

        for field in ("character_attributes", "characters_relationships"):
            item_schema = schema["properties"][field]["items"]
            self.assertEqual(item_schema["type"], "array")
            self.assertIn("items", item_schema)
            self.assertNotIn("prefixItems", item_schema)
