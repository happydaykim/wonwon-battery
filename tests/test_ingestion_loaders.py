from __future__ import annotations

import unittest

from ingestion.loaders import build_combined_text, flatten_markdown_table


class LoaderTableFlatteningTests(unittest.TestCase):
    def test_flatten_markdown_table_returns_rowwise_text(self) -> None:
        table_lines = [
            "| Company | Market Share | Growth |",
            "| --- | --- | --- |",
            "| LGES | 20% | 8% |",
            "| CATL | 37% | 12% |",
        ]

        flattened = flatten_markdown_table(table_lines, title="Battery share")

        self.assertEqual(
            [
                "Battery share | row 1 | Company: LGES; Market Share: 20%; Growth: 8%",
                "Battery share | row 2 | Company: CATL; Market Share: 37%; Growth: 12%",
            ],
            flattened,
        )

    def test_build_combined_text_appends_flattened_table_section(self) -> None:
        source_text = """
Battery market table
| Company | Market Share |
| --- | --- |
| LGES | 20% |
""".strip()

        combined = build_combined_text(source_text, None)

        self.assertIn("[TABLE FLATTENED]", combined)
        self.assertIn("Battery market table | row 1 | Company: LGES; Market Share: 20%", combined)


if __name__ == "__main__":
    unittest.main()
