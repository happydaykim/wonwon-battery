from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app import _write_final_report_markdown, build_initial_state
from config.settings import load_settings


class AppOutputTests(unittest.TestCase):
    def test_write_final_report_markdown_persists_markdown_file(self) -> None:
        state = build_initial_state("query")
        state["final_report"] = "## SUMMARY\nreport body"

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(load_settings(), outputs_dir=Path(temp_dir))

            report_path = _write_final_report_markdown(
                state,
                settings=settings,
                thread_id="battery-strategy-test",
            )

            self.assertIsNotNone(report_path)
            self.assertTrue(report_path.exists())
            self.assertEqual("## SUMMARY\nreport body\n", report_path.read_text(encoding="utf-8"))
            self.assertTrue(report_path.name.startswith("battery-strategy-test_"))
            self.assertEqual(".md", report_path.suffix)

    def test_write_final_report_markdown_skips_empty_report(self) -> None:
        state = build_initial_state("query")

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(load_settings(), outputs_dir=Path(temp_dir))

            report_path = _write_final_report_markdown(
                state,
                settings=settings,
                thread_id="battery-strategy-test",
            )

            self.assertIsNone(report_path)
            self.assertEqual([], list(Path(temp_dir).iterdir()))


if __name__ == "__main__":
    unittest.main()
