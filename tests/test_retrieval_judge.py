from __future__ import annotations

import unittest

from retrieval.judge import _format_results_for_judge


class RetrievalJudgeFormattingTests(unittest.TestCase):
    def test_judge_snapshot_keeps_numeric_context_and_metadata(self) -> None:
        results = [
            {
                "title": "LGES ESS backlog expands",
                "source_name": "SourceA",
                "published_at": "2026-03-18",
                "stance": "positive",
                "query": "LGES ESS backlog 128GWh",
                "topic_tags": ["strategy", "expansion"],
                "article_excerpt": (
                    "Background context extends the sentence before the hard numbers appear. " * 4
                    + "The report says backlog reached 128GWh and operating margin improved to 8.4% in Q4."
                ),
                "page_or_chunk": "chunk-7",
                "relevance_score": 0.91234,
            }
        ]

        formatted = _format_results_for_judge(results, limit=5)

        self.assertIn("128GWh", formatted)
        self.assertIn("8.4% in Q4", formatted)
        self.assertIn("locator: chunk-7", formatted)
        self.assertIn("relevance_score: 0.912", formatted)


if __name__ == "__main__":
    unittest.main()
