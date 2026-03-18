from __future__ import annotations

import unittest

from agents.writer import _build_references
from app import build_initial_state


class WriterReferenceTests(unittest.TestCase):
    def test_build_references_skips_unresolved_google_news_wrapper(self) -> None:
        state = build_initial_state("query")
        state["documents"] = {
            "doc_google": {
                "doc_id": "doc_google",
                "title": "Wrapped result - GoogleNews",
                "source_name": "GoogleNews RSS",
                "source_url": "https://news.google.com/rss/articles/example",
                "published_at": "2026-03-18",
                "doc_type": "news",
                "company_scope": "MARKET",
                "stance": "positive",
            }
        }

        references = _build_references(state)

        self.assertEqual({}, references)

    def test_build_references_uses_resolved_publisher_url(self) -> None:
        state = build_initial_state("query")
        state["documents"] = {
            "doc_publisher": {
                "doc_id": "doc_publisher",
                "title": "EV 넘어 ESS·로봇·AI로...K-배터리",
                "source_name": "전기신문",
                "source_url": "https://www.electimes.com/news/articleView.html?idxno=365645",
                "published_at": "2026-03-18",
                "doc_type": "news",
                "company_scope": "MARKET",
                "stance": "positive",
            }
        }

        references = _build_references(state)

        self.assertEqual(1, len(references))
        citation_text = next(iter(references.values()))["citation_text"]
        self.assertIn("전기신문(2026-03-18)", citation_text)
        self.assertIn("https://www.electimes.com/news/articleView.html?idxno=365645", citation_text)
        self.assertNotIn("GoogleNews", citation_text)


if __name__ == "__main__":
    unittest.main()
