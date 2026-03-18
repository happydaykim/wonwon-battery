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

    def test_build_references_keeps_unknown_date_when_missing(self) -> None:
        state = build_initial_state("query")
        state["documents"] = {
            "doc_missing_date": {
                "doc_id": "doc_missing_date",
                "title": "발행일 없는 기사",
                "source_name": "전기신문",
                "source_url": "https://www.electimes.com/news/articleView.html?idxno=365645",
                "published_at": None,
                "doc_type": "news",
                "company_scope": "MARKET",
                "stance": "positive",
            }
        }

        references = _build_references(state)

        citation_text = next(iter(references.values()))["citation_text"]
        self.assertIn("전기신문(날짜 미상)", citation_text)
        self.assertNotIn("2026-03-18", citation_text)

    def test_build_references_only_includes_docs_backed_by_used_evidence(self) -> None:
        state = build_initial_state("query")
        state["documents"] = {
            "doc_used": {
                "doc_id": "doc_used",
                "title": "실제 본문에 사용된 기사",
                "source_name": "전기신문",
                "source_url": "https://example.com/used",
                "published_at": "2026-03-18",
                "doc_type": "news",
                "company_scope": "MARKET",
                "stance": "positive",
            },
            "doc_unused": {
                "doc_id": "doc_unused",
                "title": "수집만 되고 미사용된 기사",
                "source_name": "배터리데일리",
                "source_url": "https://example.com/unused",
                "published_at": "2026-03-17",
                "doc_type": "news",
                "company_scope": "MARKET",
                "stance": "risk",
            },
        }
        state["evidence"] = {
            "evidence_used": {
                "evidence_id": "evidence_used",
                "doc_id": "doc_used",
                "topic": "배터리 시장 전략",
                "topic_tags": ["market_structure", "demand", "risk"],
                "claim": "실제 본문에 사용된 기사",
                "excerpt": "used excerpt",
                "full_text": None,
                "page_or_chunk": None,
                "relevance_score": None,
                "used_for": "market_background",
            },
            "evidence_unused": {
                "evidence_id": "evidence_unused",
                "doc_id": "doc_unused",
                "topic": "배터리 시장 전략",
                "topic_tags": ["risk"],
                "claim": "수집만 되고 미사용된 기사",
                "excerpt": "unused excerpt",
                "full_text": None,
                "page_or_chunk": None,
                "relevance_score": None,
                "used_for": "market_background",
            },
        }

        references = _build_references(
            state,
            section_evidence_map={
                "summary": ["evidence_used"],
                "market_background": ["evidence_used"],
                "lges_strategy": [],
                "catl_strategy": [],
                "strategy_comparison": [],
                "swot": [],
                "implications": [],
                "references": [],
            },
        )

        self.assertEqual(1, len(references))
        reference = next(iter(references.values()))
        self.assertEqual("doc_used", reference["doc_id"])
        self.assertEqual(["summary", "market_background"], reference["used_in_sections"])


if __name__ == "__main__":
    unittest.main()
