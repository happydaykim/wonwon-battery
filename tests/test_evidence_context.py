from __future__ import annotations

import unittest

from app import build_initial_state
from utils.evidence_context import (
    format_evidence_packet,
    format_quantitative_evidence_packet,
)


class EvidenceContextTests(unittest.TestCase):
    def test_format_evidence_packet_includes_balanced_stance_context(self) -> None:
        state = build_initial_state("query")
        state["documents"] = {
            "doc_pos_1": {
                "doc_id": "doc_pos_1",
                "title": "LGES expands ESS business",
                "source_name": "SourceA",
                "source_url": "https://example.com/pos1",
                "published_at": "2026-03-18",
                "doc_type": "news",
                "company_scope": "LGES",
                "stance": "positive",
            },
            "doc_pos_2": {
                "doc_id": "doc_pos_2",
                "title": "LGES explores robotics batteries",
                "source_name": "SourceA",
                "source_url": "https://example.com/pos2",
                "published_at": "2026-03-17",
                "doc_type": "news",
                "company_scope": "LGES",
                "stance": "positive",
            },
            "doc_risk_1": {
                "doc_id": "doc_risk_1",
                "title": "LGES profitability pressure grows",
                "source_name": "SourceB",
                "source_url": "https://example.com/risk1",
                "published_at": "2026-03-18",
                "doc_type": "news",
                "company_scope": "LGES",
                "stance": "risk",
            },
        }
        state["evidence"] = {
            "evidence_doc_pos_1": {
                "evidence_id": "evidence_doc_pos_1",
                "doc_id": "doc_pos_1",
                "topic": "LG에너지솔루션 포트폴리오 다각화",
                "topic_tags": ["strategy", "expansion"],
                "claim": "LGES expands ESS business",
                "excerpt": "LGES is pushing ESS and adjacent applications.",
                "full_text": "LGES is pushing ESS and adjacent applications with a stated focus on long-cycle supply agreements.",
                "page_or_chunk": "p.12",
                "relevance_score": 0.876,
                "used_for": "lges_analysis",
            },
            "evidence_doc_pos_2": {
                "evidence_id": "evidence_doc_pos_2",
                "doc_id": "doc_pos_2",
                "topic": "LG에너지솔루션 ESS HEV 로봇 확장",
                "topic_tags": ["expansion"],
                "claim": "LGES explores robotics batteries",
                "excerpt": "The company is testing robotics-linked battery demand.",
                "full_text": None,
                "page_or_chunk": None,
                "relevance_score": None,
                "used_for": "lges_analysis",
            },
            "evidence_doc_risk_1": {
                "evidence_id": "evidence_doc_risk_1",
                "doc_id": "doc_risk_1",
                "topic": "LG에너지솔루션 수익성 리스크",
                "topic_tags": ["risk"],
                "claim": "LGES profitability pressure grows",
                "excerpt": "Margins remain under pressure amid slowing EV demand.",
                "full_text": None,
                "page_or_chunk": None,
                "relevance_score": None,
                "used_for": "lges_analysis",
            },
        }

        packet = format_evidence_packet(
            state,
            [
                "evidence_doc_pos_1",
                "evidence_doc_pos_2",
                "evidence_doc_risk_1",
            ],
            limit=2,
        )

        self.assertIn("[positive | SourceA | 2026-03-18]", packet)
        self.assertIn("[risk | SourceB | 2026-03-18]", packet)
        self.assertIn("query: LG에너지솔루션 수익성 리스크", packet)
        self.assertIn("tags: strategy, expansion", packet)
        self.assertIn("locator: p.12", packet)
        self.assertIn("relevance_score: 0.876", packet)

    def test_format_evidence_packet_preserves_long_numeric_context(self) -> None:
        state = build_initial_state("query")
        state["documents"] = {
            "doc_long": {
                "doc_id": "doc_long",
                "title": "LGES long-form analysis on diversification",
                "source_name": "SourceLong",
                "source_url": "https://example.com/long",
                "published_at": "2026-03-18",
                "doc_type": "news",
                "company_scope": "LGES",
                "stance": "positive",
            }
        }
        long_excerpt = (
            "Background context extends the sentence before the hard numbers appear. " * 4
            + "The article adds that ESS backlog reached 128GWh and operating margin improved to 8.4% in Q4 due to mix shift."
        )
        state["evidence"] = {
            "evidence_doc_long": {
                "evidence_id": "evidence_doc_long",
                "doc_id": "doc_long",
                "topic": "LGES ESS backlog",
                "topic_tags": ["strategy", "expansion"],
                "claim": "LGES reports a larger ESS backlog with improved profitability.",
                "excerpt": long_excerpt,
                "full_text": None,
                "page_or_chunk": None,
                "relevance_score": None,
                "used_for": "lges_analysis",
            }
        }

        packet = format_evidence_packet(
            state,
            ["evidence_doc_long"],
            limit=1,
        )

        self.assertIn("128GWh", packet)
        self.assertIn("8.4% in Q4", packet)

    def test_format_quantitative_evidence_packet_extracts_numeric_snippets(self) -> None:
        state = build_initial_state("query")
        state["documents"] = {
            "doc_quant": {
                "doc_id": "doc_quant",
                "title": "LGES 2026 outlook",
                "source_name": "SourceA",
                "source_url": "https://example.com/quant",
                "published_at": "2026-03-18",
                "doc_type": "news",
                "company_scope": "LGES",
                "stance": "positive",
            }
        }
        state["evidence"] = {
            "evidence_doc_quant": {
                "evidence_id": "evidence_doc_quant",
                "doc_id": "doc_quant",
                "topic": "LGES ESS 전망",
                "topic_tags": ["strategy", "expansion"],
                "claim": "ESS revenue forecast raised by 55%.",
                "excerpt": "Non-Chinese global EV battery usage reached 32.7GWh, up 13.7% YoY.",
                "full_text": "Average pack price fell 18% while the secured project pipeline reached 41GWh.",
                "page_or_chunk": None,
                "relevance_score": None,
                "used_for": "lges_analysis",
            }
        }

        packet = format_quantitative_evidence_packet(
            state,
            ["evidence_doc_quant"],
            limit=3,
        )

        self.assertIn("55%", packet)
        self.assertIn("32.7GWh", packet)
        self.assertIn("13.7% YoY", packet)
        self.assertIn("41GWh", packet)


if __name__ == "__main__":
    unittest.main()
