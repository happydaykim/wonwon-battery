from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.skeptic import skeptic_node
from app import build_initial_state


class _FakeWebSearchClient:
    def __init__(self, results: dict[str, list[dict[str, object]]]) -> None:
        self._results = results

    def search(
        self,
        *,
        positive_queries: list[str],
        risk_queries: list[str],
        max_results_per_query: int = 3,
    ) -> dict[str, list[dict[str, object]]]:
        _ = (positive_queries, risk_queries, max_results_per_query)
        return self._results


class SkepticAgentTests(unittest.TestCase):
    def test_skeptic_adds_counter_evidence_and_recomputes_gaps(self) -> None:
        state = _build_lges_state()
        fake_results = {
            "positive_results": [],
            "risk_results": [
                {
                    "title": "LGES profitability pressure",
                    "source": "RiskSource",
                    "link": "https://example.com/lges-risk",
                    "published_at": "2026-03-18",
                    "query": "LGES 수익성 리스크",
                    "snippet": "risk item",
                    "stance": "risk",
                    "topic_tags": ["risk"],
                },
                {
                    "title": "LGES profitability pressure",
                    "source": "RiskSource",
                    "link": "https://example.com/lges-risk",
                    "published_at": "2026-03-18",
                    "query": "LGES 수익성 리스크",
                    "snippet": "duplicate risk item",
                    "stance": "risk",
                    "topic_tags": ["risk"],
                },
            ],
        }

        with patch(
            "agents.skeptic.BalancedWebSearchClient.from_settings",
            return_value=_FakeWebSearchClient(fake_results),
        ):
            result = skeptic_node(state)

        company_state = result["companies"]["LGES"]
        self.assertTrue(company_state["skeptic_review_completed"])
        self.assertEqual(1, len(company_state["counter_evidence_ids"]))
        self.assertEqual(2, len(company_state["evidence_ids"]))
        self.assertFalse(any(gap.startswith("stance_balance:") for gap in company_state["retrieval_gaps"]))
        self.assertTrue(any(gap.startswith("evidence_count:") for gap in company_state["retrieval_gaps"]))
        self.assertEqual(["compare", "write", "validate"], result["plan"])

    def test_skeptic_keeps_gap_when_no_counter_evidence_is_found(self) -> None:
        state = _build_lges_state()
        fake_results = {
            "positive_results": [],
            "risk_results": [],
        }

        with patch(
            "agents.skeptic.BalancedWebSearchClient.from_settings",
            return_value=_FakeWebSearchClient(fake_results),
        ):
            result = skeptic_node(state)

        company_state = result["companies"]["LGES"]
        self.assertTrue(company_state["skeptic_review_completed"])
        self.assertEqual([], company_state["counter_evidence_ids"])
        self.assertTrue(
            any(
                gap.startswith("skeptic_counter_evidence:")
                for gap in company_state["retrieval_gaps"]
            )
        )
        self.assertEqual("compare", result["runtime"]["current_phase"])


def _build_lges_state() -> dict:
    state = build_initial_state("query")
    state["plan"] = ["skeptic_lges", "compare", "write", "validate"]
    state["companies"]["LGES"]["document_ids"] = ["doc_existing"]
    state["companies"]["LGES"]["evidence_ids"] = ["evidence_doc_existing"]
    state["companies"]["LGES"]["counter_evidence_ids"] = []
    state["companies"]["LGES"]["synthesized_summary"] = "lges summary"
    state["companies"]["LGES"]["retrieval_sufficient"] = False
    state["companies"]["LGES"]["retrieval_gaps"] = [
        "stance_balance: missing risk evidence."
    ]
    state["companies"]["LGES"]["skeptic_review_required"] = True
    state["documents"]["doc_existing"] = {
        "doc_id": "doc_existing",
        "title": "LGES expansion strategy",
        "source_name": "PositiveSource",
        "source_url": "https://example.com/lges-positive",
        "published_at": "2026-03-18",
        "doc_type": "news",
        "company_scope": "LGES",
        "stance": "positive",
    }
    state["evidence"]["evidence_doc_existing"] = {
        "evidence_id": "evidence_doc_existing",
        "doc_id": "doc_existing",
        "topic": "LGES 배터리 전략 ESS 확장",
        "claim": "LGES expansion strategy",
        "excerpt": None,
        "page_or_chunk": None,
        "relevance_score": None,
        "used_for": "lges_analysis",
    }
    return state


if __name__ == "__main__":
    unittest.main()
