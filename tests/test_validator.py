from __future__ import annotations

import unittest

from agents.validator import validator_node
from app import build_initial_state


class ValidatorTests(unittest.TestCase):
    def test_validator_passes_when_no_issues_remain(self) -> None:
        state = _build_valid_state()

        result = validator_node(state)

        self.assertEqual("done", result["runtime"]["current_phase"])
        self.assertEqual("validated", result["runtime"]["termination_reason"])
        self.assertEqual([], result["validation_issues"])

    def test_validator_retries_when_retryable_issues_exist_and_budget_remains(self) -> None:
        state = _build_valid_state()
        state["final_report"] = ""

        result = validator_node(state)

        self.assertEqual("write", result["runtime"]["current_phase"])
        self.assertIsNone(result["runtime"]["termination_reason"])
        self.assertEqual(["write", "validate"], result["plan"])
        self.assertEqual(1, result["runtime"]["revision_count"])
        self.assertTrue(any(issue["retryable"] for issue in result["validation_issues"]))

    def test_validator_finishes_with_gaps_when_only_non_retryable_issues_remain(self) -> None:
        state = _build_valid_state()
        state["market"]["retrieval_sufficient"] = False
        state["market"]["retrieval_gaps"] = ["market gap"]

        result = validator_node(state)

        self.assertEqual("done", result["runtime"]["current_phase"])
        self.assertEqual("done_with_gaps", result["runtime"]["termination_reason"])
        self.assertTrue(all(not issue["retryable"] for issue in result["validation_issues"]))

    def test_validator_stops_when_revision_budget_is_exhausted(self) -> None:
        state = _build_valid_state()
        state["runtime"]["revision_count"] = state["runtime"]["max_revisions"]
        state["final_report"] = ""

        result = validator_node(state)

        self.assertEqual("done", result["runtime"]["current_phase"])
        self.assertEqual("max_revisions_reached", result["runtime"]["termination_reason"])
        self.assertTrue(any(issue["retryable"] for issue in result["validation_issues"]))


def _build_valid_state() -> dict:
    state = build_initial_state("query")
    state["plan"] = ["validate"]
    state["runtime"]["max_revisions"] = 2
    state["market"]["retrieval_sufficient"] = True
    state["market"]["synthesized_summary"] = "market"
    state["companies"]["LGES"]["retrieval_sufficient"] = True
    state["companies"]["LGES"]["synthesized_summary"] = "lges"
    state["companies"]["CATL"]["retrieval_sufficient"] = True
    state["companies"]["CATL"]["synthesized_summary"] = "catl"

    for section in state["section_drafts"].values():
        section["content"] = "draft"
        section["status"] = "drafted"

    state["final_report"] = "compiled report"
    state["references"] = {
        "ref_doc_1": {
            "ref_id": "ref_doc_1",
            "doc_id": "doc_1",
            "citation_text": "- citation",
            "reference_type": "webpage",
            "used_in_sections": ["summary"],
        }
    }
    state["documents"]["doc_1"] = {
        "doc_id": "doc_1",
        "title": "Doc 1",
        "source_name": "Source",
        "source_url": "https://example.com/doc-1",
        "published_at": "2026-03-18",
        "doc_type": "news",
        "company_scope": "MARKET",
        "stance": "positive",
    }
    return state


if __name__ == "__main__":
    unittest.main()
