from __future__ import annotations

import unittest

from agents.validator import validator_node
from app import build_initial_state


class ValidatorTests(unittest.TestCase):
    def test_validator_passes_when_no_issues_remain(self) -> None:
        state = _build_valid_state()

        result = validator_node(state)

        self.assertEqual("validate", result["runtime"]["current_phase"])
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

        self.assertEqual("validate", result["runtime"]["current_phase"])
        self.assertEqual("done_with_gaps", result["runtime"]["termination_reason"])
        self.assertTrue(all(not issue["retryable"] for issue in result["validation_issues"]))

    def test_validator_surfaces_retrieval_execution_failures(self) -> None:
        state = _build_valid_state()
        state["market"]["retrieval_sufficient"] = False
        state["market"]["retrieval_gaps"] = ["market gap"]
        state["market"]["retrieval_failures"] = [
            "web_search_failure: failed after 1 attempt(s): google news down"
        ]

        result = validator_node(state)

        self.assertTrue(
            any(issue["issue_id"] == "market_retrieval_failure" for issue in result["validation_issues"])
        )

    def test_validator_stops_when_revision_budget_is_exhausted(self) -> None:
        state = _build_valid_state()
        state["runtime"]["revision_count"] = state["runtime"]["max_revisions"]
        state["final_report"] = ""

        result = validator_node(state)

        self.assertEqual("validate", result["runtime"]["current_phase"])
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

    state["section_drafts"]["summary"]["content"] = (
        "전기차 캐즘이 장기화되는 가운데 LGES와 CATL은 모두 포트폴리오 다각화에 속도를 내고 있다. "
        "LGES는 신규 응용처와 북미 대응 축이, CATL은 원가·기술 우위를 활용한 확장 축이 상대적으로 부각된다. "
        "다만 기사형 웹 자료 중심이라 일부 gap은 후속 검증이 필요하다."
    )
    state["section_drafts"]["market_background"]["content"] = "\n".join(
        [
            "### II.I 전기차 캐즘과 HEV 피벗",
            "시장 서술",
            "### II.II K-배터리 업계의 포트폴리오 다각화 배경",
            "배경 서술",
            "### II.III CATL의 원가/기술 전략 변화",
            "CATL 서술",
        ]
    )
    state["section_drafts"]["lges_strategy"]["content"] = "LGES 본문"
    state["section_drafts"]["catl_strategy"]["content"] = "CATL 본문"
    state["section_drafts"]["strategy_comparison"]["content"] = "\n".join(
        [
            "### V.I 전략 방향 차이",
            "비교 본문",
            "### V.II 데이터 기반 비교표",
            "| 회사 | 전략 |",
            "| --- | --- |",
            "| LGES | 확장 |",
            "| CATL | 원가 |",
        ]
    )
    state["section_drafts"]["swot"]["content"] = "\n".join(
        [
            "### V.III SWOT 분석",
            "#### LGES",
            "- Strengths: 강점",
            "#### CATL",
            "- Strengths: 강점",
        ]
    )
    state["section_drafts"]["implications"]["content"] = "시사점 본문"
    state["section_drafts"]["references"]["content"] = "- Source(2026-03-18). *Doc 1*. Source, https://example.com/doc-1"
    for section in state["section_drafts"].values():
        section["status"] = "drafted"

    state["final_report"] = "\n".join(
        [
            "# 배터리 시장 전략 분석 보고서",
            "## I. EXECUTIVE SUMMARY",
            state["section_drafts"]["summary"]["content"],
            "## II. 시장 배경",
            state["section_drafts"]["market_background"]["content"],
            "## III. LG에너지솔루션의 포트폴리오 다각화 전략과 핵심 경쟁력",
            state["section_drafts"]["lges_strategy"]["content"],
            "## IV. CATL의 포트폴리오 다각화 전략과 핵심 경쟁력",
            state["section_drafts"]["catl_strategy"]["content"],
            "## V. 핵심 전략 비교 분석",
            state["section_drafts"]["strategy_comparison"]["content"],
            "## V.III SWOT 분석",
            state["section_drafts"]["swot"]["content"],
            "## VI. 종합 시사점",
            state["section_drafts"]["implications"]["content"],
            "## VII. REFERENCE",
            state["section_drafts"]["references"]["content"],
        ]
    )
    state["references"] = {
        "ref_doc_1": {
            "ref_id": "ref_doc_1",
            "doc_id": "doc_1",
            "citation_text": "- Source(2026-03-18). *Doc 1*. Source, https://example.com/doc-1",
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
