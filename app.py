from __future__ import annotations

from typing import Literal

from config.settings import Settings, load_settings
from graph.builder import build_graph
from schemas.state import (
    CompanyResearchState,
    ReportState,
    SectionDraft,
    SWOTState,
    TopicResearchState,
)


def _build_default_section_drafts() -> dict[str, SectionDraft]:
    return {
        "summary": {
            "section_id": "summary",
            "title": "SUMMARY",
            "content": "",
            "evidence_ids": [],
            "status": "pending",
        },
        "market_background": {
            "section_id": "market_background",
            "title": "시장 배경",
            "content": "",
            "evidence_ids": [],
            "status": "pending",
        },
        "lges_strategy": {
            "section_id": "lges_strategy",
            "title": "LG에너지솔루션의 포트폴리오 다각화 전략과 핵심 경쟁력",
            "content": "",
            "evidence_ids": [],
            "status": "pending",
        },
        "catl_strategy": {
            "section_id": "catl_strategy",
            "title": "CATL의 포트폴리오 다각화 전략과 핵심 경쟁력",
            "content": "",
            "evidence_ids": [],
            "status": "pending",
        },
        "strategy_comparison": {
            "section_id": "strategy_comparison",
            "title": "핵심 전략 비교 분석",
            "content": "",
            "evidence_ids": [],
            "status": "pending",
        },
        "swot": {
            "section_id": "swot",
            "title": "SWOT 분석",
            "content": "",
            "evidence_ids": [],
            "status": "pending",
        },
        "implications": {
            "section_id": "implications",
            "title": "종합 시사점",
            "content": "",
            "evidence_ids": [],
            "status": "pending",
        },
        "references": {
            "section_id": "references",
            "title": "REFERENCE",
            "content": "",
            "evidence_ids": [],
            "status": "pending",
        },
    }


def _build_default_market_state() -> TopicResearchState:
    return {
        "document_ids": [],
        "evidence_ids": [],
        "synthesized_summary": None,
    }


def _build_default_company_state(
    company: Literal["LGES", "CATL"],
) -> CompanyResearchState:
    return {
        "company": company,
        "document_ids": [],
        "evidence_ids": [],
        "counter_evidence_ids": [],
        "synthesized_summary": None,
    }


def _build_default_swot_state() -> SWOTState:
    return {
        "strengths": [],
        "weaknesses": [],
        "opportunities": [],
        "threats": [],
    }


def build_initial_state(
    user_query: str,
    settings: Settings | None = None,
) -> ReportState:
    resolved = settings or load_settings()
    return {
        "user_query": user_query,
        "plan": [],
        "messages": [],
        "runtime": {
            "current_phase": "plan",
            "revision_count": 0,
            "max_revisions": resolved.report_max_revisions,
        },
        "documents": {},
        "evidence": {},
        "market": _build_default_market_state(),
        "companies": {
            "LGES": _build_default_company_state("LGES"),
            "CATL": _build_default_company_state("CATL"),
        },
        "comparison_summary": None,
        "swot": {
            "LGES": _build_default_swot_state(),
            "CATL": _build_default_swot_state(),
        },
        "section_drafts": _build_default_section_drafts(),
        "references": {},
        "validation_issues": [],
        "final_report": None,
    }


def main() -> None:
    settings = load_settings()
    example_query = (
        "전기차 캐즘 환경에서 LG에너지솔루션과 CATL의 포트폴리오 다각화 전략을 비교 분석해줘."
    )
    initial_state = build_initial_state(example_query, settings=settings)

    try:
        graph = build_graph(settings=settings)
    except RuntimeError as exc:
        print(f"[placeholder] Graph compilation skipped: {exc}")
        print(
            "[placeholder] Install requirements.txt and configure .env before invoking the graph."
        )
        print(f"[placeholder] Initial state ready for query: {initial_state['user_query']}")
        return

    print("Graph compiled successfully.")
    print(f"Initial phase: {initial_state['runtime']['current_phase']}")
    print(f"Example graph object: {type(graph).__name__}")
    print("TODO: wire an LLM provider and retrieval backends before calling graph.invoke.")
    # Example only:
    # result = graph.invoke(initial_state, config={"configurable": {"thread_id": "demo"}})


if __name__ == "__main__":
    main()
