from __future__ import annotations

from typing import Literal
from uuid import uuid4

from config.settings import Settings, load_settings
from graph.builder import build_graph
from schemas.state import (
    CompanyResearchState,
    ReportState,
    SectionDraft,
    SWOTState,
    TopicResearchState,
)
from utils.logging import configure_langsmith
from utils.report_export import ReportArtifacts, write_report_artifacts

PLAN_STEP_LABELS = {
    "parallel_retrieval": "Parallel Retrieval Bundle (Market + LGES + CATL)",
    "skeptic_lges": "Skeptic Review for LGES",
    "skeptic_catl": "Skeptic Review for CATL",
    "compare": "Compare / SWOT",
    "write": "Writer",
    "validate": "Validator",
}


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
        "retrieval_sufficient": False,
        "retrieval_gaps": [],
        "used_web_search": False,
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
        "retrieval_sufficient": False,
        "retrieval_gaps": [],
        "used_web_search": False,
        "skeptic_review_required": False,
        "skeptic_review_completed": False,
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
            "termination_reason": None,
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


def _write_final_report_artifacts(
    result: ReportState,
    *,
    settings: Settings,
    thread_id: str,
) -> ReportArtifacts | None:
    """Persist the final report as HTML/PDF artifacts when available."""
    return write_report_artifacts(
        result,
        settings=settings,
        thread_id=thread_id,
    )


def main() -> None:
    settings = load_settings()
    configure_langsmith(
        settings.langsmith_project,
        enabled=settings.langsmith_enabled,
    )
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

    thread_id = f"battery-strategy-{uuid4().hex[:8]}"
    result = graph.invoke(
        initial_state,
        config={
            "recursion_limit": 30,
            "configurable": {"thread_id": thread_id},
        },
    )
    report_artifacts = _write_final_report_artifacts(
        result,
        settings=settings,
        thread_id=thread_id,
    )

    print("Graph compiled successfully.")
    print(f"Initial phase: {initial_state['runtime']['current_phase']}")
    print(f"Final phase: {result['runtime']['current_phase']}")
    print(f"Revision count: {result['runtime']['revision_count']}")
    print(f"Termination reason: {result['runtime']['termination_reason']}")
    print(f"Remaining plan steps: {len(result['plan'])}")
    print(f"Validation issues: {len(result['validation_issues'])}")
    print(f"Messages collected: {len(result['messages'])}")
    if report_artifacts is not None:
        print(f"HTML report saved to: {report_artifacts.html_path}")
        if report_artifacts.pdf_path is not None:
            print(f"PDF report saved to: {report_artifacts.pdf_path}")
        else:
            print(
                "PDF report was not saved because export failed: "
                f"{report_artifacts.pdf_error}"
            )
    else:
        print("Report artifacts were not saved because final_report is empty.")
    planner_message = next(
        (
            message["content"]
            for message in result["messages"]
            if message.get("name") == "planner_node"
        ),
        None,
    )
    if planner_message:
        print("Planner summary:")
        print(planner_message)
    if result["plan"]:
        print("Remaining workflow queue:")
        for index, step in enumerate(result["plan"], start=1):
            print(f"{index}. {PLAN_STEP_LABELS.get(step, step)}")
    if result["messages"]:
        print("Last message:")
        print(result["messages"][-1]["content"])


if __name__ == "__main__":
    main()
