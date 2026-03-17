from __future__ import annotations

from agents.base import create_agent_blueprint
from config.settings import load_settings
from retrieval.balanced_web_search import BalancedWebSearchClient
from retrieval.local_rag import LocalRAGRetriever
from retrieval.pipeline import (
    build_retrieval_artifacts,
    run_two_stage_retrieval,
    summarize_retrieval,
)
from retrieval.query_policy import build_company_query_policy
from schemas.state import ReportState
from utils.logging import get_logger


# LGES Agent: collect LGES strategy evidence and risk points.
LGES_BLUEPRINT = create_agent_blueprint(
    name="lges_agent",
    prompt_name="lges.md",
    tools=["local_rag", "balanced_web_search"],
)

logger = get_logger(__name__)


def lges_node(state: ReportState) -> dict:
    """Run the two-stage retrieval policy for LGES research."""
    settings = load_settings()
    query_policy = build_company_query_policy("LGES")
    local_rag = LocalRAGRetriever.from_settings(settings)
    web_search = BalancedWebSearchClient.from_settings(settings)
    retrieval_execution = run_two_stage_retrieval(
        rag_retriever=local_rag,
        web_search_client=web_search,
        query_policy=query_policy,
        company_scope="LGES",
        max_results_per_query=settings.google_news_max_results_per_query,
    )
    artifacts = build_retrieval_artifacts(
        merged_results=retrieval_execution.merged_results,
        company_scope="LGES",
    )
    summary = summarize_retrieval(
        agent_name="lges",
        local_results=retrieval_execution.local_results,
        merged_results=retrieval_execution.merged_results,
        used_web_search=retrieval_execution.used_web_search,
    )
    logger.info(
        "[LGES] documents=%d, evidence=%d, preview_titles=%s",
        len(artifacts.document_ids),
        len(artifacts.evidence_ids),
        _preview_titles(retrieval_execution.merged_results),
    )

    return {
        "documents": artifacts.documents,
        "evidence": artifacts.evidence,
        "companies": {
            "LGES": {
                **state["companies"]["LGES"],
                "document_ids": artifacts.document_ids,
                "evidence_ids": artifacts.evidence_ids,
                "synthesized_summary": summary,
            },
        },
    }


def _preview_titles(merged_results: dict[str, list[dict]]) -> list[str]:
    candidates = merged_results["positive_results"][:2] + merged_results["risk_results"][:2]
    return [item.get("title", "Untitled") for item in candidates]
