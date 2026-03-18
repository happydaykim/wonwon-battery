from __future__ import annotations

from agents.base import create_agent_blueprint
from config.settings import load_settings
from retrieval.article_fetcher import ArticleContentFetcher
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


# CATL Agent: collect CATL strategy evidence and risk points.
CATL_BLUEPRINT = create_agent_blueprint(
    name="catl_agent",
    prompt_name="catl.md",
    tools=["local_rag", "balanced_web_search"],
)

logger = get_logger(__name__)


def catl_node(state: ReportState) -> dict:
    """Run the two-stage retrieval policy for CATL research."""
    settings = load_settings()
    query_policy = build_company_query_policy("CATL")
    local_rag = LocalRAGRetriever.from_settings(settings)
    web_search = BalancedWebSearchClient.from_settings(settings)
    article_fetcher = ArticleContentFetcher.from_settings(settings)
    retrieval_execution = run_two_stage_retrieval(
        rag_retriever=local_rag,
        web_search_client=web_search,
        article_fetcher=article_fetcher,
        query_policy=query_policy,
        company_scope="CATL",
        max_results_per_query=settings.google_news_max_results_per_query,
        article_fetch_max_documents=settings.article_fetch_max_documents,
        document_search_max_retries=settings.document_search_max_retries,
        web_search_max_retries=settings.web_search_max_retries,
        max_refinement_rounds=settings.retrieval_refinement_max_rounds,
        max_new_queries_per_bucket=settings.retrieval_refinement_max_queries_per_bucket,
    )
    artifacts = build_retrieval_artifacts(
        merged_results=retrieval_execution.merged_results,
        company_scope="CATL",
    )
    summary = summarize_retrieval(
        company_scope="CATL",
        agent_name="catl",
        local_results=retrieval_execution.local_results,
        merged_results=retrieval_execution.merged_results,
        used_web_search=retrieval_execution.used_web_search,
        final_assessment=retrieval_execution.final_assessment,
        query_history=retrieval_execution.query_history,
        refinement_rounds=retrieval_execution.refinement_rounds,
    )
    logger.info(
        "[CATL] documents=%d, evidence=%d, final_sufficient=%s, gaps=%s, preview_titles=%s",
        len(artifacts.document_ids),
        len(artifacts.evidence_ids),
        retrieval_execution.final_assessment.sufficient,
        retrieval_execution.final_assessment.gaps,
        _preview_titles(retrieval_execution.merged_results),
    )

    return {
        "documents": artifacts.documents,
        "evidence": artifacts.evidence,
        "companies": {
            "CATL": {
                **state["companies"]["CATL"],
                "document_ids": artifacts.document_ids,
                "evidence_ids": artifacts.evidence_ids,
                "synthesized_summary": summary,
                "retrieval_sufficient": retrieval_execution.final_assessment.sufficient,
                "retrieval_gaps": retrieval_execution.final_assessment.gaps,
                "used_web_search": retrieval_execution.used_web_search,
                "skeptic_review_required": not retrieval_execution.final_assessment.sufficient,
                "skeptic_review_completed": False,
                "query_history": retrieval_execution.query_history,
                "refinement_rounds": retrieval_execution.refinement_rounds,
            },
        },
    }


def _preview_titles(merged_results: dict[str, list[dict]]) -> list[str]:
    candidates = merged_results["positive_results"][:2] + merged_results["risk_results"][:2]
    return [item.get("title", "Untitled") for item in candidates]
