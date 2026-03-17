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
from retrieval.query_policy import build_market_query_policy
from schemas.state import ReportState


# Market Agent: gather industry background with local RAG first.
MARKET_BLUEPRINT = create_agent_blueprint(
    name="market_agent",
    prompt_name="market.md",
    tools=["local_rag", "balanced_web_search"],
)


def market_node(state: ReportState) -> dict:
    """Run the two-stage retrieval policy for market research."""
    settings = load_settings()
    query_policy = build_market_query_policy()
    local_rag = LocalRAGRetriever.from_settings(settings)
    web_search = BalancedWebSearchClient.from_settings(settings)
    retrieval_execution = run_two_stage_retrieval(
        rag_retriever=local_rag,
        web_search_client=web_search,
        query_policy=query_policy,
        company_scope="MARKET",
        max_results_per_query=settings.google_news_max_results_per_query,
    )
    artifacts = build_retrieval_artifacts(
        merged_results=retrieval_execution.merged_results,
        company_scope="MARKET",
    )
    summary = summarize_retrieval(
        agent_name="market",
        local_results=retrieval_execution.local_results,
        merged_results=retrieval_execution.merged_results,
        used_web_search=retrieval_execution.used_web_search,
    )

    return {
        "documents": artifacts.documents,
        "evidence": artifacts.evidence,
        "market": {
            **state["market"],
            "document_ids": artifacts.document_ids,
            "evidence_ids": artifacts.evidence_ids,
            "synthesized_summary": summary,
        },
    }
