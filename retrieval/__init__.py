"""Retrieval stubs for local RAG and balanced web search."""

from retrieval.balanced_web_search import BalancedWebSearchClient
from retrieval.local_rag import LocalRAGRetriever
from retrieval.pipeline import (
    build_retrieval_artifacts,
    is_retrieval_sufficient,
    merge_retrieval_results,
    run_two_stage_retrieval,
    summarize_retrieval,
)
from retrieval.query_policy import (
    build_balanced_query_policy,
    build_company_query_policy,
    build_market_query_policy,
)

__all__ = [
    "BalancedWebSearchClient",
    "LocalRAGRetriever",
    "build_retrieval_artifacts",
    "build_balanced_query_policy",
    "build_company_query_policy",
    "build_market_query_policy",
    "is_retrieval_sufficient",
    "merge_retrieval_results",
    "run_two_stage_retrieval",
    "summarize_retrieval",
]
