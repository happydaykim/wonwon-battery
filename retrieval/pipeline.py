from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Literal

from schemas.state import EvidenceItem, SourceDocument
from utils.logging import get_logger


NormalizedResult = dict[str, Any]
MergedRetrievalResults = dict[str, list[NormalizedResult]]

MIN_EVIDENCE_COUNT = 3
MIN_SOURCE_COUNT = 2

REQUIRED_TOPIC_TAGS: dict[str, set[str]] = {
    "MARKET": {"market_structure", "demand", "risk"},
    "LGES": {"strategy", "expansion", "risk"},
    "CATL": {"strategy", "expansion", "risk"},
}

USED_FOR_BY_SCOPE: dict[str, str] = {
    "MARKET": "market_background",
    "LGES": "lges_analysis",
    "CATL": "catl_analysis",
}

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalArtifacts:
    documents: dict[str, SourceDocument]
    evidence: dict[str, EvidenceItem]
    document_ids: list[str]
    evidence_ids: list[str]


@dataclass(frozen=True, slots=True)
class RetrievalExecution:
    local_results: list[NormalizedResult]
    merged_results: MergedRetrievalResults
    used_web_search: bool


def is_retrieval_sufficient(
    local_results: list[NormalizedResult],
    *,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    min_evidence_count: int = MIN_EVIDENCE_COUNT,
    min_source_count: int = MIN_SOURCE_COUNT,
) -> bool:
    """Apply the mixed sufficiency rule for phase-1 local retrieval."""
    if len(local_results) < min_evidence_count:
        return False

    sources = {
        result.get("source_name") or result.get("source") or result.get("media")
        for result in local_results
        if result.get("source_name") or result.get("source") or result.get("media")
    }
    if len(sources) < min_source_count:
        return False

    stances = {
        result.get("stance")
        for result in local_results
        if result.get("stance") in {"positive", "risk"}
    }
    if not {"positive", "risk"}.issubset(stances):
        return False

    topic_tags = set()
    for result in local_results:
        tags = result.get("topic_tags", [])
        if isinstance(tags, str):
            topic_tags.add(tags)
        elif isinstance(tags, list):
            topic_tags.update(tag for tag in tags if isinstance(tag, str))

    required_tags = REQUIRED_TOPIC_TAGS[company_scope]
    return required_tags.issubset(topic_tags)


def merge_retrieval_results(
    *,
    local_results: list[NormalizedResult],
    web_results: MergedRetrievalResults,
) -> MergedRetrievalResults:
    """Merge local and web results into stance buckets with URL-based dedupe."""
    merged: MergedRetrievalResults = {
        "positive_results": [],
        "risk_results": [],
    }
    seen_keys: set[str] = set()

    for result in local_results:
        stance = "risk" if result.get("stance") == "risk" else "positive"
        bucket = "risk_results" if stance == "risk" else "positive_results"
        _append_deduped_result(merged[bucket], result, seen_keys)

    for bucket in ("positive_results", "risk_results"):
        for result in web_results.get(bucket, []):
            _append_deduped_result(merged[bucket], result, seen_keys)

    return merged


def run_two_stage_retrieval(
    *,
    rag_retriever: Any,
    web_search_client: Any,
    query_policy: dict[str, list[str]],
    company_scope: Literal["MARKET", "LGES", "CATL"],
    max_results_per_query: int,
) -> RetrievalExecution:
    """Run local retrieval first, then fall back to web search when insufficient."""
    local_results: list[NormalizedResult] = []
    logger.info(
        "[%s] Starting retrieval: positive_queries=%d, risk_queries=%d",
        company_scope,
        len(query_policy.get("positive_queries", [])),
        len(query_policy.get("risk_queries", [])),
    )

    for stance, bucket in (("positive", "positive_queries"), ("risk", "risk_queries")):
        for query in query_policy.get(bucket, []):
            results = rag_retriever.retrieve(
                query,
                company_scope=company_scope,
                top_k=max_results_per_query,
            )
            for result in results:
                normalized_result = dict(result)
                normalized_result.setdefault("query", query)
                normalized_result.setdefault("stance", stance)
                local_results.append(normalized_result)

    retrieval_sufficient = is_retrieval_sufficient(
        local_results,
        company_scope=company_scope,
    )
    logger.info(
        "[%s] Local RAG hits=%d, sufficient=%s",
        company_scope,
        len(local_results),
        retrieval_sufficient,
    )
    web_results: MergedRetrievalResults = {
        "positive_results": [],
        "risk_results": [],
    }
    if not retrieval_sufficient:
        if not local_results:
            logger.info("[%s] Local RAG returned 0 hits. Falling back to web search.", company_scope)
        else:
            logger.info("[%s] Local RAG was insufficient. Falling back to web search.", company_scope)
        web_results = web_search_client.search(
            positive_queries=query_policy.get("positive_queries", []),
            risk_queries=query_policy.get("risk_queries", []),
            max_results_per_query=max_results_per_query,
        )
        logger.info(
            "[%s] Web search hits: positive=%d, risk=%d",
            company_scope,
            len(web_results["positive_results"]),
            len(web_results["risk_results"]),
        )
    else:
        logger.info("[%s] Local RAG was sufficient. Web search skipped.", company_scope)

    merged_results = merge_retrieval_results(
        local_results=local_results,
        web_results=web_results,
    )
    logger.info(
        "[%s] Merged retrieval results: positive=%d, risk=%d",
        company_scope,
        len(merged_results["positive_results"]),
        len(merged_results["risk_results"]),
    )
    return RetrievalExecution(
        local_results=local_results,
        merged_results=merged_results,
        used_web_search=not retrieval_sufficient,
    )


def build_retrieval_artifacts(
    *,
    merged_results: MergedRetrievalResults,
    company_scope: Literal["MARKET", "LGES", "CATL"],
) -> RetrievalArtifacts:
    """Convert normalized search results into SourceDocument/EvidenceItem entries."""
    documents: dict[str, SourceDocument] = {}
    evidence: dict[str, EvidenceItem] = {}
    document_ids: list[str] = []
    evidence_ids: list[str] = []
    used_for = USED_FOR_BY_SCOPE[company_scope]

    for bucket in ("positive_results", "risk_results"):
        for result in merged_results.get(bucket, []):
            doc_id = _build_doc_id(result)
            evidence_id = f"evidence_{doc_id}"

            documents[doc_id] = {
                "doc_id": doc_id,
                "title": result.get("title") or "Untitled news result",
                "source_name": result.get("source") or "GoogleNews",
                "source_url": result.get("link"),
                "published_at": result.get("published_at"),
                "doc_type": "news",
                "company_scope": company_scope,
                "stance": result.get("stance", "neutral"),
            }
            evidence[evidence_id] = {
                "evidence_id": evidence_id,
                "doc_id": doc_id,
                "topic": result.get("query") or "news_result",
                "claim": result.get("title") or "Untitled claim",
                "excerpt": result.get("snippet"),
                "page_or_chunk": None,
                "relevance_score": None,
                "used_for": used_for,
            }
            document_ids.append(doc_id)
            evidence_ids.append(evidence_id)

    return RetrievalArtifacts(
        documents=documents,
        evidence=evidence,
        document_ids=document_ids,
        evidence_ids=evidence_ids,
    )


def summarize_retrieval(
    *,
    agent_name: str,
    local_results: list[NormalizedResult],
    merged_results: MergedRetrievalResults,
    used_web_search: bool,
) -> str:
    """Build a short placeholder summary for the current retrieval stage."""
    return (
        f"{agent_name} retrieval completed. "
        f"local_hits={len(local_results)}, "
        f"positive_hits={len(merged_results['positive_results'])}, "
        f"risk_hits={len(merged_results['risk_results'])}, "
        f"web_search_used={used_web_search}."
    )


def _append_deduped_result(
    bucket: list[NormalizedResult],
    result: NormalizedResult,
    seen_keys: set[str],
) -> None:
    unique_key = _result_unique_key(result)
    if unique_key in seen_keys:
        return

    seen_keys.add(unique_key)
    bucket.append(result)


def _result_unique_key(result: NormalizedResult) -> str:
    if result.get("link"):
        return str(result["link"]).strip().lower()

    fallback = "|".join(
        [
            str(result.get("title", "")).strip().lower(),
            str(result.get("source", "")).strip().lower(),
            str(result.get("published_at", "")).strip().lower(),
        ]
    )
    return fallback


def _build_doc_id(result: NormalizedResult) -> str:
    seed = _result_unique_key(result)
    return f"doc_{sha1(seed.encode('utf-8')).hexdigest()[:12]}"
