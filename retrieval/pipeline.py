from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Literal

from retrieval.balanced_web_search import infer_topic_tags
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
class RetrievalAssessment:
    sufficient: bool
    gaps: list[str]
    evidence_count: int
    source_count: int
    positive_count: int
    risk_count: int
    topic_tags: list[str]


@dataclass(frozen=True, slots=True)
class RetrievalExecution:
    local_results: list[NormalizedResult]
    merged_results: MergedRetrievalResults
    used_web_search: bool
    local_assessment: RetrievalAssessment
    final_assessment: RetrievalAssessment


def is_retrieval_sufficient(
    results: list[NormalizedResult],
    *,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    min_evidence_count: int = MIN_EVIDENCE_COUNT,
    min_source_count: int = MIN_SOURCE_COUNT,
) -> bool:
    """Apply the mixed sufficiency rule for phase-1 local retrieval."""
    return evaluate_retrieval_results(
        results,
        company_scope=company_scope,
        min_evidence_count=min_evidence_count,
        min_source_count=min_source_count,
    ).sufficient


def evaluate_retrieval_results(
    results: list[NormalizedResult],
    *,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    min_evidence_count: int = MIN_EVIDENCE_COUNT,
    min_source_count: int = MIN_SOURCE_COUNT,
) -> RetrievalAssessment:
    """Evaluate retrieval coverage and return explicit insufficiency reasons."""
    evidence_count = len(results)
    sources = {
        result.get("source_name") or result.get("source") or result.get("media")
        for result in results
        if result.get("source_name") or result.get("source") or result.get("media")
    }
    positive_count = sum(1 for result in results if result.get("stance") == "positive")
    risk_count = sum(1 for result in results if result.get("stance") == "risk")

    topic_tags: set[str] = set()
    for result in results:
        tags = result.get("topic_tags", [])
        if isinstance(tags, str):
            topic_tags.add(tags)
        elif isinstance(tags, list):
            topic_tags.update(tag for tag in tags if isinstance(tag, str))

    gaps: list[str] = []
    if evidence_count < min_evidence_count:
        gaps.append(
            f"evidence_count: found {evidence_count} items but need at least {min_evidence_count}."
        )
    if len(sources) < min_source_count:
        gaps.append(
            f"source_diversity: found {len(sources)} sources but need at least {min_source_count}."
        )
    if positive_count == 0 or risk_count == 0:
        missing_stances = []
        if positive_count == 0:
            missing_stances.append("positive")
        if risk_count == 0:
            missing_stances.append("risk")
        gaps.append(
            "stance_balance: missing "
            + ", ".join(missing_stances)
            + " evidence."
        )

    required_tags = REQUIRED_TOPIC_TAGS[company_scope]
    missing_tags = sorted(required_tags - topic_tags)
    if missing_tags:
        gaps.append(
            "required_topics: missing "
            + ", ".join(missing_tags)
            + "."
        )

    return RetrievalAssessment(
        sufficient=not gaps,
        gaps=gaps,
        evidence_count=evidence_count,
        source_count=len(sources),
        positive_count=positive_count,
        risk_count=risk_count,
        topic_tags=sorted(topic_tags),
    )


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
    document_search_max_retries: int = 0,
    web_search_max_retries: int = 0,
) -> RetrievalExecution:
    """Run local retrieval first, then fall back to web search when insufficient."""
    logger.info(
        "[%s] Starting retrieval: positive_queries=%d, risk_queries=%d",
        company_scope,
        len(query_policy.get("positive_queries", [])),
        len(query_policy.get("risk_queries", [])),
    )

    local_results = _run_local_retrieval_with_retries(
        rag_retriever=rag_retriever,
        query_policy=query_policy,
        company_scope=company_scope,
        max_results_per_query=max_results_per_query,
        max_retries=document_search_max_retries,
    )
    local_assessment = evaluate_retrieval_results(
        local_results,
        company_scope=company_scope,
    )
    logger.info(
        "[%s] Local RAG hits=%d, sufficient=%s",
        company_scope,
        len(local_results),
        local_assessment.sufficient,
    )
    web_results: MergedRetrievalResults = {
        "positive_results": [],
        "risk_results": [],
    }
    if not local_assessment.sufficient:
        if not local_results:
            logger.info("[%s] Local RAG returned 0 hits. Falling back to web search.", company_scope)
        else:
            logger.info("[%s] Local RAG was insufficient. Falling back to web search.", company_scope)
        web_results = _run_web_search_with_retries(
            web_search_client=web_search_client,
            positive_queries=query_policy.get("positive_queries", []),
            risk_queries=query_policy.get("risk_queries", []),
            max_results_per_query=max_results_per_query,
            max_retries=web_search_max_retries,
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
    final_assessment = evaluate_retrieval_results(
        merged_results["positive_results"] + merged_results["risk_results"],
        company_scope=company_scope,
    )
    return RetrievalExecution(
        local_results=local_results,
        merged_results=merged_results,
        used_web_search=not local_assessment.sufficient,
        local_assessment=local_assessment,
        final_assessment=final_assessment,
    )


def run_skeptic_counter_retrieval(
    *,
    web_search_client: Any,
    risk_queries: list[str],
    max_results_per_query: int,
    web_search_max_retries: int = 0,
) -> MergedRetrievalResults:
    """Run a single risk-focused web search pass for skeptic counter-evidence."""
    return _run_web_search_with_retries(
        web_search_client=web_search_client,
        positive_queries=[],
        risk_queries=risk_queries,
        max_results_per_query=max_results_per_query,
        max_retries=web_search_max_retries,
    )


def build_retrieval_artifacts(
    *,
    merged_results: MergedRetrievalResults,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    used_for_override: Literal[
        "market_background",
        "lges_analysis",
        "catl_analysis",
        "counter_evidence",
        "comparison",
        "swot",
    ]
    | None = None,
) -> RetrievalArtifacts:
    """Convert normalized search results into SourceDocument/EvidenceItem entries."""
    documents: dict[str, SourceDocument] = {}
    evidence: dict[str, EvidenceItem] = {}
    document_ids: list[str] = []
    evidence_ids: list[str] = []
    used_for = used_for_override or USED_FOR_BY_SCOPE[company_scope]

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
    final_assessment: RetrievalAssessment,
) -> str:
    """Build a short placeholder summary for the current retrieval stage."""
    return (
        f"{agent_name} retrieval completed. "
        f"local_hits={len(local_results)}, "
        f"positive_hits={len(merged_results['positive_results'])}, "
        f"risk_hits={len(merged_results['risk_results'])}, "
        f"web_search_used={used_web_search}, "
        f"final_sufficient={final_assessment.sufficient}, "
        f"gaps={'; '.join(final_assessment.gaps) if final_assessment.gaps else 'none'}."
    )


def build_normalized_results_from_artifacts(
    *,
    documents: dict[str, SourceDocument],
    evidence: dict[str, EvidenceItem],
    evidence_ids: list[str],
) -> list[NormalizedResult]:
    """Rebuild normalized results from state artifacts for re-evaluation."""
    normalized_results: list[NormalizedResult] = []

    for evidence_id in evidence_ids:
        evidence_item = evidence.get(evidence_id)
        if evidence_item is None:
            continue

        document = documents.get(evidence_item["doc_id"])
        if document is None:
            continue

        normalized_results.append(
            {
                "title": document["title"],
                "link": document["source_url"],
                "source": document["source_name"],
                "source_name": document["source_name"],
                "published_at": document["published_at"],
                "query": evidence_item["topic"],
                "stance": document["stance"],
                "topic_tags": infer_topic_tags(
                    evidence_item["topic"],
                    stance=document["stance"],
                ),
            }
        )

    return normalized_results


def _run_local_retrieval_with_retries(
    *,
    rag_retriever: Any,
    query_policy: dict[str, list[str]],
    company_scope: Literal["MARKET", "LGES", "CATL"],
    max_results_per_query: int,
    max_retries: int,
) -> list[NormalizedResult]:
    attempts = 0
    while True:
        try:
            return _collect_local_results(
                rag_retriever=rag_retriever,
                query_policy=query_policy,
                company_scope=company_scope,
                max_results_per_query=max_results_per_query,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime/provider failures
            if attempts >= max_retries:
                logger.warning(
                    "[%s] Local retrieval failed after %d attempt(s): %s",
                    company_scope,
                    attempts + 1,
                    exc,
                )
                return []
            attempts += 1
            logger.warning(
                "[%s] Local retrieval failed. Retrying (%d/%d): %s",
                company_scope,
                attempts,
                max_retries,
                exc,
            )


def _collect_local_results(
    *,
    rag_retriever: Any,
    query_policy: dict[str, list[str]],
    company_scope: Literal["MARKET", "LGES", "CATL"],
    max_results_per_query: int,
) -> list[NormalizedResult]:
    local_results: list[NormalizedResult] = []

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
                if "topic_tags" not in normalized_result:
                    normalized_result["topic_tags"] = infer_topic_tags(query, stance=stance)
                local_results.append(normalized_result)

    return local_results


def _run_web_search_with_retries(
    *,
    web_search_client: Any,
    positive_queries: list[str],
    risk_queries: list[str],
    max_results_per_query: int,
    max_retries: int,
) -> MergedRetrievalResults:
    attempts = 0
    while True:
        try:
            return web_search_client.search(
                positive_queries=positive_queries,
                risk_queries=risk_queries,
                max_results_per_query=max_results_per_query,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime/provider failures
            if attempts >= max_retries:
                logger.warning(
                    "Web search failed after %d attempt(s): %s",
                    attempts + 1,
                    exc,
                )
                return {
                    "positive_results": [],
                    "risk_results": [],
                }
            attempts += 1
            logger.warning(
                "Web search failed. Retrying (%d/%d): %s",
                attempts,
                max_retries,
                exc,
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
