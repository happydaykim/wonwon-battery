from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Literal

from retrieval.article_fetcher import ArticleContentFetcher
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
    follow_up_positive_queries: list[str] = field(default_factory=list)
    follow_up_risk_queries: list[str] = field(default_factory=list)
    judge_summary: str | None = None


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
    article_fetcher: ArticleContentFetcher | None,
    retrieval_judge: Any | None,
    query_policy: dict[str, list[str]],
    company_scope: Literal["MARKET", "LGES", "CATL"],
    max_results_per_query: int,
    article_fetch_max_documents: int,
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
    local_assessment = _assess_retrieval_results(
        local_results,
        company_scope=company_scope,
        query_policy=query_policy,
        stage="local",
        retrieval_judge=retrieval_judge,
    )
    logger.info(
        "[%s] Local RAG hits=%d, sufficient=%s, gaps=%s",
        company_scope,
        len(local_results),
        local_assessment.sufficient,
        local_assessment.gaps,
    )
    web_results: MergedRetrievalResults = {
        "positive_results": [],
        "risk_results": [],
    }
    if not local_assessment.sufficient:
        fallback_query_policy = _merge_query_policy_with_follow_up_queries(
            query_policy,
            positive_queries=local_assessment.follow_up_positive_queries,
            risk_queries=local_assessment.follow_up_risk_queries,
        )
        logger.info(
            "[%s] Judge suggested fallback queries: positive=%s risk=%s",
            company_scope,
            fallback_query_policy["positive_queries"],
            fallback_query_policy["risk_queries"],
        )
        if not local_results:
            logger.info("[%s] Local RAG returned 0 hits. Falling back to web search.", company_scope)
        else:
            logger.info("[%s] Local RAG was insufficient. Falling back to web search.", company_scope)
        web_results = _run_web_search_with_retries(
            web_search_client=web_search_client,
            positive_queries=fallback_query_policy.get("positive_queries", []),
            risk_queries=fallback_query_policy.get("risk_queries", []),
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
    merged_results = enrich_results_with_article_content(
        merged_results,
        article_fetcher=article_fetcher,
        max_documents=article_fetch_max_documents,
        company_scope=company_scope,
    )
    logger.info(
        "[%s] Merged retrieval results: positive=%d, risk=%d",
        company_scope,
        len(merged_results["positive_results"]),
        len(merged_results["risk_results"]),
    )
    final_assessment = _assess_retrieval_results(
        merged_results["positive_results"] + merged_results["risk_results"],
        company_scope=company_scope,
        query_policy=query_policy,
        stage="final",
        retrieval_judge=retrieval_judge,
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
    article_fetcher: ArticleContentFetcher | None,
    company_scope: Literal["LGES", "CATL"],
    risk_queries: list[str],
    max_results_per_query: int,
    article_fetch_max_documents: int,
    web_search_max_retries: int = 0,
) -> MergedRetrievalResults:
    """Run a single risk-focused web search pass for skeptic counter-evidence."""
    results = _run_web_search_with_retries(
        web_search_client=web_search_client,
        positive_queries=[],
        risk_queries=risk_queries,
        max_results_per_query=max_results_per_query,
        max_retries=web_search_max_retries,
    )
    return enrich_results_with_article_content(
        results,
        article_fetcher=article_fetcher,
        max_documents=article_fetch_max_documents,
        company_scope=company_scope,
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
                "topic_tags": list(result.get("topic_tags", [])),
                "claim": result.get("article_excerpt") or result.get("title") or "Untitled claim",
                "excerpt": result.get("article_excerpt") or result.get("snippet"),
                "full_text": result.get("article_text"),
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
    company_scope: Literal["MARKET", "LGES", "CATL"],
    agent_name: str,
    local_results: list[NormalizedResult],
    merged_results: MergedRetrievalResults,
    used_web_search: bool,
    final_assessment: RetrievalAssessment,
) -> str:
    """Build a structured content summary from the current retrieval stage."""
    positive_results = merged_results["positive_results"]
    risk_results = merged_results["risk_results"]
    all_results = positive_results + risk_results

    if not all_results:
        gap_text = "; ".join(final_assessment.gaps) if final_assessment.gaps else "근거 없음"
        return "\n".join(
            [
                "[핵심 요약]",
                f"- {agent_name.upper()} 관련 검색 결과가 수집되지 않아 실제 내용 요약을 만들 수 없다.",
                "[검증 상태]",
                f"- local_hits={len(local_results)}",
                f"- web_search_used={used_web_search}",
                f"- 남은 gap: {gap_text}",
            ]
        )

    return "\n".join(
        [
            "[핵심 요약]",
            _build_focus_line(all_results, company_scope=company_scope),
            (
                f"- 현재 수집본은 긍정 근거 {len(positive_results)}건, 리스크 근거 {len(risk_results)}건으로 구성되며, "
                + (
                    "local 결과 부족으로 web search까지 사용했다."
                    if used_web_search
                    else "local retrieval만으로 구성되었다."
                )
            ),
            "[주요 긍정 근거]",
            _format_result_digest(positive_results, limit=2),
            "[주요 리스크 근거]",
            _format_result_digest(risk_results, limit=2),
            "[검증 상태]",
            (
                "- 현재 수집본 기준 주요 coverage gap 없음."
                if final_assessment.sufficient
                else "- 남은 gap: " + "; ".join(final_assessment.gaps)
            ),
        ]
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
                "snippet": evidence_item.get("full_text") or evidence_item["excerpt"],
                "stance": document["stance"],
                "topic_tags": evidence_item.get("topic_tags")
                or infer_topic_tags(
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
                metadata = normalized_result.get("metadata")
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        normalized_result.setdefault(key, value)

                normalized_result["query"] = normalized_result.get("query") or query
                normalized_result["stance"] = normalized_result.get("stance") or stance
                normalized_result["source"] = (
                    normalized_result.get("source")
                    or normalized_result.get("source_name")
                )
                normalized_result["link"] = (
                    normalized_result.get("link")
                    or normalized_result.get("source_url")
                )
                normalized_result["title"] = (
                    normalized_result.get("title")
                    or normalized_result.get("section_title")
                    or "Untitled local result"
                )
                if not normalized_result.get("topic_tags"):
                    normalized_result["topic_tags"] = infer_topic_tags(
                        query,
                        stance=normalized_result["stance"],
                    )
                local_results.append(normalized_result)

    return local_results


def _assess_retrieval_results(
    results: list[NormalizedResult],
    *,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    query_policy: dict[str, list[str]],
    stage: Literal["local", "final"],
    retrieval_judge: Any | None,
) -> RetrievalAssessment:
    base_assessment = evaluate_retrieval_results(
        results,
        company_scope=company_scope,
    )
    if retrieval_judge is None:
        return base_assessment

    try:
        judge_decision = retrieval_judge.judge(
            results=results,
            company_scope=company_scope,
            query_policy=query_policy,
            stage=stage,
            rule_based_summary=_format_rule_based_summary(base_assessment),
        )
    except Exception as exc:  # pragma: no cover - depends on runtime credentials/network
        logger.warning(
            "[%s] Retrieval judge failed at %s stage. Falling back to rule-based assessment: %s",
            company_scope,
            stage,
            exc,
        )
        return base_assessment

    sufficient = bool(results) and judge_decision.sufficient
    gaps = judge_decision.gaps or ([] if sufficient else base_assessment.gaps)
    return RetrievalAssessment(
        sufficient=sufficient,
        gaps=gaps,
        evidence_count=base_assessment.evidence_count,
        source_count=base_assessment.source_count,
        positive_count=base_assessment.positive_count,
        risk_count=base_assessment.risk_count,
        topic_tags=base_assessment.topic_tags,
        follow_up_positive_queries=(
            judge_decision.positive_queries if stage == "local" and not sufficient else []
        ),
        follow_up_risk_queries=(
            judge_decision.risk_queries if stage == "local" and not sufficient else []
        ),
        judge_summary=judge_decision.reasoning_summary,
    )


def _format_rule_based_summary(assessment: RetrievalAssessment) -> str:
    gaps = "; ".join(assessment.gaps) if assessment.gaps else "없음"
    topic_tags = ", ".join(assessment.topic_tags) if assessment.topic_tags else "없음"
    return "\n".join(
        [
            f"- evidence_count: {assessment.evidence_count}",
            f"- source_count: {assessment.source_count}",
            f"- positive_count: {assessment.positive_count}",
            f"- risk_count: {assessment.risk_count}",
            f"- topic_tags: {topic_tags}",
            f"- gaps: {gaps}",
        ]
    )


def _merge_query_policy_with_follow_up_queries(
    query_policy: dict[str, list[str]],
    *,
    positive_queries: list[str],
    risk_queries: list[str],
) -> dict[str, list[str]]:
    return {
        "positive_queries": _dedupe_queries(
            [*query_policy.get("positive_queries", []), *positive_queries]
        ),
        "risk_queries": _dedupe_queries(
            [*query_policy.get("risk_queries", []), *risk_queries]
        ),
    }


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(str(query).split()).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        deduped.append(normalized)
        seen.add(lowered)
    return deduped


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


def enrich_results_with_article_content(
    merged_results: MergedRetrievalResults,
    *,
    article_fetcher: ArticleContentFetcher | None,
    max_documents: int,
    company_scope: Literal["MARKET", "LGES", "CATL"],
) -> MergedRetrievalResults:
    if article_fetcher is None or max_documents <= 0:
        return merged_results

    enriched: MergedRetrievalResults = {
        "positive_results": [dict(result) for result in merged_results.get("positive_results", [])],
        "risk_results": [dict(result) for result in merged_results.get("risk_results", [])],
    }
    selected_locations = _select_article_fetch_locations(
        enriched,
        max_documents=max_documents,
    )
    fetched_count = 0
    for bucket, index in selected_locations:
        result = enriched[bucket][index]
        fetch_result = article_fetcher.fetch(result.get("link"))
        if fetch_result is None:
            continue

        if fetch_result.resolved_url:
            result["link"] = fetch_result.resolved_url
        if fetch_result.publisher_name:
            result["source"] = fetch_result.publisher_name
            result["source_name"] = fetch_result.publisher_name
        if fetch_result.title:
            result["title"] = fetch_result.title
        if fetch_result.published_at:
            result["published_at"] = fetch_result.published_at
        if fetch_result.excerpt:
            result["article_excerpt"] = fetch_result.excerpt
            result["snippet"] = fetch_result.excerpt
        if fetch_result.full_text:
            result["article_text"] = fetch_result.full_text
            fetched_count += 1

    logger.info(
        "[%s] Article enrichment completed for %d/%d selected result(s).",
        company_scope,
        fetched_count,
        len(selected_locations),
    )
    return enriched


def _build_focus_line(
    results: list[NormalizedResult],
    *,
    company_scope: Literal["MARKET", "LGES", "CATL"],
) -> str:
    key_queries = _pick_distinct_queries(results, limit=3)
    if key_queries:
        if company_scope == "MARKET":
            return "- 수집 자료는 주로 " + ", ".join(f"'{query}'" for query in key_queries) + " 축의 시장 배경을 다룬다."
        return "- 수집 자료는 주로 " + ", ".join(f"'{query}'" for query in key_queries) + " 축에서 전략과 리스크를 다룬다."

    topic_tags = _collect_topic_tags(results)
    if not topic_tags:
        return "- 수집 자료의 핵심 축을 식별할 만큼 태그 정보가 충분하지 않다."

    return "- 수집 자료는 " + ", ".join(topic_tags[:3]) + " 관련 쟁점에 집중되어 있다."


def _select_article_fetch_locations(
    merged_results: MergedRetrievalResults,
    *,
    max_documents: int,
) -> list[tuple[str, int]]:
    positive_indexes = list(range(len(merged_results.get("positive_results", []))))
    risk_indexes = list(range(len(merged_results.get("risk_results", []))))

    selected: list[tuple[str, int]] = []
    positive_quota = min(len(positive_indexes), (max_documents + 1) // 2)
    risk_quota = min(len(risk_indexes), max_documents // 2)

    selected.extend(("positive_results", index) for index in positive_indexes[:positive_quota])
    selected.extend(("risk_results", index) for index in risk_indexes[:risk_quota])

    remaining_capacity = max_documents - len(selected)
    if remaining_capacity > 0:
        remaining = [
            ("positive_results", index)
            for index in positive_indexes[positive_quota:]
        ] + [
            ("risk_results", index)
            for index in risk_indexes[risk_quota:]
        ]
        selected.extend(remaining[:remaining_capacity])

    return selected


def _pick_distinct_queries(results: list[NormalizedResult], *, limit: int) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for result in results:
        query = str(result.get("query") or "").strip()
        if not query or query in seen:
            continue
        queries.append(query)
        seen.add(query)
        if len(queries) >= limit:
            break
    return queries


def _collect_topic_tags(results: list[NormalizedResult]) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for result in results:
        tags = result.get("topic_tags", [])
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if not isinstance(tag, str) or tag in seen:
                continue
            collected.append(tag)
            seen.add(tag)
    return collected


def _format_result_digest(results: list[NormalizedResult], *, limit: int) -> str:
    selected_results = _select_representative_results(results, limit=limit)
    if not selected_results:
        return "- 정보 부족"

    lines: list[str] = []
    for result in selected_results:
        source_name = result.get("source_name") or result.get("source") or "출처 미상"
        published_at = result.get("published_at") or "날짜 미상"
        title = _compact_text(str(result.get("title") or "Untitled news result"), limit=160)
        query = str(result.get("query") or "질의 미상")
        topic_tags = result.get("topic_tags", [])
        if isinstance(topic_tags, str):
            topic_tags = [topic_tags]
        tags_text = ", ".join(tag for tag in topic_tags if isinstance(tag, str)) or "없음"
        signal = _compact_text(
            str(result.get("snippet") or result.get("title") or "signal 없음"),
            limit=220,
        )
        lines.extend(
            [
                f"- [{source_name} | {published_at}] {title}",
                f"  - query: {query}",
                f"  - tags: {tags_text}",
                f"  - signal: {signal}",
            ]
        )

    return "\n".join(lines)


def _select_representative_results(
    results: list[NormalizedResult],
    *,
    limit: int,
) -> list[NormalizedResult]:
    if limit <= 0:
        return []

    selected: list[NormalizedResult] = []
    seen_sources: set[str] = set()
    seen_queries: set[str] = set()
    remaining = list(results)

    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda result: (
                1
                if (result.get("source_name") or result.get("source"))
                and (result.get("source_name") or result.get("source")) not in seen_sources
                else 0,
                1 if result.get("query") and result.get("query") not in seen_queries else 0,
                1 if result.get("snippet") else 0,
            ),
        )
        selected.append(best)
        remaining.remove(best)

        source_name = best.get("source_name") or best.get("source")
        if isinstance(source_name, str) and source_name:
            seen_sources.add(source_name)
        query = best.get("query")
        if isinstance(query, str) and query:
            seen_queries.add(query)

    return selected


def _compact_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
