from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
import re
from typing import Any, Literal

from config.settings import load_settings
from retrieval.article_fetcher import ArticleContentFetcher
from retrieval.balanced_web_search import infer_topic_tags
from retrieval.retrieval_decider import decide_retrieval_action
from retrieval.query_refiner import refine_query_policy
from schemas.state import EvidenceItem, SourceDocument
from utils.logging import get_logger


NormalizedResult = dict[str, Any]
MergedRetrievalResults = dict[str, list[NormalizedResult]]

NUMERIC_PATTERN = re.compile(
    r"(?i)(\d[\d,\.]*\s?(?:%|배|건|개|명|대|억|조|만|천|원|달러|억원|조원|GWh|MWh|kWh|Wh|GW|MW|kW|Ah|mAh|x|X|YoY|yoy|bp|bps)?)"
)

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
VALID_DOC_TYPES = {
    "industry_report",
    "annual_report",
    "ir_deck",
    "press_release",
    "news",
    "paper",
    "other",
}
VALID_COMPANY_SCOPES = {"MARKET", "LGES", "CATL", "BOTH"}
VALID_STANCES = {"neutral", "positive", "risk"}

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
    coverage_count: int
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
    query_history: list[str]
    refinement_rounds: int
    decision_notes: list[str]
    refinement_notes: list[str]
    failure_notes: list[str]


@dataclass(frozen=True, slots=True)
class FlatRetrievalCallResult:
    results: list[NormalizedResult]
    failure_note: str | None = None


@dataclass(frozen=True, slots=True)
class BucketedRetrievalCallResult:
    results: MergedRetrievalResults
    failure_note: str | None = None


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
    coverage_count = len(
        {
            _coverage_unit_key(result)
            for result in results
            if _coverage_unit_key(result)
        }
    )
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
    if coverage_count < min_evidence_count:
        gaps.append(
            "evidence_count: found "
            f"{coverage_count} distinct evidence unit(s) but need at least {min_evidence_count}."
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
        coverage_count=coverage_count,
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
    """Merge local and web results into stance buckets with origin-aware dedupe."""
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
    retrieval_judge: Any | None = None,
    query_policy: dict[str, list[str]],
    company_scope: Literal["MARKET", "LGES", "CATL"],
    max_results_per_query: int,
    article_fetch_max_documents: int,
    document_search_max_retries: int = 0,
    web_search_max_retries: int = 0,
    max_refinement_rounds: int | None = None,
    max_new_queries_per_bucket: int | None = None,
) -> RetrievalExecution:
    """Run local retrieval with decider-controlled web expansion and refinement."""
    settings = load_settings()
    refinement_budget = (
        settings.retrieval_refinement_max_rounds
        if max_refinement_rounds is None
        else max_refinement_rounds
    )
    refinement_query_limit = (
        settings.retrieval_refinement_max_queries_per_bucket
        if max_new_queries_per_bucket is None
        else max_new_queries_per_bucket
    )
    logger.info(
        "[%s] Starting retrieval: positive_queries=%d, risk_queries=%d",
        company_scope,
        len(query_policy.get("positive_queries", [])),
        len(query_policy.get("risk_queries", [])),
    )
    cumulative_local_results: list[NormalizedResult] = []
    cumulative_web_results: MergedRetrievalResults = {
        "positive_results": [],
        "risk_results": [],
    }
    query_history: list[str] = []
    used_queries: set[str] = set()
    decision_notes: list[str] = []
    refinement_notes: list[str] = []
    failure_notes: list[str] = []
    refinement_rounds = 0
    web_search_attempted = False
    active_query_policy = _sanitize_query_policy(query_policy)
    round_index = 0
    local_assessment = _build_empty_assessment(company_scope)
    final_assessment = _build_empty_assessment(company_scope)

    while True:
        _record_query_history(query_history, used_queries, active_query_policy)
        logger.info(
            "[%s] Retrieval round %d executing queries: positive=%s, risk=%s",
            company_scope,
            round_index + 1,
            active_query_policy.get("positive_queries", []),
            active_query_policy.get("risk_queries", []),
        )

        local_call = _run_local_retrieval_with_retries(
            rag_retriever=rag_retriever,
            query_policy=active_query_policy,
            company_scope=company_scope,
            max_results_per_query=max_results_per_query,
            max_retries=document_search_max_retries,
        )
        round_local_results = local_call.results
        _append_failure_note(
            failure_notes,
            decision_notes,
            local_call.failure_note,
        )
        cumulative_local_results = _merge_flat_results(
            cumulative_local_results,
            round_local_results,
        )
        local_assessment = _assess_retrieval_results(
            cumulative_local_results,
            company_scope=company_scope,
            query_policy=active_query_policy,
            stage="local",
            retrieval_judge=retrieval_judge,
        )
        logger.info(
            "[%s] Local RAG hits=%d, sufficient=%s, gaps=%s",
            company_scope,
            len(cumulative_local_results),
            local_assessment.sufficient,
            local_assessment.gaps,
        )
        _append_judge_assessment_note(
            decision_notes,
            stage="post_local",
            assessment=local_assessment,
        )

        round_web_results: MergedRetrievalResults = {
            "positive_results": [],
            "risk_results": [],
        }

        merged_query_policy = _merge_query_policy_with_follow_up_queries(
            active_query_policy,
            positive_queries=local_assessment.follow_up_positive_queries,
            risk_queries=local_assessment.follow_up_risk_queries,
        )
        if merged_query_policy != active_query_policy:
            logger.info(
                "[%s] Judge suggested follow-up queries: positive=%s risk=%s",
                company_scope,
                merged_query_policy["positive_queries"],
                merged_query_policy["risk_queries"],
            )
            decision_notes.append(
                "post_local:judge_queries:"
                f"positive={local_assessment.follow_up_positive_queries}:"
                f"risk={local_assessment.follow_up_risk_queries}"
            )
            active_query_policy = merged_query_policy

        local_decision = decide_retrieval_action(
            stage="post_local",
            company_scope=company_scope,
            assessment=local_assessment,
            observed_results=cumulative_local_results,
            current_query_policy=active_query_policy,
            query_history=query_history,
            used_web_search=web_search_attempted,
            refinement_rounds=refinement_rounds,
            refinement_budget=refinement_budget,
            settings=settings,
        )
        local_action = local_decision.action
        decision_notes.append(
            (
                f"post_local:{local_decision.decision_mode}:"
                f"{local_decision.action}:{local_decision.rationale}"
            )
        )
        logger.info(
            "[%s] Retrieval decision post_local: mode=%s, action=%s, rationale=%s",
            company_scope,
            local_decision.decision_mode,
            local_decision.action,
            local_decision.rationale,
        )
        if local_action == "search_web":
            web_search_attempted = True
            if not cumulative_local_results:
                logger.info(
                    "[%s] Local RAG returned 0 hits. Falling back to web search.",
                    company_scope,
                )
            else:
                logger.info(
                    "[%s] Local-first policy escalated to web search.",
                    company_scope,
                )
            # Judge-suggested follow-up queries should count as history only if we actually execute them.
            _record_query_history(query_history, used_queries, active_query_policy)
            web_call = _run_web_search_with_retries(
                web_search_client=web_search_client,
                positive_queries=active_query_policy.get("positive_queries", []),
                risk_queries=active_query_policy.get("risk_queries", []),
                max_results_per_query=max_results_per_query,
                max_retries=web_search_max_retries,
            )
            round_web_results = web_call.results
            _append_failure_note(
                failure_notes,
                decision_notes,
                web_call.failure_note,
            )
            logger.info(
                "[%s] Web search hits: positive=%d, risk=%d",
                company_scope,
                len(round_web_results["positive_results"]),
                len(round_web_results["risk_results"]),
            )
        else:
            logger.info("[%s] Web search skipped after local-first decision.", company_scope)

        cumulative_web_results = _merge_bucketed_results(
            cumulative_web_results,
            round_web_results,
        )
        merged_results = merge_retrieval_results(
            local_results=cumulative_local_results,
            web_results=cumulative_web_results,
        )
        final_assessment = _assess_retrieval_results(
            merged_results["positive_results"] + merged_results["risk_results"],
            company_scope=company_scope,
            query_policy=active_query_policy,
            stage="final",
            retrieval_judge=retrieval_judge,
        )
        _append_judge_assessment_note(
            decision_notes,
            stage="post_merge",
            assessment=final_assessment,
        )

        if local_action != "search_web":
            break
        merge_decision = decide_retrieval_action(
            stage="post_merge",
            company_scope=company_scope,
            assessment=final_assessment,
            observed_results=merged_results["positive_results"] + merged_results["risk_results"],
            current_query_policy=active_query_policy,
            query_history=query_history,
            used_web_search=True,
            refinement_rounds=refinement_rounds,
            refinement_budget=refinement_budget,
            settings=settings,
        )
        decision_notes.append(
            (
                f"post_merge:{merge_decision.decision_mode}:"
                f"{merge_decision.action}:{merge_decision.rationale}"
            )
        )
        logger.info(
            "[%s] Retrieval decision post_merge: mode=%s, action=%s, rationale=%s",
            company_scope,
            merge_decision.decision_mode,
            merge_decision.action,
            merge_decision.rationale,
        )
        if merge_decision.action != "refine":
            break
        refinement_result = refine_query_policy(
            company_scope=company_scope,
            current_query_policy=active_query_policy,
            assessment=final_assessment,
            observed_results=merged_results["positive_results"] + merged_results["risk_results"],
            used_queries=used_queries,
            max_new_queries_per_bucket=refinement_query_limit,
            settings=settings,
        )
        if not refinement_result.positive_queries and not refinement_result.risk_queries:
            refinement_notes.append(refinement_result.rationale)
            break

        refinement_rounds += 1
        refinement_notes.append(
            (
                f"round_{refinement_rounds}:{refinement_result.refinement_mode}: "
                f"positive={refinement_result.positive_queries}, risk={refinement_result.risk_queries}"
            )
        )
        active_query_policy = {
            "positive_queries": refinement_result.positive_queries,
            "risk_queries": refinement_result.risk_queries,
        }
        round_index += 1

    merged_results = merge_retrieval_results(
        local_results=cumulative_local_results,
        web_results=cumulative_web_results,
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
    if retrieval_judge is None:
        local_assessment = _assess_retrieval_results(
            cumulative_local_results,
            company_scope=company_scope,
            query_policy=active_query_policy,
            stage="local",
            retrieval_judge=retrieval_judge,
        )
        final_assessment = _assess_retrieval_results(
            merged_results["positive_results"] + merged_results["risk_results"],
            company_scope=company_scope,
            query_policy=active_query_policy,
            stage="final",
            retrieval_judge=retrieval_judge,
        )
    return RetrievalExecution(
        local_results=cumulative_local_results,
        merged_results=merged_results,
        used_web_search=web_search_attempted,
        local_assessment=local_assessment,
        final_assessment=final_assessment,
        query_history=query_history,
        refinement_rounds=refinement_rounds,
        decision_notes=decision_notes,
        refinement_notes=refinement_notes,
        failure_notes=failure_notes,
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
    max_refinement_rounds: int | None = None,
    max_new_queries_per_bucket: int | None = None,
) -> RetrievalExecution:
    """Run a risk-focused web retrieval loop for skeptic counter-evidence."""
    settings = load_settings()
    refinement_budget = (
        settings.retrieval_refinement_max_rounds
        if max_refinement_rounds is None
        else max_refinement_rounds
    )
    refinement_query_limit = (
        settings.retrieval_refinement_max_queries_per_bucket
        if max_new_queries_per_bucket is None
        else max_new_queries_per_bucket
    )
    cumulative_web_results: MergedRetrievalResults = {
        "positive_results": [],
        "risk_results": [],
    }
    query_history: list[str] = []
    used_queries: set[str] = set()
    decision_notes: list[str] = []
    refinement_notes: list[str] = []
    failure_notes: list[str] = []
    refinement_rounds = 0
    web_search_attempted = False
    active_query_policy = {
        "positive_queries": [],
        "risk_queries": _sanitize_queries(risk_queries),
    }
    round_index = 0

    while True:
        _record_query_history(query_history, used_queries, active_query_policy)
        web_search_attempted = True
        web_call = _run_web_search_with_retries(
            web_search_client=web_search_client,
            positive_queries=[],
            risk_queries=active_query_policy["risk_queries"],
            max_results_per_query=max_results_per_query,
            max_retries=web_search_max_retries,
        )
        round_results = web_call.results
        _append_failure_note(
            failure_notes,
            decision_notes,
            web_call.failure_note,
        )
        cumulative_web_results = _merge_bucketed_results(
            cumulative_web_results,
            round_results,
        )
        final_assessment = evaluate_retrieval_results(
            cumulative_web_results["positive_results"] + cumulative_web_results["risk_results"],
            company_scope=company_scope,
        )
        if round_index >= refinement_budget:
            decision_notes.append(
                "risk_review:fallback:stop:Refinement budget is exhausted after this round."
            )
            break
        risk_decision = decide_retrieval_action(
            stage="risk_review",
            company_scope=company_scope,
            assessment=final_assessment,
            observed_results=cumulative_web_results["positive_results"] + cumulative_web_results["risk_results"],
            current_query_policy=active_query_policy,
            query_history=query_history,
            used_web_search=web_search_attempted,
            refinement_rounds=refinement_rounds,
            refinement_budget=refinement_budget,
            settings=settings,
            risk_only=True,
        )
        decision_notes.append(
            (
                f"risk_review:{risk_decision.decision_mode}:"
                f"{risk_decision.action}:{risk_decision.rationale}"
            )
        )
        logger.info(
            "[%s] Retrieval decision risk_review: mode=%s, action=%s, rationale=%s",
            company_scope,
            risk_decision.decision_mode,
            risk_decision.action,
            risk_decision.rationale,
        )
        if risk_decision.action != "refine":
            break

        refinement_result = refine_query_policy(
            company_scope=company_scope,
            current_query_policy=active_query_policy,
            assessment=final_assessment,
            observed_results=cumulative_web_results["positive_results"] + cumulative_web_results["risk_results"],
            used_queries=used_queries,
            max_new_queries_per_bucket=refinement_query_limit,
            settings=settings,
            risk_only=True,
        )
        if not refinement_result.risk_queries:
            refinement_notes.append(refinement_result.rationale)
            break

        refinement_rounds += 1
        refinement_notes.append(
            f"round_{refinement_rounds}:{refinement_result.refinement_mode}: risk={refinement_result.risk_queries}"
        )
        active_query_policy = {
            "positive_queries": [],
            "risk_queries": refinement_result.risk_queries,
        }
        round_index += 1

    merged_results = enrich_results_with_article_content(
        cumulative_web_results,
        article_fetcher=article_fetcher,
        max_documents=article_fetch_max_documents,
        company_scope=company_scope,
    )
    final_assessment = evaluate_retrieval_results(
        merged_results["positive_results"] + merged_results["risk_results"],
        company_scope=company_scope,
    )
    return RetrievalExecution(
        local_results=[],
        merged_results=merged_results,
        used_web_search=web_search_attempted,
        local_assessment=_build_empty_assessment(company_scope),
        final_assessment=final_assessment,
        query_history=query_history,
        refinement_rounds=refinement_rounds,
        decision_notes=decision_notes,
        refinement_notes=refinement_notes,
        failure_notes=failure_notes,
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
            doc_id = _resolve_document_id(result)
            evidence_id = _build_evidence_id(result, doc_id=doc_id)

            if doc_id not in documents:
                documents[doc_id] = {
                    "doc_id": doc_id,
                    "title": result.get("title") or "Untitled result",
                    "source_name": _resolve_source_name(result),
                    "source_url": result.get("link"),
                    "published_at": result.get("published_at"),
                    "doc_type": _resolve_doc_type(result),
                    "company_scope": _resolve_company_scope(
                        result,
                        default_scope=company_scope,
                    ),
                    "stance": _resolve_stance(result),
                }
                document_ids.append(doc_id)

            if evidence_id in evidence:
                continue

            evidence[evidence_id] = {
                "evidence_id": evidence_id,
                "doc_id": doc_id,
                "topic": result.get("query") or "retrieval_result",
                "topic_tags": _normalize_topic_tags(result.get("topic_tags")),
                "claim": (
                    result.get("article_excerpt")
                    or result.get("snippet")
                    or result.get("title")
                    or "Untitled claim"
                ),
                "excerpt": result.get("article_excerpt") or result.get("snippet"),
                "full_text": result.get("article_text"),
                "page_or_chunk": _resolve_page_or_chunk(result),
                "relevance_score": _resolve_relevance_score(result),
                "used_for": used_for,
            }
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
    query_history: list[str],
    refinement_rounds: int,
    failure_notes: list[str] | None = None,
) -> str:
    """Build a structured content summary from the current retrieval stage."""
    positive_results = merged_results["positive_results"]
    risk_results = merged_results["risk_results"]
    all_results = positive_results + risk_results
    query_note = ", ".join(f"'{query}'" for query in query_history[:6]) or "기록 없음"
    source_count = _count_distinct_sources(all_results)

    if not all_results:
        gap_text = "; ".join(final_assessment.gaps) if final_assessment.gaps else "근거 없음"
        lines = [
            "[핵심 요약]",
            f"- {agent_name.upper()} 관련 검색 결과가 수집되지 않아 실제 내용 요약을 만들 수 없다.",
            "[검증 상태]",
            f"- local_hits={len(local_results)}",
            f"- web_search_used={used_web_search}",
            f"- query_history={query_note}",
            f"- refinement_rounds={refinement_rounds}",
            f"- 남은 gap: {gap_text}",
        ]
        if failure_notes:
            lines.append("- retrieval_failures: " + "; ".join(failure_notes))
        return "\n".join(lines)

    lines = [
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
        f"- 대표 출처는 {source_count}곳이며, 동일 출처 반복보다 서로 다른 근거 축이 우선 반영되었다.",
        "[검색 루프]",
        f"- 실행 질의: {query_note}",
        f"- refinement_rounds={refinement_rounds}",
        "[주요 긍정 근거]",
        _format_result_digest(positive_results, limit=3),
        "[주요 리스크 근거]",
        _format_result_digest(risk_results, limit=3),
        "[검증 상태]",
        (
            "- 현재 수집본 기준 주요 coverage gap 없음."
            if final_assessment.sufficient
            else "- 남은 gap: " + "; ".join(final_assessment.gaps)
        ),
    ]
    if failure_notes and not final_assessment.sufficient:
        lines.append("- retrieval_failures: " + "; ".join(failure_notes))
    return "\n".join(lines)


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
                "doc_id": document["doc_id"],
                "title": document["title"],
                "link": document["source_url"],
                "source": document["source_name"],
                "source_name": document["source_name"],
                "published_at": document["published_at"],
                "doc_type": document["doc_type"],
                "company_scope": document["company_scope"],
                "query": evidence_item["topic"],
                "snippet": evidence_item["excerpt"] or evidence_item.get("full_text"),
                "article_excerpt": evidence_item["excerpt"],
                "article_text": evidence_item.get("full_text"),
                "stance": document["stance"],
                "topic_tags": evidence_item.get("topic_tags")
                or infer_topic_tags(
                    evidence_item["topic"],
                    stance=document["stance"],
                ),
                "page_or_chunk": evidence_item.get("page_or_chunk"),
                "relevance_score": evidence_item.get("relevance_score"),
                "retrieval_origin": (
                    "local_rag"
                    if evidence_item.get("relevance_score") is not None
                    else "web_search"
                ),
            }
        )
        if evidence_item.get("relevance_score") is not None and evidence_id.startswith("evidence_"):
            normalized_results[-1]["chunk_id"] = evidence_id.removeprefix("evidence_")

    return normalized_results


def _build_empty_assessment(
    company_scope: Literal["MARKET", "LGES", "CATL"],
) -> RetrievalAssessment:
    return evaluate_retrieval_results([], company_scope=company_scope)


def _sanitize_query_policy(
    query_policy: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        "positive_queries": _sanitize_queries(query_policy.get("positive_queries", [])),
        "risk_queries": _sanitize_queries(query_policy.get("risk_queries", [])),
    }


def _sanitize_queries(queries: list[str]) -> list[str]:
    sanitized: list[str] = []
    seen: set[str] = set()
    for raw_query in queries:
        query = " ".join(raw_query.split())
        if not query:
            continue
        normalized = query.lower()
        if normalized in seen:
            continue
        sanitized.append(query)
        seen.add(normalized)
    return sanitized


def _record_query_history(
    query_history: list[str],
    used_queries: set[str],
    query_policy: dict[str, list[str]],
) -> None:
    for bucket in ("positive_queries", "risk_queries"):
        for query in query_policy.get(bucket, []):
            normalized = query.lower()
            if normalized in used_queries:
                continue
            query_history.append(query)
            used_queries.add(normalized)


def _merge_flat_results(
    existing_results: list[NormalizedResult],
    new_results: list[NormalizedResult],
) -> list[NormalizedResult]:
    merged_results: list[NormalizedResult] = []
    seen_keys: set[str] = set()
    for result in [*existing_results, *new_results]:
        _append_deduped_result(merged_results, result, seen_keys)
    return merged_results


def _merge_bucketed_results(
    existing_results: MergedRetrievalResults,
    new_results: MergedRetrievalResults,
) -> MergedRetrievalResults:
    merged: MergedRetrievalResults = {
        "positive_results": [],
        "risk_results": [],
    }
    seen_keys: set[str] = set()

    for bucket in ("positive_results", "risk_results"):
        for result in existing_results.get(bucket, []):
            _append_deduped_result(merged[bucket], result, seen_keys)
        for result in new_results.get(bucket, []):
            _append_deduped_result(merged[bucket], result, seen_keys)

    return merged


def _run_local_retrieval_with_retries(
    *,
    rag_retriever: Any,
    query_policy: dict[str, list[str]],
    company_scope: Literal["MARKET", "LGES", "CATL"],
    max_results_per_query: int,
    max_retries: int,
) -> FlatRetrievalCallResult:
    attempts = 0
    while True:
        try:
            return FlatRetrievalCallResult(
                results=_collect_local_results(
                    rag_retriever=rag_retriever,
                    query_policy=query_policy,
                    company_scope=company_scope,
                    max_results_per_query=max_results_per_query,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on runtime/provider failures
            failure_note = (
                "local_retrieval_failure: "
                f"failed after {attempts + 1} attempt(s): {exc}"
            )
            if attempts >= max_retries:
                logger.warning(
                    "[%s] Local retrieval failed after %d attempt(s): %s",
                    company_scope,
                    attempts + 1,
                    exc,
                )
                return FlatRetrievalCallResult(
                    results=[],
                    failure_note=failure_note,
                )
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
                resolved_stance = normalized_result.get("stance")
                if not isinstance(resolved_stance, str) or not resolved_stance.strip():
                    resolved_stance = stance
                normalized_result["stance"] = resolved_stance
                normalized_result["source_name"] = (
                    normalized_result.get("source_name")
                    or normalized_result.get("source")
                )
                normalized_result["source"] = (
                    normalized_result.get("source")
                    or normalized_result.get("source_name")
                )
                normalized_result["link"] = (
                    normalized_result.get("link")
                    or normalized_result.get("source_url")
                )
                normalized_result["source_url"] = (
                    normalized_result.get("source_url")
                    or normalized_result.get("link")
                )
                normalized_result["title"] = (
                    normalized_result.get("title")
                    or normalized_result.get("section_title")
                    or "Untitled local result"
                )
                if not normalized_result.get("topic_tags"):
                    normalized_result["topic_tags"] = infer_topic_tags(
                        query,
                        stance=(
                            resolved_stance
                            if resolved_stance in {"positive", "risk"}
                            else stance
                        ),
                    )
                normalized_result.setdefault("retrieval_origin", "local_rag")
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
        coverage_count=base_assessment.coverage_count,
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
            f"- coverage_count: {assessment.coverage_count}",
            f"- source_count: {assessment.source_count}",
            f"- positive_count: {assessment.positive_count}",
            f"- risk_count: {assessment.risk_count}",
            f"- topic_tags: {topic_tags}",
            f"- gaps: {gaps}",
        ]
    )


def _append_judge_assessment_note(
    decision_notes: list[str],
    *,
    stage: str,
    assessment: RetrievalAssessment,
) -> None:
    if (
        assessment.judge_summary is None
        and not assessment.follow_up_positive_queries
        and not assessment.follow_up_risk_queries
    ):
        return

    details = [
        assessment.judge_summary
        or (
            "Judge marked current evidence sufficient."
            if assessment.sufficient
            else "Judge flagged remaining coverage gaps."
        )
    ]
    if assessment.follow_up_positive_queries:
        details.append(f"positive_queries={assessment.follow_up_positive_queries}")
    if assessment.follow_up_risk_queries:
        details.append(f"risk_queries={assessment.follow_up_risk_queries}")

    decision_notes.append(f"{stage}:judge_assessment:{' | '.join(details)}")


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
) -> BucketedRetrievalCallResult:
    attempts = 0
    while True:
        try:
            return BucketedRetrievalCallResult(
                results=web_search_client.search(
                    positive_queries=positive_queries,
                    risk_queries=risk_queries,
                    max_results_per_query=max_results_per_query,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on runtime/provider failures
            failure_note = (
                "web_search_failure: "
                f"failed after {attempts + 1} attempt(s): {exc}"
            )
            if attempts >= max_retries:
                logger.warning(
                    "Web search failed after %d attempt(s): %s",
                    attempts + 1,
                    exc,
                )
                return BucketedRetrievalCallResult(
                    results={
                        "positive_results": [],
                        "risk_results": [],
                    },
                    failure_note=failure_note,
                )
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


def _append_failure_note(
    failure_notes: list[str],
    decision_notes: list[str],
    failure_note: str | None,
) -> None:
    if not failure_note or failure_note in failure_notes:
        return

    failure_notes.append(failure_note)
    decision_notes.append(f"runtime_failure:{failure_note}")


def _result_unique_key(result: NormalizedResult) -> str:
    if _is_local_result(result):
        chunk_id = result.get("chunk_id")
        if chunk_id:
            return f"local_chunk:{chunk_id}"

        local_fallback = "|".join(
            [
                str(result.get("doc_id", "")).strip().lower(),
                str(result.get("page_or_chunk", "")).strip().lower(),
                str(result.get("query", "")).strip().lower(),
            ]
        )
        if local_fallback.strip("|"):
            return f"local_result:{local_fallback}"

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


def _coverage_unit_key(result: NormalizedResult) -> str:
    if _is_local_result(result):
        chunk_id = str(result.get("chunk_id") or "").strip().lower()
        if chunk_id:
            return f"local_chunk:{chunk_id}"

        local_locator = "|".join(
            [
                str(result.get("doc_id") or "").strip().lower(),
                str(result.get("page_or_chunk") or "").strip().lower(),
            ]
        ).strip("|")
        if local_locator:
            return f"local_unit:{local_locator}"

        local_url = str(result.get("source_url") or result.get("link") or "").strip().lower()
        if local_url:
            return f"local_doc:{local_url}"

    title_key = _normalize_story_title(result.get("title"))
    published_key = str(result.get("published_at") or "").strip().lower()
    if title_key:
        if published_key:
            return f"story:{title_key}|{published_key}"
        return f"story:{title_key}"

    return _result_unique_key(result)


def _normalize_story_title(value: Any) -> str:
    return re.sub(r"\W+", "", str(value or "").strip().lower())


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
    positive_indexes = [
        index
        for index, result in enumerate(merged_results.get("positive_results", []))
        if not result.get("article_text")
    ]
    risk_indexes = [
        index
        for index, result in enumerate(merged_results.get("risk_results", []))
        if not result.get("article_text")
    ]

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


def _resolve_document_id(result: NormalizedResult) -> str:
    if _is_local_result(result):
        local_doc_id = str(result.get("doc_id") or "").strip()
        if local_doc_id:
            return local_doc_id
    return _build_doc_id(result)


def _build_evidence_id(result: NormalizedResult, *, doc_id: str) -> str:
    if _is_local_result(result):
        chunk_id = str(result.get("chunk_id") or "").strip()
        if chunk_id:
            return f"evidence_{chunk_id}"

        seed = "|".join(
            [
                doc_id,
                str(result.get("page_or_chunk") or "").strip(),
                str(result.get("query") or "").strip(),
            ]
        )
        return f"evidence_{sha1(seed.encode('utf-8')).hexdigest()[:12]}"

    return f"evidence_{doc_id}"


def _is_local_result(result: NormalizedResult) -> bool:
    if result.get("retrieval_origin") == "local_rag":
        return True
    if result.get("chunk_id"):
        return True
    return result.get("relevance_score") is not None


def _resolve_source_name(result: NormalizedResult) -> str:
    return (
        str(result.get("source_name") or "").strip()
        or str(result.get("source") or "").strip()
        or "GoogleNews"
    )


def _resolve_doc_type(result: NormalizedResult) -> str:
    doc_type = str(result.get("doc_type") or "").strip()
    if doc_type in VALID_DOC_TYPES:
        return doc_type
    if _is_local_result(result):
        return "other"
    return "news"


def _resolve_company_scope(
    result: NormalizedResult,
    *,
    default_scope: Literal["MARKET", "LGES", "CATL"],
) -> str:
    resolved_scope = str(result.get("company_scope") or "").strip()
    if resolved_scope in VALID_COMPANY_SCOPES:
        return resolved_scope
    return default_scope


def _resolve_stance(result: NormalizedResult) -> str:
    stance = str(result.get("stance") or "").strip()
    if stance in VALID_STANCES:
        return stance
    return "neutral"


def _normalize_topic_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        return [
            tag.strip()
            for tag in value
            if isinstance(tag, str) and tag.strip()
        ]
    return []


def _resolve_page_or_chunk(result: NormalizedResult) -> str | None:
    page_or_chunk = str(result.get("page_or_chunk") or "").strip()
    return page_or_chunk or None


def _resolve_relevance_score(result: NormalizedResult) -> float | None:
    score = result.get("relevance_score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


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


def _count_distinct_sources(results: list[NormalizedResult]) -> int:
    sources = {
        str(result.get("source_name") or result.get("source") or "").strip()
        for result in results
        if str(result.get("source_name") or result.get("source") or "").strip()
    }
    return len(sources)


def _format_result_digest(results: list[NormalizedResult], *, limit: int) -> str:
    selected_results = _select_representative_results(results, limit=limit)
    if not selected_results:
        return "- 정보 부족"

    lines: list[str] = []
    for result in selected_results:
        source_name = result.get("source_name") or result.get("source") or "출처 미상"
        published_at = result.get("published_at") or "날짜 미상"
        title = _compact_text(
            str(result.get("title") or "Untitled news result"),
            limit=200,
            numeric_limit=260,
        )
        query = _compact_text(
            str(result.get("query") or "질의 미상"),
            limit=180,
            numeric_limit=240,
        )
        topic_tags = result.get("topic_tags", [])
        if isinstance(topic_tags, str):
            topic_tags = [topic_tags]
        tags_text = ", ".join(tag for tag in topic_tags if isinstance(tag, str)) or "없음"
        signal = _compact_text(
            str(result.get("snippet") or result.get("title") or "signal 없음"),
            limit=340,
            numeric_limit=460,
        )
        lines.extend(
            [
                f"- [{source_name} | {published_at}] {title}",
                f"  - query: {query}",
                f"  - tags: {tags_text}",
                f"  - signal: {signal}",
            ]
        )
        page_or_chunk = str(result.get("page_or_chunk") or "").strip()
        relevance_score = result.get("relevance_score")
        if page_or_chunk:
            lines.append(f"  - locator: {page_or_chunk}")
        if isinstance(relevance_score, (int, float)):
            lines.append(f"  - relevance_score: {float(relevance_score):.3f}")

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


def _compact_text(
    value: str,
    *,
    limit: int,
    numeric_limit: int | None = None,
) -> str:
    normalized = " ".join(value.split())
    effective_limit = limit
    if numeric_limit is not None and NUMERIC_PATTERN.search(normalized):
        effective_limit = max(limit, numeric_limit)
    if len(normalized) <= effective_limit:
        return normalized
    return normalized[: effective_limit - 3].rstrip() + "..."
