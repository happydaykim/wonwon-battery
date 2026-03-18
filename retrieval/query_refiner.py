from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import Settings, load_settings
from utils.prompt_loader import load_prompt


QUERY_REFINER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", load_prompt("query_refiner.md")),
        ("human", "{refinement_context}"),
    ]
)


class QueryRefinerOutput(BaseModel):
    positive_queries: list[str] = Field(description="추가 positive 검색 질의")
    risk_queries: list[str] = Field(description="추가 risk 검색 질의")


@dataclass(frozen=True, slots=True)
class QueryRefinementResult:
    positive_queries: list[str]
    risk_queries: list[str]
    refinement_mode: Literal["llm", "fallback", "skipped"]
    rationale: str


class RetrievalAssessmentLike(Protocol):
    gaps: list[str]


def refine_query_policy(
    *,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    current_query_policy: dict[str, list[str]],
    assessment: RetrievalAssessmentLike,
    observed_results: list[dict[str, Any]],
    used_queries: set[str],
    max_new_queries_per_bucket: int,
    settings: Settings | None = None,
    risk_only: bool = False,
) -> QueryRefinementResult:
    if not assessment.gaps or max_new_queries_per_bucket <= 0:
        return QueryRefinementResult(
            positive_queries=[],
            risk_queries=[],
            refinement_mode="skipped",
            rationale="No remaining retrieval gaps or refinement budget.",
        )

    resolved_settings = settings or load_settings()
    if _can_use_llm_refinement(resolved_settings):
        try:
            output = _create_query_refiner_chain(resolved_settings).invoke(
                {
                    "refinement_context": _build_refinement_context(
                        company_scope=company_scope,
                        current_query_policy=current_query_policy,
                        assessment=assessment,
                        observed_results=observed_results,
                        used_queries=used_queries,
                        risk_only=risk_only,
                    )
                }
            )
            positive_queries, risk_queries = _sanitize_queries(
                output=output,
                used_queries=used_queries,
                max_new_queries_per_bucket=max_new_queries_per_bucket,
                risk_only=risk_only,
            )
            if positive_queries or risk_queries:
                return QueryRefinementResult(
                    positive_queries=positive_queries,
                    risk_queries=risk_queries,
                    refinement_mode="llm",
                    rationale="Refined via llm gap analysis.",
                )
        except Exception:
            pass

    positive_queries, risk_queries = _build_fallback_queries(
        company_scope=company_scope,
        assessment=assessment,
        used_queries=used_queries,
        max_new_queries_per_bucket=max_new_queries_per_bucket,
        risk_only=risk_only,
    )
    if positive_queries or risk_queries:
        return QueryRefinementResult(
            positive_queries=positive_queries,
            risk_queries=risk_queries,
            refinement_mode="fallback",
            rationale="Refined via deterministic gap heuristics.",
        )

    return QueryRefinementResult(
        positive_queries=[],
        risk_queries=[],
        refinement_mode="skipped",
        rationale="No new refinement queries were available.",
    )


def _create_query_refiner_chain(settings: Settings) -> Any:
    query_refiner_llm = init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=0,
    )
    return QUERY_REFINER_PROMPT | query_refiner_llm.with_structured_output(QueryRefinerOutput)


def _can_use_llm_refinement(settings: Settings) -> bool:
    if settings.llm_provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return True


def _build_refinement_context(
    *,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    current_query_policy: dict[str, list[str]],
    assessment: RetrievalAssessmentLike,
    observed_results: list[dict[str, Any]],
    used_queries: set[str],
    risk_only: bool,
) -> str:
    preview_lines: list[str] = []
    for result in observed_results[:6]:
        preview_lines.append(
            "- "
            + " | ".join(
                [
                    str(result.get("source_name") or result.get("source") or "출처 미상"),
                    str(result.get("published_at") or "날짜 미상"),
                    str(result.get("title") or "Untitled"),
                    str(result.get("query") or "질의 미상"),
                ]
            )
        )

    return "\n".join(
        [
            f"company_scope={company_scope}",
            f"risk_only={risk_only}",
            "[current_positive_queries]",
            *current_query_policy.get("positive_queries", []),
            "[current_risk_queries]",
            *current_query_policy.get("risk_queries", []),
            "[used_queries]",
            *sorted(used_queries),
            "[remaining_gaps]",
            *assessment.gaps,
            "[observed_results_preview]",
            *(preview_lines or ["- 관측 결과 없음"]),
        ]
    )


def _sanitize_queries(
    *,
    output: QueryRefinerOutput,
    used_queries: set[str],
    max_new_queries_per_bucket: int,
    risk_only: bool,
) -> tuple[list[str], list[str]]:
    positive_queries = _sanitize_query_bucket(
        [] if risk_only else output.positive_queries,
        used_queries=used_queries,
        limit=max_new_queries_per_bucket,
    )
    risk_queries = _sanitize_query_bucket(
        output.risk_queries,
        used_queries=used_queries,
        limit=max_new_queries_per_bucket,
    )
    return positive_queries, risk_queries


def _sanitize_query_bucket(
    queries: list[str],
    *,
    used_queries: set[str],
    limit: int,
) -> list[str]:
    sanitized: list[str] = []
    seen: set[str] = set()
    for raw_query in queries:
        query = " ".join(raw_query.split())
        if not query:
            continue
        normalized = query.lower()
        if normalized in seen or normalized in used_queries:
            continue
        sanitized.append(query)
        seen.add(normalized)
        if len(sanitized) >= limit:
            break
    return sanitized


def _build_fallback_queries(
    *,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    assessment: RetrievalAssessmentLike,
    used_queries: set[str],
    max_new_queries_per_bucket: int,
    risk_only: bool,
) -> tuple[list[str], list[str]]:
    scope_label = "배터리 시장" if company_scope == "MARKET" else ("LG에너지솔루션" if company_scope == "LGES" else "CATL")
    missing_topics = _extract_missing_topics(assessment.gaps)
    positive_candidates: list[str] = []
    risk_candidates: list[str] = []

    if company_scope == "MARKET":
        if "market_structure" in missing_topics:
            positive_candidates.append("전기차 캐즘 HEV ESS 시장 구조 변화")
        if "demand" in missing_topics:
            positive_candidates.append("배터리 HEV ESS 로봇 수요 전환")
        if "risk" in missing_topics:
            risk_candidates.append("전기차 캐즘 배터리 수요 둔화 리스크")
    else:
        if "strategy" in missing_topics:
            positive_candidates.append(f"{scope_label} 포트폴리오 다각화 전략")
        if "expansion" in missing_topics:
            positive_candidates.append(f"{scope_label} ESS HEV 로봇 신규사업 확장")
        if "risk" in missing_topics:
            risk_candidates.append(f"{scope_label} 수익성 압박 경쟁 리스크")

    gap_text = " ".join(assessment.gaps)
    if "source_diversity" in gap_text:
        if company_scope == "MARKET":
            positive_candidates.append("battery industry report EV HEV ESS shift")
        else:
            positive_candidates.append(f"{scope_label} annual report diversification")
            risk_candidates.append(f"{scope_label} earnings profitability risk")

    if "stance_balance: missing positive" in gap_text and company_scope != "MARKET":
        positive_candidates.append(f"{scope_label} 성장 전략 핵심 경쟁력")
    if "stance_balance: missing risk" in gap_text:
        risk_candidates.append(f"{scope_label} 리스크 제약 수익성 압박")

    if "evidence_count" in gap_text and company_scope == "MARKET":
        positive_candidates.append("글로벌 배터리 시장 포트폴리오 다변화")
    if "evidence_count" in gap_text and company_scope != "MARKET":
        positive_candidates.append(f"{scope_label} 배터리 전략 사업 다각화")

    positive_queries = _sanitize_query_bucket(
        [] if risk_only else positive_candidates,
        used_queries=used_queries,
        limit=max_new_queries_per_bucket,
    )
    risk_queries = _sanitize_query_bucket(
        risk_candidates,
        used_queries=used_queries,
        limit=max_new_queries_per_bucket,
    )
    return positive_queries, risk_queries


def _extract_missing_topics(gaps: list[str]) -> set[str]:
    missing_topics: set[str] = set()
    for gap in gaps:
        if not gap.startswith("required_topics: missing "):
            continue
        tail = gap.removeprefix("required_topics: missing ").rstrip(".")
        for topic in tail.split(","):
            normalized = topic.strip()
            if normalized:
                missing_topics.add(normalized)
    return missing_topics
