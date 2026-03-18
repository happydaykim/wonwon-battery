from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import Settings, load_settings
from utils.prompt_loader import load_prompt


DecisionStage = Literal["post_local", "post_merge", "risk_review"]
DecisionAction = Literal["stop", "search_web", "refine"]

RETRIEVAL_DECIDER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", load_prompt("retrieval_decider.md")),
        ("human", "{decision_context}"),
    ]
)

ALLOWED_ACTIONS_BY_STAGE: dict[DecisionStage, tuple[DecisionAction, ...]] = {
    "post_local": ("stop", "search_web"),
    "post_merge": ("stop", "refine"),
    "risk_review": ("stop", "refine"),
}


class RetrievalAssessmentLike(Protocol):
    sufficient: bool
    gaps: list[str]
    evidence_count: int
    source_count: int
    positive_count: int
    risk_count: int
    topic_tags: list[str]


class RetrievalDecisionOutput(BaseModel):
    action: str = Field(description="선택된 retrieval action")
    rationale: str = Field(description="선택 이유")


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    action: DecisionAction
    decision_mode: Literal["llm", "fallback"]
    rationale: str


def decide_retrieval_action(
    *,
    stage: DecisionStage,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    assessment: RetrievalAssessmentLike,
    observed_results: list[dict[str, Any]],
    current_query_policy: dict[str, list[str]],
    query_history: list[str],
    used_web_search: bool,
    refinement_rounds: int,
    refinement_budget: int,
    settings: Settings | None = None,
    risk_only: bool = False,
) -> RetrievalDecision:
    resolved_settings = settings or load_settings()
    if _can_use_llm_decider(resolved_settings):
        try:
            output = _create_decider_chain(resolved_settings).invoke(
                {
                    "decision_context": _build_decision_context(
                        stage=stage,
                        company_scope=company_scope,
                        assessment=assessment,
                        observed_results=observed_results,
                        current_query_policy=current_query_policy,
                        query_history=query_history,
                        used_web_search=used_web_search,
                        refinement_rounds=refinement_rounds,
                        refinement_budget=refinement_budget,
                        risk_only=risk_only,
                    )
                }
            )
            action = _sanitize_action(output.action, stage=stage)
            if action is not None:
                return RetrievalDecision(
                    action=action,
                    decision_mode="llm",
                    rationale=output.rationale.strip() or "LLM selected the next retrieval action.",
                )
        except Exception:
            pass

    return _build_fallback_decision(
        stage=stage,
        assessment=assessment,
        refinement_rounds=refinement_rounds,
        refinement_budget=refinement_budget,
    )


def _create_decider_chain(settings: Settings) -> Any:
    decider_llm = init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=0,
    )
    return RETRIEVAL_DECIDER_PROMPT | decider_llm.with_structured_output(RetrievalDecisionOutput)


def _can_use_llm_decider(settings: Settings) -> bool:
    if settings.llm_provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return True


def _sanitize_action(raw_action: str, *, stage: DecisionStage) -> DecisionAction | None:
    normalized = raw_action.strip().lower()
    if normalized in ALLOWED_ACTIONS_BY_STAGE[stage]:
        return normalized  # type: ignore[return-value]
    return None


def _build_fallback_decision(
    *,
    stage: DecisionStage,
    assessment: RetrievalAssessmentLike,
    refinement_rounds: int,
    refinement_budget: int,
) -> RetrievalDecision:
    if stage == "post_local":
        if assessment.sufficient:
            return RetrievalDecision(
                action="stop",
                decision_mode="fallback",
                rationale="Local retrieval satisfied the current coverage checks.",
            )
        return RetrievalDecision(
            action="search_web",
            decision_mode="fallback",
            rationale="Local retrieval left explicit gaps, so web search is warranted.",
        )

    if assessment.sufficient:
        return RetrievalDecision(
            action="stop",
            decision_mode="fallback",
            rationale="Current retrieval results satisfy the coverage checks.",
        )
    if refinement_rounds >= refinement_budget:
        return RetrievalDecision(
            action="stop",
            decision_mode="fallback",
            rationale="Refinement budget is exhausted.",
        )
    return RetrievalDecision(
        action="refine",
        decision_mode="fallback",
        rationale="Coverage gaps remain and another refined retrieval round is allowed.",
    )


def _build_decision_context(
    *,
    stage: DecisionStage,
    company_scope: Literal["MARKET", "LGES", "CATL"],
    assessment: RetrievalAssessmentLike,
    observed_results: list[dict[str, Any]],
    current_query_policy: dict[str, list[str]],
    query_history: list[str],
    used_web_search: bool,
    refinement_rounds: int,
    refinement_budget: int,
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
            f"stage={stage}",
            f"company_scope={company_scope}",
            f"risk_only={risk_only}",
            f"used_web_search={used_web_search}",
            f"refinement_rounds={refinement_rounds}",
            f"refinement_budget={refinement_budget}",
            f"sufficient={assessment.sufficient}",
            f"evidence_count={assessment.evidence_count}",
            f"source_count={assessment.source_count}",
            f"positive_count={assessment.positive_count}",
            f"risk_count={assessment.risk_count}",
            f"topic_tags={assessment.topic_tags}",
            "[remaining_gaps]",
            *(assessment.gaps or ["none"]),
            "[current_positive_queries]",
            *current_query_policy.get("positive_queries", []),
            "[current_risk_queries]",
            *current_query_policy.get("risk_queries", []),
            "[query_history]",
            *(query_history or ["none"]),
            "[observed_results_preview]",
            *(preview_lines or ["- 관측 결과 없음"]),
            "[allowed_actions]",
            *ALLOWED_ACTIONS_BY_STAGE[stage],
        ]
    )
