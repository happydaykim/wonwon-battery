from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.base import build_agent_message, create_agent_blueprint
from config.settings import load_settings
from schemas.state import ReportState
from utils.logging import get_logger


# Planner Node: convert the user goal into a stable section plan.
PLANNER_BLUEPRINT = create_agent_blueprint(
    name="planner_node",
    prompt_name="planner.md",
)

logger = get_logger(__name__)


class PlannerOutput(BaseModel):
    """Ordered execution plan for the battery strategy report."""

    steps: list[str] = Field(description="실행할 단계 목록")


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_BLUEPRINT.system_prompt),
        ("human", "{user_query}"),
    ]
)


DEFAULT_REPORT_PLAN = [
    "시장 배경 분석 범위를 정의하고 핵심 조사 축을 정리한다.",
    "로컬 RAG 우선 원칙에 따라 EV 캐즘, HEV 피벗, ESS/로봇/원가 전략 관련 시장 근거를 수집한다.",
    "LGES의 포트폴리오 다각화 전략, 핵심 경쟁력, 리스크 근거를 수집한다.",
    "CATL의 포트폴리오 다각화 전략, 핵심 경쟁력, 리스크 근거를 수집한다.",
    "양사에 대해 skeptic 단계로 반대 근거와 제약 조건을 재점검한다.",
    "LGES와 CATL을 동일 프레임으로 비교하고 SWOT 구조를 정리한다.",
    "SUMMARY와 REFERENCE를 포함한 보고서 초안을 작성하고 validator 기준으로 수정 여부를 점검한다.",
]


def _create_planner_chain() -> Any:
    """Create the planner runnable using the configured chat model."""
    settings = load_settings()
    planner_llm = init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=0,
    )
    return PLANNER_PROMPT | planner_llm.with_structured_output(PlannerOutput)


def _build_fallback_plan(user_query: str) -> list[str]:
    """Return a deterministic fallback plan when LLM planning is unavailable."""
    _ = user_query
    return DEFAULT_REPORT_PLAN.copy()


def _generate_plan(user_query: str) -> tuple[list[str], str]:
    """Generate a plan via LLM and fall back safely when unavailable."""
    try:
        planner_chain = _create_planner_chain()
        output = planner_chain.invoke({"user_query": user_query})
        steps = [step.strip() for step in output.steps if step.strip()]
        if not steps:
            raise ValueError("Planner returned an empty steps list.")
        return steps, "llm"
    except Exception as exc:  # pragma: no cover - depends on runtime credentials/network
        logger.warning("Planner LLM unavailable. Falling back to default plan: %s", exc)
        return _build_fallback_plan(user_query), "fallback"


def planner_node(state: ReportState) -> dict:
    """Create a section plan and hand off to the retrieval phase."""
    next_plan, plan_source = _generate_plan(state["user_query"])
    message = build_agent_message(
        PLANNER_BLUEPRINT.name,
        (
            f"Section plan prepared via {plan_source} planner with "
            f"{len(next_plan)} ordered steps."
        ),
    )

    return {
        "plan": next_plan,
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": "retrieve_market",
        },
    }
