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


ALLOWED_PLAN_STEPS = {
    "parallel_retrieval",
    "skeptic_lges",
    "skeptic_catl",
    "compare",
    "write",
    "validate",
}

DEFAULT_EXECUTION_PLAN = [
    "parallel_retrieval",
    "skeptic_lges",
    "skeptic_catl",
    "compare",
    "write",
    "validate",
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
    return DEFAULT_EXECUTION_PLAN.copy()


def _sanitize_plan_steps(raw_steps: list[str]) -> list[str]:
    """Keep only supported broad steps and preserve the canonical execution order."""
    requested_steps = {step.strip() for step in raw_steps if step.strip() in ALLOWED_PLAN_STEPS}
    ordered_steps = [step for step in DEFAULT_EXECUTION_PLAN if step in requested_steps]
    return ordered_steps or DEFAULT_EXECUTION_PLAN.copy()


def _generate_plan(user_query: str) -> tuple[list[str], str]:
    """Generate a plan via LLM and fall back safely when unavailable."""
    try:
        planner_chain = _create_planner_chain()
        output = planner_chain.invoke({"user_query": user_query})
        steps = _sanitize_plan_steps(output.steps)
        if not steps:
            raise ValueError("Planner returned an empty steps list.")
        return steps, "llm"
    except Exception as exc:  # pragma: no cover - depends on runtime credentials/network
        logger.warning("Planner LLM unavailable. Falling back to default plan: %s", exc)
        return _build_fallback_plan(user_query), "fallback"


def planner_node(state: ReportState) -> dict:
    """Create an execution queue and hand off control to the supervisor."""
    next_plan, plan_source = _generate_plan(state["user_query"])
    message = build_agent_message(
        PLANNER_BLUEPRINT.name,
        (
            f"Execution plan prepared via {plan_source} planner with "
            f"{len(next_plan)} broad steps: {', '.join(next_plan)}."
        ),
    )

    return {
        "plan": next_plan,
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": "plan",
        },
    }
