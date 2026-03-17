from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from schemas.state import ReportState


# Planner Node: convert the user goal into a stable section plan.
PLANNER_BLUEPRINT = create_agent_blueprint(
    name="planner_node",
    prompt_name="planner.md",
)

DEFAULT_REPORT_PLAN = [
    "SUMMARY",
    "시장 배경",
    "LGES 전략 분석",
    "CATL 전략 분석",
    "핵심 전략 비교 및 SWOT",
    "종합 시사점",
    "REFERENCE",
]


def planner_node(state: ReportState) -> dict:
    """Create a section plan and hand off to the retrieval phase."""
    next_plan = state["plan"] or DEFAULT_REPORT_PLAN.copy()
    message = build_agent_message(
        PLANNER_BLUEPRINT.name,
        "Section plan prepared. TODO: replace the static plan with prompt-driven planning.",
    )

    return {
        "plan": next_plan,
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": "retrieve_market",
        },
    }
