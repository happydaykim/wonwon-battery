from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from schemas.state import ReportState


# Supervisor Agent: decide which specialist should run next.
SUPERVISOR_BLUEPRINT = create_agent_blueprint(
    name="supervisor_agent",
    prompt_name="supervisor.md",
)

STEP_TO_PHASE_MAP = {
    "parallel_retrieval": "retrieve_market",
    "skeptic_lges": "skeptic_lges",
    "skeptic_catl": "skeptic_catl",
    "compare": "compare",
    "write": "write",
    "validate": "validate",
}


def supervisor_node(state: ReportState) -> dict:
    """Interpret the planner queue and prepare the next execution phase."""
    runtime = {**state["runtime"]}
    plan = state["plan"].copy()
    current_step = plan[0] if plan else None

    if current_step == "parallel_retrieval" and _parallel_retrieval_completed(state):
        plan = plan[1:]
        current_step = plan[0] if plan else None
        note = "Parallel retrieval bundle completed and the next execution step was prepared."
    elif current_step is None:
        runtime["current_phase"] = "done"
        note = "Planner queue is empty. Supervisor is marking the workflow as done."
    else:
        note = f"Supervisor prepared the next execution step: '{current_step}'."

    if current_step is not None:
        runtime["current_phase"] = STEP_TO_PHASE_MAP[current_step]

    message = build_agent_message(
        SUPERVISOR_BLUEPRINT.name,
        note,
    )
    return {
        "plan": plan,
        "messages": state["messages"] + [message],
        "runtime": runtime,
    }


def _parallel_retrieval_completed(state: ReportState) -> bool:
    """Check whether market/LGES/CATL retrieval bundle has finished."""
    current_step = state["plan"][0] if state["plan"] else None
    if current_step != "parallel_retrieval":
        return False

    market_ready = state["market"]["synthesized_summary"] is not None
    lges_ready = state["companies"]["LGES"]["synthesized_summary"] is not None
    catl_ready = state["companies"]["CATL"]["synthesized_summary"] is not None
    return market_ready and lges_ready and catl_ready
