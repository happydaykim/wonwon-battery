from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from schemas.state import ReportState


# Supervisor Agent: decide which specialist should run next.
SUPERVISOR_BLUEPRINT = create_agent_blueprint(
    name="supervisor_agent",
    prompt_name="supervisor.md",
)


def supervisor_node(state: ReportState) -> dict:
    """Record the current handoff point for the next routed specialist."""
    phase = state["runtime"]["current_phase"]
    message = build_agent_message(
        SUPERVISOR_BLUEPRINT.name,
        f"Supervisor reviewed phase '{phase}'. TODO: connect real handoff logic.",
    )
    return {
        "messages": state["messages"] + [message],
    }
