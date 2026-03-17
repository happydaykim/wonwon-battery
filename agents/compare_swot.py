from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from schemas.state import ReportState


# Compare / SWOT Agent: align both companies on a shared comparison frame.
COMPARE_SWOT_BLUEPRINT = create_agent_blueprint(
    name="compare_swot_agent",
    prompt_name="compare_swot.md",
)


def compare_swot_node(state: ReportState) -> dict:
    """Prepare comparison and SWOT placeholders before report drafting."""
    message = build_agent_message(
        COMPARE_SWOT_BLUEPRINT.name,
        (
            "Comparison and SWOT stage reached. "
            "TODO: synthesize aligned comparison points and populate SWOT entries."
        ),
    )

    return {
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": "write",
        },
    }
