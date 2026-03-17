from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from schemas.state import ReportState


# Writer Agent: organize the final report structure and placeholders.
WRITER_BLUEPRINT = create_agent_blueprint(
    name="writer_agent",
    prompt_name="writer.md",
)


def writer_node(state: ReportState) -> dict:
    """Move the report state into validation without generating fake content."""
    message = build_agent_message(
        WRITER_BLUEPRINT.name,
        (
            "Writer stage reached. TODO: generate section drafts and final report text "
            "after evidence and citation logic are implemented."
        ),
    )

    return {
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": "validate",
        },
    }
