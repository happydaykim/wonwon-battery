from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from retrieval.query_policy import build_company_query_policy
from schemas.state import ReportState


# Skeptic Agent: force counter-evidence and risk review for each company.
SKEPTIC_BLUEPRINT = create_agent_blueprint(
    name="skeptic_agent",
    prompt_name="skeptic.md",
    tools=["balanced_web_search"],
)


def skeptic_node(state: ReportState) -> dict:
    """Prepare counter-evidence review for the current company step."""
    current_step = state["plan"][0] if state["plan"] else None

    if current_step == "skeptic_lges":
        company = "LGES"
    elif current_step == "skeptic_catl":
        company = "CATL"
    else:
        raise ValueError(f"Unsupported skeptic step: {current_step}")

    query_policy = build_company_query_policy(company)
    remaining_plan = state["plan"][1:]
    next_step = remaining_plan[0] if remaining_plan else None
    message = build_agent_message(
        SKEPTIC_BLUEPRINT.name,
        (
            f"Prepared skeptic review for {company} with "
            f"{len(query_policy['risk_queries'])} risk queries. "
            "TODO: implement counter-evidence retrieval and contradiction checks."
        ),
    )

    return {
        "plan": remaining_plan,
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": next_step or "done",
        },
    }
