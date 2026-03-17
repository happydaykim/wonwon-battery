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
    """Prepare counter-evidence review for LGES and CATL in sequence."""
    current_phase = state["runtime"]["current_phase"]

    if current_phase == "skeptic_lges":
        company = "LGES"
        next_phase = "skeptic_catl"
    elif current_phase == "skeptic_catl":
        company = "CATL"
        next_phase = "compare"
    else:
        raise ValueError(f"Unsupported skeptic phase: {current_phase}")

    query_policy = build_company_query_policy(company)
    message = build_agent_message(
        SKEPTIC_BLUEPRINT.name,
        (
            f"Prepared skeptic review for {company} with "
            f"{len(query_policy['risk_queries'])} risk queries. "
            "TODO: implement counter-evidence retrieval and contradiction checks."
        ),
    )

    return {
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": next_phase,
        },
    }
