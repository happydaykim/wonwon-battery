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
    """Re-check existing company evidence from a skeptic perspective."""
    current_step = state["plan"][0] if state["plan"] else None

    if current_step == "skeptic_lges":
        company = "LGES"
    elif current_step == "skeptic_catl":
        company = "CATL"
    else:
        raise ValueError(f"Unsupported skeptic step: {current_step}")

    query_policy = build_company_query_policy(company)
    company_state = state["companies"][company]
    risk_evidence_ids = [
        evidence_id
        for evidence_id in company_state["evidence_ids"]
        if _is_risk_evidence(state, evidence_id)
    ]
    source_names = {
        _lookup_source_name(state, evidence_id)
        for evidence_id in company_state["evidence_ids"]
        if _lookup_source_name(state, evidence_id)
    }
    risk_source_names = {
        _lookup_source_name(state, evidence_id)
        for evidence_id in risk_evidence_ids
        if _lookup_source_name(state, evidence_id)
    }
    remaining_plan = state["plan"][1:]
    next_step = remaining_plan[0] if remaining_plan else None
    message = build_agent_message(
        SKEPTIC_BLUEPRINT.name,
        _build_skeptic_note(
            company=company,
            risk_query_count=len(query_policy["risk_queries"]),
            risk_evidence_count=len(risk_evidence_ids),
            total_source_count=len(source_names),
            risk_source_count=len(risk_source_names),
        ),
    )

    return {
        "plan": remaining_plan,
        "companies": {
            company: {
                **company_state,
                "counter_evidence_ids": risk_evidence_ids,
            }
        },
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": next_step or "done",
        },
    }


def _is_risk_evidence(state: ReportState, evidence_id: str) -> bool:
    evidence = state["evidence"].get(evidence_id)
    if evidence is None:
        return False

    document = state["documents"].get(evidence["doc_id"])
    if document is None:
        return False

    return document["stance"] == "risk"


def _lookup_source_name(state: ReportState, evidence_id: str) -> str | None:
    evidence = state["evidence"].get(evidence_id)
    if evidence is None:
        return None

    document = state["documents"].get(evidence["doc_id"])
    if document is None:
        return None

    return document["source_name"]


def _build_skeptic_note(
    *,
    company: str,
    risk_query_count: int,
    risk_evidence_count: int,
    total_source_count: int,
    risk_source_count: int,
) -> str:
    if risk_evidence_count == 0:
        return (
            f"Skeptic reviewed {company} with {risk_query_count} risk queries and found "
            "no risk-tagged evidence in the current retrieval set. Additional re-check is needed."
        )

    if risk_source_count < 2:
        return (
            f"Skeptic reviewed {company} with {risk_query_count} risk queries and found "
            f"{risk_evidence_count} risk items, but they are concentrated in {risk_source_count} source."
        )

    return (
        f"Skeptic reviewed {company} with {risk_query_count} risk queries and identified "
        f"{risk_evidence_count} counter-evidence items across {total_source_count} total sources."
    )
