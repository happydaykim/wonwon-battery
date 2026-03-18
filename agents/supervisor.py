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

POST_RETRIEVAL_TAIL = ("compare", "write", "validate")


def supervisor_node(state: ReportState) -> dict:
    """Interpret the planner queue and prepare the next execution phase."""
    runtime = {**state["runtime"]}
    plan = state["plan"].copy()
    companies = {name: {**company_state} for name, company_state in state["companies"].items()}
    current_step = plan[0] if plan else None

    if current_step == "parallel_retrieval" and _parallel_retrieval_completed(state):
        plan = _rewrite_post_retrieval_plan(state, plan[1:])
        current_step = plan[0] if plan else None
        companies = _sync_skeptic_requirements(companies)
        note = _build_post_retrieval_note(state, plan)
    elif current_step is None:
        runtime["current_phase"] = "done"
        note = "Planner queue is empty. Supervisor is marking the workflow as done."
    else:
        note = f"Supervisor prepared the next execution step: '{current_step}'."

    if current_step is not None:
        runtime["current_phase"] = STEP_TO_PHASE_MAP[current_step]
        runtime["termination_reason"] = None

    message = build_agent_message(
        SUPERVISOR_BLUEPRINT.name,
        note,
    )
    return {
        "plan": plan,
        "companies": companies,
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


def _rewrite_post_retrieval_plan(state: ReportState, remaining_plan: list[str]) -> list[str]:
    tail = [step for step in POST_RETRIEVAL_TAIL if step in remaining_plan] or list(POST_RETRIEVAL_TAIL)
    skeptic_steps: list[str] = []

    for company, skeptic_step in (("LGES", "skeptic_lges"), ("CATL", "skeptic_catl")):
        if not state["companies"][company]["retrieval_sufficient"]:
            skeptic_steps.append(skeptic_step)

    return [*skeptic_steps, *tail]


def _sync_skeptic_requirements(
    companies: dict[str, dict],
) -> dict[str, dict]:
    updated_companies = {name: {**company_state} for name, company_state in companies.items()}

    for company_name, company_state in updated_companies.items():
        requires_skeptic = not company_state["retrieval_sufficient"]
        company_state["skeptic_review_required"] = requires_skeptic
        if not requires_skeptic:
            company_state["skeptic_review_completed"] = False
            company_state["counter_evidence_ids"] = []

    return updated_companies


def _build_post_retrieval_note(state: ReportState, plan: list[str]) -> str:
    company_notes = []
    for company in ("LGES", "CATL"):
        company_state = state["companies"][company]
        status = "sufficient" if company_state["retrieval_sufficient"] else "needs_skeptic_review"
        company_notes.append(f"{company}={status}")

    market_note = (
        "MARKET=sufficient"
        if state["market"]["retrieval_sufficient"]
        else "MARKET=insufficient"
    )
    return (
        "Parallel retrieval bundle completed. "
        f"Post-retrieval branch prepared with {market_note}, "
        f"{', '.join(company_notes)}. "
        f"Next queue: {', '.join(plan) if plan else 'done'}."
    )
