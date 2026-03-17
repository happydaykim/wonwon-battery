from __future__ import annotations

from typing import Final

from schemas.state import ReportState


SUPERVISOR_ROUTE_MAP: Final[dict[str, str]] = {
    "retrieve_market": "market_agent",
    "retrieve_lges": "lges_agent",
    "retrieve_catl": "catl_agent",
    "skeptic_lges": "skeptic_agent",
    "skeptic_catl": "skeptic_agent",
    "compare": "compare_swot_agent",
    "write": "writer_agent",
    "validate": "validator_agent",
}


def route_supervisor(state: ReportState) -> str:
    """Route the supervisor to the next specialist node."""
    phase = state["runtime"]["current_phase"]
    if phase not in SUPERVISOR_ROUTE_MAP:
        raise ValueError(f"Unsupported supervisor phase: {phase}")
    return SUPERVISOR_ROUTE_MAP[phase]


def has_revision_budget(state: ReportState) -> bool:
    """Check whether the graph can schedule another revision attempt."""
    runtime = state["runtime"]
    return runtime["revision_count"] < runtime["max_revisions"]


def should_retry_revision(state: ReportState) -> bool:
    """Check whether validator output should branch back into the graph."""
    runtime = state["runtime"]
    return (
        bool(state["validation_issues"])
        and runtime["current_phase"] != "done"
        and runtime["revision_count"] <= runtime["max_revisions"]
    )


def route_validator(state: ReportState) -> str:
    """Branch to END on pass or back to Supervisor on revise."""
    return "revise" if should_retry_revision(state) else "pass"
