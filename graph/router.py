from __future__ import annotations

from typing import Final

from schemas.state import ReportState


SUPERVISOR_ROUTE_MAP: Final[dict[str, str]] = {
    "parallel_retrieval": "parallel_retrieval_dispatch",
    "skeptic_lges": "skeptic_agent",
    "skeptic_catl": "skeptic_agent",
    "compare": "compare_swot_agent",
    "write": "writer_agent",
    "validate": "validator_agent",
}


def route_supervisor(state: ReportState) -> str:
    """Route the supervisor to the next specialist node."""
    if state["runtime"]["current_phase"] == "done" or not state["plan"]:
        return "done"

    current_step = state["plan"][0]
    if current_step not in SUPERVISOR_ROUTE_MAP:
        raise ValueError(f"Unsupported supervisor step: {current_step}")
    return SUPERVISOR_ROUTE_MAP[current_step]


def has_revision_budget(state: ReportState) -> bool:
    """Check whether the graph can schedule another revision attempt."""
    runtime = state["runtime"]
    return runtime["revision_count"] < runtime["max_revisions"]


def should_retry_revision(state: ReportState) -> bool:
    """Check whether validator output should branch back into the graph."""
    return state["runtime"]["current_phase"] != "done" and bool(state["plan"])


def route_validator(state: ReportState) -> str:
    """Branch to END on pass or back to Supervisor on revise."""
    return "revise" if should_retry_revision(state) else "pass"
