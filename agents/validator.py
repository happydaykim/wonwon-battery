from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from graph.router import has_revision_budget
from schemas.state import ReportState, ValidationIssue


# Validator Agent: inspect completeness and manage the revision loop.
VALIDATOR_BLUEPRINT = create_agent_blueprint(
    name="validator_agent",
    prompt_name="validator.md",
)

REQUIRED_SECTION_IDS = (
    "summary",
    "market_background",
    "lges_strategy",
    "catl_strategy",
    "strategy_comparison",
    "swot",
    "implications",
    "references",
)


def _build_validation_issues(state: ReportState) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for section_id in REQUIRED_SECTION_IDS:
        section = state["section_drafts"].get(section_id)
        if section is None:
            issues.append(
                {
                    "issue_id": f"missing_section_{section_id}",
                    "section_id": section_id,
                    "severity": "error",
                    "message": "Required section draft is missing from section_drafts.",
                    "related_evidence_ids": [],
                    "suggested_action": "Add the section scaffold before report generation.",
                }
            )
            continue

        if section["status"] not in {"drafted", "approved"}:
            issues.append(
                {
                    "issue_id": f"undrafted_section_{section_id}",
                    "section_id": section_id,
                    "severity": "error",
                    "message": "Required section is not drafted yet.",
                    "related_evidence_ids": section["evidence_ids"],
                    "suggested_action": "Generate or revise this section with evidence links.",
                }
            )

    if not state["final_report"]:
        issues.append(
            {
                "issue_id": "missing_final_report",
                "section_id": "summary",
                "severity": "error",
                "message": "final_report is empty.",
                "related_evidence_ids": [],
                "suggested_action": "Compile approved section drafts into the final report.",
            }
        )

    if not state["references"]:
        issues.append(
            {
                "issue_id": "missing_references",
                "section_id": "references",
                "severity": "warning",
                "message": "No reference entries are registered yet.",
                "related_evidence_ids": [],
                "suggested_action": "Add citation entries once evidence extraction is implemented.",
            }
        )

    return issues


def validator_node(state: ReportState) -> dict:
    """Validate draft completeness and control the revision loop."""
    issues = _build_validation_issues(state)
    runtime = {**state["runtime"]}
    remaining_plan = (
        state["plan"][1:] if state["plan"] and state["plan"][0] == "validate" else state["plan"]
    )

    if issues and has_revision_budget(state):
        runtime["revision_count"] += 1
        runtime["current_phase"] = "write"
        next_plan = ["write", "validate", *remaining_plan]
        note = "Validation found issues. Routing back to Writer for another revision attempt."
    else:
        runtime["current_phase"] = "done"
        next_plan = remaining_plan
        note = "Validation finished. TODO: finalize success/failure reporting with real criteria."

    message = build_agent_message(VALIDATOR_BLUEPRINT.name, note)
    return {
        "plan": next_plan,
        "validation_issues": issues,
        "messages": state["messages"] + [message],
        "runtime": runtime,
    }
