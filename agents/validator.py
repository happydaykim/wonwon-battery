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
                    "retryable": True,
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
                    "retryable": True,
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
                "suggested_action": "Compile drafted section outputs into the final report.",
                "retryable": True,
            }
        )

    if not state["references"]:
        issues.append(
            {
                "issue_id": "missing_references",
                "section_id": "references",
                "severity": "warning",
                "message": (
                    "Reference entries are missing even though source documents exist."
                    if state["documents"]
                    else "No verifiable references were collected from the current retrieval path."
                ),
                "related_evidence_ids": [],
                "suggested_action": (
                    "Assemble references from the existing documents."
                    if state["documents"]
                    else "Keep the limitation visible in the report and avoid unsupported claims."
                ),
                "retryable": bool(state["documents"]),
            }
        )

    issues.extend(_build_retrieval_gap_issues(state))
    return issues


def validator_node(state: ReportState) -> dict:
    """Validate draft completeness and control the revision loop."""
    issues = _build_validation_issues(state)
    retryable_issues = [issue for issue in issues if issue["retryable"]]
    non_retryable_issues = [issue for issue in issues if not issue["retryable"]]
    runtime = {**state["runtime"]}
    remaining_plan = (
        state["plan"][1:] if state["plan"] and state["plan"][0] == "validate" else state["plan"]
    )

    if not issues:
        runtime["current_phase"] = "done"
        runtime["termination_reason"] = "validated"
        next_plan = remaining_plan
        note = "Validation passed with no remaining issues."
    elif retryable_issues and has_revision_budget(state):
        runtime["revision_count"] += 1
        runtime["current_phase"] = "write"
        runtime["termination_reason"] = None
        next_plan = ["write", "validate", *remaining_plan]
        note = (
            "Validation found retryable issues. "
            f"Scheduling another writer pass ({len(retryable_issues)} retryable / "
            f"{len(non_retryable_issues)} non-retryable)."
        )
    else:
        runtime["current_phase"] = "done"
        next_plan = remaining_plan
        if retryable_issues:
            runtime["termination_reason"] = "max_revisions_reached"
            note = (
                "Validation stopped because retryable issues remain but the maximum "
                "revision budget has been reached."
            )
        else:
            runtime["termination_reason"] = "done_with_gaps"
            note = (
                "Validation finished with non-retryable information gaps. "
                "The report is being finalized with explicit caveats."
            )

    message = build_agent_message(VALIDATOR_BLUEPRINT.name, note)
    return {
        "plan": next_plan,
        "validation_issues": issues,
        "messages": state["messages"] + [message],
        "runtime": runtime,
    }


def _build_retrieval_gap_issues(state: ReportState) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not state["market"]["retrieval_sufficient"]:
        issues.append(
            {
                "issue_id": "market_retrieval_gap",
                "section_id": "market_background",
                "severity": "warning",
                "message": "Market background evidence is still insufficient.",
                "related_evidence_ids": state["market"]["evidence_ids"],
                "suggested_action": _build_gap_action(state["market"]["retrieval_gaps"]),
                "retryable": False,
            }
        )

    for company, section_id in (("LGES", "lges_strategy"), ("CATL", "catl_strategy")):
        company_state = state["companies"][company]
        if not company_state["retrieval_sufficient"]:
            issues.append(
                {
                    "issue_id": f"{company.lower()}_retrieval_gap",
                    "section_id": section_id,
                    "severity": "warning",
                    "message": f"{company} evidence remains insufficient after retrieval/skeptic flow.",
                    "related_evidence_ids": company_state["evidence_ids"],
                    "suggested_action": _build_gap_action(company_state["retrieval_gaps"]),
                    "retryable": False,
                }
            )

        if company_state["skeptic_review_required"] and not company_state["skeptic_review_completed"]:
            issues.append(
                {
                    "issue_id": f"{company.lower()}_skeptic_pending",
                    "section_id": section_id,
                    "severity": "error",
                    "message": f"{company} required skeptic review but it did not complete.",
                    "related_evidence_ids": company_state["evidence_ids"],
                    "suggested_action": "Run the skeptic step before finalizing the report.",
                    "retryable": False,
                }
            )

        if (
            company_state["skeptic_review_required"]
            and company_state["skeptic_review_completed"]
            and not company_state["counter_evidence_ids"]
        ):
            issues.append(
                {
                    "issue_id": f"{company.lower()}_counter_evidence_gap",
                    "section_id": "swot",
                    "severity": "warning",
                    "message": f"{company} skeptic review did not recover counter-evidence.",
                    "related_evidence_ids": company_state["evidence_ids"],
                    "suggested_action": "Keep weakness/threat language conservative and note the missing risk coverage.",
                    "retryable": False,
                }
            )

    return issues


def _build_gap_action(gaps: list[str]) -> str:
    if not gaps:
        return "Preserve the limitation in the report and avoid unsupported claims."
    return "Preserve the limitation in the report: " + "; ".join(gaps)
