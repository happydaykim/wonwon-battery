from __future__ import annotations

from agents.base import build_agent_message
from graph.router import has_revision_budget
from schemas.state import ReportState, ValidationIssue

VALIDATOR_AGENT_NAME = "validator_agent"

SUMMARY_MAX_CHARS = 900
SUMMARY_MIN_CHARS = 150
FINAL_REPORT_REQUIRED_HEADINGS = (
    "## I. EXECUTIVE SUMMARY",
    "## II. 시장 배경",
    "## III. LG에너지솔루션의 포트폴리오 다각화 전략과 핵심 경쟁력",
    "## IV. CATL의 포트폴리오 다각화 전략과 핵심 경쟁력",
    "## V. 핵심 전략 비교 분석",
    "## V.III SWOT 분석",
    "## VI. 종합 시사점",
    "## VII. REFERENCE",
)
COMPARISON_REQUIRED_SUBHEADINGS = (
    "### V.I 전략 방향 차이",
    "### V.II 데이터 기반 비교표",
)
PLACEHOLDER_PHRASES = (
    "TODO",
    "추가 정리가 필요하다",
    "비교 요약이 아직 생성되지 않아",
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

    issues.extend(_build_retrieval_failure_issues(state))
    issues.extend(_build_retrieval_gap_issues(state))
    issues.extend(_build_content_quality_issues(state))
    issues.extend(_build_citation_issues(state))
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
        runtime["current_phase"] = remaining_plan[0] if remaining_plan else "validate"
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
        runtime["current_phase"] = remaining_plan[0] if remaining_plan else "validate"
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

    message = build_agent_message(VALIDATOR_AGENT_NAME, note)
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


def _build_retrieval_failure_issues(state: ReportState) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if state["market"].get("retrieval_failures") and not state["market"]["retrieval_sufficient"]:
        issues.append(
            {
                "issue_id": "market_retrieval_failure",
                "section_id": "market_background",
                "severity": "warning",
                "message": "Market retrieval encountered execution failures before coverage became sufficient.",
                "related_evidence_ids": state["market"]["evidence_ids"],
                "suggested_action": _build_failure_action(state["market"]["retrieval_failures"]),
                "retryable": False,
            }
        )

    for company, section_id in (("LGES", "lges_strategy"), ("CATL", "catl_strategy")):
        company_state = state["companies"][company]
        if company_state.get("retrieval_failures") and not company_state["retrieval_sufficient"]:
            issues.append(
                {
                    "issue_id": f"{company.lower()}_retrieval_failure",
                    "section_id": section_id,
                    "severity": "warning",
                    "message": f"{company} retrieval encountered execution failures before coverage became sufficient.",
                    "related_evidence_ids": company_state["evidence_ids"],
                    "suggested_action": _build_failure_action(company_state["retrieval_failures"]),
                    "retryable": False,
                }
            )

    return issues


def _build_gap_action(gaps: list[str]) -> str:
    if not gaps:
        return "Preserve the limitation in the report and avoid unsupported claims."
    return "Preserve the limitation in the report: " + "; ".join(gaps)


def _build_failure_action(failures: list[str]) -> str:
    if not failures:
        return "Review the retrieval logs and LangSmith traces before trusting the remaining gaps."
    return (
        "Review the retrieval logs and LangSmith traces before trusting the remaining gaps: "
        + "; ".join(failures)
    )


def _build_content_quality_issues(state: ReportState) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    summary = state["section_drafts"]["summary"]["content"].strip()
    if summary and len(summary) > SUMMARY_MAX_CHARS:
        issues.append(
            {
                "issue_id": "summary_too_long",
                "section_id": "summary",
                "severity": "error",
                "message": "EXECUTIVE SUMMARY exceeds the configured maximum length.",
                "related_evidence_ids": state["section_drafts"]["summary"]["evidence_ids"],
                "suggested_action": "Condense the executive summary to half-page length.",
                "retryable": True,
            }
        )
    if summary and len(summary) < SUMMARY_MIN_CHARS:
        issues.append(
            {
                "issue_id": "summary_too_short",
                "section_id": "summary",
                "severity": "warning",
                "message": "EXECUTIVE SUMMARY is too short to function as an executive summary.",
                "related_evidence_ids": state["section_drafts"]["summary"]["evidence_ids"],
                "suggested_action": "Expand the summary with key findings and caveats.",
                "retryable": True,
            }
        )

    market_content = state["section_drafts"]["market_background"]["content"]
    market_subheadings = _extract_numbered_subheadings(market_content, prefix="### II.")
    if len(market_subheadings) < 2:
        issues.append(
            {
                "issue_id": "market_subheadings_missing",
                "section_id": "market_background",
                "severity": "error",
                "message": "Market background should include multiple writer-chosen Roman-numbered subsections.",
                "related_evidence_ids": state["section_drafts"]["market_background"]["evidence_ids"],
                "suggested_action": "Add at least two meaningful `### II.I` style subsections and expand the section with evidence-backed prose.",
                "retryable": True,
            }
        )

    comparison_content = state["section_drafts"]["strategy_comparison"]["content"]
    if any(heading not in comparison_content for heading in COMPARISON_REQUIRED_SUBHEADINGS):
        issues.append(
            {
                "issue_id": "comparison_subheadings_missing",
                "section_id": "strategy_comparison",
                "severity": "error",
                "message": "Strategy comparison is missing required subsection headings.",
                "related_evidence_ids": state["section_drafts"]["strategy_comparison"]["evidence_ids"],
                "suggested_action": "Include V.I strategy difference and V.II data table subsections.",
                "retryable": True,
            }
        )

    if state["final_report"]:
        missing_report_headings = [
            heading for heading in FINAL_REPORT_REQUIRED_HEADINGS if heading not in state["final_report"]
        ]
        if missing_report_headings:
            issues.append(
                {
                    "issue_id": "final_report_headings_missing",
                    "section_id": "summary",
                    "severity": "error",
                    "message": "final_report is missing one or more required Roman-numbered headings.",
                    "related_evidence_ids": [],
                    "suggested_action": "Rebuild the final report with the exact required top-level Roman headings.",
                    "retryable": True,
                }
            )

    reference_content = state["section_drafts"]["references"]["content"]
    invalid_reference_lines = [
        line
        for line in reference_content.splitlines()
        if line.strip() and line.strip() != "수집된 참고문헌이 없어 추가 검증 필요." and not _is_reference_line_valid(line)
    ]
    if invalid_reference_lines:
        issues.append(
            {
                "issue_id": "reference_format_invalid",
                "section_id": "references",
                "severity": "error",
                "message": "One or more reference lines do not match the expected Markdown citation style.",
                "related_evidence_ids": [],
                "suggested_action": "Format references as report/paper/webpage citations with italicized titles and URLs.",
                "retryable": True,
            }
        )

    for section_id in REQUIRED_SECTION_IDS:
        content = state["section_drafts"][section_id]["content"]
        if any(phrase in content for phrase in PLACEHOLDER_PHRASES):
            issues.append(
                {
                    "issue_id": f"{section_id}_placeholder_language",
                    "section_id": section_id,
                    "severity": "warning",
                    "message": "Section still contains placeholder language instead of finished report prose.",
                    "related_evidence_ids": state["section_drafts"][section_id]["evidence_ids"],
                    "suggested_action": "Rewrite the section as reader-facing report prose.",
                    "retryable": True,
                }
            )

    return issues


def _build_citation_issues(state: ReportState) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for section_id in REQUIRED_SECTION_IDS:
        if section_id == "references":
            continue

        section = state["section_drafts"][section_id]
        content = section["content"].strip()
        if not content or not section["evidence_ids"]:
            continue

        if not section.get("citations"):
            issues.append(
                {
                    "issue_id": f"{section_id}_citations_missing",
                    "section_id": section_id,
                    "severity": "error",
                    "message": "Section content does not retain sentence-level evidence citations.",
                    "related_evidence_ids": section["evidence_ids"],
                    "suggested_action": "Attach inline citations and sentence-evidence traces before finalizing the report.",
                    "retryable": True,
                }
            )
            continue

        unresolved_reference_ids = sorted(
            reference_id
            for reference_id in {
                reference_id
                for citation in section.get("citations", [])
                for reference_id in citation.get("reference_ids", [])
            }
            if reference_id not in state["references"]
        )
        if unresolved_reference_ids:
            issues.append(
                {
                    "issue_id": f"{section_id}_inline_refs_unresolved",
                    "section_id": section_id,
                    "severity": "error",
                    "message": "Section contains inline references that do not resolve in the REFERENCE section.",
                    "related_evidence_ids": section["evidence_ids"],
                    "suggested_action": "Rebuild references from the actually cited evidence/documents.",
                    "retryable": True,
                }
            )

    return issues


def _is_reference_line_valid(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("- ") and "*" in stripped and "http" in stripped


def _extract_numbered_subheadings(content: str, *, prefix: str) -> list[str]:
    return [
        line.strip()
        for line in content.splitlines()
        if line.strip().startswith(prefix)
    ]
