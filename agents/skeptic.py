from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from config.settings import load_settings
from retrieval.balanced_web_search import BalancedWebSearchClient
from retrieval.pipeline import (
    build_normalized_results_from_artifacts,
    build_retrieval_artifacts,
    evaluate_retrieval_results,
    run_skeptic_counter_retrieval,
)
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

    settings = load_settings()
    query_policy = build_company_query_policy(company)
    web_search = BalancedWebSearchClient.from_settings(settings)
    company_state = state["companies"][company]
    skeptic_results = run_skeptic_counter_retrieval(
        web_search_client=web_search,
        risk_queries=query_policy["risk_queries"],
        max_results_per_query=settings.google_news_max_results_per_query,
        web_search_max_retries=settings.web_search_max_retries,
    )
    skeptic_artifacts = build_retrieval_artifacts(
        merged_results=skeptic_results,
        company_scope=company,
        used_for_override="counter_evidence",
    )

    merged_documents = {**state["documents"], **skeptic_artifacts.documents}
    merged_evidence = {**state["evidence"], **skeptic_artifacts.evidence}
    merged_document_ids = _dedupe_ids(
        [*company_state["document_ids"], *skeptic_artifacts.document_ids]
    )
    merged_evidence_ids = _dedupe_ids(
        [*company_state["evidence_ids"], *skeptic_artifacts.evidence_ids]
    )
    risk_evidence_ids = _dedupe_ids(
        [
            *company_state["counter_evidence_ids"],
            *[
                evidence_id
                for evidence_id in merged_evidence_ids
                if _is_risk_evidence(
                    documents=merged_documents,
                    evidence=merged_evidence,
                    evidence_id=evidence_id,
                )
            ],
        ]
    )

    source_names = {
        source_name
        for evidence_id in merged_evidence_ids
        if (
            source_name := _lookup_source_name(
                documents=merged_documents,
                evidence=merged_evidence,
                evidence_id=evidence_id,
            )
        )
    }
    risk_source_names = {
        source_name
        for evidence_id in risk_evidence_ids
        if (
            source_name := _lookup_source_name(
                documents=merged_documents,
                evidence=merged_evidence,
                evidence_id=evidence_id,
            )
        )
    }
    normalized_results = build_normalized_results_from_artifacts(
        documents=merged_documents,
        evidence=merged_evidence,
        evidence_ids=merged_evidence_ids,
    )
    final_assessment = evaluate_retrieval_results(
        normalized_results,
        company_scope=company,
    )
    final_gaps = final_assessment.gaps.copy()
    if not risk_evidence_ids:
        final_gaps = _append_gap(
            final_gaps,
            "skeptic_counter_evidence: no additional risk evidence was recovered during skeptic review.",
        )

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
            added_risk_evidence_count=len(skeptic_artifacts.evidence_ids),
            remaining_gap_count=len(final_gaps),
        ),
    )

    return {
        "plan": remaining_plan,
        "documents": skeptic_artifacts.documents,
        "evidence": skeptic_artifacts.evidence,
        "companies": {
            company: {
                **company_state,
                "document_ids": merged_document_ids,
                "evidence_ids": merged_evidence_ids,
                "counter_evidence_ids": risk_evidence_ids,
                "retrieval_sufficient": final_assessment.sufficient,
                "retrieval_gaps": final_gaps,
                "used_web_search": True,
                "skeptic_review_completed": True,
                "synthesized_summary": _append_skeptic_summary(
                    company_state["synthesized_summary"],
                    added_risk_evidence_count=len(skeptic_artifacts.evidence_ids),
                    final_sufficient=final_assessment.sufficient,
                    gaps=final_gaps,
                ),
            }
        },
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": next_step or "done",
            "termination_reason": None,
        },
    }


def _is_risk_evidence(
    *,
    documents: dict[str, dict],
    evidence: dict[str, dict],
    evidence_id: str,
) -> bool:
    evidence_item = evidence.get(evidence_id)
    if evidence_item is None:
        return False

    document = documents.get(evidence_item["doc_id"])
    if document is None:
        return False

    return document["stance"] == "risk"


def _lookup_source_name(
    *,
    documents: dict[str, dict],
    evidence: dict[str, dict],
    evidence_id: str,
) -> str | None:
    evidence_item = evidence.get(evidence_id)
    if evidence_item is None:
        return None

    document = documents.get(evidence_item["doc_id"])
    if document is None:
        return None

    return document["source_name"]


def _dedupe_ids(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _append_gap(gaps: list[str], gap: str) -> list[str]:
    if gap in gaps:
        return gaps
    return [*gaps, gap]


def _append_skeptic_summary(
    summary: str | None,
    *,
    added_risk_evidence_count: int,
    final_sufficient: bool,
    gaps: list[str],
) -> str:
    base_summary = summary or "Skeptic review summary initialized."
    suffix = (
        f" Skeptic review added {added_risk_evidence_count} risk evidence item(s), "
        f"final_sufficient={final_sufficient}, "
        f"remaining_gaps={'; '.join(gaps) if gaps else 'none'}."
    )
    return base_summary + suffix


def _build_skeptic_note(
    *,
    company: str,
    risk_query_count: int,
    risk_evidence_count: int,
    total_source_count: int,
    risk_source_count: int,
    added_risk_evidence_count: int,
    remaining_gap_count: int,
) -> str:
    if risk_evidence_count == 0:
        return (
            f"Skeptic reviewed {company} with {risk_query_count} risk queries and found "
            "no risk-tagged evidence in the current retrieval set. Additional verification remains necessary."
        )

    if risk_source_count < 2:
        return (
            f"Skeptic reviewed {company} with {risk_query_count} risk queries and found "
            f"{risk_evidence_count} risk items, added {added_risk_evidence_count} new item(s), "
            f"but they are concentrated in {risk_source_count} source."
        )

    return (
        f"Skeptic reviewed {company} with {risk_query_count} risk queries and identified "
        f"{risk_evidence_count} counter-evidence items across {total_source_count} total sources. "
        f"New risk items={added_risk_evidence_count}, remaining_gaps={remaining_gap_count}."
    )
