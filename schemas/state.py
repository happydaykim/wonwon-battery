from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class RuntimeState(TypedDict):
    current_phase: Literal[
        "plan",
        "retrieve_market",
        "retrieve_lges",
        "retrieve_catl",
        "skeptic_lges",
        "skeptic_catl",
        "compare",
        "write",
        "validate",
        "done",
    ]
    revision_count: int
    max_revisions: int
    termination_reason: Literal[
        "validated",
        "done_with_gaps",
        "max_revisions_reached",
    ] | None


class SourceDocument(TypedDict):
    doc_id: str
    title: str
    source_name: str
    source_url: str | None
    published_at: str | None
    doc_type: Literal[
        "industry_report",
        "annual_report",
        "ir_deck",
        "press_release",
        "news",
        "paper",
        "other",
    ]
    company_scope: Literal["MARKET", "LGES", "CATL", "BOTH"]
    stance: Literal["neutral", "positive", "risk"]


class EvidenceItem(TypedDict):
    evidence_id: str
    doc_id: str
    topic: str
    claim: str
    excerpt: str | None
    page_or_chunk: str | None
    relevance_score: float | None
    used_for: Literal[
        "market_background",
        "lges_analysis",
        "catl_analysis",
        "counter_evidence",
        "comparison",
        "swot",
    ]


class TopicResearchState(TypedDict):
    document_ids: list[str]
    evidence_ids: list[str]
    synthesized_summary: str | None
    retrieval_sufficient: bool
    retrieval_gaps: list[str]
    used_web_search: bool


class CompanyResearchState(TypedDict):
    company: Literal["LGES", "CATL"]
    document_ids: list[str]
    evidence_ids: list[str]
    counter_evidence_ids: list[str]
    synthesized_summary: str | None
    retrieval_sufficient: bool
    retrieval_gaps: list[str]
    used_web_search: bool
    skeptic_review_required: bool
    skeptic_review_completed: bool


class SectionDraft(TypedDict):
    section_id: str
    title: str
    content: str
    evidence_ids: list[str]
    status: Literal["pending", "drafted", "needs_revision", "approved"]


class ReferenceEntry(TypedDict):
    ref_id: str
    doc_id: str
    citation_text: str
    reference_type: Literal["report", "paper", "webpage"]
    used_in_sections: list[str]


class ValidationIssue(TypedDict):
    issue_id: str
    section_id: str
    severity: Literal["warning", "error"]
    message: str
    related_evidence_ids: list[str]
    suggested_action: str
    retryable: bool


class SWOTState(TypedDict):
    strengths: list[str]
    weaknesses: list[str]
    opportunities: list[str]
    threats: list[str]


class ReportState(TypedDict):
    user_query: str
    plan: list[str]
    messages: list[Any]
    runtime: RuntimeState
    documents: Annotated[dict[str, SourceDocument], operator.or_]
    evidence: Annotated[dict[str, EvidenceItem], operator.or_]
    market: TopicResearchState
    companies: Annotated[dict[str, CompanyResearchState], operator.or_]
    comparison_summary: str | None
    swot: dict[str, SWOTState]
    section_drafts: dict[str, SectionDraft]
    references: dict[str, ReferenceEntry]
    validation_issues: list[ValidationIssue]
    final_report: str | None
