from __future__ import annotations

from typing import Any, Literal, TypedDict


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


class CompanyResearchState(TypedDict):
    company: Literal["LGES", "CATL"]
    document_ids: list[str]
    evidence_ids: list[str]
    counter_evidence_ids: list[str]
    synthesized_summary: str | None


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
    documents: dict[str, SourceDocument]
    evidence: dict[str, EvidenceItem]
    market: TopicResearchState
    companies: dict[str, CompanyResearchState]
    comparison_summary: str | None
    swot: dict[str, SWOTState]
    section_drafts: dict[str, SectionDraft]
    references: dict[str, ReferenceEntry]
    validation_issues: list[ValidationIssue]
    final_report: str | None
