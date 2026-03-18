from __future__ import annotations

from typing import Literal, TypedDict


DEFAULT_COLLECTION_NAME = "battery_document"

VECTOR_METADATA_FIELDS = (
    "chunk_id",
    "doc_id",
    "chunk_index",
    "page_number",
    "page_or_chunk",
    "section_title",
    "title",
    "author",
    "source_name",
    "source_url",
    "published_at",
    "doc_type",
    "company_scope",
    "stance",
    "language",
    "source",
    "pdf_path",
    "metadata_path",
)


class VectorDocumentMetadata(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    chunk_index: int
    page_number: int
    page_or_chunk: str
    section_title: str
    title: str
    author: str
    source_name: str
    source_url: str
    published_at: str
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
    language: Literal["en", "ko", "zh"]
    source: str
    pdf_path: str
    metadata_path: str

