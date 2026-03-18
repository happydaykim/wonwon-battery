from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import Settings, load_settings
from retrieval.embeddings import load_embedding_backend
from retrieval.vector_schema import DEFAULT_COLLECTION_NAME
from retrieval.vector_store import get_chroma_collection, query_collection


@dataclass(slots=True)
class LocalRAGRetriever:
    """Chroma-backed local retriever that returns normalized retrieval results."""

    embedding_model: str
    vector_store: str
    persist_directory: Path
    collection_name: str = DEFAULT_COLLECTION_NAME

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "LocalRAGRetriever":
        resolved = settings or load_settings()
        return cls(
            embedding_model=resolved.embedding_model,
            vector_store=resolved.vector_store,
            persist_directory=resolved.chroma_persist_directory,
        )

    def retrieve(
        self,
        query: str,
        *,
        company_scope: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query the shared Chroma collection and normalize each local hit."""
        collection = get_chroma_collection(
            chroma_dir=self.persist_directory,
            collection_name=self.collection_name,
        )
        backend = load_embedding_backend(self.embedding_model)
        where = _build_company_scope_filter(company_scope)
        overfetch_k = max(top_k * 4, top_k)
        result = query_collection(
            query,
            collection=collection,
            embedding_backend=backend,
            where=where,
            top_k=overfetch_k,
        )

        documents = _first_query_batch(result.get("documents"))
        metadatas = _first_query_batch(result.get("metadatas"))
        distances = _first_query_batch(result.get("distances"))

        matches: list[dict[str, Any]] = []
        seen_doc_keys: set[str] = set()
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            normalized_metadata = dict(metadata or {})
            doc_key = (
                normalized_metadata.get("doc_id")
                or normalized_metadata.get("source_url")
                or normalized_metadata.get("chunk_id")
            )
            if doc_key in seen_doc_keys:
                continue
            if doc_key:
                seen_doc_keys.add(str(doc_key))
            matches.append(
                {
                    **normalized_metadata,
                    **_normalize_local_match(
                        document=document,
                        metadata=normalized_metadata,
                        distance=distance,
                    ),
                    "page_content": document,
                    "metadata": normalized_metadata,
                    "distance": distance,
                }
            )
            if len(matches) >= top_k:
                break
        return matches


def _normalize_local_match(
    *,
    document: str,
    metadata: dict[str, Any],
    distance: float | int | None,
) -> dict[str, Any]:
    content = document.strip()
    excerpt = _build_excerpt(content)
    source_name = (
        metadata.get("source_name")
        or metadata.get("source")
        or "Local corpus"
    )

    return {
        "doc_id": metadata.get("doc_id"),
        "chunk_id": metadata.get("chunk_id"),
        "page_or_chunk": metadata.get("page_or_chunk"),
        "title": metadata.get("title")
        or metadata.get("section_title")
        or metadata.get("doc_id")
        or "Untitled local document",
        "source_name": source_name,
        "source": source_name,
        "link": metadata.get("source_url"),
        "published_at": metadata.get("published_at"),
        "doc_type": metadata.get("doc_type"),
        "company_scope": metadata.get("company_scope"),
        "stance": metadata.get("stance"),
        "snippet": excerpt,
        "article_excerpt": excerpt,
        "article_text": content,
        "relevance_score": float(distance) if distance is not None else None,
        "retrieval_origin": "local_rag",
    }


def _first_query_batch(payload: Any) -> list[Any]:
    if not isinstance(payload, list) or not payload:
        return []

    first_batch = payload[0]
    if isinstance(first_batch, list):
        return first_batch
    return payload


def _build_excerpt(content: str, *, limit: int = 400) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _build_company_scope_filter(company_scope: str | None) -> dict[str, Any] | None:
    if not company_scope:
        return None
    if company_scope in {"LGES", "CATL"}:
        return {"company_scope": {"$in": [company_scope, "BOTH"]}}
    return {"company_scope": company_scope}
