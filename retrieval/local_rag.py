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
    """Placeholder interface for Chroma-backed local retrieval."""

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
<<<<<<< Updated upstream
        """Return no local hits until the real Chroma/Qwen3 RAG pipeline is implemented."""
        _ = (query, company_scope, top_k)
        # TODO: Wire Chroma, document ingestion, metadata filtering, and reranking.
        # Current policy intentionally returns zero hits so every retrieval agent falls
        # back to web search after the sufficiency check.
        return []
=======
        """Query the shared Chroma collection with the shared embedding backend."""
        collection = get_chroma_collection(
            chroma_dir=self.persist_directory,
            collection_name=self.collection_name,
        )
        backend = load_embedding_backend(self.embedding_model)
        where = {"company_scope": company_scope} if company_scope else None
        result = query_collection(
            query,
            collection=collection,
            embedding_backend=backend,
            where=where,
            top_k=top_k,
        )

        documents = result.get("documents", [[]])
        metadatas = result.get("metadatas", [[]])
        distances = result.get("distances", [[]])

        matches: list[dict[str, Any]] = []
        for document, metadata, distance in zip(
            documents[0],
            metadatas[0],
            distances[0],
        ):
            matches.append(
                {
                    "page_content": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return matches
>>>>>>> Stashed changes
