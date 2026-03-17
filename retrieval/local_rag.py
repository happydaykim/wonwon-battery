from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import Settings, load_settings


@dataclass(slots=True)
class LocalRAGRetriever:
    """Placeholder interface for Chroma-backed local retrieval."""

    embedding_model: str
    vector_store: str
    persist_directory: Path
    collection_name: str = "battery_strategy_agent"

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
        """Return no results until real Chroma/Qwen3 integration is added."""
        _ = (query, company_scope, top_k)
        # TODO: Wire Chroma, document ingestion, metadata filtering, and reranking.
        return []
