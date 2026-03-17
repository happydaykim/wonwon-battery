from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import Settings, load_settings


@dataclass(slots=True)
class BalancedWebSearchClient:
    """Placeholder interface for positive/risk balanced web search."""

    provider_name: str

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None
    ) -> "BalancedWebSearchClient":
        resolved = settings or load_settings()
        return cls(provider_name=f"{resolved.llm_provider}-search-placeholder")

    def search(
        self,
        *,
        positive_queries: list[str],
        risk_queries: list[str],
        max_results_per_query: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return empty buckets until a real web search backend is connected."""
        _ = (positive_queries, risk_queries, max_results_per_query)
        # TODO: Integrate a search provider and normalize result metadata.
        return {
            "positive_results": [],
            "risk_results": [],
        }
