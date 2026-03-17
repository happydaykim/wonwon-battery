from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
from typing import Any

from config.settings import Settings, load_settings
from utils.logging import get_logger


logger = get_logger(__name__)

ENGLISH_QUERY_REPLACEMENTS = (
    ("전기차 캐즘", "EV chasm"),
    ("배터리 전략", "battery strategy"),
    ("배터리", "battery"),
    ("포트폴리오 다각화", "portfolio diversification"),
    ("ESS", "ESS"),
    ("HEV", "HEV"),
    ("로봇", "robotics"),
    ("확장", "expansion"),
    ("전략", "strategy"),
    ("신규 사업", "new business"),
    ("수요 둔화", "demand slowdown"),
    ("공급과잉", "oversupply"),
    ("수익성", "profitability"),
    ("압박", "pressure"),
    ("경쟁", "competition"),
    ("리스크", "risk"),
    ("LG에너지솔루션", "LG Energy Solution"),
    ("CATL", "CATL"),
)


@dataclass(slots=True)
class BalancedWebSearchClient:
    """GoogleNews-backed interface for positive/risk balanced web search."""

    provider_name: str
    period: str
    languages: tuple[str, ...]
    default_max_results_per_query: int

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None
    ) -> "BalancedWebSearchClient":
        resolved = settings or load_settings()
        return cls(
            provider_name=resolved.web_search_provider,
            period=resolved.google_news_period,
            languages=resolved.google_news_languages,
            default_max_results_per_query=resolved.google_news_max_results_per_query,
        )

    def search(
        self,
        *,
        positive_queries: list[str],
        risk_queries: list[str],
        max_results_per_query: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run GoogleNews searches for both positive and risk buckets."""
        resolved_max_results = max_results_per_query or self.default_max_results_per_query
        return {
            "positive_results": self._search_bucket(
                queries=positive_queries,
                stance="positive",
                max_results_per_query=resolved_max_results,
            ),
            "risk_results": self._search_bucket(
                queries=risk_queries,
                stance="risk",
                max_results_per_query=resolved_max_results,
            ),
        }

    def _search_bucket(
        self,
        *,
        queries: list[str],
        stance: str,
        max_results_per_query: int,
    ) -> list[dict[str, Any]]:
        aggregated_results: list[dict[str, Any]] = []
        seen_links: set[str] = set()

        for query in queries:
            for query_language, expanded_query in self._expand_query_variants(query):
                try:
                    raw_results = self._run_google_news_search(
                        query=expanded_query,
                        language=query_language,
                        max_results_per_query=max_results_per_query,
                    )
                except Exception as exc:  # pragma: no cover - depends on provider/runtime
                    logger.warning(
                        "GoogleNews search failed for query '%s' (%s): %s",
                        expanded_query,
                        query_language,
                        exc,
                    )
                    continue

                for item in raw_results[:max_results_per_query]:
                    normalized = self._normalize_result(
                        item=item,
                        original_query=query,
                        query_language=query_language,
                        stance=stance,
                    )
                    link = normalized.get("link")
                    if link and link in seen_links:
                        continue
                    if link:
                        seen_links.add(link)
                    aggregated_results.append(normalized)

        return aggregated_results

    def _run_google_news_search(
        self,
        *,
        query: str,
        language: str,
        max_results_per_query: int,
    ) -> list[dict[str, Any]]:
        try:
            from langchain_teddynote.tools import GoogleNews
        except ImportError as exc:  # pragma: no cover - depends on local install
            logger.warning(
                "langchain_teddynote.tools.GoogleNews is not available. Web search results will be empty."
            )
            raise RuntimeError("Teddynote GoogleNews dependency is missing.") from exc

        client = GoogleNews()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            _ = language
            return client.search_by_keyword(query, k=max_results_per_query)
        return []

    def _expand_query_variants(self, query: str) -> list[tuple[str, str]]:
        variants: list[tuple[str, str]] = []

        for language in self.languages:
            if language == "ko":
                variants.append(("ko", query))
                continue

            if language == "en":
                variants.append(("en", _translate_query_to_english(query)))
                continue

            variants.append((language, query))

        return variants

    def _normalize_result(
        self,
        *,
        item: dict[str, Any],
        original_query: str,
        query_language: str,
        stance: str,
    ) -> dict[str, Any]:
        published_at = item.get("datetime") or item.get("date")
        source_name = item.get("media") or item.get("source") or "GoogleNews RSS"
        title = item.get("title") or item.get("content") or "Untitled news result"
        link = item.get("link") or item.get("url")
        snippet = item.get("desc") or item.get("summary") or item.get("content")

        return {
            "title": title,
            "link": link,
            "source": source_name,
            "published_at": str(published_at) if published_at else None,
            "snippet": snippet,
            "query": original_query,
            "query_language": query_language,
            "stance": stance,
            "topic_tags": _infer_topic_tags(original_query, stance=stance),
        }


def _translate_query_to_english(query: str) -> str:
    translated = query
    for korean, english in ENGLISH_QUERY_REPLACEMENTS:
        translated = translated.replace(korean, english)
    return translated


def _infer_topic_tags(query: str, *, stance: str) -> list[str]:
    normalized = query.lower()
    topic_tags: list[str] = []

    if any(keyword in normalized for keyword in ("시장", "market", "공급과잉", "oversupply")):
        topic_tags.append("market_structure")
    if any(keyword in normalized for keyword in ("수요", "demand", "ess", "hev", "로봇", "robotics")):
        topic_tags.append("demand")
        topic_tags.append("expansion")
    if any(keyword in normalized for keyword in ("전략", "strategy", "다각화", "diversification")):
        topic_tags.append("strategy")
    if stance == "risk" or any(
        keyword in normalized
        for keyword in ("리스크", "risk", "수익성", "profitability", "압박", "pressure")
    ):
        topic_tags.append("risk")

    return sorted(set(topic_tags))
