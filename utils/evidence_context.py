from __future__ import annotations

import re

from schemas.state import ReportState


def format_evidence_packet(
    state: ReportState,
    evidence_ids: list[str],
    *,
    limit: int,
) -> str:
    selected_ids = _select_representative_evidence_ids(
        state,
        evidence_ids,
        limit=limit,
    )
    if not selected_ids:
        return "- 정보 부족"

    lines: list[str] = []
    for evidence_id in selected_ids:
        evidence_item = state["evidence"].get(evidence_id)
        if evidence_item is None:
            continue

        document = state["documents"].get(evidence_item["doc_id"])
        if document is None:
            continue

        stance = document["stance"]
        source_name = document["source_name"] or "출처 미상"
        published_at = document["published_at"] or "날짜 미상"
        query = evidence_item["topic"] or "질의 미상"
        tags = ", ".join(evidence_item.get("topic_tags", [])) or "없음"
        title = _compact_text(document["title"], limit=160)
        claim = _compact_text(evidence_item["claim"], limit=180)
        excerpt = _compact_text(
            evidence_item.get("full_text")
            or evidence_item["excerpt"]
            or evidence_item["claim"]
            or "excerpt 없음",
            limit=240,
        )
        lines.extend(
            [
                f"- [{stance} | {source_name} | {published_at}] {title}",
                f"  - query: {query}",
                f"  - tags: {tags}",
                f"  - claim: {claim}",
                f"  - excerpt: {excerpt}",
            ]
        )

    return "\n".join(lines) if lines else "- 정보 부족"


def format_quantitative_evidence_packet(
    state: ReportState,
    evidence_ids: list[str],
    *,
    limit: int,
) -> str:
    selected_ids = _select_representative_evidence_ids(
        state,
        evidence_ids,
        limit=max(limit * 2, limit),
    )
    if not selected_ids:
        return "- 정량 근거 없음"

    lines: list[str] = []
    seen_snippets: set[str] = set()

    for evidence_id in selected_ids:
        evidence_item = state["evidence"].get(evidence_id)
        if evidence_item is None:
            continue

        document = state["documents"].get(evidence_item["doc_id"])
        if document is None:
            continue

        numeric_snippets = _extract_numeric_snippets(
            [
                evidence_item["claim"],
                evidence_item.get("excerpt"),
                evidence_item.get("full_text"),
            ]
        )
        if not numeric_snippets:
            continue

        source_name = document["source_name"] or "출처 미상"
        published_at = document["published_at"] or "날짜 미상"
        title = _compact_text(document["title"], limit=140)
        for snippet in numeric_snippets:
            normalized = " ".join(snippet.split())
            if normalized in seen_snippets:
                continue
            seen_snippets.add(normalized)
            lines.append(f"- [{source_name} | {published_at}] {title}: {normalized}")
            if len(lines) >= limit:
                return "\n".join(lines)

    return "\n".join(lines) if lines else "- 정량 근거 없음"


def _select_representative_evidence_ids(
    state: ReportState,
    evidence_ids: list[str],
    *,
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []

    positive_ids = [
        evidence_id
        for evidence_id in evidence_ids
        if _lookup_stance(state, evidence_id) != "risk"
    ]
    risk_ids = [
        evidence_id
        for evidence_id in evidence_ids
        if _lookup_stance(state, evidence_id) == "risk"
    ]

    selected: list[str] = []
    seen_sources: set[str] = set()
    seen_queries: set[str] = set()

    if positive_ids and risk_ids:
        positive_quota = min(len(positive_ids), (limit + 1) // 2)
        risk_quota = min(len(risk_ids), limit // 2)
        selected.extend(
            _pick_diverse_ids(
                state,
                positive_ids,
                count=positive_quota,
                seen_sources=seen_sources,
                seen_queries=seen_queries,
            )
        )
        selected.extend(
            _pick_diverse_ids(
                state,
                risk_ids,
                count=risk_quota,
                seen_sources=seen_sources,
                seen_queries=seen_queries,
            )
        )

    remaining_capacity = limit - len(selected)
    if remaining_capacity > 0:
        remaining_ids = [evidence_id for evidence_id in evidence_ids if evidence_id not in selected]
        selected.extend(
            _pick_diverse_ids(
                state,
                remaining_ids,
                count=remaining_capacity,
                seen_sources=seen_sources,
                seen_queries=seen_queries,
            )
        )

    return selected[:limit]


def _pick_diverse_ids(
    state: ReportState,
    candidate_ids: list[str],
    *,
    count: int,
    seen_sources: set[str],
    seen_queries: set[str],
) -> list[str]:
    if count <= 0:
        return []

    selected: list[str] = []
    remaining = [evidence_id for evidence_id in candidate_ids if _is_valid_candidate(state, evidence_id)]

    while remaining and len(selected) < count:
        best_id = max(
            remaining,
            key=lambda evidence_id: _candidate_score(
                state,
                evidence_id,
                seen_sources=seen_sources,
                seen_queries=seen_queries,
            ),
        )
        selected.append(best_id)
        remaining.remove(best_id)

        source_name = _lookup_source_name(state, best_id)
        if source_name:
            seen_sources.add(source_name)
        query = _lookup_query(state, best_id)
        if query:
            seen_queries.add(query)

    return selected


def _candidate_score(
    state: ReportState,
    evidence_id: str,
    *,
    seen_sources: set[str],
    seen_queries: set[str],
) -> tuple[int, int, int]:
    source_name = _lookup_source_name(state, evidence_id)
    query = _lookup_query(state, evidence_id)
    has_excerpt = 1 if _lookup_excerpt(state, evidence_id) or _lookup_full_text(state, evidence_id) else 0
    has_new_source = 1 if source_name and source_name not in seen_sources else 0
    has_new_query = 1 if query and query not in seen_queries else 0
    return (has_new_source, has_new_query, has_excerpt)


def _lookup_stance(state: ReportState, evidence_id: str) -> str | None:
    evidence_item = state["evidence"].get(evidence_id)
    if evidence_item is None:
        return None
    document = state["documents"].get(evidence_item["doc_id"])
    if document is None:
        return None
    return document["stance"]


def _lookup_source_name(state: ReportState, evidence_id: str) -> str | None:
    evidence_item = state["evidence"].get(evidence_id)
    if evidence_item is None:
        return None
    document = state["documents"].get(evidence_item["doc_id"])
    if document is None:
        return None
    return document["source_name"]


def _lookup_query(state: ReportState, evidence_id: str) -> str | None:
    evidence_item = state["evidence"].get(evidence_id)
    if evidence_item is None:
        return None
    return evidence_item["topic"]


def _lookup_excerpt(state: ReportState, evidence_id: str) -> str | None:
    evidence_item = state["evidence"].get(evidence_id)
    if evidence_item is None:
        return None
    return evidence_item["excerpt"]


def _lookup_full_text(state: ReportState, evidence_id: str) -> str | None:
    evidence_item = state["evidence"].get(evidence_id)
    if evidence_item is None:
        return None
    return evidence_item.get("full_text")


def _is_valid_candidate(state: ReportState, evidence_id: str) -> bool:
    evidence_item = state["evidence"].get(evidence_id)
    if evidence_item is None:
        return False
    return evidence_item["doc_id"] in state["documents"]


def _compact_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


NUMERIC_PATTERN = re.compile(
    r"(?i)(\d[\d,\.]*\s?(?:%|배|건|개|명|대|억|조|만|천|원|달러|억원|조원|GWh|MWh|kWh|Wh|GW|MW|kW|Ah|mAh|x|X|YoY|yoy|bp|bps)?)"
)


def _extract_numeric_snippets(values: list[str | None]) -> list[str]:
    snippets: list[str] = []
    for value in values:
        if not value:
            continue
        for chunk in re.split(r"(?<=[\.\!\?])\s+|\n+", value):
            normalized = " ".join(chunk.split())
            if not normalized or not NUMERIC_PATTERN.search(normalized):
                continue
            snippets.append(_compact_text(normalized, limit=180))
            if len(snippets) >= 2:
                break
        if len(snippets) >= 2:
            break
    return snippets
