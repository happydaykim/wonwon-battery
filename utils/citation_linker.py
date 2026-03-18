from __future__ import annotations

import re
from collections.abc import Iterable

from schemas.state import ReportState, SentenceCitation


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9가-힣]{2,}")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
RISK_KEYWORDS = {
    "risk",
    "리스크",
    "위험",
    "압박",
    "둔화",
    "부족",
    "제약",
    "위협",
    "약점",
}
COMPANY_KEYWORDS = {
    "LGES": {"lges", "lg", "에너지솔루션", "lg에너지솔루션"},
    "CATL": {"catl", "닝더스다이", "宁德时代"},
    "MARKET": {"시장", "market", "ev", "hev", "ess", "수요", "공급"},
}


def apply_inline_citations(
    state: ReportState,
    content: str,
    candidate_evidence_ids: Iterable[str],
    *,
    max_citations_per_unit: int = 2,
) -> tuple[str, list[SentenceCitation], set[str]]:
    candidate_ids = _build_candidate_ids(state, candidate_evidence_ids)
    if not content.strip():
        return content, [], set()

    cited_lines: list[str] = []
    traces: list[SentenceCitation] = []
    used_reference_ids: set[str] = set()
    lines = content.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]

        if not line.strip():
            cited_lines.append(line)
            index += 1
            continue

        if _is_table_line(line):
            table_lines: list[str] = []
            while index < len(lines) and _is_table_line(lines[index]):
                table_lines.append(lines[index])
                index += 1

            cited_lines.extend(table_lines)
            trace = _build_trace_for_unit(
                state,
                unit_text="\n".join(table_lines),
                unit_kind="table",
                candidate_ids=candidate_ids,
                max_citations_per_unit=max_citations_per_unit,
            )
            if trace is not None:
                traces.append(trace)
                used_reference_ids.update(trace["reference_ids"])
            continue

        cited_line, line_traces = _cite_line(
            state,
            line,
            candidate_ids=candidate_ids,
            max_citations_per_unit=max_citations_per_unit,
        )
        cited_lines.append(cited_line)
        traces.extend(line_traces)
        for trace in line_traces:
            used_reference_ids.update(trace["reference_ids"])
        index += 1

    return "\n".join(cited_lines), traces, used_reference_ids


def _cite_line(
    state: ReportState,
    line: str,
    *,
    candidate_ids: list[str],
    max_citations_per_unit: int,
) -> tuple[str, list[SentenceCitation]]:
    stripped = line.strip()
    if _is_heading_line(stripped):
        return line, []

    bullet_prefix = _extract_bullet_prefix(line)
    body = line[len(bullet_prefix) :] if bullet_prefix else line
    if not _looks_like_meaningful_unit(body):
        return line, []

    if bullet_prefix:
        trace = _build_trace_for_unit(
            state,
            unit_text=body.strip(),
            unit_kind="bullet",
            candidate_ids=candidate_ids,
            max_citations_per_unit=max_citations_per_unit,
        )
        if trace is None:
            return line, []
        return line, [trace]

    segments = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(body.strip()) if segment.strip()]
    if not segments:
        segments = [body.strip()]

    cited_segments: list[str] = []
    traces: list[SentenceCitation] = []
    for segment in segments:
        trace = _build_trace_for_unit(
            state,
            unit_text=segment,
            unit_kind="sentence",
            candidate_ids=candidate_ids,
            max_citations_per_unit=max_citations_per_unit,
        )
        if trace is None:
            cited_segments.append(segment)
            continue
        cited_segments.append(segment)
        traces.append(trace)

    return line, traces


def _build_trace_for_unit(
    state: ReportState,
    *,
    unit_text: str,
    unit_kind: str,
    candidate_ids: list[str],
    max_citations_per_unit: int,
) -> SentenceCitation | None:
    if not candidate_ids or not _looks_like_meaningful_unit(unit_text):
        return None

    ranked = sorted(
        candidate_ids,
        key=lambda evidence_id: _score_evidence(state, unit_text, evidence_id),
        reverse=True,
    )
    selected_evidence_ids = _select_diverse_evidence_ids(
        state,
        ranked,
        unit_text=unit_text,
        max_citations_per_unit=max_citations_per_unit,
    )
    if not selected_evidence_ids:
        return None

    reference_ids = _build_reference_ids_for_evidence(state, selected_evidence_ids)
    if not reference_ids:
        return None

    return {
        "unit_text": unit_text.strip(),
        "unit_kind": unit_kind,
        "evidence_ids": selected_evidence_ids,
        "reference_ids": reference_ids,
    }


def _select_diverse_evidence_ids(
    state: ReportState,
    ranked_evidence_ids: list[str],
    *,
    unit_text: str,
    max_citations_per_unit: int,
) -> list[str]:
    selected: list[str] = []
    seen_doc_ids: set[str] = set()

    for evidence_id in ranked_evidence_ids:
        evidence_item = state["evidence"].get(evidence_id)
        if evidence_item is None:
            continue
        doc_id = evidence_item["doc_id"]
        if doc_id in seen_doc_ids:
            continue

        score = _score_evidence(state, unit_text, evidence_id)
        if score <= 0 and selected:
            break

        selected.append(evidence_id)
        seen_doc_ids.add(doc_id)
        if len(selected) >= max_citations_per_unit:
            break

    if selected:
        return selected

    for evidence_id in ranked_evidence_ids:
        evidence_item = state["evidence"].get(evidence_id)
        if evidence_item is None:
            continue
        doc_id = evidence_item["doc_id"]
        if doc_id in seen_doc_ids:
            continue
        selected.append(evidence_id)
        seen_doc_ids.add(doc_id)
        if len(selected) >= max_citations_per_unit:
            break

    return selected


def _score_evidence(state: ReportState, unit_text: str, evidence_id: str) -> int:
    evidence_item = state["evidence"].get(evidence_id)
    if evidence_item is None:
        return -1

    document = state["documents"].get(evidence_item["doc_id"])
    if document is None:
        return -1

    unit_tokens = set(_tokenize(unit_text))
    evidence_tokens = set(
        _tokenize(
            " ".join(
                [
                    document["title"],
                    document["source_name"],
                    evidence_item["topic"],
                    evidence_item["claim"],
                    evidence_item.get("excerpt") or "",
                    evidence_item.get("full_text") or "",
                    " ".join(evidence_item.get("topic_tags", [])),
                ]
            )
        )
    )

    overlap_score = len(unit_tokens & evidence_tokens) * 3
    stance_score = 2 if _mentions_risk(unit_text) and document["stance"] == "risk" else 0
    scope_score = 2 if _scope_matches(unit_tokens, document["company_scope"]) else 0
    excerpt_score = 1 if evidence_item.get("excerpt") or evidence_item.get("full_text") else 0
    return overlap_score + stance_score + scope_score + excerpt_score


def _scope_matches(unit_tokens: set[str], company_scope: str) -> bool:
    keywords = COMPANY_KEYWORDS.get(company_scope)
    if not keywords:
        return False
    return bool(unit_tokens & keywords)


def _mentions_risk(unit_text: str) -> bool:
    unit_tokens = set(_tokenize(unit_text))
    return bool(unit_tokens & RISK_KEYWORDS)


def _build_reference_ids_for_evidence(
    state: ReportState,
    evidence_ids: list[str],
) -> list[str]:
    reference_ids: list[str] = []
    seen_reference_ids: set[str] = set()

    for evidence_id in evidence_ids:
        evidence_item = state["evidence"].get(evidence_id)
        if evidence_item is None:
            continue
        document = state["documents"].get(evidence_item["doc_id"])
        if document is None or not _is_verifiable_document(document):
            continue
        reference_id = f"ref_{document['doc_id']}"
        if reference_id in seen_reference_ids:
            continue
        reference_ids.append(reference_id)
        seen_reference_ids.add(reference_id)

    return reference_ids


def _is_verifiable_document(document: dict[str, str | None]) -> bool:
    source_url = (document.get("source_url") or "").strip()
    return bool(source_url) and "news.google.com" not in source_url


def _build_candidate_ids(state: ReportState, evidence_ids: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for evidence_id in evidence_ids:
        if evidence_id in seen:
            continue
        evidence_item = state["evidence"].get(evidence_id)
        if evidence_item is None:
            continue
        if evidence_item["doc_id"] not in state["documents"]:
            continue
        deduped.append(evidence_id)
        seen.add(evidence_id)
    return deduped


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _extract_bullet_prefix(line: str) -> str:
    match = re.match(r"^(\s*(?:-|\d+\.)\s+)", line)
    return match.group(1) if match else ""


def _is_heading_line(line: str) -> bool:
    return line.startswith("#")


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _looks_like_meaningful_unit(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("근거:"):
        return False
    if _is_heading_line(stripped):
        return False
    if all(character in "-|: " for character in stripped):
        return False
    return True
