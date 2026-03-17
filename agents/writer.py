from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from schemas.state import ReferenceEntry, ReportState, SectionDraft


# Writer Agent: organize the final report structure and placeholders.
WRITER_BLUEPRINT = create_agent_blueprint(
    name="writer_agent",
    prompt_name="writer.md",
)

SECTION_ORDER = (
    "summary",
    "market_background",
    "lges_strategy",
    "catl_strategy",
    "strategy_comparison",
    "swot",
    "implications",
    "references",
)


def writer_node(state: ReportState) -> dict:
    """Draft all required report sections using current evidence and gap annotations."""
    references = _build_references(state)
    section_drafts = {
        "summary": _build_section(
            state,
            "summary",
            _build_summary_content(state),
            list(state["evidence"].keys()),
        ),
        "market_background": _build_section(
            state,
            "market_background",
            _build_market_background_content(state),
            state["market"]["evidence_ids"],
        ),
        "lges_strategy": _build_section(
            state,
            "lges_strategy",
            _build_company_section_content(state, "LGES"),
            state["companies"]["LGES"]["evidence_ids"],
        ),
        "catl_strategy": _build_section(
            state,
            "catl_strategy",
            _build_company_section_content(state, "CATL"),
            state["companies"]["CATL"]["evidence_ids"],
        ),
        "strategy_comparison": _build_section(
            state,
            "strategy_comparison",
            state["comparison_summary"]
            or "비교 요약이 아직 생성되지 않아 추가 정리가 필요하다.",
            _dedupe_ids(
                [
                    *state["companies"]["LGES"]["evidence_ids"],
                    *state["companies"]["CATL"]["evidence_ids"],
                ]
            ),
        ),
        "swot": _build_section(
            state,
            "swot",
            _build_swot_content(state),
            _dedupe_ids(
                [
                    *state["companies"]["LGES"]["evidence_ids"],
                    *state["companies"]["CATL"]["evidence_ids"],
                ]
            ),
        ),
        "implications": _build_section(
            state,
            "implications",
            _build_implications_content(state),
            _dedupe_ids(
                [
                    *state["market"]["evidence_ids"],
                    *state["companies"]["LGES"]["evidence_ids"],
                    *state["companies"]["CATL"]["evidence_ids"],
                ]
            ),
        ),
        "references": _build_section(
            state,
            "references",
            _build_references_content(references),
            [],
        ),
    }
    final_report = _build_final_report(section_drafts)
    remaining_plan = state["plan"][1:]
    message = build_agent_message(
        WRITER_BLUEPRINT.name,
        (
            "Writer stage completed with section drafts for all required headings. "
            "Evidence gaps were preserved as explicit caveats instead of being filled with unsupported claims."
        ),
    )

    return {
        "plan": remaining_plan,
        "section_drafts": section_drafts,
        "references": references,
        "final_report": final_report,
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": "validate" if remaining_plan else "done",
            "termination_reason": None,
        },
    }


def _build_section(
    state: ReportState,
    section_id: str,
    content: str,
    evidence_ids: list[str],
) -> SectionDraft:
    existing = state["section_drafts"][section_id]
    return {
        **existing,
        "content": content,
        "evidence_ids": _dedupe_ids(evidence_ids),
        "status": "drafted",
    }


def _build_summary_content(state: ReportState) -> str:
    global_gaps = _collect_global_gaps(state)
    return "\n".join(
        [
            "이 초안은 LGES와 CATL의 포트폴리오 다각화 전략을 비교하기 위한 구조화된 보고서 초안이다.",
            (
                f"수집 현황: market={len(state['market']['evidence_ids'])}건, "
                f"LGES={len(state['companies']['LGES']['evidence_ids'])}건, "
                f"CATL={len(state['companies']['CATL']['evidence_ids'])}건."
            ),
            (
                "검증 메모: "
                + (
                    "; ".join(global_gaps)
                    if global_gaps
                    else "현재 수집본 기준으로 필수 섹션 초안을 구성할 수 있다."
                )
            ),
        ]
    )


def _build_market_background_content(state: ReportState) -> str:
    market_state = state["market"]
    lines = [
        market_state["synthesized_summary"]
        or "시장 배경 요약이 아직 비어 있어 추가 정리가 필요하다.",
    ]
    if market_state["retrieval_gaps"]:
        lines.append(
            "시장 배경 한계: "
            + "; ".join(market_state["retrieval_gaps"])
            + " 추가 검증 필요."
        )
    return "\n".join(lines)


def _build_company_section_content(state: ReportState, company: str) -> str:
    company_state = state["companies"][company]
    lines = [
        company_state["synthesized_summary"]
        or f"{company} 요약이 아직 비어 있어 추가 정리가 필요하다.",
        (
            f"근거 현황: evidence={len(company_state['evidence_ids'])}, "
            f"counter_evidence={len(company_state['counter_evidence_ids'])}, "
            f"skeptic_review_completed={company_state['skeptic_review_completed']}."
        ),
    ]
    if company_state["retrieval_gaps"]:
        lines.append(
            "정보 부족 메모: "
            + "; ".join(company_state["retrieval_gaps"])
            + " 검증이 끝나지 않은 항목은 보수적으로 해석한다."
        )
    return "\n".join(lines)


def _build_swot_content(state: ReportState) -> str:
    lines: list[str] = []
    for company in ("LGES", "CATL"):
        swot = state["swot"][company]
        lines.extend(
            [
                f"{company} SWOT",
                "S: " + " / ".join(swot["strengths"]),
                "W: " + " / ".join(swot["weaknesses"]),
                "O: " + " / ".join(swot["opportunities"]),
                "T: " + " / ".join(swot["threats"]),
            ]
        )
    return "\n".join(lines)


def _build_implications_content(state: ReportState) -> str:
    global_gaps = _collect_global_gaps(state)
    lines = [
        "종합 시사점은 동일한 비교 프레임을 유지하되, 부족한 근거를 결론으로 과장하지 않는 방향으로 정리한다.",
    ]
    if global_gaps:
        lines.append("남은 검증 과제: " + "; ".join(global_gaps))
    else:
        lines.append("현재 수집본 기준으로 비교/시사점 섹션을 추가 검증 단계로 넘길 수 있다.")
    return "\n".join(lines)


def _build_references(state: ReportState) -> dict[str, ReferenceEntry]:
    references: dict[str, ReferenceEntry] = {}
    for doc_id, document in sorted(state["documents"].items()):
        ref_id = f"ref_{doc_id}"
        references[ref_id] = {
            "ref_id": ref_id,
            "doc_id": doc_id,
            "citation_text": _build_citation_text(document),
            "reference_type": "webpage",
            "used_in_sections": _infer_used_sections(document["company_scope"]),
        }
    return references


def _build_references_content(references: dict[str, ReferenceEntry]) -> str:
    if not references:
        return "수집된 참고문헌이 없어 추가 검증 필요."
    return "\n".join(
        reference["citation_text"] for _, reference in sorted(references.items())
    )


def _build_citation_text(document: dict[str, str | None]) -> str:
    source_name = document["source_name"] or "Unknown source"
    published_at = document["published_at"] or "날짜 미상"
    source_url = document["source_url"] or "URL 미확인"
    return f"- {document['title']} | {source_name} | {published_at} | {source_url}"


def _infer_used_sections(company_scope: str) -> list[str]:
    if company_scope == "MARKET":
        return ["summary", "market_background", "implications", "references"]
    if company_scope == "LGES":
        return [
            "summary",
            "lges_strategy",
            "strategy_comparison",
            "swot",
            "implications",
            "references",
        ]
    if company_scope == "CATL":
        return [
            "summary",
            "catl_strategy",
            "strategy_comparison",
            "swot",
            "implications",
            "references",
        ]
    return ["summary", "strategy_comparison", "swot", "implications", "references"]


def _build_final_report(section_drafts: dict[str, SectionDraft]) -> str:
    sections = []
    for section_id in SECTION_ORDER:
        section = section_drafts[section_id]
        sections.append(f"## {section['title']}\n{section['content']}")
    return "\n\n".join(sections)


def _collect_global_gaps(state: ReportState) -> list[str]:
    gaps = [
        *state["market"]["retrieval_gaps"],
        *state["companies"]["LGES"]["retrieval_gaps"],
        *state["companies"]["CATL"]["retrieval_gaps"],
    ]
    return _dedupe_ids(gaps)


def _dedupe_ids(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped
