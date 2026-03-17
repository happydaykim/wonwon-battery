from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from schemas.state import ReportState


# Compare / SWOT Agent: align both companies on a shared comparison frame.
COMPARE_SWOT_BLUEPRINT = create_agent_blueprint(
    name="compare_swot_agent",
    prompt_name="compare_swot.md",
)


def compare_swot_node(state: ReportState) -> dict:
    """Prepare comparison and SWOT placeholders before report drafting."""
    remaining_plan = state["plan"][1:]
    next_step = remaining_plan[0] if remaining_plan else None
    comparison_summary = _build_comparison_summary(state)
    swot = {
        "LGES": _build_company_swot(state, "LGES"),
        "CATL": _build_company_swot(state, "CATL"),
    }
    message = build_agent_message(
        COMPARE_SWOT_BLUEPRINT.name,
        (
            "Comparison and SWOT stage completed with aligned evidence counts, "
            "counter-evidence notes, and explicit information-gap placeholders."
        ),
    )

    return {
        "plan": remaining_plan,
        "comparison_summary": comparison_summary,
        "swot": swot,
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": next_step or "done",
            "termination_reason": None,
        },
    }


def _build_comparison_summary(state: ReportState) -> str:
    market_state = state["market"]
    lines = [
        "핵심 전략 비교 메모",
        (
            f"- LGES: evidence={len(state['companies']['LGES']['evidence_ids'])}, "
            f"counter_evidence={len(state['companies']['LGES']['counter_evidence_ids'])}, "
            f"gaps={_format_gaps(state['companies']['LGES']['retrieval_gaps'])}"
        ),
        (
            f"- CATL: evidence={len(state['companies']['CATL']['evidence_ids'])}, "
            f"counter_evidence={len(state['companies']['CATL']['counter_evidence_ids'])}, "
            f"gaps={_format_gaps(state['companies']['CATL']['retrieval_gaps'])}"
        ),
        (
            f"- MARKET: evidence={len(market_state['evidence_ids'])}, "
            f"gaps={_format_gaps(market_state['retrieval_gaps'])}"
        ),
    ]

    if not market_state["retrieval_sufficient"]:
        lines.append("- 시장 배경 근거가 충분하지 않아 opportunity/threat 해석에는 보수적 문구를 유지한다.")

    if any(
        not state["companies"][company]["retrieval_sufficient"]
        for company in ("LGES", "CATL")
    ):
        lines.append("- 회사별 비교는 동일 프레임을 유지하되, 미충분 항목은 '정보 부족/추가 검증 필요'로 남긴다.")

    return "\n".join(lines)


def _build_company_swot(state: ReportState, company: str) -> dict[str, list[str]]:
    company_state = state["companies"][company]
    market_state = state["market"]
    evidence_count = len(company_state["evidence_ids"])
    counter_evidence_count = len(company_state["counter_evidence_ids"])
    gaps_text = _format_gaps(company_state["retrieval_gaps"])

    strengths = [
        (
            f"{company} 관련 근거 {evidence_count}건을 기준으로 전략 검토를 시작할 수 있다."
            if evidence_count
            else f"{company} 강점 판단에 필요한 직접 근거가 부족하다."
        )
    ]
    weaknesses = [
        (
            f"{company} 반대 근거 {counter_evidence_count}건이 확보되어 weakness 후보를 정리할 수 있다."
            if counter_evidence_count
            else f"{company} weakness 근거가 부족해 추가 검증이 필요하다."
        )
    ]
    opportunities = [
        (
            f"시장 배경 근거 {len(market_state['evidence_ids'])}건을 바탕으로 {company}의 확장 기회를 연결해 볼 수 있다."
            if market_state["evidence_ids"]
            else f"시장 배경 근거 부족으로 {company} opportunity 판단을 보수적으로 유지한다."
        )
    ]
    threats = [
        (
            f"미해결 검증 항목: {gaps_text}."
            if company_state["retrieval_gaps"]
            else f"{company} 위협 항목은 추가 validator 점검까지 현재 수집본 기준으로 유지한다."
        )
    ]

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "opportunities": opportunities,
        "threats": threats,
    }


def _format_gaps(gaps: list[str]) -> str:
    return "; ".join(gaps) if gaps else "none"
