from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.base import build_agent_message, create_agent_blueprint
from config.settings import load_settings
from schemas.state import ReportState
from utils.evidence_context import (
    format_evidence_packet,
    format_quantitative_evidence_packet,
)
from utils.logging import get_logger


# Compare / SWOT Agent: align both companies on a shared comparison frame.
COMPARE_SWOT_BLUEPRINT = create_agent_blueprint(
    name="compare_swot_agent",
    prompt_name="compare_swot.md",
)

logger = get_logger(__name__)


class SWOTItems(BaseModel):
    strengths: list[str] = Field(description="강점")
    weaknesses: list[str] = Field(description="약점")
    opportunities: list[str] = Field(description="기회")
    threats: list[str] = Field(description="위협")


class CompareOutput(BaseModel):
    strategy_direction_diff: str = Field(description="전략 방향 차이에 대한 한국어 비교 서술")
    data_table_markdown: str = Field(description="5.2에 들어갈 Markdown 비교표")
    lges_swot: SWOTItems = Field(description="LGES SWOT")
    catl_swot: SWOTItems = Field(description="CATL SWOT")


COMPARE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", COMPARE_SWOT_BLUEPRINT.system_prompt),
        ("human", "{comparison_context}"),
    ]
)


def compare_swot_node(state: ReportState) -> dict:
    """Prepare comparison and SWOT content before report drafting."""
    remaining_plan = state["plan"][1:]
    next_step = remaining_plan[0] if remaining_plan else None

    try:
        output = _create_compare_chain().invoke(
            {"comparison_context": _build_comparison_context(state)}
        )
        comparison_summary = _render_comparison_summary(output)
        swot = {
            "LGES": output.lges_swot.model_dump(),
            "CATL": output.catl_swot.model_dump(),
        }
        note = "Comparison and SWOT stage completed via llm synthesis."
    except Exception as exc:  # pragma: no cover - depends on runtime credentials/network
        logger.warning("Compare/SWOT LLM unavailable. Falling back to deterministic summary: %s", exc)
        comparison_summary = _build_fallback_comparison_summary(state)
        swot = {
            "LGES": _build_fallback_company_swot(state, "LGES"),
            "CATL": _build_fallback_company_swot(state, "CATL"),
        }
        note = "Comparison and SWOT stage completed via fallback synthesis."

    message = build_agent_message(COMPARE_SWOT_BLUEPRINT.name, note)

    return {
        "plan": remaining_plan,
        "comparison_summary": comparison_summary,
        "swot": swot,
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": next_step or state["runtime"]["current_phase"],
            "termination_reason": None,
        },
    }


def _create_compare_chain() -> Any:
    settings = load_settings()
    logger.info(
        "Compare/SWOT report generation model configured: provider=%s, model=%s",
        settings.report_llm_provider,
        settings.report_llm_model,
    )
    compare_llm = init_chat_model(
        settings.report_llm_model,
        model_provider=settings.report_llm_provider,
        temperature=0,
    )
    return COMPARE_PROMPT | compare_llm.with_structured_output(CompareOutput)


def _build_comparison_context(state: ReportState) -> str:
    return "\n\n".join(
        [
            "다음 근거만 사용해서 5장 비교 분석용 결과를 작성하라.",
            "규칙:",
            "- 전략 방향 차이는 한국어 문단 2~4개 분량으로 작성한다.",
            "- 데이터 기반 비교표는 Markdown 표로 작성한다.",
            "- SWOT 각 항목은 근거가 충분하면 2~4개까지 제시할 수 있고, 근거가 부족하면 억지로 채우지 않는다.",
            "- 근거가 부족하면 '정보 부족/추가 검증 필요'를 명시한다.",
            "- 정량 수치, 기준 시점, 비교 조건이 판단에 중요하면 요약 과정에서 생략하지 않는다.",
            "[시장 배경 요약]",
            state["market"]["synthesized_summary"] or "정보 부족",
            "[시장 배경 근거]",
            format_evidence_packet(state, state["market"]["evidence_ids"], limit=10),
            "[시장 배경 정량 근거]",
            format_quantitative_evidence_packet(state, state["market"]["evidence_ids"], limit=7),
            "[LGES 요약]",
            state["companies"]["LGES"]["synthesized_summary"] or "정보 부족",
            "[LGES 근거]",
            format_evidence_packet(state, state["companies"]["LGES"]["evidence_ids"], limit=12),
            "[LGES 정량 근거]",
            format_quantitative_evidence_packet(
                state,
                state["companies"]["LGES"]["evidence_ids"],
                limit=8,
            ),
            "[LGES counter evidence]",
            format_evidence_packet(
                state,
                state["companies"]["LGES"]["counter_evidence_ids"],
                limit=6,
            ),
            "[CATL 요약]",
            state["companies"]["CATL"]["synthesized_summary"] or "정보 부족",
            "[CATL 근거]",
            format_evidence_packet(state, state["companies"]["CATL"]["evidence_ids"], limit=12),
            "[CATL 정량 근거]",
            format_quantitative_evidence_packet(
                state,
                state["companies"]["CATL"]["evidence_ids"],
                limit=8,
            ),
            "[CATL counter evidence]",
            format_evidence_packet(
                state,
                state["companies"]["CATL"]["counter_evidence_ids"],
                limit=6,
            ),
            "[검증 gap]",
            "\n".join(
                [
                    f"MARKET: {_format_gaps(state['market']['retrieval_gaps'])}",
                    f"LGES: {_format_gaps(state['companies']['LGES']['retrieval_gaps'])}",
                    f"CATL: {_format_gaps(state['companies']['CATL']['retrieval_gaps'])}",
                ]
            ),
        ]
    )


def _render_comparison_summary(output: CompareOutput) -> str:
    return "\n\n".join(
        [
            "### V.I 전략 방향 차이",
            output.strategy_direction_diff.strip(),
            "### V.II 데이터 기반 비교표",
            output.data_table_markdown.strip(),
        ]
    )


def _build_fallback_comparison_summary(state: ReportState) -> str:
    market_state = state["market"]
    lines = [
        "### V.I 전략 방향 차이",
        (
            f"LGES는 현재 {len(state['companies']['LGES']['evidence_ids'])}건의 근거가 수집되었고, "
            f"CATL은 {len(state['companies']['CATL']['evidence_ids'])}건의 근거가 수집되었다. "
            "두 기업 모두 포트폴리오 다각화를 추진하고 있으나, 현재 자동 비교는 수집된 기사 제목과 요약 수준의 근거를 기반으로 한다."
        ),
        (
            "시장 배경상 "
            + (
                "근거 부족이 남아 있어 전략 방향 해석은 보수적으로 유지한다."
                if not market_state["retrieval_sufficient"]
                else "비교를 진행할 최소 배경 정보는 확보되었다."
            )
        ),
        "### V.II 데이터 기반 비교표",
        "| 회사 | 확보 근거 수 | 반대 근거 수 | 남은 gap |",
        "| --- | ---: | ---: | --- |",
        (
            f"| LGES | {len(state['companies']['LGES']['evidence_ids'])} | "
            f"{len(state['companies']['LGES']['counter_evidence_ids'])} | "
            f"{_format_gaps(state['companies']['LGES']['retrieval_gaps'])} |"
        ),
        (
            f"| CATL | {len(state['companies']['CATL']['evidence_ids'])} | "
            f"{len(state['companies']['CATL']['counter_evidence_ids'])} | "
            f"{_format_gaps(state['companies']['CATL']['retrieval_gaps'])} |"
        ),
    ]
    return "\n".join(lines)


def _build_fallback_company_swot(state: ReportState, company: str) -> dict[str, list[str]]:
    company_state = state["companies"][company]
    market_state = state["market"]
    evidence_count = len(company_state["evidence_ids"])
    counter_evidence_count = len(company_state["counter_evidence_ids"])
    gaps_text = _format_gaps(company_state["retrieval_gaps"])

    return {
        "strengths": [
            (
                f"{company} 관련 근거 {evidence_count}건이 확보되어 전략 방향을 정리할 수 있다."
                if evidence_count
                else f"{company} 강점 판단 근거가 부족하다."
            )
        ],
        "weaknesses": [
            (
                f"{company} 반대 근거 {counter_evidence_count}건이 확보되었다."
                if counter_evidence_count
                else f"{company} weakness 근거는 정보 부족 상태다."
            )
        ],
        "opportunities": [
            (
                f"시장 배경 근거 {len(market_state['evidence_ids'])}건을 바탕으로 확장 기회를 검토할 수 있다."
                if market_state["evidence_ids"]
                else "시장 배경 근거가 부족해 opportunity 판단을 유보한다."
            )
        ],
        "threats": [
            (
                f"남은 검증 gap: {gaps_text}"
                if company_state["retrieval_gaps"]
                else "현재 수집본 기준 위협 요인은 추가 검증이 필요하다."
            )
        ],
    }


def _format_gaps(gaps: list[str]) -> str:
    return "; ".join(gaps) if gaps else "none"
