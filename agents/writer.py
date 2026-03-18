from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.base import build_agent_message, create_agent_blueprint
from config.settings import load_settings
from schemas.state import ReferenceEntry, ReportState, SectionDraft
from utils.evidence_context import format_evidence_packet
from utils.logging import get_logger


# Writer Agent: organize the final report structure and placeholders.
WRITER_BLUEPRINT = create_agent_blueprint(
    name="writer_agent",
    prompt_name="writer.md",
)

logger = get_logger(__name__)

SUMMARY_MAX_CHARS = 900

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

REPORT_HEADING_BY_SECTION = {
    "summary": "1. SUMMARY",
    "market_background": "2. 시장 배경",
    "lges_strategy": "3. LG에너지솔루션의 포트폴리오 다각화 전략과 핵심 경쟁력",
    "catl_strategy": "4. CATL의 포트폴리오 다각화 전략과 핵심 경쟁력",
    "strategy_comparison": "5. 핵심 전략 비교 분석",
    "swot": "5.3 SWOT 분석",
    "implications": "6. 종합 시사점",
    "references": "7. REFERENCE",
}


class WriterOutput(BaseModel):
    summary: str = Field(description="1장 SUMMARY 본문")
    market_background: str = Field(description="2장 본문. 2.1, 2.2, 2.3 소제목 포함")
    lges_strategy: str = Field(description="3장 본문")
    catl_strategy: str = Field(description="4장 본문")
    implications: str = Field(description="6장 본문")


WRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_BLUEPRINT.system_prompt),
        ("human", "{report_context}"),
    ]
)


def writer_node(state: ReportState) -> dict:
    """Generate a human-readable Korean report draft from the collected evidence."""
    references = _build_references(state)

    try:
        output = _create_writer_chain().invoke(
            {"report_context": _build_writer_context(state, references)}
        )
        note = "Writer stage completed via llm report synthesis."
        section_drafts = _build_section_drafts_from_output(state, output, references)
    except Exception as exc:  # pragma: no cover - depends on runtime credentials/network
        logger.warning("Writer LLM unavailable. Falling back to deterministic report draft: %s", exc)
        note = "Writer stage completed via fallback report synthesis."
        section_drafts = _build_fallback_section_drafts(state, references)

    final_report = _build_final_report(section_drafts)
    remaining_plan = state["plan"][1:]
    message = build_agent_message(WRITER_BLUEPRINT.name, note)

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


def _create_writer_chain() -> Any:
    settings = load_settings()
    logger.info(
        "Writer report generation model configured: provider=%s, model=%s",
        settings.report_llm_provider,
        settings.report_llm_model,
    )
    writer_llm = init_chat_model(
        settings.report_llm_model,
        model_provider=settings.report_llm_provider,
        temperature=0,
    )
    return WRITER_PROMPT | writer_llm.with_structured_output(WriterOutput)


def _build_section_drafts_from_output(
    state: ReportState,
    output: WriterOutput,
    references: dict[str, ReferenceEntry],
) -> dict[str, SectionDraft]:
    return {
        "summary": _build_section(
            state,
            "summary",
            output.summary.strip(),
            list(state["evidence"].keys()),
        ),
        "market_background": _build_section(
            state,
            "market_background",
            output.market_background.strip(),
            state["market"]["evidence_ids"],
        ),
        "lges_strategy": _build_section(
            state,
            "lges_strategy",
            output.lges_strategy.strip(),
            state["companies"]["LGES"]["evidence_ids"],
        ),
        "catl_strategy": _build_section(
            state,
            "catl_strategy",
            output.catl_strategy.strip(),
            state["companies"]["CATL"]["evidence_ids"],
        ),
        "strategy_comparison": _build_section(
            state,
            "strategy_comparison",
            state["comparison_summary"]
            or _build_fallback_strategy_comparison(state),
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
            output.implications.strip(),
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


def _build_fallback_section_drafts(
    state: ReportState,
    references: dict[str, ReferenceEntry],
) -> dict[str, SectionDraft]:
    return {
        "summary": _build_section(
            state,
            "summary",
            _build_fallback_summary_content(state),
            list(state["evidence"].keys()),
        ),
        "market_background": _build_section(
            state,
            "market_background",
            _build_fallback_market_background_content(state),
            state["market"]["evidence_ids"],
        ),
        "lges_strategy": _build_section(
            state,
            "lges_strategy",
            _build_fallback_company_section_content(state, "LGES"),
            state["companies"]["LGES"]["evidence_ids"],
        ),
        "catl_strategy": _build_section(
            state,
            "catl_strategy",
            _build_fallback_company_section_content(state, "CATL"),
            state["companies"]["CATL"]["evidence_ids"],
        ),
        "strategy_comparison": _build_section(
            state,
            "strategy_comparison",
            state["comparison_summary"] or _build_fallback_strategy_comparison(state),
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
            _build_fallback_implications_content(state),
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


def _build_writer_context(
    state: ReportState,
    references: dict[str, ReferenceEntry],
) -> str:
    return "\n\n".join(
        [
            "다음 근거만 사용해 사람이 읽을 수 있는 한국어 전략 분석 보고서를 작성하라.",
            f"- SUMMARY는 EXECUTIVE SUMMARY이며 {SUMMARY_MAX_CHARS}자를 넘기지 않는다.",
            "- MARKET BACKGROUND는 반드시 다음 소제목을 포함한다:",
            "  - ### 2.1 전기차 캐즘과 HEV 피벗",
            "  - ### 2.2 K-배터리 업계의 포트폴리오 다각화 배경",
            "  - ### 2.3 CATL의 원가/기술 전략 변화",
            "- 3장과 4장은 각각 LGES/CATL 전략을 실제 보고서 문체로 작성한다.",
            "- 6장은 종합 시사점을 작성한다.",
            "- unsupported fact는 만들지 말고, 정보가 부족하면 그 사실을 자연스럽게 명시한다.",
            "- 비교표, SWOT, REFERENCE는 별도로 조립되므로 여기서는 1/2/3/4/6장 본문만 생성한다.",
            "[시장 배경 요약]",
            state["market"]["synthesized_summary"] or "정보 부족",
            "[시장 배경 근거]",
            format_evidence_packet(state, state["market"]["evidence_ids"], limit=10),
            "[LGES 요약]",
            state["companies"]["LGES"]["synthesized_summary"] or "정보 부족",
            "[LGES 근거]",
            format_evidence_packet(state, state["companies"]["LGES"]["evidence_ids"], limit=12),
            "[LGES counter evidence]",
            format_evidence_packet(
                state,
                state["companies"]["LGES"]["counter_evidence_ids"],
                limit=5,
            ),
            "[CATL 요약]",
            state["companies"]["CATL"]["synthesized_summary"] or "정보 부족",
            "[CATL 근거]",
            format_evidence_packet(state, state["companies"]["CATL"]["evidence_ids"], limit=12),
            "[CATL counter evidence]",
            format_evidence_packet(
                state,
                state["companies"]["CATL"]["counter_evidence_ids"],
                limit=5,
            ),
            "[핵심 전략 비교 메모]",
            state["comparison_summary"] or "정보 부족",
            "[SWOT 메모]",
            _build_swot_context(state),
            "[REFERENCE 후보]",
            _build_references_content(references),
        ]
    )


def _build_swot_context(state: ReportState) -> str:
    lines: list[str] = []
    for company in ("LGES", "CATL"):
        swot = state["swot"][company]
        lines.extend(
            [
                f"[{company}]",
                "S: " + " / ".join(swot["strengths"]),
                "W: " + " / ".join(swot["weaknesses"]),
                "O: " + " / ".join(swot["opportunities"]),
                "T: " + " / ".join(swot["threats"]),
            ]
        )
    return "\n".join(lines)


def _build_swot_content(state: ReportState) -> str:
    lines: list[str] = []
    for company in ("LGES", "CATL"):
        swot = state["swot"][company]
        lines.extend(
            [
                f"#### {company}",
                "- Strengths: " + " / ".join(swot["strengths"]),
                "- Weaknesses: " + " / ".join(swot["weaknesses"]),
                "- Opportunities: " + " / ".join(swot["opportunities"]),
                "- Threats: " + " / ".join(swot["threats"]),
            ]
        )
    return "\n".join(lines)


def _build_references(state: ReportState) -> dict[str, ReferenceEntry]:
    references: dict[str, ReferenceEntry] = {}
    for doc_id, document in sorted(state["documents"].items()):
        if not _is_verifiable_reference(document):
            continue
        ref_id = f"ref_{doc_id}"
        reference_type = _infer_reference_type(document["doc_type"])
        references[ref_id] = {
            "ref_id": ref_id,
            "doc_id": doc_id,
            "citation_text": _build_citation_text(document, reference_type=reference_type),
            "reference_type": reference_type,
            "used_in_sections": _infer_used_sections(document["company_scope"]),
        }
    return references


def _build_references_content(references: dict[str, ReferenceEntry]) -> str:
    if not references:
        return "수집된 참고문헌이 없어 추가 검증 필요."
    return "\n".join(
        reference["citation_text"] for _, reference in sorted(references.items())
    )


def _infer_reference_type(doc_type: str) -> str:
    if doc_type in {"industry_report", "annual_report", "ir_deck", "press_release"}:
        return "report"
    if doc_type == "paper":
        return "paper"
    return "webpage"


def _build_citation_text(
    document: dict[str, str | None],
    *,
    reference_type: str,
) -> str:
    cleaned_title, inferred_site = _clean_title_and_site(document["title"])
    source_url = document["source_url"] or "URL 미확인"
    source_name = (
        _normalize_source_name(document["source_name"], source_url=source_url)
        or inferred_site
        or _infer_source_name_from_url(source_url)
        or "Unknown source"
    )
    published_at = document["published_at"] or datetime.now().strftime("%Y-%m-%d")
    year = published_at[:4] if len(published_at) >= 4 else "YYYY"
    title = cleaned_title

    if reference_type == "report":
        return f"- {source_name}({year}). *{title}*. {source_url}"
    if reference_type == "paper":
        return f"- {source_name}({year}). {title}. *학술자료명 미상*, 페이지 정보 미상."
    return f"- {source_name}({published_at}). *{title}*. {source_name}, {source_url}"


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
    sections = ["# 배터리 시장 전략 분석 보고서"]
    for section_id in SECTION_ORDER:
        section = section_drafts[section_id]
        sections.append(f"## {REPORT_HEADING_BY_SECTION[section_id]}\n{section['content']}")
    return "\n\n".join(sections)


def _build_fallback_summary_content(state: ReportState) -> str:
    global_gaps = _collect_global_gaps(state)
    return "\n".join(
        [
            "전기차 캐즘 장기화로 배터리 수요의 무게중심이 EV 단일 시장에서 HEV, ESS, 신규 응용처로 분산되고 있다.",
            "이 환경에서 LG에너지솔루션과 CATL은 모두 포트폴리오 다각화를 추진하지만, 현재 수집된 기사 기반 근거를 보면 LGES는 사업 포트폴리오 확장과 북미/신규 응용처 대응이, CATL은 원가 경쟁력과 기술-공급망 우위를 활용한 확장이 상대적으로 부각된다.",
            (
                "다만 본 보고서는 기사형 웹 자료 중심으로 작성되어 source diversity와 일부 topic coverage 한계가 남아 있다. "
                + (
                    "남은 주요 gap: " + "; ".join(global_gaps)
                    if global_gaps
                    else "현재 수집본 기준으로 핵심 비교는 가능하다."
                )
            ),
        ]
    )[:SUMMARY_MAX_CHARS]


def _build_fallback_market_background_content(state: ReportState) -> str:
    market_evidence = format_evidence_packet(state, state["market"]["evidence_ids"], limit=8)
    gaps = "; ".join(state["market"]["retrieval_gaps"]) or "none"
    return "\n".join(
        [
            "### 2.1 전기차 캐즘과 HEV 피벗",
            "최근 수집된 기사에서는 EV 수요 둔화와 함께 HEV·ESS 등 대체 수요처로 관심이 이동하는 흐름이 반복적으로 나타난다.",
            "### 2.2 K-배터리 업계의 포트폴리오 다각화 배경",
            "국내 배터리 업계는 EV 단일 수요 의존을 낮추기 위해 ESS, 로봇, 신규 폼팩터, 북미 거점 확대 같은 다각화 전략을 병행하는 것으로 해석된다.",
            "### 2.3 CATL의 원가/기술 전략 변화",
            "CATL 관련 자료에서는 원가 경쟁력, 기술 고도화, 공급망 활용이 핵심 축으로 자주 언급된다.",
            "근거 메모:",
            market_evidence,
            f"검증 한계: {gaps}",
        ]
    )


def _build_fallback_company_section_content(state: ReportState, company: str) -> str:
    company_state = state["companies"][company]
    evidence_block = format_evidence_packet(state, company_state["evidence_ids"], limit=10)
    counter_block = format_evidence_packet(state, company_state["counter_evidence_ids"], limit=5)
    gaps = "; ".join(company_state["retrieval_gaps"]) or "none"
    return "\n".join(
        [
            f"{company} 관련 수집 근거를 기준으로 보면, 회사의 전략 방향은 EV 단일 수요 의존을 낮추고 신규 응용처 및 수익성 방어 축을 확보하려는 흐름으로 읽힌다.",
            "기사 기반 근거에서는 사업 다각화, 기술 경쟁력, 공급망/원가 대응, 실적 및 수익성 리스크가 함께 나타난다.",
            "주요 근거:",
            evidence_block,
            "반대 근거:",
            counter_block,
            f"남은 검증 gap: {gaps}",
        ]
    )


def _build_fallback_strategy_comparison(state: ReportState) -> str:
    return "\n".join(
        [
            "### 5.1 전략 방향 차이",
            "LGES와 CATL 모두 포트폴리오 다각화를 추진하고 있지만, 현재 자동 수집 근거만으로 보면 LGES는 신규 응용처와 지역 확장 대응이, CATL은 원가/기술 우위를 활용한 확장이 상대적으로 더 강조된다.",
            "### 5.2 데이터 기반 비교표",
            "| 회사 | 확보 근거 수 | 반대 근거 수 | 남은 gap |",
            "| --- | ---: | ---: | --- |",
            (
                f"| LGES | {len(state['companies']['LGES']['evidence_ids'])} | "
                f"{len(state['companies']['LGES']['counter_evidence_ids'])} | "
                f"{'; '.join(state['companies']['LGES']['retrieval_gaps']) or 'none'} |"
            ),
            (
                f"| CATL | {len(state['companies']['CATL']['evidence_ids'])} | "
                f"{len(state['companies']['CATL']['counter_evidence_ids'])} | "
                f"{'; '.join(state['companies']['CATL']['retrieval_gaps']) or 'none'} |"
            ),
        ]
    )


def _build_fallback_implications_content(state: ReportState) -> str:
    global_gaps = _collect_global_gaps(state)
    lines = [
        "종합적으로 보면 배터리 산업의 전략 초점은 EV 단일 성장률 회복을 기다리는 것보다 다변화된 수요처와 수익성 방어 장치를 확보하는 쪽으로 이동하고 있다.",
        "따라서 LGES와 CATL 비교에서도 단순 점유율보다 원가 구조, 응용처 확장성, 지역별 공급망 대응력, 리스크 흡수력이 더 중요한 판단 기준이 된다.",
    ]
    if global_gaps:
        lines.append("다만 현재 보고서는 기사형 웹 근거 중심이므로 다음 gap은 후속 검증이 필요하다: " + "; ".join(global_gaps))
    return "\n".join(lines)


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


def _clean_title_and_site(title: str) -> tuple[str, str | None]:
    if " - " not in title:
        return title, None
    clean_title, site = title.rsplit(" - ", maxsplit=1)
    return clean_title.strip(), site.strip()


def _is_verifiable_reference(document: dict[str, str | None]) -> bool:
    source_url = document.get("source_url")
    if not source_url:
        return False
    parsed = urlparse(source_url)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    return hostname != "news.google.com"


def _normalize_source_name(
    source_name: str | None,
    *,
    source_url: str | None,
) -> str | None:
    cleaned = (source_name or "").strip()
    if cleaned and cleaned not in {"GoogleNews RSS", "GoogleNews"}:
        return cleaned
    return _infer_source_name_from_url(source_url)


def _infer_source_name_from_url(source_url: str | None) -> str | None:
    if not source_url:
        return None
    parsed = urlparse(source_url)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return None
    if hostname.startswith("www."):
        hostname = hostname[4:]
    if hostname.startswith("m."):
        hostname = hostname[2:]
    if hostname.endswith(".co.kr"):
        hostname = hostname[: -len(".co.kr")]
    elif "." in hostname:
        hostname = hostname.rsplit(".", maxsplit=1)[0]

    label = hostname.split(".")[-1]
    if not label:
        return None
    return label.replace("-", " ").strip().title()
