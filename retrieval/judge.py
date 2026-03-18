from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import Settings, load_settings


MAX_RESULTS_IN_CONTEXT = 12
MAX_QUERY_SUGGESTIONS_PER_BUCKET = 2


class RetrievalJudgeOutput(BaseModel):
    sufficient: bool = Field(description="현재 근거만으로 해당 범위의 보고서 서술이 가능한지 여부")
    reasoning_summary: str = Field(description="판단 근거를 한두 문장으로 요약")
    gaps: list[str] = Field(default_factory=list, description="남아 있는 핵심 정보 공백")
    positive_queries: list[str] = Field(
        default_factory=list,
        description="부족할 때 추가로 검색할 긍정/전략 관점 질의",
    )
    risk_queries: list[str] = Field(
        default_factory=list,
        description="부족할 때 추가로 검색할 리스크 관점 질의",
    )


@dataclass(frozen=True, slots=True)
class RetrievalJudgeDecision:
    sufficient: bool
    reasoning_summary: str
    gaps: list[str]
    positive_queries: list[str]
    risk_queries: list[str]


RETRIEVAL_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "\n".join(
                [
                    "You are a retrieval sufficiency judge for a battery strategy reporting workflow.",
                    "Assess whether the provided evidence is sufficient for the requested scope.",
                    "Be conservative: duplicated chunks, same-story repetition, and vague snippets do not count as strong coverage.",
                    "If the evidence is insufficient and the stage is local, propose up to 2 high-signal follow-up positive queries and up to 2 high-signal follow-up risk queries for web search.",
                    "If the stage is final or the evidence is already sufficient, return empty query lists.",
                    "Use only the provided evidence and query policy. Do not invent facts.",
                ]
            ),
        ),
        ("human", "{judge_context}"),
    ]
)


@dataclass(slots=True)
class RetrievalJudge:
    provider_name: str
    model_name: str

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RetrievalJudge":
        resolved = settings or load_settings()
        return cls(
            provider_name=resolved.llm_provider,
            model_name=resolved.llm_model,
        )

    def judge(
        self,
        *,
        results: list[dict[str, object]],
        company_scope: Literal["MARKET", "LGES", "CATL"],
        query_policy: dict[str, list[str]],
        stage: Literal["local", "final"],
        rule_based_summary: str,
    ) -> RetrievalJudgeDecision:
        chain = self._create_chain()
        output = chain.invoke(
            {
                "judge_context": _build_judge_context(
                    results=results,
                    company_scope=company_scope,
                    query_policy=query_policy,
                    stage=stage,
                    rule_based_summary=rule_based_summary,
                )
            }
        )
        return RetrievalJudgeDecision(
            sufficient=bool(output.sufficient),
            reasoning_summary=output.reasoning_summary.strip(),
            gaps=_normalize_string_list(output.gaps),
            positive_queries=_normalize_string_list(
                output.positive_queries,
                limit=MAX_QUERY_SUGGESTIONS_PER_BUCKET,
            ),
            risk_queries=_normalize_string_list(
                output.risk_queries,
                limit=MAX_QUERY_SUGGESTIONS_PER_BUCKET,
            ),
        )

    def _create_chain(self):
        judge_llm = init_chat_model(
            self.model_name,
            model_provider=self.provider_name,
            temperature=0,
        )
        return RETRIEVAL_JUDGE_PROMPT | judge_llm.with_structured_output(
            RetrievalJudgeOutput
        )


def _build_judge_context(
    *,
    results: list[dict[str, object]],
    company_scope: Literal["MARKET", "LGES", "CATL"],
    query_policy: dict[str, list[str]],
    stage: Literal["local", "final"],
    rule_based_summary: str,
) -> str:
    return "\n\n".join(
        [
            f"[scope]\n{company_scope}",
            f"[stage]\n{stage}",
            "[current query policy]",
            "positive_queries: " + ", ".join(query_policy.get("positive_queries", [])),
            "risk_queries: " + ", ".join(query_policy.get("risk_queries", [])),
            "[rule-based coverage summary]",
            rule_based_summary,
            "[evidence snapshot]",
            _format_results_for_judge(results, limit=MAX_RESULTS_IN_CONTEXT),
            "[instructions]",
            "\n".join(
                [
                    "- sufficient는 '현재 근거만으로 해당 범위의 보고서 서술이 가능하다'면 true로 판단한다.",
                    "- gaps는 남아 있는 핵심 공백만 짧게 적는다.",
                    "- local 단계에서 insufficient이면 positive_queries/risk_queries를 각각 최대 2개 제안한다.",
                    "- final 단계이거나 sufficient이면 positive_queries/risk_queries는 빈 배열로 둔다.",
                ]
            ),
        ]
    )


def _format_results_for_judge(results: list[dict[str, object]], *, limit: int) -> str:
    if not results:
        return "- no local evidence"

    lines: list[str] = []
    seen_keys: set[str] = set()
    index = 0
    for result in results:
        unique_key = str(
            result.get("link")
            or result.get("source_url")
            or "|".join(
                [
                    str(result.get("title", "")),
                    str(result.get("source_name") or result.get("source") or ""),
                    str(result.get("published_at", "")),
                ]
            )
        )
        if unique_key in seen_keys:
            continue
        seen_keys.add(unique_key)
        index += 1
        if index > limit:
            break

        title = _compact_text(str(result.get("title") or "Untitled"), limit=160)
        source = str(
            result.get("source_name")
            or result.get("source")
            or result.get("media")
            or "unknown source"
        )
        published_at = str(result.get("published_at") or "날짜 미상")
        stance = str(result.get("stance") or "unknown")
        query = _compact_text(str(result.get("query") or "질의 미상"), limit=120)
        raw_tags = result.get("topic_tags", [])
        if isinstance(raw_tags, str):
            tags = raw_tags
        elif isinstance(raw_tags, list):
            tags = ", ".join(str(tag) for tag in raw_tags if isinstance(tag, str)) or "없음"
        else:
            tags = "없음"
        excerpt = _compact_text(
            str(
                result.get("article_excerpt")
                or result.get("snippet")
                or result.get("page_content")
                or ""
            ),
            limit=220,
        ) or "excerpt 없음"
        lines.extend(
            [
                f"{index}. [{stance} | {source} | {published_at}] {title}",
                f"   - query: {query}",
                f"   - tags: {tags}",
                f"   - excerpt: {excerpt}",
            ]
        )

    return "\n".join(lines) if lines else "- no unique evidence"


def _normalize_string_list(values: list[str], *, limit: int | None = None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        normalized.append(cleaned)
        seen.add(lowered)
        if limit is not None and len(normalized) >= limit:
            break
    return normalized


def _compact_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
