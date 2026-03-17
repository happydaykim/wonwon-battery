from __future__ import annotations

from typing import Literal


QueryPolicy = dict[str, list[str]]


def build_market_query_policy(topic: str | None = None) -> QueryPolicy:
    market_topic = topic or "전기차 캐즘과 배터리 산업 포트폴리오 다각화"
    return {
        "positive_queries": [
            f"{market_topic} 산업 보고서",
            f"{market_topic} ESS 로봇 HEV 기회",
        ],
        "risk_queries": [
            f"{market_topic} 수익성 압박 리스크",
            f"{market_topic} 공급과잉 경쟁 심화",
        ],
    }


def build_company_query_policy(
    company: Literal["LGES", "CATL"],
    topic: str | None = None,
) -> QueryPolicy:
    company_name = "LG에너지솔루션" if company == "LGES" else "CATL"
    company_topic = topic or "포트폴리오 다각화 전략"
    return {
        "positive_queries": [
            f"{company_name} {company_topic}",
            f"{company_name} ESS 로봇 HEV 신규 사업",
        ],
        "risk_queries": [
            f"{company_name} {company_topic} 리스크",
            f"{company_name} 수익성 경쟁 압력 약점",
        ],
    }


def build_balanced_query_policy(
    target: Literal["market", "LGES", "CATL"],
    topic: str | None = None,
) -> QueryPolicy:
    if target == "market":
        return build_market_query_policy(topic=topic)

    return build_company_query_policy(company=target, topic=topic)
