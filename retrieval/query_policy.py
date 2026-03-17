from __future__ import annotations

from typing import Literal


QueryPolicy = dict[str, list[str]]


def build_market_query_policy(topic: str | None = None) -> QueryPolicy:
    market_topic = topic or "전기차 캐즘 배터리 전략"
    return {
        "positive_queries": [
            market_topic,
            "배터리 ESS HEV 로봇 수요",
        ],
        "risk_queries": [
            "전기차 캐즘 배터리 수요 둔화",
            "배터리 공급과잉 수익성 압박",
        ],
    }


def build_company_query_policy(
    company: Literal["LGES", "CATL"],
    topic: str | None = None,
) -> QueryPolicy:
    company_name = "LG에너지솔루션" if company == "LGES" else "CATL"
    company_topic = topic or "포트폴리오 다각화"
    return {
        "positive_queries": [
            f"{company_name} {company_topic}",
            f"{company_name} ESS HEV 로봇 확장",
        ],
        "risk_queries": [
            f"{company_name} 수익성 리스크",
            f"{company_name} 경쟁 압박",
        ],
    }


def build_balanced_query_policy(
    target: Literal["market", "LGES", "CATL"],
    topic: str | None = None,
) -> QueryPolicy:
    if target == "market":
        return build_market_query_policy(topic=topic)

    return build_company_query_policy(company=target, topic=topic)
