from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
from config.settings import load_settings
from retrieval.balanced_web_search import BalancedWebSearchClient
from retrieval.local_rag import LocalRAGRetriever
from retrieval.query_policy import build_market_query_policy
from schemas.state import ReportState


# Market Agent: gather industry background with local RAG first.
MARKET_BLUEPRINT = create_agent_blueprint(
    name="market_agent",
    prompt_name="market.md",
    tools=["local_rag", "balanced_web_search"],
)


def market_node(state: ReportState) -> dict:
    """Prepare market retrieval policy and move to LGES analysis."""
    settings = load_settings()
    query_policy = build_market_query_policy()
    local_rag = LocalRAGRetriever.from_settings(settings)
    web_search = BalancedWebSearchClient.from_settings(settings)
    _ = (local_rag, web_search)

    message = build_agent_message(
        MARKET_BLUEPRINT.name,
        (
            "Prepared market retrieval policy with "
            f"{len(query_policy['positive_queries'])} positive and "
            f"{len(query_policy['risk_queries'])} risk queries. "
            "TODO: implement local RAG, web fallback, and evidence synthesis."
        ),
    )

    return {
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": "retrieve_lges",
        },
    }
