from __future__ import annotations

from agents.base import create_agent_blueprint
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
    """Prepare market retrieval outputs for the parallel retrieval bundle."""
    settings = load_settings()
    query_policy = build_market_query_policy()
    local_rag = LocalRAGRetriever.from_settings(settings)
    web_search = BalancedWebSearchClient.from_settings(settings)
    _ = (local_rag, web_search)

    return {
        "market": {
            **state["market"],
            "synthesized_summary": (
                "TODO: parallel retrieval placeholder for market background. "
                f"Prepared {len(query_policy['positive_queries'])} positive and "
                f"{len(query_policy['risk_queries'])} risk queries."
            ),
        },
    }
