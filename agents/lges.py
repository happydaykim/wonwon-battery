from __future__ import annotations

from agents.base import create_agent_blueprint
from config.settings import load_settings
from retrieval.balanced_web_search import BalancedWebSearchClient
from retrieval.local_rag import LocalRAGRetriever
from retrieval.query_policy import build_company_query_policy
from schemas.state import ReportState


# LGES Agent: collect LGES strategy evidence and risk points.
LGES_BLUEPRINT = create_agent_blueprint(
    name="lges_agent",
    prompt_name="lges.md",
    tools=["local_rag", "balanced_web_search"],
)


def lges_node(state: ReportState) -> dict:
    """Prepare LGES retrieval outputs for the parallel retrieval bundle."""
    settings = load_settings()
    query_policy = build_company_query_policy("LGES")
    local_rag = LocalRAGRetriever.from_settings(settings)
    web_search = BalancedWebSearchClient.from_settings(settings)
    _ = (local_rag, web_search)

    return {
        "companies": {
            "LGES": {
                **state["companies"]["LGES"],
                "synthesized_summary": (
                    "TODO: parallel retrieval placeholder for LGES analysis. "
                    f"Prepared {len(query_policy['positive_queries'])} positive and "
                    f"{len(query_policy['risk_queries'])} risk queries."
                ),
            },
        },
    }
