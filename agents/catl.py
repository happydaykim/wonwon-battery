from __future__ import annotations

from agents.base import create_agent_blueprint
from config.settings import load_settings
from retrieval.balanced_web_search import BalancedWebSearchClient
from retrieval.local_rag import LocalRAGRetriever
from retrieval.query_policy import build_company_query_policy
from schemas.state import ReportState


# CATL Agent: collect CATL strategy evidence and risk points.
CATL_BLUEPRINT = create_agent_blueprint(
    name="catl_agent",
    prompt_name="catl.md",
    tools=["local_rag", "balanced_web_search"],
)


def catl_node(state: ReportState) -> dict:
    """Prepare CATL retrieval outputs for the parallel retrieval bundle."""
    settings = load_settings()
    query_policy = build_company_query_policy("CATL")
    local_rag = LocalRAGRetriever.from_settings(settings)
    web_search = BalancedWebSearchClient.from_settings(settings)
    _ = (local_rag, web_search)

    return {
        "companies": {
            "CATL": {
                **state["companies"]["CATL"],
                "synthesized_summary": (
                    "TODO: parallel retrieval placeholder for CATL analysis. "
                    f"Prepared {len(query_policy['positive_queries'])} positive and "
                    f"{len(query_policy['risk_queries'])} risk queries."
                ),
            },
        },
    }
