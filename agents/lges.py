from __future__ import annotations

from agents.base import build_agent_message, create_agent_blueprint
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
    """Prepare LGES retrieval policy and move to CATL analysis."""
    settings = load_settings()
    query_policy = build_company_query_policy("LGES")
    local_rag = LocalRAGRetriever.from_settings(settings)
    web_search = BalancedWebSearchClient.from_settings(settings)
    _ = (local_rag, web_search)

    message = build_agent_message(
        LGES_BLUEPRINT.name,
        (
            "Prepared LGES retrieval policy with "
            f"{len(query_policy['positive_queries'])} positive and "
            f"{len(query_policy['risk_queries'])} risk queries. "
            "TODO: implement company-specific retrieval, synthesis, and citation mapping."
        ),
    )

    return {
        "messages": state["messages"] + [message],
        "runtime": {
            **state["runtime"],
            "current_phase": "retrieve_catl",
        },
    }
