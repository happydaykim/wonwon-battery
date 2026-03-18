from __future__ import annotations

from typing import Any

from agents.catl import catl_node
from agents.compare_swot import compare_swot_node
from agents.lges import lges_node
from agents.market import market_node
from agents.planner import planner_node
from agents.skeptic import skeptic_node
from agents.supervisor import supervisor_node
from agents.validator import validator_node
from agents.writer import writer_node
from config.settings import Settings, load_settings
from graph.router import route_supervisor
from schemas.state import ReportState


def parallel_retrieval_dispatch(_: ReportState) -> dict:
    """Trigger the parallel market/LGES/CATL retrieval bundle."""
    return {}


def build_graph(settings: Settings | None = None) -> Any:
    """Build and compile the Supervisor-based LangGraph skeleton."""
    _ = settings or load_settings()

    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "LangGraph is not installed. Install requirements.txt before compiling the graph."
        ) from exc

    workflow = StateGraph(ReportState)

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("supervisor_agent", supervisor_node)
    workflow.add_node("parallel_retrieval_dispatch", parallel_retrieval_dispatch)
    workflow.add_node("market_agent", market_node)
    workflow.add_node("lges_agent", lges_node)
    workflow.add_node("catl_agent", catl_node)
    workflow.add_node("skeptic_agent", skeptic_node)
    workflow.add_node("compare_swot_agent", compare_swot_node)
    workflow.add_node("writer_agent", writer_node)
    workflow.add_node("validator_agent", validator_node)

    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "supervisor_agent")
    workflow.add_edge("parallel_retrieval_dispatch", "market_agent")
    workflow.add_edge("parallel_retrieval_dispatch", "lges_agent")
    workflow.add_edge("parallel_retrieval_dispatch", "catl_agent")
    workflow.add_edge(
        ["market_agent", "lges_agent", "catl_agent"],
        "supervisor_agent",
    )

    workflow.add_conditional_edges(
        "supervisor_agent",
        route_supervisor,
        {
            "parallel_retrieval_dispatch": "parallel_retrieval_dispatch",
            "skeptic_agent": "skeptic_agent",
            "compare_swot_agent": "compare_swot_agent",
            "writer_agent": "writer_agent",
            "validator_agent": "validator_agent",
            "done": END,
        },
    )

    for specialist in ("skeptic_agent", "compare_swot_agent", "writer_agent", "validator_agent"):
        workflow.add_edge(specialist, "supervisor_agent")

    return workflow.compile(checkpointer=MemorySaver())
