"""Lazy agent exports to avoid circular imports during package initialization."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_AGENT_EXPORTS = {
    "catl_node": ("agents.catl", "catl_node"),
    "compare_swot_node": ("agents.compare_swot", "compare_swot_node"),
    "lges_node": ("agents.lges", "lges_node"),
    "market_node": ("agents.market", "market_node"),
    "planner_node": ("agents.planner", "planner_node"),
    "skeptic_node": ("agents.skeptic", "skeptic_node"),
    "supervisor_node": ("agents.supervisor", "supervisor_node"),
    "validator_node": ("agents.validator", "validator_node"),
    "writer_node": ("agents.writer", "writer_node"),
}

__all__ = list(_AGENT_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _AGENT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _AGENT_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
