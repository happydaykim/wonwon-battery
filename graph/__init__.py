"""Lazy graph exports to avoid importing the full graph during router access."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_GRAPH_EXPORTS = {
    "build_graph": ("graph.builder", "build_graph"),
    "route_supervisor": ("graph.router", "route_supervisor"),
    "display_graph": ("graph.visualization", "display_graph"),
}

__all__ = list(_GRAPH_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _GRAPH_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _GRAPH_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
