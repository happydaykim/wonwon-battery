from __future__ import annotations

from typing import Any


def display_graph(graph: Any, *, xray: bool = False, ascii: bool = False) -> None:
    """Delegate graph visualization to langchain_teddynote's built-in helper."""
    from langchain_teddynote.graphs import visualize_graph

    visualize_graph(graph, xray=xray, ascii=ascii)
