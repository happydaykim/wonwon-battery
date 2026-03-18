from __future__ import annotations

from typing import Any, Callable


def _resolve_drawable_graph(graph: Any) -> Any:
    """Accept a compiled LangGraph or an already drawable graph view."""
    if hasattr(graph, "get_graph"):
        return graph.get_graph()
    return graph


def _load_ipython_display() -> tuple[Callable[[bytes], Any], Callable[[Any], None]]:
    from IPython.display import Image, display

    return Image, display


def display_graph(graph: Any) -> None:
    """Render a LangGraph Mermaid image and fall back to ASCII when unavailable."""
    drawable_graph = _resolve_drawable_graph(graph)

    try:
        image_factory, display = _load_ipython_display()
        display(image_factory(drawable_graph.draw_mermaid_png()))
    except Exception:
        print(drawable_graph.draw_ascii())
