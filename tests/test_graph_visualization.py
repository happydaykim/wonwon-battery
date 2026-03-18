from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from graph.visualization import display_graph


class _FakeDrawableGraph:
    def __init__(
        self,
        *,
        mermaid_png: bytes = b"png-bytes",
        ascii_diagram: str = "ASCII GRAPH",
        mermaid_error: Exception | None = None,
    ) -> None:
        self._mermaid_png = mermaid_png
        self._ascii_diagram = ascii_diagram
        self._mermaid_error = mermaid_error

    def draw_mermaid_png(self) -> bytes:
        if self._mermaid_error is not None:
            raise self._mermaid_error
        return self._mermaid_png

    def draw_ascii(self) -> str:
        return self._ascii_diagram


class _FakeCompiledGraph:
    def __init__(self, drawable_graph: _FakeDrawableGraph) -> None:
        self._drawable_graph = drawable_graph

    def get_graph(self) -> _FakeDrawableGraph:
        return self._drawable_graph


class GraphVisualizationTests(unittest.TestCase):
    def test_display_graph_renders_mermaid_png_for_compiled_graph(self) -> None:
        compiled_graph = _FakeCompiledGraph(_FakeDrawableGraph())
        image_factory = Mock(return_value="image-object")
        display = Mock()

        with patch(
            "graph.visualization._load_ipython_display",
            return_value=(image_factory, display),
        ):
            display_graph(compiled_graph)

        image_factory.assert_called_once_with(b"png-bytes")
        display.assert_called_once_with("image-object")

    def test_display_graph_falls_back_to_ascii_when_mermaid_rendering_fails(self) -> None:
        compiled_graph = _FakeCompiledGraph(
            _FakeDrawableGraph(mermaid_error=RuntimeError("mermaid unavailable"))
        )

        with patch("builtins.print") as mock_print:
            display_graph(compiled_graph)

        mock_print.assert_called_once_with("ASCII GRAPH")


if __name__ == "__main__":
    unittest.main()
