from __future__ import annotations

import unittest
from unittest.mock import patch

from graph.visualization import display_graph


class GraphVisualizationTests(unittest.TestCase):
    def test_display_graph_delegates_to_langchain_teddynote_visualize_graph(self) -> None:
        compiled_graph = object()

        with patch(
            "langchain_teddynote.graphs.visualize_graph",
        ) as mock_visualize_graph:
            display_graph(compiled_graph, xray=True, ascii=True)

        mock_visualize_graph.assert_called_once_with(
            compiled_graph,
            xray=True,
            ascii=True,
        )


if __name__ == "__main__":
    unittest.main()
