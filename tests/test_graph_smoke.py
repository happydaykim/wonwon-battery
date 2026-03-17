from __future__ import annotations

import unittest
from unittest.mock import patch

from app import build_initial_state
from graph.builder import build_graph


class GraphSmokeTests(unittest.TestCase):
    def test_graph_finishes_safely_when_retrieval_returns_no_results(self) -> None:
        with patch(
            "agents.planner._generate_plan",
            return_value=(
                [
                    "parallel_retrieval",
                    "skeptic_lges",
                    "skeptic_catl",
                    "compare",
                    "write",
                    "validate",
                ],
                "test",
            ),
        ), patch(
            "retrieval.balanced_web_search.BalancedWebSearchClient.search",
            return_value={"positive_results": [], "risk_results": []},
        ):
            graph = build_graph()
            result = graph.invoke(
                build_initial_state("query"),
                config={
                    "recursion_limit": 30,
                    "configurable": {"thread_id": "graph-smoke-test"},
                },
            )

        self.assertEqual("done", result["runtime"]["current_phase"])
        self.assertEqual("done_with_gaps", result["runtime"]["termination_reason"])
        self.assertEqual([], result["plan"])
        self.assertEqual(0, result["runtime"]["revision_count"])
        self.assertGreater(len(result["validation_issues"]), 0)


if __name__ == "__main__":
    unittest.main()
