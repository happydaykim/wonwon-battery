from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import build_initial_state
from graph.builder import build_graph


class GraphSmokeTests(unittest.TestCase):
    def test_graph_finishes_safely_when_retrieval_returns_no_results(self) -> None:
        class _FakeSWOT:
            def __init__(self, label: str) -> None:
                self._label = label

            def model_dump(self) -> dict[str, list[str]]:
                return {
                    "strengths": [f"{self._label} strength"],
                    "weaknesses": [f"{self._label} weakness"],
                    "opportunities": [f"{self._label} opportunity"],
                    "threats": [f"{self._label} threat"],
                }

        class _FakeChain:
            def __init__(self, result: object) -> None:
                self._result = result

            def invoke(self, payload: object) -> object:
                _ = payload
                return self._result

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
            "agents.supervisor._generate_supervisor_plan",
            side_effect=lambda state: (
                (
                    ["compare", "write", "validate"]
                    if (
                        state["plan"]
                        and state["plan"][0] == "parallel_retrieval"
                        and state["market"]["synthesized_summary"] is not None
                        and state["companies"]["LGES"]["synthesized_summary"] is not None
                        and state["companies"]["CATL"]["synthesized_summary"] is not None
                    )
                    else state["plan"]
                ),
                "test",
                "graph smoke patch",
            ),
        ), patch(
            "retrieval.balanced_web_search.BalancedWebSearchClient.search",
            return_value={"positive_results": [], "risk_results": []},
        ), patch(
            "retrieval.pipeline.decide_retrieval_action",
            return_value=SimpleNamespace(
                action="stop",
                decision_mode="test",
                rationale="Stop retrieval for smoke test.",
            ),
        ), patch(
            "retrieval.pipeline.refine_query_policy",
            return_value=SimpleNamespace(
                positive_queries=[],
                risk_queries=[],
                refinement_mode="test",
                rationale="No refinement for smoke test.",
            ),
        ), patch(
            "agents.compare_swot._create_compare_chain",
            return_value=_FakeChain(
                SimpleNamespace(
                    strategy_direction_diff="전략 방향 차이 본문",
                    data_table_markdown="| 회사 | 전략 |\n| --- | --- |\n| LGES | 확장 |\n| CATL | 원가 |",
                    lges_swot=_FakeSWOT("LGES"),
                    catl_swot=_FakeSWOT("CATL"),
                )
            ),
        ), patch(
            "agents.writer._create_writer_chain",
            return_value=_FakeChain(
                SimpleNamespace(
                    summary=(
                        "전기차 캐즘 장기화 속에서 LGES와 CATL은 모두 포트폴리오 다각화를 추진하고 있다. "
                        "현재 수집본 기준으로 LGES는 응용처 확장과 북미 대응 축이, CATL은 원가·기술 우위 활용 축이 상대적으로 부각된다. "
                        "다만 기사형 웹 근거 중심이라 source diversity와 일부 topic coverage gap은 남아 있으며, 보고서는 이 한계를 명시한 상태로 마무리된다."
                    ),
                    market_background="\n".join(
                        [
                            "### 2.1 전기차 캐즘과 HEV 피벗",
                            "시장 본문",
                            "### 2.2 K-배터리 업계의 포트폴리오 다각화 배경",
                            "배경 본문",
                            "### 2.3 CATL의 원가/기술 전략 변화",
                            "CATL 본문",
                        ]
                    ),
                    lges_strategy="LGES 본문",
                    catl_strategy="CATL 본문",
                    implications="시사점 본문",
                )
            ),
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
        self.assertIn(result["runtime"]["termination_reason"], {"validated", "done_with_gaps"})
        self.assertEqual([], result["plan"])
        self.assertGreaterEqual(result["runtime"]["revision_count"], 0)


if __name__ == "__main__":
    unittest.main()
