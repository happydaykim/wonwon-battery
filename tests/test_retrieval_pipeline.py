from __future__ import annotations

import unittest

from retrieval.pipeline import (
    _collect_local_results,
    build_normalized_results_from_artifacts,
    build_retrieval_artifacts,
    evaluate_retrieval_results,
    is_retrieval_sufficient,
    run_two_stage_retrieval,
    summarize_retrieval,
)
from retrieval.query_policy import build_balanced_query_policy


def _result(
    *,
    title: str,
    source: str,
    stance: str,
    topic_tags: list[str],
) -> dict[str, object]:
    return {
        "title": title,
        "source": source,
        "stance": stance,
        "topic_tags": topic_tags,
        "link": f"https://example.com/{title}",
    }


class RetrievalPipelineTests(unittest.TestCase):
    def test_run_two_stage_retrieval_uses_judge_generated_fallback_queries(self) -> None:
        class StubRetriever:
            def retrieve(
                self,
                query: str,
                *,
                company_scope: str | None = None,
                top_k: int = 5,
            ) -> list[dict[str, object]]:
                return [
                    {
                        "title": f"local-{query}",
                        "source": "LocalSource",
                        "source_name": "LocalSource",
                        "stance": "positive",
                        "topic_tags": ["strategy"],
                        "link": f"https://example.com/{query}",
                    }
                ]

        class StubWebSearch:
            def __init__(self) -> None:
                self.calls: list[tuple[list[str], list[str]]] = []

            def search(
                self,
                *,
                positive_queries: list[str],
                risk_queries: list[str],
                max_results_per_query: int = 3,
            ) -> dict[str, list[dict[str, object]]]:
                self.calls.append((positive_queries, risk_queries))
                return {
                    "positive_results": [
                        _result(
                            title="web-positive",
                            source="WebSourceA",
                            stance="positive",
                            topic_tags=["strategy", "expansion"],
                        )
                    ],
                    "risk_results": [
                        _result(
                            title="web-risk",
                            source="WebSourceB",
                            stance="risk",
                            topic_tags=["risk"],
                        )
                    ],
                }

        class StubJudge:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def judge(
                self,
                *,
                results: list[dict[str, object]],
                company_scope: str,
                query_policy: dict[str, list[str]],
                stage: str,
                rule_based_summary: str,
            ):
                del results, query_policy, rule_based_summary
                self.calls.append((company_scope, stage))

                class Decision:
                    def __init__(
                        self,
                        *,
                        sufficient: bool,
                        gaps: list[str],
                        positive_queries: list[str],
                        risk_queries: list[str],
                        reasoning_summary: str,
                    ) -> None:
                        self.sufficient = sufficient
                        self.gaps = gaps
                        self.positive_queries = positive_queries
                        self.risk_queries = risk_queries
                        self.reasoning_summary = reasoning_summary

                if stage == "local":
                    return Decision(
                        sufficient=False,
                        gaps=["Need broader LGES evidence."],
                        positive_queries=["LG에너지솔루션 ESS 수주 확대"],
                        risk_queries=["LG에너지솔루션 북미 수익성 리스크"],
                        reasoning_summary="local evidence too narrow",
                    )

                return Decision(
                    sufficient=True,
                    gaps=[],
                    positive_queries=[],
                    risk_queries=[],
                    reasoning_summary="merged evidence is sufficient",
                )

        web_search = StubWebSearch()
        judge = StubJudge()
        execution = run_two_stage_retrieval(
            rag_retriever=StubRetriever(),
            web_search_client=web_search,
            article_fetcher=None,
            retrieval_judge=judge,
            query_policy=build_balanced_query_policy("LGES"),
            company_scope="LGES",
            max_results_per_query=2,
            article_fetch_max_documents=0,
        )

        self.assertTrue(execution.used_web_search)
        self.assertEqual([("LGES", "local"), ("LGES", "final")], judge.calls)
        self.assertEqual(1, len(web_search.calls))
        positive_queries, risk_queries = web_search.calls[0]
        self.assertIn("LG에너지솔루션 포트폴리오 다각화", positive_queries)
        self.assertIn("LG에너지솔루션 ESS 수주 확대", positive_queries)
        self.assertIn("LG에너지솔루션 북미 수익성 리스크", risk_queries)
        self.assertEqual(["Need broader LGES evidence."], execution.local_assessment.gaps)
        self.assertTrue(execution.final_assessment.sufficient)

    def test_run_two_stage_retrieval_skips_web_when_judge_marks_local_results_sufficient(self) -> None:
        class StubRetriever:
            def retrieve(
                self,
                query: str,
                *,
                company_scope: str | None = None,
                top_k: int = 5,
            ) -> list[dict[str, object]]:
                return [
                    {
                        "title": f"local-{query}",
                        "source": "LocalSource",
                        "source_name": "LocalSource",
                        "stance": "positive" if "리스크" not in query and "압박" not in query else "risk",
                        "topic_tags": ["strategy", "expansion", "risk"],
                        "link": f"https://example.com/{query}",
                    }
                ]

        class StubWebSearch:
            def search(self, **kwargs):
                raise AssertionError("web search should not run when judge marks local evidence sufficient")

        class StubJudge:
            def judge(
                self,
                *,
                results: list[dict[str, object]],
                company_scope: str,
                query_policy: dict[str, list[str]],
                stage: str,
                rule_based_summary: str,
            ):
                del results, company_scope, query_policy, rule_based_summary

                class Decision:
                    sufficient = True
                    gaps: list[str] = []
                    positive_queries: list[str] = []
                    risk_queries: list[str] = []
                    reasoning_summary = "sufficient"

                return Decision()

        execution = run_two_stage_retrieval(
            rag_retriever=StubRetriever(),
            web_search_client=StubWebSearch(),
            article_fetcher=None,
            retrieval_judge=StubJudge(),
            query_policy=build_balanced_query_policy("LGES"),
            company_scope="LGES",
            max_results_per_query=2,
            article_fetch_max_documents=0,
        )

        self.assertFalse(execution.used_web_search)
        self.assertTrue(execution.local_assessment.sufficient)
        self.assertTrue(execution.final_assessment.sufficient)

    def test_collect_local_results_flattens_metadata_for_evaluation(self) -> None:
        class StubRetriever:
            def retrieve(
                self,
                query: str,
                *,
                company_scope: str | None = None,
                top_k: int = 5,
            ) -> list[dict[str, object]]:
                if query == "전기차 캐즘 배터리 전략":
                    return [
                        {
                            "page_content": "market-positive-a",
                            "metadata": {
                                "doc_id": "market-a",
                                "title": "Market A",
                                "source_name": "SourceA",
                                "source_url": "https://example.com/market-a",
                            },
                            "distance": 0.1,
                        },
                        {
                            "page_content": "market-positive-b",
                            "metadata": {
                                "doc_id": "market-b",
                                "title": "Market B",
                                "source_name": "SourceB",
                                "source_url": "https://example.com/market-b",
                            },
                            "distance": 0.2,
                        },
                    ]
                if query == "배터리 ESS HEV 로봇 수요":
                    return [
                        {
                            "page_content": "market-positive-demand",
                            "metadata": {
                                "doc_id": "market-demand",
                                "title": "Market Demand",
                                "source_name": "SourceD",
                                "source_url": "https://example.com/market-demand",
                            },
                            "distance": 0.25,
                        }
                    ]
                if query == "전기차 캐즘 배터리 수요 둔화":
                    return [
                        {
                            "page_content": "market-risk-c",
                            "metadata": {
                                "doc_id": "market-c",
                                "title": "Market C",
                                "source_name": "SourceC",
                                "source_url": "https://example.com/market-c",
                            },
                            "distance": 0.3,
                        }
                    ]
                if query == "배터리 공급과잉 수익성 압박":
                    return [
                        {
                            "page_content": "market-risk-oversupply",
                            "metadata": {
                                "doc_id": "market-oversupply",
                                "title": "Market Oversupply",
                                "source_name": "SourceE",
                                "source_url": "https://example.com/market-oversupply",
                            },
                            "distance": 0.35,
                        }
                    ]
                return []

        results = _collect_local_results(
            rag_retriever=StubRetriever(),
            query_policy=build_balanced_query_policy("market"),
            company_scope="MARKET",
            max_results_per_query=3,
        )

        assessment = evaluate_retrieval_results(results, company_scope="MARKET")

        self.assertEqual("SourceA", results[0]["source_name"])
        self.assertEqual("SourceA", results[0]["source"])
        self.assertEqual("https://example.com/market-a", results[0]["link"])
        self.assertEqual("positive", results[0]["stance"])
        self.assertTrue(assessment.sufficient)

    def test_sufficient_results_pass_evaluation(self) -> None:
        results = [
            _result(
                title="market-positive-1",
                source="SourceA",
                stance="positive",
                topic_tags=["market_structure", "demand"],
            ),
            _result(
                title="market-risk-1",
                source="SourceB",
                stance="risk",
                topic_tags=["risk", "market_structure"],
            ),
            _result(
                title="market-positive-2",
                source="SourceA",
                stance="positive",
                topic_tags=["demand", "risk"],
            ),
        ]

        assessment = evaluate_retrieval_results(results, company_scope="MARKET")

        self.assertTrue(assessment.sufficient)
        self.assertEqual([], assessment.gaps)
        self.assertTrue(is_retrieval_sufficient(results, company_scope="MARKET"))

    def test_missing_risk_results_create_stance_gap(self) -> None:
        results = [
            _result(
                title="lges-positive-1",
                source="SourceA",
                stance="positive",
                topic_tags=["strategy", "expansion"],
            ),
            _result(
                title="lges-positive-2",
                source="SourceB",
                stance="positive",
                topic_tags=["strategy", "expansion"],
            ),
            _result(
                title="lges-positive-3",
                source="SourceA",
                stance="positive",
                topic_tags=["strategy", "expansion"],
            ),
        ]

        assessment = evaluate_retrieval_results(results, company_scope="LGES")

        self.assertFalse(assessment.sufficient)
        self.assertTrue(any(gap.startswith("stance_balance:") for gap in assessment.gaps))

    def test_source_diversity_gap_is_reported(self) -> None:
        results = [
            _result(
                title="catl-positive-1",
                source="SourceA",
                stance="positive",
                topic_tags=["strategy", "expansion"],
            ),
            _result(
                title="catl-risk-1",
                source="SourceA",
                stance="risk",
                topic_tags=["risk"],
            ),
            _result(
                title="catl-positive-2",
                source="SourceA",
                stance="positive",
                topic_tags=["strategy", "expansion"],
            ),
        ]

        assessment = evaluate_retrieval_results(results, company_scope="CATL")

        self.assertFalse(assessment.sufficient)
        self.assertTrue(any(gap.startswith("source_diversity:") for gap in assessment.gaps))

    def test_missing_topic_tags_are_reported(self) -> None:
        results = [
            _result(
                title="market-positive-1",
                source="SourceA",
                stance="positive",
                topic_tags=["market_structure"],
            ),
            _result(
                title="market-risk-1",
                source="SourceB",
                stance="risk",
                topic_tags=["risk"],
            ),
            _result(
                title="market-positive-2",
                source="SourceA",
                stance="positive",
                topic_tags=["market_structure"],
            ),
        ]

        assessment = evaluate_retrieval_results(results, company_scope="MARKET")

        self.assertFalse(assessment.sufficient)
        self.assertTrue(any(gap.startswith("required_topics:") for gap in assessment.gaps))
        self.assertIn("demand", " ".join(assessment.gaps))

    def test_zero_results_report_all_major_gaps(self) -> None:
        assessment = evaluate_retrieval_results([], company_scope="LGES")

        self.assertFalse(assessment.sufficient)
        self.assertEqual(0, assessment.evidence_count)
        self.assertTrue(any(gap.startswith("evidence_count:") for gap in assessment.gaps))
        self.assertTrue(any(gap.startswith("source_diversity:") for gap in assessment.gaps))
        self.assertTrue(any(gap.startswith("stance_balance:") for gap in assessment.gaps))
        self.assertTrue(any(gap.startswith("required_topics:") for gap in assessment.gaps))

    def test_summarize_retrieval_returns_structured_content_digest(self) -> None:
        merged_results = {
            "positive_results": [
                _result(
                    title="LGES expands ESS portfolio",
                    source="SourceA",
                    stance="positive",
                    topic_tags=["strategy", "expansion"],
                )
            ],
            "risk_results": [
                _result(
                    title="LGES faces profitability pressure",
                    source="SourceB",
                    stance="risk",
                    topic_tags=["risk"],
                )
            ],
        }
        assessment = evaluate_retrieval_results(
            merged_results["positive_results"] + merged_results["risk_results"],
            company_scope="LGES",
        )

        summary = summarize_retrieval(
            company_scope="LGES",
            agent_name="lges",
            local_results=[],
            merged_results=merged_results,
            used_web_search=True,
            final_assessment=assessment,
        )

        self.assertIn("[핵심 요약]", summary)
        self.assertIn("[주요 긍정 근거]", summary)
        self.assertIn("[주요 리스크 근거]", summary)
        self.assertIn("LGES expands ESS portfolio", summary)
        self.assertIn("LGES faces profitability pressure", summary)

    def test_topic_tags_survive_artifact_roundtrip(self) -> None:
        merged_results = {
            "positive_results": [
                _result(
                    title="CATL expands LFP strategy",
                    source="SourceA",
                    stance="positive",
                    topic_tags=["strategy", "expansion"],
                )
            ],
            "risk_results": [],
        }

        artifacts = build_retrieval_artifacts(
            merged_results=merged_results,
            company_scope="CATL",
        )
        normalized_results = build_normalized_results_from_artifacts(
            documents=artifacts.documents,
            evidence=artifacts.evidence,
            evidence_ids=artifacts.evidence_ids,
        )

        self.assertEqual(["strategy", "expansion"], normalized_results[0]["topic_tags"])


if __name__ == "__main__":
    unittest.main()
