from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

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
    query: str | None = None,
    link: str | None = None,
    published_at: str = "2026-03-18",
) -> dict[str, object]:
    return {
        "title": title,
        "source": source,
        "source_name": source,
        "stance": stance,
        "topic_tags": topic_tags,
        "query": query or title,
        "link": link or f"https://example.com/{title}",
        "published_at": published_at,
    }


def _local_result(
    *,
    doc_id: str,
    chunk_id: str,
    title: str,
    source_name: str,
    link: str,
    page_or_chunk: str,
    company_scope: str,
    doc_type: str = "industry_report",
    published_at: str = "2026-03-18",
    article_text: str | None = None,
    stance: str | None = None,
    relevance_score: float = 0.12,
) -> dict[str, object]:
    text = article_text or f"{title} local supporting evidence."
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "page_or_chunk": page_or_chunk,
        "title": title,
        "source_name": source_name,
        "source": source_name,
        "link": link,
        "published_at": published_at,
        "doc_type": doc_type,
        "company_scope": company_scope,
        "stance": stance,
        "snippet": text[:120],
        "article_excerpt": text[:120],
        "article_text": text,
        "relevance_score": relevance_score,
        "retrieval_origin": "local_rag",
    }


def _clone_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in results]


def _clone_merged_results(
    merged_results: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "positive_results": _clone_results(merged_results.get("positive_results", [])),
        "risk_results": _clone_results(merged_results.get("risk_results", [])),
    }


class _FakeRAGRetriever:
    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self._results_by_query = results_by_query
        self.calls: list[tuple[str, str | None, int]] = []

    def retrieve(
        self,
        query: str,
        *,
        company_scope: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        self.calls.append((query, company_scope, top_k))
        return _clone_results(self._results_by_query.get(query, [])[:top_k])


class _FakeWebSearchClient:
    def __init__(self, results: dict[str, list[dict[str, Any]]]) -> None:
        self._results = results
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        *,
        positive_queries: list[str],
        risk_queries: list[str],
        max_results_per_query: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        self.calls.append(
            {
                "positive_queries": list(positive_queries),
                "risk_queries": list(risk_queries),
                "max_results_per_query": max_results_per_query,
            }
        )
        return _clone_merged_results(self._results)


class _SequentialWebSearchClient:
    def __init__(self, responses: list[dict[str, list[dict[str, Any]]]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        *,
        positive_queries: list[str],
        risk_queries: list[str],
        max_results_per_query: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        self.calls.append(
            {
                "positive_queries": list(positive_queries),
                "risk_queries": list(risk_queries),
                "max_results_per_query": max_results_per_query,
            }
        )
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return _clone_merged_results(self._responses[index])


class _RecordingArticleFetcher:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def fetch(self, url: str | None) -> None:
        self.calls.append(url)
        return None


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
        with patch(
            "retrieval.pipeline.decide_retrieval_action",
            side_effect=[
                SimpleNamespace(
                    action="search_web",
                    decision_mode="test",
                    rationale="Judge gaps are meaningful enough to expand on web.",
                ),
                SimpleNamespace(
                    action="stop",
                    decision_mode="test",
                    rationale="Merged coverage is sufficient after web expansion.",
                ),
            ],
        ) as decider_mock:
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
        self.assertEqual(2, decider_mock.call_count)
        first_call = decider_mock.call_args_list[0].kwargs
        self.assertEqual("post_local", first_call["stage"])
        self.assertIn("LG에너지솔루션 ESS 수주 확대", first_call["current_query_policy"]["positive_queries"])
        self.assertIn("LG에너지솔루션 북미 수익성 리스크", first_call["current_query_policy"]["risk_queries"])
        second_call = decider_mock.call_args_list[1].kwargs
        self.assertEqual("post_merge", second_call["stage"])
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

        with patch(
            "retrieval.pipeline.decide_retrieval_action",
            return_value=SimpleNamespace(
                action="stop",
                decision_mode="test",
                rationale="Local evidence is already sufficient.",
            ),
        ) as decider_mock:
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
        self.assertEqual(1, decider_mock.call_count)

    def test_judge_cannot_force_web_when_decider_stops_after_local(self) -> None:
        class StubRetriever:
            def retrieve(
                self,
                query: str,
                *,
                company_scope: str | None = None,
                top_k: int = 5,
            ) -> list[dict[str, object]]:
                del company_scope, top_k
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
            def search(self, **kwargs):
                raise AssertionError("web search should not run when decider stops after local review")

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
                del results, company_scope, query_policy, stage, rule_based_summary

                class Decision:
                    sufficient = False
                    gaps = ["Need broader LGES evidence."]
                    positive_queries = ["LG에너지솔루션 ESS 수주 확대"]
                    risk_queries = ["LG에너지솔루션 북미 수익성 리스크"]
                    reasoning_summary = "judge recommends broader web evidence"

                return Decision()

        with patch(
            "retrieval.pipeline.decide_retrieval_action",
            return_value=SimpleNamespace(
                action="stop",
                decision_mode="test",
                rationale="Additional search is not worth the cost for this run.",
            ),
        ) as decider_mock:
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
        self.assertEqual(1, decider_mock.call_count)
        self.assertFalse(execution.final_assessment.sufficient)
        self.assertNotIn("LG에너지솔루션 ESS 수주 확대", execution.query_history)
        self.assertNotIn("LG에너지솔루션 북미 수익성 리스크", execution.query_history)

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

    def test_run_two_stage_retrieval_skips_web_search_when_local_is_sufficient(self) -> None:
        query_policy = {
            "positive_queries": [
                "LG에너지솔루션 포트폴리오 다각화",
                "LG에너지솔루션 ESS HEV 로봇 확장",
            ],
            "risk_queries": [
                "LG에너지솔루션 수익성 리스크",
            ],
        }
        rag_retriever = _FakeRAGRetriever(
            {
                "LG에너지솔루션 포트폴리오 다각화": [
                    _local_result(
                        doc_id="lges_report_a",
                        chunk_id="lges_report_a_p001_c001",
                        title="LGES portfolio diversification plan",
                        source_name="LocalSourceA",
                        link="https://example.com/lges/report-a",
                        page_or_chunk="p.1",
                        company_scope="LGES",
                    )
                ],
                "LG에너지솔루션 ESS HEV 로봇 확장": [
                    _local_result(
                        doc_id="lges_report_b",
                        chunk_id="lges_report_b_p002_c001",
                        title="LGES expansion into ESS and robotics",
                        source_name="LocalSourceB",
                        link="https://example.com/lges/report-b",
                        page_or_chunk="p.2",
                        company_scope="LGES",
                    )
                ],
                "LG에너지솔루션 수익성 리스크": [
                    _local_result(
                        doc_id="lges_report_a",
                        chunk_id="lges_report_a_p003_c001",
                        title="LGES profitability risk overview",
                        source_name="LocalSourceA",
                        link="https://example.com/lges/report-a",
                        page_or_chunk="p.3",
                        company_scope="LGES",
                    )
                ],
            }
        )
        web_search_client = _FakeWebSearchClient(
            {
                "positive_results": [
                    _result(
                        title="web-should-not-run",
                        source="WebSource",
                        stance="positive",
                        topic_tags=["strategy"],
                    )
                ],
                "risk_results": [],
            }
        )
        article_fetcher = _RecordingArticleFetcher()

        with patch(
            "retrieval.pipeline.decide_retrieval_action",
            return_value=SimpleNamespace(
                action="stop",
                decision_mode="test",
                rationale="Local evidence is sufficient.",
            ),
        ):
            execution = run_two_stage_retrieval(
                rag_retriever=rag_retriever,
                web_search_client=web_search_client,
                article_fetcher=article_fetcher,
                query_policy=query_policy,
                company_scope="LGES",
                max_results_per_query=3,
                article_fetch_max_documents=4,
            )

        self.assertTrue(execution.local_assessment.sufficient)
        self.assertTrue(execution.final_assessment.sufficient)
        self.assertFalse(execution.used_web_search)
        self.assertEqual(0, len(web_search_client.calls))
        self.assertEqual(0, len(article_fetcher.calls))
        self.assertEqual(2, len(execution.merged_results["positive_results"]))
        self.assertEqual(1, len(execution.merged_results["risk_results"]))
        self.assertTrue(
            all(
                result["retrieval_origin"] == "local_rag"
                for result in execution.local_results
            )
        )

    def test_run_two_stage_retrieval_falls_back_to_web_when_local_is_insufficient(self) -> None:
        query_policy = {
            "positive_queries": [
                "LG에너지솔루션 포트폴리오 다각화",
                "LG에너지솔루션 ESS HEV 로봇 확장",
            ],
            "risk_queries": [
                "LG에너지솔루션 수익성 리스크",
            ],
        }
        rag_retriever = _FakeRAGRetriever(
            {
                "LG에너지솔루션 포트폴리오 다각화": [
                    _local_result(
                        doc_id="lges_report_a",
                        chunk_id="lges_report_a_p001_c001",
                        title="LGES portfolio diversification plan",
                        source_name="LocalSourceA",
                        link="https://example.com/lges/report-a",
                        page_or_chunk="p.1",
                        company_scope="LGES",
                    )
                ]
            }
        )
        web_search_client = _FakeWebSearchClient(
            {
                "positive_results": [
                    _result(
                        title="LGES expands ESS footprint",
                        source="WebSourceA",
                        stance="positive",
                        topic_tags=["demand", "expansion"],
                        query="LG에너지솔루션 ESS HEV 로봇 확장",
                        link="https://example.com/lges/web-positive",
                    )
                ],
                "risk_results": [
                    _result(
                        title="LGES faces profitability pressure",
                        source="WebSourceB",
                        stance="risk",
                        topic_tags=["risk"],
                        query="LG에너지솔루션 수익성 리스크",
                        link="https://example.com/lges/web-risk",
                    )
                ],
            }
        )
        article_fetcher = _RecordingArticleFetcher()

        with patch(
            "retrieval.pipeline.decide_retrieval_action",
            side_effect=[
                SimpleNamespace(
                    action="search_web",
                    decision_mode="test",
                    rationale="Need web coverage.",
                ),
                SimpleNamespace(
                    action="stop",
                    decision_mode="test",
                    rationale="Merged coverage is sufficient.",
                ),
            ],
        ):
            execution = run_two_stage_retrieval(
                rag_retriever=rag_retriever,
                web_search_client=web_search_client,
                article_fetcher=article_fetcher,
                query_policy=query_policy,
                company_scope="LGES",
                max_results_per_query=3,
                article_fetch_max_documents=4,
            )

        self.assertFalse(execution.local_assessment.sufficient)
        self.assertTrue(execution.used_web_search)
        self.assertEqual(1, len(web_search_client.calls))
        self.assertEqual(
            [
                "https://example.com/lges/web-positive",
                "https://example.com/lges/web-risk",
            ],
            article_fetcher.calls,
        )
        self.assertEqual(2, len(execution.merged_results["positive_results"]))
        self.assertEqual(1, len(execution.merged_results["risk_results"]))
        self.assertTrue(execution.final_assessment.sufficient)

    def test_run_two_stage_retrieval_refines_queries_when_gaps_remain(self) -> None:
        query_policy = {
            "positive_queries": [
                "LG에너지솔루션 포트폴리오 다각화",
            ],
            "risk_queries": [],
        }
        rag_retriever = _FakeRAGRetriever({})
        web_search_client = _SequentialWebSearchClient(
            [
                {
                    "positive_results": [
                        _result(
                            title="LGES expands ESS footprint",
                            source="WebSourceA",
                            stance="positive",
                            topic_tags=["strategy", "expansion"],
                            query="LG에너지솔루션 포트폴리오 다각화",
                            link="https://example.com/lges/web-positive",
                        )
                    ],
                    "risk_results": [],
                },
                {
                    "positive_results": [
                        _result(
                            title="LGES expands HEV and ESS demand",
                            source="WebSourceC",
                            stance="positive",
                            topic_tags=["demand", "expansion"],
                            query="LG에너지솔루션 ESS HEV 로봇 신규사업 확장",
                            link="https://example.com/lges/web-positive-2",
                        )
                    ],
                    "risk_results": [
                        _result(
                            title="LGES profitability pressure",
                            source="WebSourceB",
                            stance="risk",
                            topic_tags=["risk"],
                            query="LG에너지솔루션 수익성 압박 경쟁 리스크",
                            link="https://example.com/lges/web-risk",
                        )
                    ],
                },
            ]
        )
        article_fetcher = _RecordingArticleFetcher()

        with patch(
            "retrieval.pipeline.decide_retrieval_action",
            side_effect=[
                SimpleNamespace(
                    action="search_web",
                    decision_mode="test",
                    rationale="Need web coverage.",
                ),
                SimpleNamespace(
                    action="refine",
                    decision_mode="test",
                    rationale="Need one more targeted round.",
                ),
                SimpleNamespace(
                    action="search_web",
                    decision_mode="test",
                    rationale="Run the refined queries on web.",
                ),
                SimpleNamespace(
                    action="stop",
                    decision_mode="test",
                    rationale="Coverage is now sufficient.",
                ),
            ],
        ), patch(
            "retrieval.pipeline.refine_query_policy",
            return_value=SimpleNamespace(
                positive_queries=["LG에너지솔루션 ESS HEV 로봇 신규사업 확장"],
                risk_queries=["LG에너지솔루션 수익성 압박 경쟁 리스크"],
                refinement_mode="test",
                rationale="Refined via test patch.",
            ),
        ):
            execution = run_two_stage_retrieval(
                rag_retriever=rag_retriever,
                web_search_client=web_search_client,
                article_fetcher=article_fetcher,
                query_policy=query_policy,
                company_scope="LGES",
                max_results_per_query=3,
                article_fetch_max_documents=4,
                max_refinement_rounds=1,
                max_new_queries_per_bucket=2,
            )

        self.assertEqual(2, len(web_search_client.calls))
        self.assertEqual(1, execution.refinement_rounds)
        self.assertGreater(len(execution.query_history), len(query_policy["positive_queries"]))
        self.assertTrue(any("리스크" in query or "risk" in query.lower() for query in execution.query_history))
        self.assertTrue(execution.final_assessment.sufficient)

    def test_local_artifacts_preserve_doc_and_chunk_level_context(self) -> None:
        merged_results = {
            "positive_results": [
                _local_result(
                    doc_id="lges_report",
                    chunk_id="lges_report_p001_c001",
                    title="LGES portfolio diversification plan",
                    source_name="MIRAE ASSET",
                    link="https://example.com/lges/report",
                    page_or_chunk="p.1",
                    company_scope="LGES",
                    relevance_score=0.11,
                ),
                _local_result(
                    doc_id="lges_report",
                    chunk_id="lges_report_p001_c002",
                    title="LGES portfolio diversification plan",
                    source_name="MIRAE ASSET",
                    link="https://example.com/lges/report",
                    page_or_chunk="p.1",
                    company_scope="LGES",
                    relevance_score=0.22,
                ),
            ],
            "risk_results": [],
        }

        artifacts = build_retrieval_artifacts(
            merged_results=merged_results,
            company_scope="LGES",
        )

        self.assertEqual(["lges_report"], artifacts.document_ids)
        self.assertEqual(1, len(artifacts.documents))
        self.assertEqual(2, len(artifacts.evidence))
        self.assertEqual(
            "industry_report",
            artifacts.documents["lges_report"]["doc_type"],
        )
        self.assertEqual(
            "https://example.com/lges/report",
            artifacts.documents["lges_report"]["source_url"],
        )
        self.assertEqual(
            "p.1",
            artifacts.evidence["evidence_lges_report_p001_c001"]["page_or_chunk"],
        )
        self.assertEqual(
            0.11,
            artifacts.evidence["evidence_lges_report_p001_c001"]["relevance_score"],
        )
        self.assertEqual(
            0.22,
            artifacts.evidence["evidence_lges_report_p001_c002"]["relevance_score"],
        )

    def test_local_artifact_roundtrip_preserves_chunk_identity(self) -> None:
        merged_results = {
            "positive_results": [
                _local_result(
                    doc_id="catl_report",
                    chunk_id="catl_report_p004_c001",
                    title="CATL expands LFP strategy",
                    source_name="Business Chemistry",
                    link="https://example.com/catl/report",
                    page_or_chunk="p.4",
                    company_scope="CATL",
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

        self.assertEqual("catl_report", normalized_results[0]["doc_id"])
        self.assertEqual("catl_report_p004_c001", normalized_results[0]["chunk_id"])
        self.assertEqual("p.4", normalized_results[0]["page_or_chunk"])
        self.assertEqual("local_rag", normalized_results[0]["retrieval_origin"])

    def test_summarize_retrieval_returns_structured_content_digest(self) -> None:
        merged_results = {
            "positive_results": [
                _result(
                    title="LGES expands ESS portfolio",
                    source="SourceA",
                    stance="positive",
                    topic_tags=["strategy", "expansion"],
                ),
                _result(
                    title="LGES secures 128GWh backlog",
                    source="SourceC",
                    stance="positive",
                    topic_tags=["strategy", "expansion"],
                ),
                _result(
                    title="LGES raises ESS revenue target by 55%",
                    source="SourceD",
                    stance="positive",
                    topic_tags=["strategy", "growth"],
                ),
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
            query_history=[
                "LG에너지솔루션 포트폴리오 다각화",
                "LG에너지솔루션 수익성 리스크",
            ],
            refinement_rounds=1,
        )

        self.assertIn("[핵심 요약]", summary)
        self.assertIn("[검색 루프]", summary)
        self.assertIn("[주요 긍정 근거]", summary)
        self.assertIn("[주요 리스크 근거]", summary)
        self.assertIn("LGES expands ESS portfolio", summary)
        self.assertIn("LGES secures 128GWh backlog", summary)
        self.assertIn("LGES raises ESS revenue target by 55%", summary)
        self.assertIn("LGES faces profitability pressure", summary)
        self.assertIn("대표 출처는 4곳", summary)

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
