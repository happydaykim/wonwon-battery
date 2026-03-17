from __future__ import annotations

import unittest

from retrieval.pipeline import evaluate_retrieval_results, is_retrieval_sufficient


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


if __name__ == "__main__":
    unittest.main()
