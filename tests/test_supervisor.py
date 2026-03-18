from __future__ import annotations

import unittest

from agents.supervisor import supervisor_node
from app import build_initial_state


class SupervisorBranchingTests(unittest.TestCase):
    def test_post_retrieval_plan_is_rewritten_by_company_sufficiency(self) -> None:
        cases = (
            (True, True, True, ["compare", "write", "validate"]),
            (True, False, True, ["skeptic_lges", "compare", "write", "validate"]),
            (True, True, False, ["skeptic_catl", "compare", "write", "validate"]),
            (True, False, False, ["skeptic_lges", "skeptic_catl", "compare", "write", "validate"]),
            (False, True, True, ["compare", "write", "validate"]),
        )

        for market_sufficient, lges_sufficient, catl_sufficient, expected_plan in cases:
            with self.subTest(
                market_sufficient=market_sufficient,
                lges_sufficient=lges_sufficient,
                catl_sufficient=catl_sufficient,
            ):
                state = build_initial_state("query")
                state["plan"] = [
                    "parallel_retrieval",
                    "skeptic_lges",
                    "skeptic_catl",
                    "compare",
                    "write",
                    "validate",
                ]
                state["market"]["synthesized_summary"] = "market done"
                state["market"]["retrieval_sufficient"] = market_sufficient
                state["companies"]["LGES"]["synthesized_summary"] = "lges done"
                state["companies"]["LGES"]["retrieval_sufficient"] = lges_sufficient
                state["companies"]["CATL"]["synthesized_summary"] = "catl done"
                state["companies"]["CATL"]["retrieval_sufficient"] = catl_sufficient

                result = supervisor_node(state)

                self.assertEqual(expected_plan, result["plan"])
                self.assertEqual(
                    "compare" if expected_plan[0] == "compare" else expected_plan[0],
                    result["runtime"]["current_phase"],
                )
                self.assertEqual(
                    not lges_sufficient,
                    result["companies"]["LGES"]["skeptic_review_required"],
                )
                self.assertEqual(
                    not catl_sufficient,
                    result["companies"]["CATL"]["skeptic_review_required"],
                )


if __name__ == "__main__":
    unittest.main()
