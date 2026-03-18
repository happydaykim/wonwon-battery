from __future__ import annotations

import unittest

from config.settings import load_settings
from utils.prompt_loader import load_prompt


BANNED_PLACEHOLDER_PHRASES = (
    "TODO 수준의 skeleton",
    "skeleton만 남긴다",
    "실제 검색 구현은 하지 않는다",
    "TODO-model-name",
)

PROMPT_FILES = (
    "planner.md",
    "supervisor.md",
    "market.md",
    "lges.md",
    "catl.md",
    "skeptic.md",
    "compare_swot.md",
    "writer.md",
    "validator.md",
)


class PromptContractTests(unittest.TestCase):
    def test_all_prompts_are_non_placeholder_contracts(self) -> None:
        for prompt_name in PROMPT_FILES:
            prompt_text = load_prompt(prompt_name)
            self.assertGreaterEqual(
                len([line for line in prompt_text.splitlines() if line.strip()]),
                5,
                msg=f"{prompt_name} is too thin to function as a meaningful prompt/contract.",
            )
            for phrase in BANNED_PLACEHOLDER_PHRASES:
                self.assertNotIn(
                    phrase,
                    prompt_text,
                    msg=f"{prompt_name} still contains placeholder language: {phrase}",
                )

    def test_writer_prompt_contains_required_report_constraints(self) -> None:
        prompt_text = load_prompt("writer.md")
        self.assertIn("900자", prompt_text)
        self.assertIn("### 2.1 전기차 캐즘과 HEV 피벗", prompt_text)
        self.assertIn("### 2.2 K-배터리 업계의 포트폴리오 다각화 배경", prompt_text)
        self.assertIn("### 2.3 CATL의 원가/기술 전략 변화", prompt_text)
        self.assertIn("정보 부족/추가 검증 필요", prompt_text)

    def test_compare_prompt_requires_gap_disclosure_and_no_fabrication(self) -> None:
        prompt_text = load_prompt("compare_swot.md")
        self.assertIn("정보 부족/추가 검증 필요", prompt_text)
        self.assertIn("새로 만들지 않는다", prompt_text)
        self.assertIn("counter-evidence", prompt_text)

    def test_report_model_default_is_gpt4o(self) -> None:
        settings = load_settings()
        self.assertEqual("gpt-4o", settings.report_llm_model)


if __name__ == "__main__":
    unittest.main()
