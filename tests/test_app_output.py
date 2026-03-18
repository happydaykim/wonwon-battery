from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pymupdf

from app import _write_final_report_artifacts, build_initial_state
from config.settings import load_settings


class AppOutputTests(unittest.TestCase):
    def test_write_final_report_artifacts_persist_html_and_pdf_files(self) -> None:
        state = build_initial_state("전략 비교 분석 보고서 테스트")
        state["final_report"] = "## I. EXECUTIVE SUMMARY\n보고서 본문"
        for section_id in state["section_drafts"]:
            state["section_drafts"][section_id]["status"] = "drafted"

        state["section_drafts"]["summary"]["content"] = (
            "전기차 캐즘 장기화 속에서 양사의 다각화 전략을 비교했다."
        )
        state["section_drafts"]["market_background"]["content"] = "\n\n".join(
            [
                "### II.I 전기차 캐즘과 HEV 피벗\nEV 수요 둔화와 HEV 수요 확대가 병행되고 있다.",
                "### II.II K-배터리 업계의 포트폴리오 다각화 배경\n국내 업체들은 EV 외 수요처 확장을 추진한다.",
                "### II.III CATL의 원가/기술 전략 변화\nCATL은 원가와 기술 우위를 기반으로 확장 전략을 전개한다.",
            ]
        )
        state["section_drafts"]["lges_strategy"]["content"] = (
            "LGES는 북미 생산기지와 ESS 대응을 축으로 포트폴리오를 넓히고 있다."
        )
        state["section_drafts"]["catl_strategy"]["content"] = (
            "CATL은 원가 경쟁력과 기술 포트폴리오를 결합해 시장 대응 범위를 확장하고 있다."
        )
        state["section_drafts"]["strategy_comparison"]["content"] = "\n\n".join(
            [
                "### V.I 전략 방향 차이\nLGES는 지역 확장과 응용처 다변화가 두드러지고, CATL은 원가 및 기술 축이 상대적으로 강하다.",
                "\n".join(
                    [
                        "### V.II 데이터 기반 비교표",
                        "| 회사 | 전략 축 | 확보 근거 수 |",
                        "| --- | --- | ---: |",
                        "| LGES | 북미/ESS | 3 |",
                        "| CATL | 원가/기술 | 4 |",
                    ]
                ),
            ]
        )
        state["section_drafts"]["implications"]["content"] = (
            "양사는 모두 다각화를 추진하지만 실행 축과 강점의 조합이 다르므로 투자·협력 전략도 분리해 해석할 필요가 있다."
        )
        state["section_drafts"]["references"]["content"] = (
            "- Battery Institute(2026). *Battery Strategy Outlook*. https://example.com/report"
        )
        state["references"] = {
            "ref_1": {
                "ref_id": "ref_1",
                "doc_id": "doc_1",
                "citation_text": "- Battery Institute(2026). *Battery Strategy Outlook*. https://example.com/report",
                "reference_type": "report",
                "used_in_sections": ["summary", "references"],
            }
        }
        state["swot"] = {
            "LGES": {
                "strengths": ["북미 생산기지 확대"],
                "weaknesses": ["EV 수요 회복 속도 불확실성"],
                "opportunities": ["ESS 및 비EV 응용처 확대"],
                "threats": ["가격 경쟁 심화"],
            },
            "CATL": {
                "strengths": ["원가 경쟁력"],
                "weaknesses": ["지정학적 불확실성"],
                "opportunities": ["기술 포트폴리오 확장"],
                "threats": ["글로벌 규제 변화"],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(load_settings(), outputs_dir=Path(temp_dir))

            artifacts = _write_final_report_artifacts(
                state,
                settings=settings,
                thread_id="battery-strategy-test",
            )

            self.assertIsNotNone(artifacts)
            assert artifacts is not None
            self.assertTrue(artifacts.html_path.exists())
            self.assertIsNotNone(artifacts.pdf_path)
            assert artifacts.pdf_path is not None
            self.assertTrue(artifacts.pdf_path.exists())
            self.assertTrue(artifacts.html_path.name.startswith("battery-strategy-test_"))
            self.assertEqual(".html", artifacts.html_path.suffix)
            self.assertEqual(".pdf", artifacts.pdf_path.suffix)

            html = artifacts.html_path.read_text(encoding="utf-8")
            self.assertIn("Table 5-1. 데이터 기반 비교표", html)
            self.assertIn("Table 5-2. LGES SWOT Matrix", html)
            self.assertIn('class="swot-matrix"', html)
            self.assertNotIn("Contents", html)
            self.assertNotIn("Strategic Analysis Report", html)
            self.assertNotIn("전략 비교 분석 보고서 테스트", html)
            self.assertNotIn("문서 형식", html)
            self.assertNotIn("작성 목적", html)
            self.assertNotIn("1. 1. SUMMARY", html)
            self.assertIn("I. EXECUTIVE SUMMARY", html)

            pdf = pymupdf.open(str(artifacts.pdf_path))
            page_texts = [page.get_text() for page in pdf]
            normalized_page_texts = [" ".join(page_text.split()) for page_text in page_texts]
            first_block_tops = []
            for page in pdf:
                blocks = page.get_text("blocks")
                visible_blocks = [block for block in blocks if str(block[4]).strip()]
                first_block_tops.append(min(block[1] for block in visible_blocks))
            pdf_text = "\n".join(page_texts)
            self.assertIn("배터리 시장 전략 분석 보고서", pdf_text)
            self.assertIn("LGES", pdf_text)
            self.assertIn("Battery Strategy Outlook", pdf_text)
            self.assertGreaterEqual(len(page_texts), 1)
            self.assertGreater(first_block_tops[0], 45)
            self.assertTrue(all(top > 40 for top in first_block_tops))
            self.assertIn("I. EXECUTIVE SUMMARY", normalized_page_texts[0])

            compact_pdf_text = "".join(pdf_text.split())
            expected_headings = [
                "I. EXECUTIVE SUMMARY",
                "II. 시장 배경",
                "III. LG에너지솔루션의 포트폴리오 다각화 전략과 핵심 경쟁력",
                "IV. CATL의 포트폴리오 다각화 전략과 핵심 경쟁력",
                "V. 핵심 전략 비교 분석",
                "V.III SWOT 분석",
                "VI. 종합 시사점",
                "VII. REFERENCE",
            ]
            heading_positions = [
                compact_pdf_text.index("".join(heading.split())) for heading in expected_headings
            ]
            self.assertEqual(sorted(heading_positions), heading_positions)

    def test_write_final_report_artifacts_skips_empty_report(self) -> None:
        state = build_initial_state("query")

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(load_settings(), outputs_dir=Path(temp_dir))

            artifacts = _write_final_report_artifacts(
                state,
                settings=settings,
                thread_id="battery-strategy-test",
            )

            self.assertIsNone(artifacts)
            self.assertEqual([], list(Path(temp_dir).iterdir()))


if __name__ == "__main__":
    unittest.main()
