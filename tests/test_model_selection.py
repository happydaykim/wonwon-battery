from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from agents.compare_swot import _create_compare_chain
from agents.writer import _create_writer_chain
from config.settings import load_settings


class _FakeLLM:
    def with_structured_output(self, schema: object) -> RunnableLambda:
        _ = schema
        return RunnableLambda(lambda payload: payload)


class ModelSelectionTests(unittest.TestCase):
    def test_writer_chain_uses_report_llm_model(self) -> None:
        settings = replace(
            load_settings(),
            llm_provider="openai",
            llm_model="planner-model",
            report_llm_provider="openai",
            report_llm_model="gpt-4o",
        )

        with patch("agents.writer.load_settings", return_value=settings), patch(
            "agents.writer.init_chat_model",
            return_value=_FakeLLM(),
        ) as init_chat_model_mock:
            chain = _create_writer_chain()

        self.assertIsNotNone(chain)
        init_chat_model_mock.assert_called_once_with(
            "gpt-4o",
            model_provider="openai",
            temperature=0,
        )

    def test_compare_chain_uses_report_llm_model(self) -> None:
        settings = replace(
            load_settings(),
            llm_provider="openai",
            llm_model="planner-model",
            report_llm_provider="openai",
            report_llm_model="gpt-4o",
        )

        with patch("agents.compare_swot.load_settings", return_value=settings), patch(
            "agents.compare_swot.init_chat_model",
            return_value=_FakeLLM(),
        ) as init_chat_model_mock:
            chain = _create_compare_chain()

        self.assertIsNotNone(chain)
        init_chat_model_mock.assert_called_once_with(
            "gpt-4o",
            model_provider="openai",
            temperature=0,
        )


if __name__ == "__main__":
    unittest.main()
