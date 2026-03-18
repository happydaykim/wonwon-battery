from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from utils.logging import THIRD_PARTY_LOG_LEVELS, configure_runtime_logging


class LoggingConfigTests(unittest.TestCase):
    def test_configure_runtime_logging_quiets_and_restores_third_party_loggers(self) -> None:
        logger_name = "httpx"
        logger = logging.getLogger(logger_name)
        original_level = logger.level
        self.addCleanup(logger.setLevel, original_level)

        configure_runtime_logging(quiet_third_party_logs=True)
        self.assertEqual(THIRD_PARTY_LOG_LEVELS[logger_name], logger.level)

        configure_runtime_logging(quiet_third_party_logs=False)
        self.assertEqual(logging.NOTSET, logger.level)

    def test_configure_runtime_logging_toggles_progress_bar_helpers(self) -> None:
        with patch(
            "huggingface_hub.utils.disable_progress_bars"
        ) as disable_hf_progress_mock, patch(
            "huggingface_hub.utils.enable_progress_bars"
        ) as enable_hf_progress_mock, patch(
            "transformers.utils.logging.disable_progress_bar"
        ) as disable_transformers_progress_mock, patch(
            "transformers.utils.logging.enable_progress_bar"
        ) as enable_transformers_progress_mock:
            configure_runtime_logging(quiet_third_party_logs=True)
            configure_runtime_logging(quiet_third_party_logs=False)

        disable_hf_progress_mock.assert_called_once()
        enable_hf_progress_mock.assert_called_once()
        disable_transformers_progress_mock.assert_called_once()
        enable_transformers_progress_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
