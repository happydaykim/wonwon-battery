from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config.settings import load_settings
from retrieval.embeddings import normalize_embedding_model_id


class EmbeddingConfigTests(unittest.TestCase):
    def test_load_settings_normalizes_legacy_embedding_alias(self) -> None:
        with patch("config.settings._load_dotenv", return_value=False), patch.dict(
            os.environ,
            {"EMBEDDING_MODEL": "Qwen3-Embedding-0.6B"},
            clear=True,
        ):
            settings = load_settings()

        self.assertEqual("Qwen/Qwen3-Embedding-0.6B", settings.embedding_model)
        self.assertTrue(settings.quiet_third_party_logs)
        self.assertTrue(settings.local_rag_prewarm_enabled)

    def test_normalize_embedding_model_id_keeps_canonical_id(self) -> None:
        self.assertEqual(
            "Qwen/Qwen3-Embedding-0.6B",
            normalize_embedding_model_id("Qwen/Qwen3-Embedding-0.6B"),
        )

    def test_load_settings_reads_runtime_log_and_prewarm_flags(self) -> None:
        with patch("config.settings._load_dotenv", return_value=False), patch.dict(
            os.environ,
            {
                "QUIET_THIRD_PARTY_LOGS": "false",
                "LOCAL_RAG_PREWARM_ENABLED": "false",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertFalse(settings.quiet_third_party_logs)
        self.assertFalse(settings.local_rag_prewarm_enabled)


if __name__ == "__main__":
    unittest.main()
