from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from app import _prewarm_local_rag_resources
from config.settings import load_settings


class AppStartupTests(unittest.TestCase):
    def test_prewarm_local_rag_resources_uses_collection_and_embedding_backend(self) -> None:
        settings = replace(load_settings(), local_rag_prewarm_enabled=True)

        with patch("app.LocalRAGRetriever.from_settings") as from_settings_mock, patch(
            "app.get_chroma_collection"
        ) as get_collection_mock, patch(
            "app.load_embedding_backend"
        ) as load_embedding_backend_mock:
            fake_retriever = from_settings_mock.return_value
            fake_retriever.persist_directory = settings.chroma_persist_directory
            fake_retriever.collection_name = "battery_document"
            fake_retriever.embedding_model = settings.embedding_model

            _prewarm_local_rag_resources(settings)

        from_settings_mock.assert_called_once_with(settings)
        get_collection_mock.assert_called_once_with(
            chroma_dir=settings.chroma_persist_directory,
            collection_name="battery_document",
        )
        load_embedding_backend_mock.assert_called_once_with(settings.embedding_model)

    def test_prewarm_local_rag_resources_skips_when_disabled(self) -> None:
        settings = replace(load_settings(), local_rag_prewarm_enabled=False)

        with patch("app.LocalRAGRetriever.from_settings") as from_settings_mock, patch(
            "app.get_chroma_collection"
        ) as get_collection_mock, patch(
            "app.load_embedding_backend"
        ) as load_embedding_backend_mock:
            _prewarm_local_rag_resources(settings)

        from_settings_mock.assert_not_called()
        get_collection_mock.assert_not_called()
        load_embedding_backend_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
