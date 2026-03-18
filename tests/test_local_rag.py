from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval.local_rag import LocalRAGRetriever


class LocalRAGRetrieverTests(unittest.TestCase):
    @patch("retrieval.local_rag.query_collection")
    @patch("retrieval.local_rag.load_embedding_backend")
    @patch("retrieval.local_rag.get_chroma_collection")
    def test_retrieve_flattens_metadata_and_dedupes_by_doc(
        self,
        mock_get_collection,
        mock_load_backend,
        mock_query_collection,
    ) -> None:
        mock_get_collection.return_value = object()
        mock_load_backend.return_value = {"backend": "stub"}
        mock_query_collection.return_value = {
            "documents": [[
                "doc-a chunk-1",
                "doc-a chunk-2",
                "doc-b chunk-1",
            ]],
            "metadatas": [[
                {
                    "doc_id": "doc-a",
                    "title": "Doc A",
                    "source_name": "SourceA",
                    "source_url": "https://example.com/a",
                },
                {
                    "doc_id": "doc-a",
                    "title": "Doc A",
                    "source_name": "SourceA",
                    "source_url": "https://example.com/a",
                },
                {
                    "doc_id": "doc-b",
                    "title": "Doc B",
                    "source_name": "SourceB",
                    "source_url": "https://example.com/b",
                },
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }

        retriever = LocalRAGRetriever(
            embedding_model="test-model",
            vector_store="chroma",
            persist_directory=Path("/tmp/chroma"),
        )

        results = retriever.retrieve("battery strategy", company_scope="MARKET", top_k=2)

        self.assertEqual(2, len(results))
        self.assertEqual("doc-a", results[0]["doc_id"])
        self.assertEqual("SourceA", results[0]["source_name"])
        self.assertEqual("https://example.com/a", results[0]["source_url"])
        self.assertEqual("doc-b", results[1]["doc_id"])
        self.assertEqual("SourceB", results[1]["source_name"])
        mock_query_collection.assert_called_once()
        self.assertEqual(8, mock_query_collection.call_args.kwargs["top_k"])
        self.assertEqual(
            {"company_scope": "MARKET"},
            mock_query_collection.call_args.kwargs["where"],
        )

    @patch("retrieval.local_rag.query_collection")
    @patch("retrieval.local_rag.load_embedding_backend")
    @patch("retrieval.local_rag.get_chroma_collection")
    def test_retrieve_includes_both_scope_documents_for_company_queries(
        self,
        mock_get_collection,
        mock_load_backend,
        mock_query_collection,
    ) -> None:
        mock_get_collection.return_value = object()
        mock_load_backend.return_value = {"backend": "stub"}
        mock_query_collection.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever = LocalRAGRetriever(
            embedding_model="test-model",
            vector_store="chroma",
            persist_directory=Path("/tmp/chroma"),
        )

        retriever.retrieve("battery strategy", company_scope="LGES", top_k=2)

        mock_query_collection.assert_called_once()
        self.assertEqual(
            {"company_scope": {"$in": ["LGES", "BOTH"]}},
            mock_query_collection.call_args.kwargs["where"],
        )


if __name__ == "__main__":
    unittest.main()
