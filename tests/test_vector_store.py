from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval.vector_store import _get_cached_chroma_collection, get_chroma_collection


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.collections: dict[str, object] = {}

    def get_or_create_collection(self, *, name: str) -> object:
        self.calls.append(name)
        self.collections.setdefault(name, object())
        return self.collections[name]


class VectorStoreTests(unittest.TestCase):
    def tearDown(self) -> None:
        _get_cached_chroma_collection.cache_clear()

    def test_get_chroma_collection_reuses_cached_collection_for_same_path(self) -> None:
        fake_client = _FakeClient()

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "retrieval.vector_store._create_persistent_client",
            return_value=fake_client,
        ) as create_client_mock:
            chroma_dir = Path(temp_dir)
            first = get_chroma_collection(
                chroma_dir=chroma_dir,
                collection_name="battery_document",
            )
            second = get_chroma_collection(
                chroma_dir=chroma_dir,
                collection_name="battery_document",
            )

        self.assertIs(first, second)
        create_client_mock.assert_called_once()
        self.assertEqual(["battery_document"], fake_client.calls)


if __name__ == "__main__":
    unittest.main()
