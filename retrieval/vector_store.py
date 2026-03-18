from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from retrieval.embeddings import embed_query, embed_texts, load_embedding_backend
from retrieval.vector_schema import DEFAULT_COLLECTION_NAME, VECTOR_METADATA_FIELDS


DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
_CHROMA_COLLECTION_LOCK = Lock()


def get_chroma_collection(
    *,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    """Open the shared persistent Chroma collection."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir_str = str(chroma_dir.resolve())
    with _CHROMA_COLLECTION_LOCK:
        return _get_cached_chroma_collection(chroma_dir_str, collection_name)


def _create_persistent_client(chroma_dir: str) -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("chromadb is required for vector storage.") from exc

    return chromadb.PersistentClient(path=chroma_dir)


@lru_cache(maxsize=8)
def _get_cached_chroma_collection(chroma_dir: str, collection_name: str) -> Any:
    client = _create_persistent_client(chroma_dir)
    return client.get_or_create_collection(name=collection_name)


def build_chroma_metadata(document: Any) -> dict[str, Any]:
    """Keep Chroma metadata flat and aligned with the shared vector schema."""
    metadata = document.metadata
    flat_metadata = {
        field: metadata.get(field)
        for field in VECTOR_METADATA_FIELDS
    }
    return {key: value for key, value in flat_metadata.items() if value is not None}


def upsert_chunk_documents(
    documents: list[Any],
    *,
    collection: Any,
    embedding_backend: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Embed and upsert chunk documents into Chroma, skipping existing ids."""
    if not documents:
        return {"upserted_count": 0, "skipped_count": 0}

    backend = embedding_backend or load_embedding_backend()
    existing = collection.get(
        ids=[document.metadata["chunk_id"] for document in documents],
        include=[],
    )
    existing_ids = set(existing.get("ids", []))
    documents_to_upsert = [
        document
        for document in documents
        if document.metadata["chunk_id"] not in existing_ids
    ]
    skipped_count = len(documents) - len(documents_to_upsert)
    if not documents_to_upsert:
        return {"upserted_count": 0, "skipped_count": skipped_count}

    texts = [document.page_content for document in documents_to_upsert]
    embeddings = embed_texts(texts, backend=backend)
    collection.upsert(
        ids=[document.metadata["chunk_id"] for document in documents_to_upsert],
        documents=texts,
        metadatas=[build_chroma_metadata(document) for document in documents_to_upsert],
        embeddings=embeddings,
    )
    return {
        "upserted_count": len(documents_to_upsert),
        "skipped_count": skipped_count,
    }


def query_collection(
    query: str,
    *,
    collection: Any,
    embedding_backend: dict[str, Any] | None = None,
    where: dict[str, Any] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run a Chroma similarity query using the shared embedding backend."""
    backend = embedding_backend or load_embedding_backend()
    query_embedding = embed_query(query, backend=backend)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )
