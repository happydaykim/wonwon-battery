from __future__ import annotations

from functools import lru_cache
from threading import Lock
from typing import Any


EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_ALIASES = {
    "Qwen3-Embedding-0.6B": EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_ID: EMBEDDING_MODEL_ID,
}
_EMBEDDING_BACKEND_LOCK = Lock()


def normalize_embedding_model_id(model_id: str | None) -> str:
    resolved = (model_id or "").strip()
    if not resolved:
        return EMBEDDING_MODEL_ID
    return EMBEDDING_MODEL_ALIASES.get(resolved, resolved)


@lru_cache(maxsize=1)
def load_embedding_backend(model_id: str = EMBEDDING_MODEL_ID) -> dict[str, Any]:
    """Load the shared embedding backend once per model id."""
    normalized_model_id = normalize_embedding_model_id(model_id)
    with _EMBEDDING_BACKEND_LOCK:
        return _load_embedding_backend_cached(normalized_model_id)


@lru_cache(maxsize=4)
def _load_embedding_backend_cached(model_id: str) -> dict[str, Any]:
    """Build the shared LangChain Hugging Face embedding backend."""
    try:
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "langchain_huggingface and torch are required for embedding generation."
        ) from exc

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={
            "device": device,
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )
    return {
        "device": device,
        "model_id": model_id,
        "embeddings": embeddings,
    }


def embed_texts(texts: list[str], *, backend: dict[str, Any] | None = None) -> list[list[float]]:
    """Create dense normalized embeddings for the given texts."""
    resolved_backend = backend or load_embedding_backend()
    return resolved_backend["embeddings"].embed_documents(texts)


def embed_query(query: str, *, backend: dict[str, Any] | None = None) -> list[float]:
    """Embed a single query string with the shared backend."""
    resolved_backend = backend or load_embedding_backend()
    return resolved_backend["embeddings"].embed_query(query)
