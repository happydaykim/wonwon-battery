from __future__ import annotations

from functools import lru_cache
from typing import Any


EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


@lru_cache(maxsize=1)
def load_embedding_backend(model_id: str = EMBEDDING_MODEL_ID) -> dict[str, Any]:
    """Load the shared LangChain Hugging Face embedding backend."""
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
