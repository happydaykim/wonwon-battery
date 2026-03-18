from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from langchain_core.documents import Document

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from ingestion.chunking import chunk_documents
    from ingestion.loaders import (
        DEFAULT_METADATA_DIR,
        DEFAULT_PDF_DIR,
        iter_page_documents,
        load_metadata,
        resolve_pdf_paths,
    )
    from retrieval.embeddings import (
        EMBEDDING_MODEL_ID,
        load_embedding_backend,
    )
    from retrieval.vector_schema import DEFAULT_COLLECTION_NAME
    from retrieval.vector_store import (
        DEFAULT_CHROMA_DIR,
        get_chroma_collection,
        upsert_chunk_documents,
    )
except ModuleNotFoundError:  # pragma: no cover - allows direct script execution
    from chunking import chunk_documents
    from loaders import (
        DEFAULT_METADATA_DIR,
        DEFAULT_PDF_DIR,
        iter_page_documents,
        load_metadata,
        resolve_pdf_paths,
    )
    from retrieval.embeddings import (
        EMBEDDING_MODEL_ID,
        load_embedding_backend,
    )
    from retrieval.vector_schema import DEFAULT_COLLECTION_NAME
    from retrieval.vector_store import (
        DEFAULT_CHROMA_DIR,
        get_chroma_collection,
        upsert_chunk_documents,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def serialize_document(document: Any) -> dict[str, Any]:
    """Convert a LangChain document to a JSON-serializable dict."""
    return {
        "page_content": document.page_content,
        "metadata": document.metadata,
    }


def deserialize_document(payload: dict[str, Any]) -> Document:
    """Convert a saved JSON payload back into a LangChain document."""
    return Document(
        page_content=payload["page_content"],
        metadata=payload["metadata"],
    )


def load_documents_from_jsonl(path: Path) -> list[Document]:
    """Load LangChain documents from a JSONL file."""
    if not path.exists():
        return []

    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            normalized = line.strip()
            if not normalized:
                continue
            documents.append(deserialize_document(json.loads(normalized)))
    return documents


def initialize_output_files(
    doc_id: str,
    *,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    clear_existing: bool = True,
) -> dict[str, Path]:
    """Prepare output paths and clear previous incremental artifacts."""
    doc_output_dir = output_dir / doc_id
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    pages_path = doc_output_dir / "pages.jsonl"
    chunks_path = doc_output_dir / "chunks.jsonl"
    manifest_path = doc_output_dir / "manifest.json"
    if clear_existing:
        for path in (pages_path, chunks_path, manifest_path):
            if path.exists():
                path.unlink()
    return {
        "doc_output_dir": doc_output_dir,
        "pages_path": pages_path,
        "chunks_path": chunks_path,
        "manifest_path": manifest_path,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a single JSON record as one JSONL line."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False))
        file.write("\n")


def write_manifest(
    *,
    manifest_path: Path,
    doc_id: str,
    pdf_path: str,
    pages_path: Path,
    chunks_path: Path,
    page_count: int,
    chunk_count: int,
    collection_name: str,
    chroma_path: Path,
) -> None:
    """Write or update the manifest for an incrementally processed PDF."""
    manifest_payload = {
        "doc_id": doc_id,
        "pdf_path": pdf_path,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "pages_path": str(pages_path),
        "chunks_path": str(chunks_path),
        "collection_name": collection_name,
        "chroma_path": str(chroma_path),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_pdf(
    pdf_path: Path,
    *,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    use_vision: bool = False,
    vision_model_id: str | None = None,
    vision_max_new_tokens: int = 300,
    show_progress: bool = True,
    embedding_backend: dict[str, Any] | None = None,
    collection: Any | None = None,
) -> dict[str, Any]:
    """Load metadata, extract page documents, and return chunked documents."""
    if show_progress:
        print(f"[start] pdf={pdf_path}", flush=True)
    metadata = load_metadata(pdf_path, metadata_dir=metadata_dir)
    resolved_embedding_backend = embedding_backend
    resolved_collection = collection
    if resolved_embedding_backend is None or resolved_collection is None:
        if show_progress:
            print(
                f"[embedding] loading {EMBEDDING_MODEL_ID} and opening collection={collection_name}",
                flush=True,
            )
        resolved_embedding_backend = load_embedding_backend()
        resolved_collection = get_chroma_collection(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
        )
    existing_paths = initialize_output_files(
        metadata["doc_id"],
        output_dir=output_dir,
        clear_existing=False,
    )
    existing_page_documents = load_documents_from_jsonl(existing_paths["pages_path"])
    existing_chunked_documents = load_documents_from_jsonl(existing_paths["chunks_path"])
    if existing_page_documents and existing_chunked_documents:
        if show_progress:
            print(
                f"[skip-chunking] doc_id={metadata['doc_id']} | reuse pages={len(existing_page_documents)} | chunks={len(existing_chunked_documents)}",
                flush=True,
            )
        upsert_result = upsert_chunk_documents(
            existing_chunked_documents,
            collection=resolved_collection,
            embedding_backend=resolved_embedding_backend,
        )
        write_manifest(
            manifest_path=existing_paths["manifest_path"],
            doc_id=metadata["doc_id"],
            pdf_path=str(pdf_path),
            pages_path=existing_paths["pages_path"],
            chunks_path=existing_paths["chunks_path"],
            page_count=len(existing_page_documents),
            chunk_count=len(existing_chunked_documents),
            collection_name=collection_name,
            chroma_path=chroma_dir,
        )
        if show_progress:
            print(
                f"[chroma] doc_id={metadata['doc_id']} | upserted={upsert_result['upserted_count']} | skipped_existing={upsert_result['skipped_count']}",
                flush=True,
            )
        return {
            "pdf_path": str(pdf_path),
            "metadata": metadata,
            "page_documents": existing_page_documents,
            "chunked_documents": existing_chunked_documents,
            "saved_paths": {key: str(value) for key, value in existing_paths.items()},
        }

    saved_paths = initialize_output_files(
        metadata["doc_id"],
        output_dir=output_dir,
        clear_existing=True,
    )
    page_documents: list[Any] = []
    chunked_documents: list[Any] = []

    for page_document in iter_page_documents(
        pdf_path,
        metadata,
        use_vision=use_vision,
        vision_model_id=vision_model_id,
        vision_max_new_tokens=vision_max_new_tokens,
        show_progress=show_progress,
    ):
        page_documents.append(page_document)
        append_jsonl(saved_paths["pages_path"], serialize_document(page_document))

        page_chunks = chunk_documents(
            [page_document],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        upsert_result = upsert_chunk_documents(
            page_chunks,
            collection=resolved_collection,
            embedding_backend=resolved_embedding_backend,
        )
        chunked_documents.extend(page_chunks)
        for chunk_document in page_chunks:
            append_jsonl(saved_paths["chunks_path"], serialize_document(chunk_document))

        write_manifest(
            manifest_path=saved_paths["manifest_path"],
            doc_id=metadata["doc_id"],
            pdf_path=str(pdf_path),
            pages_path=saved_paths["pages_path"],
            chunks_path=saved_paths["chunks_path"],
            page_count=len(page_documents),
            chunk_count=len(chunked_documents),
            collection_name=collection_name,
            chroma_path=chroma_dir,
        )
        if show_progress:
            print(
                f"[saved-page] doc_id={metadata['doc_id']} | page={page_document.metadata['page_number']} | total_chunks={len(chunked_documents)} | upserted={upsert_result['upserted_count']} | skipped_existing={upsert_result['skipped_count']}",
                flush=True,
            )

    if show_progress:
        print(
            f"[done] pdf={pdf_path.name} | pages={len(page_documents)} | chunks={len(chunked_documents)}",
            flush=True,
        )
        print(
            f"[saved] doc_id={metadata['doc_id']} | pages={saved_paths['pages_path']} | chunks={saved_paths['chunks_path']}",
            flush=True,
        )

    return {
        "pdf_path": str(pdf_path),
        "metadata": metadata,
        "page_documents": page_documents,
        "chunked_documents": chunked_documents,
        "saved_paths": {key: str(value) for key, value in saved_paths.items()},
    }


def process_pdfs(
    input_path: Path = DEFAULT_PDF_DIR,
    *,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    use_vision: bool = False,
    vision_model_id: str | None = None,
    vision_max_new_tokens: int = 300,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Process one PDF or a directory of PDFs through the chunking stage."""
    pdf_paths = resolve_pdf_paths(input_path)
    if show_progress:
        print(f"[batch] found {len(pdf_paths)} pdf(s)", flush=True)
        print(
            f"[embedding] loading {EMBEDDING_MODEL_ID} and opening collection={collection_name}",
            flush=True,
        )
    embedding_backend = load_embedding_backend()
    collection = get_chroma_collection(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
    )
    return [
        process_pdf(
            pdf_path,
            metadata_dir=metadata_dir,
            output_dir=output_dir,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_vision=use_vision,
            vision_model_id=vision_model_id,
            vision_max_new_tokens=vision_max_new_tokens,
            show_progress=show_progress,
            embedding_backend=embedding_backend,
            collection=collection,
        )
        for pdf_path in pdf_paths
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract PDF pages and split them into text chunks."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="Path to a PDF file or a directory containing PDFs.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Directory containing metadata JSON files matched by PDF stem.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Characters of overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory to save processed page/chunk JSON files.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Persistent directory for the single Chroma collection.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Single Chroma collection name used for all documents.",
    )
    parser.add_argument(
        "--use-vision",
        action="store_true",
        help="Render each page and append Qwen vision descriptions to the text.",
    )
    parser.add_argument(
        "--vision-model-id",
        default=None,
        help="Hugging Face model id for the Qwen vision model.",
    )
    parser.add_argument(
        "--vision-max-new-tokens",
        type=int,
        default=300,
        help="Maximum tokens to generate per page for visual descriptions.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    results = process_pdfs(
        args.input_path,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_vision=args.use_vision,
        vision_model_id=args.vision_model_id,
        vision_max_new_tokens=args.vision_max_new_tokens,
    )

    for result in results:
        print(
            " | ".join(
                [
                    f"pdf={result['pdf_path']}",
                    f"pages={len(result['page_documents'])}",
                    f"chunks={len(result['chunked_documents'])}",
                    f"doc_id={result['metadata'].get('doc_id')}",
                    f"saved={result['saved_paths']['chunks_path']}",
                    f"collection={args.collection_name}",
                ]
            )
        )


if __name__ == "__main__":
    main()
