from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_text_splitter(
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def build_chunk_id(
    *,
    doc_id: str,
    page_number: int,
    chunk_index: int,
) -> str:
    return f"{doc_id}_p{page_number:03d}_c{chunk_index:03d}"


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Keep one chunk per page by default and split only oversized pages."""
    chunked_documents: list[Document] = []
    splitter = None

    for document in documents:
        if len(document.page_content) <= chunk_size:
            page_chunks = [document]
        else:
            if splitter is None:
                splitter = build_text_splitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            page_chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(page_chunks, start=1):
            metadata = dict(chunk.metadata)
            doc_id = str(metadata["doc_id"])
            page_number = int(metadata["page_number"])

            section_title = metadata.get("section_title")
            if section_title is None:
                section_title = metadata.get("title")

            chunk_text = chunk.page_content.strip()
            chunk_id = build_chunk_id(
                doc_id=doc_id,
                page_number=page_number,
                chunk_index=chunk_index,
            )

            chunked_documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "page_number": page_number,
                        "page_or_chunk": f"p.{page_number}",
                        "section_title": section_title,
                        "title": metadata.get("title"),
                        "author": metadata.get("author"),
                        "source_name": metadata.get("source_name"),
                        "source_url": metadata.get("source_url"),
                        "published_at": metadata.get("published_at"),
                        "doc_type": metadata.get("doc_type"),
                        "company_scope": metadata.get("company_scope"),
                        "stance": metadata.get("stance"),
                        "language": metadata.get("language"),
                        "source": metadata.get("source"),
                        "metadata_path": metadata.get("metadata_path"),
                        "pdf_path": metadata.get("pdf_path"),
                        "source_text": metadata.get("source_text"),
                        "visual_text": metadata.get("visual_text"),
                        "combined_text": chunk_text,
                    },
                )
            )

    return chunked_documents
