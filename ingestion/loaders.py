from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
import time
from typing import Any

import fitz
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
VISION_PROMPTS_DIR = PROJECT_ROOT / "prompts" / "vision"


def resolve_pdf_paths(input_path: Path) -> list[Path]:
    """Resolve a single PDF or all PDFs directly under a directory."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {input_path}")
        return [input_path]

    pdf_paths = sorted(path for path in input_path.glob("*.pdf") if path.is_file())
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under: {input_path}")
    return pdf_paths


def load_metadata(
    pdf_path: Path,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
) -> dict[str, Any]:
    """Load the metadata JSON that matches the PDF stem."""
    metadata_path = metadata_dir / f"{pdf_path.stem}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata JSON not found for {pdf_path.name}: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["author"] = normalize_optional_metadata_value(metadata.get("author"))
    metadata["stance"] = normalize_optional_metadata_value(metadata.get("stance"))
    metadata["language"] = normalize_optional_metadata_value(metadata.get("language"))
    metadata["pdf_path"] = str(pdf_path)
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def normalize_optional_metadata_value(value: Any) -> Any:
    """Normalize placeholder-like optional metadata values to None."""
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.lower() in {"null", "none", "n/a", "unknown"}:
            return None
        return normalized

    return value


def extract_page_documents(
    pdf_path: Path,
    base_metadata: dict[str, Any],
    *,
    use_vision: bool = False,
    vision_model_id: str | None = None,
    vision_max_new_tokens: int = 300,
    show_progress: bool = True,
) -> list[Document]:
    """Extract one LangChain Document per PDF page."""
    return list(
        iter_page_documents(
            pdf_path,
            base_metadata,
            use_vision=use_vision,
            vision_model_id=vision_model_id,
            vision_max_new_tokens=vision_max_new_tokens,
            show_progress=show_progress,
        )
    )


def iter_page_documents(
    pdf_path: Path,
    base_metadata: dict[str, Any],
    *,
    use_vision: bool = False,
    vision_model_id: str | None = None,
    vision_max_new_tokens: int = 300,
    show_progress: bool = True,
):
    """Yield one LangChain Document per PDF page."""
    documents: list[Document] = []
    vision_backend = None
    if use_vision:
        if show_progress:
            print(f"[vision] loading model for {pdf_path.name}", flush=True)
        vision_backend = load_qwen_vision_backend(vision_model_id)

    with fitz.open(pdf_path) as pdf_document:
        total_pages = len(pdf_document)
        started_at = time.perf_counter()
        if show_progress:
            mode = "vision+text" if use_vision else "text-only"
            print(
                f"[pdf] processing {pdf_path.name} | pages={total_pages} | mode={mode}",
                flush=True,
            )
        for page_index, page in enumerate(pdf_document, start=1):
            elapsed_seconds = time.perf_counter() - started_at
            processed_pages = max(page_index - 1, 0)
            average_seconds = (
                elapsed_seconds / processed_pages if processed_pages > 0 else None
            )
            remaining_pages = total_pages - page_index + 1
            eta_seconds = (
                average_seconds * remaining_pages if average_seconds is not None else None
            )
            if show_progress:
                print(
                    f"[page] {pdf_path.name} | {page_index}/{total_pages} | elapsed={format_duration(elapsed_seconds)} | eta={format_duration(eta_seconds)}",
                    flush=True,
                )
            source_text = page.get_text("text").strip()
            visual_text = None
            if vision_backend is not None:
                visual_text = analyze_page_with_qwen_vision(
                    page,
                    processor=vision_backend["processor"],
                    model=vision_backend["model"],
                    device=vision_backend["device"],
                    prompt=build_vision_prompt(base_metadata),
                    max_new_tokens=vision_max_new_tokens,
                )

            combined_text = build_combined_text(source_text, visual_text)
            if not combined_text:
                continue

            yield Document(
                page_content=combined_text,
                metadata={
                    **base_metadata,
                    "source": pdf_path.name,
                    "page_number": page_index,
                    "page_or_chunk": f"p.{page_index}",
                    "source_text": source_text or None,
                    "visual_text": visual_text,
                    "combined_text": combined_text,
                },
            )


def format_duration(seconds: float | None) -> str:
    """Format elapsed or remaining seconds for progress logs."""
    if seconds is None:
        return "estimating"

    rounded = max(int(round(seconds)), 0)
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_combined_text(source_text: str, visual_text: str | None) -> str:
    """Combine extracted text with optional visual description."""
    normalized_source = source_text.strip()
    normalized_visual = (visual_text or "").strip()

    if normalized_source and normalized_visual:
        return (
            f"{normalized_source}\n\n"
            "[VISUAL CONTENT]\n"
            f"{normalized_visual}"
        ).strip()
    if normalized_visual:
        return f"[VISUAL CONTENT]\n{normalized_visual}".strip()
    return normalized_source


def load_qwen_vision_backend(model_id: str | None = None) -> dict[str, Any]:
    """Load the Qwen vision model once for page-level visual descriptions."""
    resolved_model_id = (
        model_id
        or os.getenv("VISION_MODEL")
        or "Qwen/Qwen2-VL-2B-Instruct"
    )
    return _load_qwen_vision_backend_cached(resolved_model_id)


@lru_cache(maxsize=2)
def _load_qwen_vision_backend_cached(model_id: str) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "transformers and torch are required for Qwen vision analysis."
        ) from exc

    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    return {
        "model_id": model_id,
        "processor": processor,
        "model": model,
        "device": device,
    }


def analyze_page_with_qwen_vision(
    page: fitz.Page,
    *,
    processor: Any,
    model: Any,
    device: str,
    prompt: str | None = None,
    max_new_tokens: int = 300,
) -> str:
    """Render a PDF page in-memory and ask Qwen vision to describe it."""
    try:
        import torch
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Pillow and torch are required for Qwen vision analysis."
        ) from exc

    if prompt is None:
        prompt = (
            "You are reading a PDF page from a business document. "
            "Describe charts, tables, diagrams, callout boxes, and visually important "
            "content in a factual way. Include titles, axes, units, legends, and key "
            "numbers when visible. Do not repeat the full body text. Keep the output concise."
        )

    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def build_vision_prompt(metadata: dict[str, Any]) -> str:
    """Return the exact vision prompt text used for page analysis."""
    _ = metadata
    return load_vision_prompt_template("common")


@lru_cache(maxsize=None)
def load_vision_prompt_template(template_name: str) -> str:
    """Load a vision prompt template by name, falling back to 'other'."""
    prompt_path = VISION_PROMPTS_DIR / f"{template_name}.md"
    if not prompt_path.exists():
        prompt_path = VISION_PROMPTS_DIR / "other.md"
    return prompt_path.read_text(encoding="utf-8").strip()
