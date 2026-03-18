from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
import re

from config.settings import Settings
from schemas.state import ReportState, SWOTState


SECTION_ORDER = (
    "summary",
    "market_background",
    "lges_strategy",
    "catl_strategy",
    "strategy_comparison",
    "swot",
    "implications",
    "references",
)

SECTION_HEADINGS = {
    "summary": "I. EXECUTIVE SUMMARY",
    "market_background": "II. 시장 배경",
    "lges_strategy": "III. LG에너지솔루션의 포트폴리오 다각화 전략과 핵심 경쟁력",
    "catl_strategy": "IV. CATL의 포트폴리오 다각화 전략과 핵심 경쟁력",
    "strategy_comparison": "V. 핵심 전략 비교 분석",
    "swot": "V.III SWOT 분석",
    "implications": "VI. 종합 시사점",
    "references": "VII. REFERENCE",
}

URL_PATTERN = re.compile(r"https?://[^\s<]+")
STRONG_PATTERN = re.compile(r"\*\*([^*\n]+)\*\*")
EMPHASIS_PATTERN = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
MM_TO_PT = 72 / 25.4
PDF_FOOTER_RESERVED_MM = 16
PDF_PAGE_NUMBER_BOTTOM_MM = 5
PDF_PAGE_NUMBER_HEIGHT_MM = 6


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    html_path: Path
    pdf_path: Path | None
    pdf_error: str | None = None


@dataclass(frozen=True, slots=True)
class MarkdownTable:
    headers: list[str]
    alignments: list[str]
    rows: list[list[str]]


def write_report_artifacts(
    result: ReportState,
    *,
    settings: Settings,
    thread_id: str,
) -> ReportArtifacts | None:
    final_report = result.get("final_report")
    if not final_report:
        return None

    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now()
    stem = f"{thread_id}_{generated_at.strftime('%Y%m%d_%H%M%S')}"
    html_path = settings.outputs_dir / f"{stem}.html"
    pdf_path = settings.outputs_dir / f"{stem}.pdf"

    screen_html = build_report_html(
        result,
        generated_at=generated_at,
        output_mode="screen",
    )
    html_path.write_text(screen_html, encoding="utf-8")

    try:
        pdf_html = build_report_html(
            result,
            generated_at=generated_at,
            output_mode="pdf",
        )
        _write_pdf_from_html(pdf_html, pdf_path)
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        if pdf_path.exists():
            pdf_path.unlink()
        return ReportArtifacts(
            html_path=html_path,
            pdf_path=None,
            pdf_error=str(exc),
        )

    return ReportArtifacts(html_path=html_path, pdf_path=pdf_path)


def build_report_html(
    result: ReportState,
    *,
    generated_at: datetime | None = None,
    output_mode: str = "screen",
) -> str:
    created_at = generated_at or datetime.now()
    report_title = "배터리 시장 전략 분석 보고서"
    sections_html = "\n".join(_render_section(result, section_id) for section_id in SECTION_ORDER)
    created_label = created_at.strftime("%Y-%m-%d")
    body_class = "pdf-export" if output_mode == "pdf" else "screen-export"

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="ko">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(report_title)}</title>",
            "<style>",
            _build_report_css(),
            "</style>",
            "</head>",
            f'<body class="{body_class}">',
            '<main class="report-sheet">',
            '<header class="title-page">',
            f"<h1>{escape(report_title)}</h1>",
            f'<p class="report-date">작성 일시 {escape(created_label)}</p>',
            "</header>",
            sections_html,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _build_report_css() -> str:
    return """
html {
  background: #f2f2f2;
}

body {
  margin: 0;
  color: #111111;
  font-family: "Nanum Myeongjo", "Noto Serif KR", "AppleMyungjo", "Batang", "Times New Roman", serif;
  font-size: 14px;
  line-height: 1.8;
}

.report-sheet {
  width: 210mm;
  min-height: 297mm;
  box-sizing: border-box;
  margin: 18px auto;
  padding: 22mm 20mm 24mm;
  background: #ffffff;
  box-shadow: 0 0 0 1px rgba(17, 17, 17, 0.12);
}

.title-page {
  padding: 16mm 0 8mm;
  margin-bottom: 12mm;
}

.title-page h1 {
  margin: 0;
  font-size: 28px;
  line-height: 1.35;
  font-weight: 700;
  text-align: center;
}

.report-table,
.swot-matrix {
  width: 100%;
  border-collapse: collapse;
}

.report-table th,
.report-table td,
.swot-matrix td {
  border: 1px solid #111111;
  padding: 8px 10px;
  vertical-align: top;
}

.report-date {
  margin: 12px 0 0;
  font-size: 12px;
  text-align: right;
}

.toc {
  padding: 8mm 0;
  margin: 0 0 12mm;
}

.toc h2 {
  margin: 0 0 10px;
  font-size: 17px;
}

.toc-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.toc-list li {
  margin: 4px 0;
}

.toc a,
a {
  color: #111111;
  text-decoration: none;
  border-bottom: 1px solid #111111;
}

.report-section {
  margin-bottom: 14mm;
}

.report-section h2 {
  margin: 0 0 6mm;
  padding-bottom: 3mm;
  border-bottom: 1px solid #111111;
  font-size: 20px;
  line-height: 1.4;
}

.report-section h3 {
  margin: 7mm 0 3mm;
  font-size: 16px;
  line-height: 1.5;
}

.report-section h4 {
  margin: 4mm 0 2mm;
  font-size: 14px;
}

.report-section p {
  margin: 0 0 4mm;
  text-align: justify;
}

.summary-section .section-body {
  border: 1px solid #111111;
  padding: 6mm 6mm 3mm;
}

.report-section ul,
.report-section ol {
  margin: 0 0 4mm 20px;
  padding: 0;
}

.report-section li {
  margin: 0 0 2mm;
}

.report-table {
  margin: 5mm 0 6mm;
  table-layout: fixed;
}

.report-table caption,
.swot-matrix caption {
  caption-side: top;
  margin-bottom: 2mm;
  font-size: 12px;
  font-style: italic;
  text-align: center;
}

.report-table th {
  font-weight: 700;
  text-align: left;
}

.report-table td.align-right,
.report-table th.align-right {
  text-align: right;
}

.report-table td.align-center,
.report-table th.align-center {
  text-align: center;
}

.swot-company {
  margin: 0 0 8mm;
}

.swot-company h3 {
  margin-top: 0;
}

.swot-matrix {
  table-layout: fixed;
}

.swot-matrix td {
  width: 50%;
  min-height: 110px;
}

.swot-label {
  margin: 0 0 2mm;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.swot-empty,
.empty-note {
  font-style: italic;
}

.references-list {
  margin-left: 18px;
}

.references-list li {
  margin-bottom: 3mm;
}

@page {
  size: A4;
  margin: 0;
}

@media print {
  html {
    background: #ffffff;
  }

  .report-sheet {
    margin: 0;
    width: auto;
    min-height: auto;
    box-shadow: none;
    padding: 0;
  }
}

body.pdf-export {
  background: #ffffff;
  font-size: 13px;
  line-height: 1.72;
}

body.pdf-export .report-sheet {
  width: auto;
  min-height: auto;
  margin: 0;
  padding: 24mm 22mm 28mm;
  box-shadow: none;
}

body.pdf-export .title-page {
  padding-top: 14mm;
}

body.pdf-export .title-page h1 {
  font-size: 24px;
}

body.pdf-export .report-section h2 {
  font-size: 18px;
}

body.pdf-export .report-section h3 {
  font-size: 15px;
}

body.pdf-export .report-section h4 {
  font-size: 13px;
}
""".strip()


def _render_table_of_contents() -> str:
    lines = [
        '<nav class="toc" aria-label="목차">',
        "<h2>Contents</h2>",
        '<ul class="toc-list">',
    ]
    for section_id in SECTION_ORDER:
        lines.append(
            (
                f'<li><a href="#{section_id}">'
                f"{escape(SECTION_HEADINGS[section_id])}"
                "</a></li>"
            )
        )
    lines.extend(["</ul>", "</nav>"])
    return "\n".join(lines)


def _render_section(result: ReportState, section_id: str) -> str:
    section_class = "report-section"
    if section_id == "summary":
        section_class += " summary-section"

    body_html = _render_section_body(result, section_id)
    return "\n".join(
        [
            f'<section class="{section_class}" id="{section_id}">',
            f"<h2>{escape(SECTION_HEADINGS[section_id])}</h2>",
            f'<div class="section-body">{body_html}</div>',
            "</section>",
        ]
    )


def _render_section_body(result: ReportState, section_id: str) -> str:
    if section_id == "swot":
        return _render_swot_section(result)
    if section_id == "references":
        return _render_references_section(result)
    if section_id == "strategy_comparison":
        return _render_markdownish_content(
            result["section_drafts"][section_id]["content"],
            default_table_caption="Table 5-1. 데이터 기반 비교표",
        )
    return _render_markdownish_content(result["section_drafts"][section_id]["content"])


def _render_swot_section(result: ReportState) -> str:
    company_sections: list[str] = []
    for index, company in enumerate(("LGES", "CATL"), start=1):
        swot = result["swot"].get(company, _empty_swot())
        company_sections.append(
            "\n".join(
                [
                    '<article class="swot-company">',
                    f"<h3>{escape(company)}</h3>",
                    '<table class="swot-matrix">',
                    f"<caption>Table 5-{index + 1}. {escape(company)} SWOT Matrix</caption>",
                    "<tbody>",
                    "<tr>",
                    f"<td>{_render_swot_cell('Strengths', swot['strengths'])}</td>",
                    f"<td>{_render_swot_cell('Weaknesses', swot['weaknesses'])}</td>",
                    "</tr>",
                    "<tr>",
                    f"<td>{_render_swot_cell('Opportunities', swot['opportunities'])}</td>",
                    f"<td>{_render_swot_cell('Threats', swot['threats'])}</td>",
                    "</tr>",
                    "</tbody>",
                    "</table>",
                    "</article>",
                ]
            )
        )

    return "\n".join(company_sections)


def _render_swot_cell(label: str, items: list[str]) -> str:
    if not items:
        items_html = '<p class="swot-empty">정보 부족/추가 검증 필요</p>'
    else:
        list_items = "".join(f"<li>{_format_inline(item)}</li>" for item in items)
        items_html = f"<ul>{list_items}</ul>"
    return f'<p class="swot-label">{escape(label)}</p>{items_html}'


def _render_references_section(result: ReportState) -> str:
    references = result["references"]
    if not references:
        return '<p class="empty-note">수집된 참고문헌이 없어 추가 검증 필요.</p>'

    items = []
    for _, reference in sorted(references.items()):
        citation = reference["citation_text"].strip()
        if citation.startswith("- "):
            citation = citation[2:]
        items.append(f"<li>{_format_inline(citation)}</li>")
    return '<ol class="references-list">' + "".join(items) + "</ol>"


def _render_markdownish_content(
    content: str,
    *,
    default_table_caption: str | None = None,
) -> str:
    stripped_content = content.strip()
    if not stripped_content:
        return '<p class="empty-note">정보 부족/추가 검증 필요.</p>'

    blocks: list[str] = []
    paragraph_lines: list[str] = []
    list_items: list[str] = []
    table_lines: list[str] = []
    table_caption = default_table_caption

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip())
        blocks.append(f"<p>{_format_inline(paragraph)}</p>")
        paragraph_lines.clear()

    def flush_list() -> None:
        if not list_items:
            return
        items_html = "".join(f"<li>{_format_inline(item)}</li>" for item in list_items)
        blocks.append(f"<ul>{items_html}</ul>")
        list_items.clear()

    def flush_table() -> None:
        nonlocal table_caption
        if not table_lines:
            return
        blocks.append(
            _render_markdown_table(
                table_lines,
                caption=table_caption,
            )
        )
        table_lines.clear()
        table_caption = None

    for raw_line in stripped_content.splitlines():
        line = raw_line.strip()
        if not line:
            flush_table()
            flush_list()
            flush_paragraph()
            continue

        if line.startswith("|"):
            flush_paragraph()
            flush_list()
            table_lines.append(line)
            continue

        flush_table()

        if line.startswith("- "):
            flush_paragraph()
            list_items.append(line[2:].strip())
            continue

        flush_list()

        if line.startswith("### "):
            flush_paragraph()
            blocks.append(f"<h3>{_format_inline(line[4:].strip())}</h3>")
            continue

        if line.startswith("#### "):
            flush_paragraph()
            blocks.append(f"<h4>{_format_inline(line[5:].strip())}</h4>")
            continue

        paragraph_lines.append(line)

    flush_table()
    flush_list()
    flush_paragraph()
    return "\n".join(blocks)


def _render_markdown_table(
    lines: list[str],
    *,
    caption: str | None = None,
) -> str:
    table = _parse_markdown_table(lines)
    if table is None:
        fallback = " ".join(line.strip() for line in lines if line.strip())
        return f"<p>{_format_inline(fallback)}</p>"

    parts = ['<table class="report-table">']
    if caption:
        parts.append(f"<caption>{escape(caption)}</caption>")
    parts.append("<thead><tr>")
    for header, alignment in zip(table.headers, table.alignments):
        parts.append(
            f'<th class="{_alignment_class(alignment)}">{_format_inline(header)}</th>'
        )
    parts.append("</tr></thead><tbody>")

    for row in table.rows:
        parts.append("<tr>")
        for index, cell in enumerate(row):
            alignment = table.alignments[index] if index < len(table.alignments) else "left"
            parts.append(
                f'<td class="{_alignment_class(alignment)}">{_format_inline(cell)}</td>'
            )
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return "".join(parts)


def _parse_markdown_table(lines: list[str]) -> MarkdownTable | None:
    normalized = [line.strip() for line in lines if line.strip()]
    if len(normalized) < 2:
        return None

    headers = _split_table_row(normalized[0])
    separators = _split_table_row(normalized[1])
    if not headers or len(headers) != len(separators):
        return None
    if any(not re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in separators):
        return None

    alignments = [_parse_alignment(cell) for cell in separators]
    rows: list[list[str]] = []
    for line in normalized[2:]:
        row = _split_table_row(line)
        if not row:
            continue
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))
        rows.append(row[: len(headers)])

    return MarkdownTable(headers=headers, alignments=alignments, rows=rows)


def _split_table_row(line: str) -> list[str]:
    row = line
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _parse_alignment(cell: str) -> str:
    compact = cell.replace(" ", "")
    if compact.startswith(":") and compact.endswith(":"):
        return "center"
    if compact.endswith(":"):
        return "right"
    return "left"


def _alignment_class(alignment: str) -> str:
    return {
        "left": "align-left",
        "center": "align-center",
        "right": "align-right",
    }.get(alignment, "align-left")


def _format_inline(text: str) -> str:
    tokens: dict[str, str] = {}

    def stash(fragment: str) -> str:
        token = f"__HTML_TOKEN_{len(tokens)}__"
        tokens[token] = fragment
        return token

    def replace_url(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        url = raw_url.rstrip(".,);")
        trailing = raw_url[len(url):]
        link = f'<a href="{escape(url, quote=True)}">{escape(url)}</a>'
        return stash(link) + trailing

    text = URL_PATTERN.sub(replace_url, text)
    text = STRONG_PATTERN.sub(
        lambda match: stash(f"<strong>{escape(match.group(1))}</strong>"),
        text,
    )
    text = EMPHASIS_PATTERN.sub(
        lambda match: stash(f"<em>{escape(match.group(1))}</em>"),
        text,
    )
    text = escape(text)
    for token, fragment in tokens.items():
        text = text.replace(token, fragment)
    return text


def _empty_swot() -> SWOTState:
    return {
        "strengths": [],
        "weaknesses": [],
        "opportunities": [],
        "threats": [],
    }


def _write_pdf_from_html(html: str, pdf_path: Path) -> None:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "PDF export requires pymupdf. Install project dependencies and run through the project environment."
        ) from exc

    writer = pymupdf.DocumentWriter(str(pdf_path))
    story = pymupdf.Story(html=html)
    mediabox = pymupdf.paper_rect("a4")
    footer_reserved_pt = PDF_FOOTER_RESERVED_MM * MM_TO_PT
    # Reserve a footer band on every page so body content never collides with page numbers.
    content_box = pymupdf.Rect(
        mediabox.x0,
        mediabox.y0,
        mediabox.x1,
        mediabox.y1 - footer_reserved_pt,
    )

    def rectfn(rect_num: int, filled: object) -> tuple[object, object, object]:
        del rect_num, filled
        return mediabox, content_box, pymupdf.Matrix(1, 1)

    story.write(writer, rectfn)
    writer.close()
    _add_pdf_page_numbers(pdf_path)


def _add_pdf_page_numbers(pdf_path: Path) -> None:
    import pymupdf

    document = pymupdf.open(str(pdf_path))
    temp_path = pdf_path.with_name(f"{pdf_path.stem}_numbered{pdf_path.suffix}")
    footer_reserved_pt = PDF_FOOTER_RESERVED_MM * MM_TO_PT
    footer_bottom_pt = PDF_PAGE_NUMBER_BOTTOM_MM * MM_TO_PT
    footer_height_pt = PDF_PAGE_NUMBER_HEIGHT_MM * MM_TO_PT
    try:
        for page_number, page in enumerate(document, start=1):
            footer_top = max(
                page.rect.height - footer_reserved_pt,
                page.rect.height - footer_bottom_pt - footer_height_pt,
            )
            footer_rect = pymupdf.Rect(
                0,
                footer_top,
                page.rect.width,
                page.rect.height - footer_bottom_pt,
            )
            page.insert_textbox(
                footer_rect,
                str(page_number),
                fontname="Times-Roman",
                fontsize=9,
                color=(0, 0, 0),
                align=1,
                overlay=True,
            )
        document.save(str(temp_path), garbage=4, deflate=True)
    finally:
        document.close()

    temp_path.replace(pdf_path)
