from __future__ import annotations

from typing import Mapping, Sequence

from ir_system.models import PositionalIndex


def format_postings_entry(doc_postings: Mapping[str, tuple[int, ...]]) -> str:
    """Format a Term's Document postings as 'd1: 1; d2: 1, 3'."""
    parts = [
        f"{doc_id}: {', '.join(str(pos) for pos in positions)}"
        for doc_id, positions in doc_postings.items()
    ]
    return "; ".join(parts)


def _render_pipe_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a clean column-aligned pipe-delimited text table."""
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    separator_line = "-+-".join("-" * width for width in widths)
    body_lines = [
        " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def format_documents_table(index: PositionalIndex) -> str:
    """Format the Corpus Documents table with File, Exact document text, and Word count."""
    headers = ("File", "Exact document text", "Word count")
    rows = [
        (
            doc.doc_id,
            " ".join(doc.text.splitlines()),
            str(doc.word_count),
        )
        for doc in index.documents
    ]
    return _render_pipe_table(headers, rows)


def format_positional_index_table(index: PositionalIndex) -> str:
    """Format the Positional Index table in 'term | d1: 1; d2: 1' reference style."""
    header = "Term | Document IDs and 1-based positions"
    separator = "-" * len(header)
    rows = [
        f"{term} | {format_postings_entry(doc_postings)}"
        for term, doc_postings in index.postings.items()
    ]
    return "\n".join([header, separator, *rows])


def format_warnings(warnings: Sequence[str]) -> str:
    """Format skipped-file ingestion warnings, or return empty string when none exist."""
    if not warnings:
        return ""
    lines = ["Warnings:"]
    lines.extend(f"  - {warning}" for warning in warnings)
    return "\n".join(lines)


def format_part_a_output(index: PositionalIndex) -> str:
    """Format the complete Part A summary, warnings, Documents table, and Positional Index."""
    sections = [
        f"Corpus Summary: N = {index.doc_count} Documents, "
        f"{len(index.postings)} Terms, {index.total_tokens} Tokens "
        f"(indexed in {index.indexing_seconds:.2f}s)"
    ]
    warnings_block = format_warnings(index.warnings)
    if warnings_block:
        sections.append(warnings_block)
    sections.append(f"Documents (N = {index.doc_count}):\n{format_documents_table(index)}")
    sections.append(
        f"Positional Index ({len(index.postings)} Terms):\n{format_positional_index_table(index)}"
    )
    return "\n\n".join(sections)

