from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ir_system.formatting import (
    clause_match_sections,
    df_idf_table_data,
    doc_lengths_table_data,
    documents_table_data,
    matched_docs_text,
    positional_index_table_data,
    query_heading_label,
    query_similarity_table_data,
    query_weights_table_data,
    term_doc_matrix_data,
)
from ir_system.models import PositionalIndex
from ir_system.retrieval import QueryResult
from ir_system.vector_space import VectorSpaceModel

DEFAULT_REPORT_FILENAME = "report.md"


def _escape_md_cell(value: str) -> str:
    """Escape pipe characters and collapse newlines for safe GitHub-Flavored Markdown cells."""
    return " ".join(value.splitlines()).replace("|", "\\|")


def _render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render headers and rows as a GitHub-Flavored Markdown table."""
    header_row = "| " + " | ".join(_escape_md_cell(h) for h in headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(_escape_md_cell(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def _render_query_markdown_section(result: QueryResult) -> str:
    """Render a single QueryResult as a Markdown subsection."""
    lines: list[str] = [f"### {query_heading_label(result)}"]
    sections = clause_match_sections(result, backticks=True)

    if result.parsed_query.query_type == "phrase":
        _, headers, rows = sections[0]
        if rows:
            lines.append(_render_markdown_table(headers, rows))
        else:
            lines.append("_No matching documents._")
    else:
        lines.append(f"**Matched documents:** `{matched_docs_text(result)}`")
        for label, headers, rows in sections:
            lines.append(f"#### {label}")
            if rows:
                lines.append(_render_markdown_table(headers, rows))
            else:
                lines.append("_No matching documents for this clause._")

    lines.append(f"Query vector length = `{result.query_length:.6f}`")
    lines.append(_render_markdown_table(*query_weights_table_data(result)))

    s_headers, s_rows = query_similarity_table_data(result)
    if s_rows:
        lines.append(_render_markdown_table(s_headers, s_rows))
        lines.append(f"Ranking (highest similarity first): `{result.ranking_expression}`")
    else:
        lines.append("_No matching documents._")

    return "\n\n".join(lines)


def generate_markdown_report(
    index: PositionalIndex,
    model: VectorSpaceModel,
    query_results: Sequence[QueryResult] = (),
    *,
    title: str = "Technology Version IR System Report",
) -> str:
    """Generate a complete Markdown report covering Part A, Part B, and all Query Results."""
    sections: list[str] = [f"# {title}"]

    summary_lines = [
        "## Corpus Summary",
        f"- **Documents (`N`)**: {index.doc_count}",
        f"- **Vocabulary Terms**: {len(index.postings)}",
        f"- **Total Tokens**: {index.total_tokens}",
        f"- **Indexing Time**: {index.indexing_seconds:.2f}s",
    ]
    if index.corpus_path:
        summary_lines.append(f"- **Corpus Path**: `{index.corpus_path}`")
    if index.warnings:
        summary_lines.append("- **Warnings**:")
        summary_lines.extend(f"  - {warning}" for warning in index.warnings)
    sections.append("\n".join(summary_lines))

    sections.append("## Part A — Positional Index")
    sections.append(f"### Documents (`N = {index.doc_count}`)\n\n{_render_markdown_table(*documents_table_data(index))}")
    sections.append(
        f"### Positional Index ({len(index.postings)} Terms)\n\n"
        f"{_render_markdown_table(*positional_index_table_data(index))}"
    )

    sections.append("## Part B — Vector-Space Model")
    sections.append(f"### Raw TF\n\n{_render_markdown_table(*term_doc_matrix_data(model, model.raw_tf))}")
    sections.append(
        f"### Weighted TF (`1 + log10(TF)`)\n\n"
        f"{_render_markdown_table(*term_doc_matrix_data(model, model.weighted_tf))}"
    )
    sections.append(f"### DF and IDF\n\n{_render_markdown_table(*df_idf_table_data(model))}")
    sections.append(f"### TF-IDF Matrix\n\n{_render_markdown_table(*term_doc_matrix_data(model, model.tf_idf))}")
    sections.append(
        f"### Document Vector Lengths\n\n{_render_markdown_table(*doc_lengths_table_data(model))}"
    )
    sections.append(
        f"### Normalized TF-IDF Matrix\n\n"
        f"{_render_markdown_table(*term_doc_matrix_data(model, model.normalized_tf_idf))}"
    )

    sections.append("## Query Results")
    if query_results:
        for result in query_results:
            sections.append(_render_query_markdown_section(result))
    else:
        sections.append("_No queries executed._")

    return "\n\n".join(sections) + "\n"


def export_markdown_report(
    index: PositionalIndex,
    model: VectorSpaceModel,
    query_results: Sequence[QueryResult],
    output_dir: Path | str,
    filename: str = DEFAULT_REPORT_FILENAME,
    *,
    title: str = "Technology Version IR System Report",
) -> Path:
    """Write the generated Markdown report to output_dir/filename."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / filename
    content = generate_markdown_report(index, model, query_results, title=title)
    target.write_text(content, encoding="utf-8")
    return target
