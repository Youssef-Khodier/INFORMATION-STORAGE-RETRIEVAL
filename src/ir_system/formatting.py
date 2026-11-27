from __future__ import annotations

from typing import Mapping, Sequence

from ir_system.models import PositionalIndex
from ir_system.retrieval import ClauseMatchResult, QueryResult
from ir_system.vector_space import VectorSpaceModel


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


def documents_table_data(index: PositionalIndex) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Return headers and rows for the Corpus Documents table."""
    headers = ("File", "Exact document text", "Word count")
    rows = [
        (
            doc.doc_id,
            " ".join(doc.text.splitlines()),
            str(doc.word_count),
        )
        for doc in index.documents
    ]
    return headers, rows


def format_documents_table(index: PositionalIndex) -> str:
    """Format the Corpus Documents table with File, Exact document text, and Word count."""
    return _render_pipe_table(*documents_table_data(index))


def positional_index_table_data(
    index: PositionalIndex,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Return headers and rows for the Positional Index table."""
    headers = ("Term", "Document IDs and 1-based positions")
    rows = [
        (term, format_postings_entry(doc_postings))
        for term, doc_postings in index.postings.items()
    ]
    return headers, rows


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


def format_matrix_cell(value: int | float) -> str:
    """Format a matrix cell as '0' when zero and to 4 decimal places when nonzero."""
    if value == 0:
        return "0"
    return f"{value:.4f}"


def term_doc_matrix_data(
    model: VectorSpaceModel,
    matrix: Mapping[str, Mapping[str, int | float]],
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Return headers and formatted rows for a Term x Document weighting matrix."""
    headers = ("Term", *model.doc_ids)
    rows = [
        (
            term,
            *(format_matrix_cell(matrix[term][doc_id]) for doc_id in model.doc_ids),
        )
        for term in model.terms
    ]
    return headers, rows


def _format_term_doc_matrix(
    model: VectorSpaceModel,
    matrix: Mapping[str, Mapping[str, int | float]],
) -> str:
    """Render a Term x Document weighting matrix using reference cell formatting."""
    return _render_pipe_table(*term_doc_matrix_data(model, matrix))


def format_raw_tf_table(model: VectorSpaceModel) -> str:
    """Format the Raw TF Term x Document matrix."""
    return _format_term_doc_matrix(model, model.raw_tf)


def format_weighted_tf_table(model: VectorSpaceModel) -> str:
    """Format the Weighted TF Term x Document matrix."""
    return _format_term_doc_matrix(model, model.weighted_tf)


def df_idf_table_data(model: VectorSpaceModel) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Return headers and rows for the DF and base-10 IDF table."""
    headers = ("Term", "DF", f"IDF = log10({model.doc_count} / DF)")
    rows = [
        (term, str(model.df[term]), f"{model.idf[term]:.4f}")
        for term in model.terms
    ]
    return headers, rows


def format_df_idf_table(model: VectorSpaceModel) -> str:
    """Format the DF and base-10 IDF table with dynamic Corpus size N."""
    return _render_pipe_table(*df_idf_table_data(model))


def format_tf_idf_table(model: VectorSpaceModel) -> str:
    """Format the TF-IDF Weight Term x Document matrix."""
    return _format_term_doc_matrix(model, model.tf_idf)


def doc_lengths_table_data(
    model: VectorSpaceModel,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Return headers and rows for the Document Vector Lengths table."""
    headers = ("Document", "Vector length")
    rows = [
        (doc_id, f"{model.doc_lengths[doc_id]:.4f}")
        for doc_id in model.doc_ids
    ]
    return headers, rows


def format_doc_lengths_table(model: VectorSpaceModel) -> str:
    """Format the Document Vector Lengths table."""
    return _render_pipe_table(*doc_lengths_table_data(model))


def format_normalized_tf_idf_table(model: VectorSpaceModel) -> str:
    """Format the Normalized TF-IDF Term x Document matrix."""
    return _format_term_doc_matrix(model, model.normalized_tf_idf)


def format_part_b_output(model: VectorSpaceModel) -> str:
    """Format all Part B vector-space weighting tables in reference order."""
    sections = [
        f"Raw TF:\n{format_raw_tf_table(model)}",
        f"Weighted TF: 1 + log10(TF):\n{format_weighted_tf_table(model)}",
        f"DF and IDF:\n{format_df_idf_table(model)}",
        f"TF-IDF Matrix:\n{format_tf_idf_table(model)}",
        f"Document Vector Lengths:\n{format_doc_lengths_table(model)}",
        f"Normalized TF-IDF Matrix:\n{format_normalized_tf_idf_table(model)}",
    ]
    return "\n\n".join(sections)


def query_heading_label(result: QueryResult) -> str:
    """Return a reference-style heading label for a Phrase or Boolean QueryResult."""
    parsed = result.parsed_query
    if parsed.query_type == "phrase":
        return f'Phrase query: "{result.raw_query}"'
    operators = [clause.operator for clause in parsed.clauses[1:] if clause.operator is not None]
    if operators == ["AND"]:
        return f'Boolean AND: "{result.raw_query}"'
    if operators == ["AND NOT"]:
        return f'Boolean AND NOT: "{result.raw_query}"'
    return f'Boolean query: "{result.raw_query}"'


def clause_matches_table_data(
    clause_result: ClauseMatchResult,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Return headers and rows for a ClauseMatchResult's phrase match positions and ranges."""
    headers = ("Matched document", "Starting positions", "Phrase token ranges")
    rows = [
        (match.doc_id, match.formatted_positions, match.formatted_ranges)
        for match in clause_result.matches
    ]
    return headers, rows


def matched_docs_text(result: QueryResult) -> str:
    """Return comma-separated matched Document IDs or 'None'."""
    return ", ".join(result.matched_doc_ids) if result.matched_doc_ids else "None"


def clause_display_label(clause_result: ClauseMatchResult, *, backticks: bool = False) -> str:
    """Return a display label such as 'Clause AND \"data science\"' or 'Clause AND `data science`'."""
    clause_phrase = " ".join(clause_result.clause.tokens)
    prefix = f"{clause_result.clause.operator} " if clause_result.clause.operator else ""
    wrapped = f"`{clause_phrase}`" if backticks else f'"{clause_phrase}"'
    return f"Clause {prefix}{wrapped}"


def clause_match_sections(
    result: QueryResult,
    *,
    backticks: bool = False,
) -> list[tuple[str, tuple[str, ...], list[tuple[str, ...]]]]:
    """Return (clause_label, headers, rows) tuples for each clause in a QueryResult."""
    return [
        (clause_display_label(clause_result, backticks=backticks), *clause_matches_table_data(clause_result))
        for clause_result in result.clause_matches
    ]


def query_weights_table_data(result: QueryResult) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Return headers and 4-decimal rows for a Query Vector's per-Term weights."""
    headers = ("Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight")
    rows = [
        (
            weight.term,
            f"{weight.weighted_tf:.4f}",
            f"{weight.tf_idf:.4f}",
            f"{weight.normalized_weight:.4f}",
        )
        for weight in result.term_weights
    ]
    return headers, rows


def query_similarity_table_data(
    result: QueryResult,
    *,
    preserve_ranking_order: bool = False,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Return headers and 4-decimal rows for matched Document Similarity Scores."""
    headers = ("Matched document", "Similarity")
    if preserve_ranking_order:
        ordered = result.ranked_documents
    else:
        by_doc = {item.doc_id: item for item in result.ranked_documents}
        ordered = tuple(by_doc[doc_id] for doc_id in result.matched_doc_ids if doc_id in by_doc)
    rows = [(item.doc_id, f"{item.similarity:.4f}") for item in ordered]
    return headers, rows


def format_query_result(result: QueryResult) -> str:
    """Format a single QueryResult with positions/ranges, 6-decimal length, and 4-decimal scores."""
    lines = [query_heading_label(result)]
    sections = clause_match_sections(result, backticks=False)

    if result.parsed_query.query_type == "phrase":
        _, headers, rows = sections[0]
        if rows:
            lines.append(_render_pipe_table(headers, rows))
        else:
            lines.append("No matching documents.")
    else:
        lines.append(f"Matched documents: {matched_docs_text(result)}")
        for label, headers, rows in sections:
            if rows:
                lines.append(f"{label}:\n{_render_pipe_table(headers, rows)}")
            else:
                lines.append(f"{label}: No matching documents.")

    lines.append(f"Query vector length = {result.query_length:.6f}")
    w_headers, w_rows = query_weights_table_data(result)
    lines.append(_render_pipe_table(w_headers, w_rows))

    s_headers, s_rows = query_similarity_table_data(result)
    if s_rows:
        lines.append(_render_pipe_table(s_headers, s_rows))
        lines.append(f"Ranking (highest similarity first): {result.ranking_expression}")
    else:
        lines.append("No matching documents.")

    return "\n\n".join(lines)


def format_query_results(results: Sequence[QueryResult]) -> str:
    """Format all QueryResults into a clean multi-section text block."""
    if not results:
        return "No queries executed."
    return "\n\n".join(format_query_result(result) for result in results)
