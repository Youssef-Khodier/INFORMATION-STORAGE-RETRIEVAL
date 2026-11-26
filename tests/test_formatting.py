from __future__ import annotations

from ir_system.formatting import (
    format_documents_table,
    format_part_a_output,
    format_positional_index_table,
    format_postings_entry,
)
from ir_system.models import DocumentRecord, PositionalIndex


def test_format_postings_entry_matches_reference_style() -> None:
    """Verify postings format as 'doc: pos1, pos2; doc2: pos1'."""
    assert format_postings_entry({"d6": (1,), "d9": (1, 3)}) == "d6: 1; d9: 1, 3"
    assert format_postings_entry({"d5": (8,)}) == "d5: 8"


def test_format_documents_and_positional_index_tables_include_warnings_and_rows() -> None:
    """Verify Documents table, Positional Index table, and warnings render cleanly."""
    index = PositionalIndex(
        documents=(
            DocumentRecord(doc_id="d1", text="machine learning", word_count=2),
            DocumentRecord(doc_id="d2", text="machine learning improves data processing", word_count=5),
        ),
        postings={
            "data": {"d2": (4,)},
            "improves": {"d2": (3,)},
            "learning": {"d1": (2,), "d2": (2,)},
            "machine": {"d1": (1,), "d2": (1,)},
            "processing": {"d2": (5,)},
        },
        warnings=("Skipped empty or whitespace-only file: blank.txt",),
        indexing_seconds=0.42,
        corpus_path="sample_corpus",
    )

    docs_table = format_documents_table(index)
    assert "d1" in docs_table
    assert "machine learning" in docs_table
    assert "2" in docs_table
    assert "d2" in docs_table
    assert "machine learning improves data processing" in docs_table

    pos_table = format_positional_index_table(index)
    assert "learning | d1: 2; d2: 2" in pos_table
    assert "machine  | d1: 1; d2: 1" in pos_table or "machine | d1: 1; d2: 1" in pos_table

    full_output = format_part_a_output(index)
    assert "Warnings:" in full_output
    assert "Skipped empty or whitespace-only file: blank.txt" in full_output
    assert "Documents (N = 2)" in full_output
    assert "Positional Index (27" not in full_output
    assert "Positional Index (5 Terms)" in full_output


def test_format_part_b_tables_match_pdf_conventions_and_dynamic_corpus_header() -> None:
    """Verify Part B formatters use '0' for zero cells, 4 decimals for weights, and dynamic N."""
    from ir_system.formatting import (
        format_df_idf_table,
        format_doc_lengths_table,
        format_matrix_cell,
        format_normalized_tf_idf_table,
        format_part_b_output,
        format_raw_tf_table,
        format_tf_idf_table,
        format_weighted_tf_table,
    )
    from ir_system.vector_space import build_vector_space_model

    index = PositionalIndex(
        documents=(
            DocumentRecord(doc_id="d1", text="shared unique", word_count=2),
            DocumentRecord(doc_id="d2", text="shared shared", word_count=2),
        ),
        postings={
            "shared": {"d1": (1,), "d2": (1, 2)},
            "unique": {"d1": (2,)},
        },
    )
    model = build_vector_space_model(index)

    assert format_matrix_cell(0) == "0"
    assert format_matrix_cell(0.0) == "0"
    assert format_matrix_cell(2) == "2.0000"
    assert format_matrix_cell(1.30102999) == "1.3010"

    raw_tf_table = format_raw_tf_table(model)
    assert "1.0000" in raw_tf_table
    assert "2.0000" in raw_tf_table
    assert "0" in raw_tf_table

    weighted_tf_table = format_weighted_tf_table(model)
    assert "1.3010" in weighted_tf_table

    df_idf_table = format_df_idf_table(model)
    assert "IDF = log10(2 / DF)" in df_idf_table
    assert "0.0000" in df_idf_table
    assert "0.3010" in df_idf_table

    tf_idf_table = format_tf_idf_table(model)
    assert "0.3010" in tf_idf_table

    doc_lengths_table = format_doc_lengths_table(model)
    assert "0.3010" in doc_lengths_table
    assert "0.0000" in doc_lengths_table

    norm_table = format_normalized_tf_idf_table(model)
    assert "1.0000" in norm_table

    part_b_output = format_part_b_output(model)
    for heading in (
        "Raw TF",
        "Weighted TF: 1 + log10(TF)",
        "DF and IDF",
        "TF-IDF Matrix",
        "Document Vector Lengths",
        "Normalized TF-IDF Matrix",
    ):
        assert heading in part_b_output

