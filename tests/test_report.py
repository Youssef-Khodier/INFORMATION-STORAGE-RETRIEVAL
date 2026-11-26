from __future__ import annotations

from pathlib import Path

from ir_system.formatting import format_query_result, format_query_results
from ir_system.models import DocumentRecord, PositionalIndex
from ir_system.report import export_markdown_report, generate_markdown_report
from ir_system.retrieval import build_vector_space_model, execute_query, run_queries


def test_format_query_results_and_generate_markdown_report() -> None:
    """Verify 6-decimal query length, 4-decimal weights/scores, and complete Markdown report."""
    index = PositionalIndex(
        documents=(
            DocumentRecord(doc_id="d1", text="machine learning", word_count=2),
            DocumentRecord(doc_id="d2", text="machine learning data science", word_count=4),
            DocumentRecord(doc_id="d3", text="data science", word_count=2),
        ),
        postings={
            "data": {"d2": (3,), "d3": (1,)},
            "learning": {"d1": (2,), "d2": (2,)},
            "machine": {"d1": (1,), "d2": (1,)},
            "science": {"d2": (4,), "d3": (2,)},
        },
        warnings=("Skipped empty or whitespace-only file: empty.txt",),
        indexing_seconds=0.25,
        corpus_path="mini_corpus",
    )
    model = build_vector_space_model(index)
    results = run_queries(
        index,
        queries=(
            "machine learning",
            "machine learning AND NOT data science",
            "nonexistent",
        ),
        model=model,
    )

    formatted_first = format_query_result(results[0])
    assert 'Phrase query: "machine learning"' in formatted_first
    assert f"Query vector length = {results[0].query_length:.6f}" in formatted_first
    assert "Weighted TF" in formatted_first
    assert "Query TF-IDF" in formatted_first
    assert "Normalized query weight" in formatted_first
    assert "1-2" in formatted_first
    assert "1.0000" in formatted_first
    assert "Ranking (highest similarity first): d1 > d2" in formatted_first

    formatted_all = format_query_results(results)
    assert 'Boolean AND NOT: "machine learning AND NOT data science"' in formatted_all
    assert "No matching documents" in formatted_all

    report = generate_markdown_report(index, model, results)
    for required_section in (
        "## Corpus Summary",
        "## Part A",
        "### Documents",
        "### Positional Index",
        "### Raw TF",
        "### Weighted TF",
        "### DF and IDF",
        "### TF-IDF Matrix",
        "### Document Vector Lengths",
        "### Normalized TF-IDF Matrix",
        "## Query Results",
    ):
        assert required_section in report

    # GFM table markers and precision checks
    assert "| Term | d1 | d2 | d3 |" in report
    assert "| --- | --- | --- | --- |" in report
    assert f"Query vector length = `{results[0].query_length:.6f}`" in report
    assert "Ranking (highest similarity first): `d1 > d2`" in report
    assert "empty.txt" in report


def test_export_markdown_report_writes_report_md(tmp_path: Path) -> None:
    """Verify export_markdown_report writes report.md inside output_dir."""
    index = PositionalIndex(
        documents=(DocumentRecord(doc_id="d1", text="alpha beta", word_count=2),),
        postings={"alpha": {"d1": (1,)}, "beta": {"d1": (2,)}},
    )
    model = build_vector_space_model(index)
    results = (execute_query(index, "alpha beta", model=model),)

    out_path = export_markdown_report(index, model, results, tmp_path / "out")
    assert out_path == tmp_path / "out" / "report.md"
    assert out_path.is_file()
    content = out_path.read_text(encoding="utf-8")
    assert "alpha beta" in content
    assert "| d1 |" in content
