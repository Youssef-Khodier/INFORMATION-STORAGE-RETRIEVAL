from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from ir_system.models import DocumentRecord, PositionalIndex
from ir_system.spark_indexer import export_positional_index, load_positional_index


def test_cli_from_index_prints_documents_warnings_and_positional_index(
    tmp_path: Path,
    capsys: object,
) -> None:
    """Verify CLI --from-index reloads Part A artifact and renders reference tables."""
    from ir_system.cli import main

    sample = PositionalIndex(
        documents=(
            DocumentRecord(doc_id="d1", text="machine learning", word_count=2),
            DocumentRecord(doc_id="d2", text="data science", word_count=2),
        ),
        postings={
            "data": {"d2": (1,)},
            "learning": {"d1": (2,)},
            "machine": {"d1": (1,)},
            "science": {"d2": (2,)},
        },
        warnings=("Skipped empty or whitespace-only file: blank.txt",),
        indexing_seconds=0.15,
        corpus_path="sample",
    )
    exported = export_positional_index(sample, tmp_path / "exported")

    exit_code = main(["--from-index", str(exported)])
    assert exit_code == 0

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Documents (N = 2):" in captured.out
    assert "machine learning" in captured.out
    assert "Warnings:" in captured.out
    assert "blank.txt" in captured.out
    assert "learning | d1: 2" in captured.out
    assert "data | d2: 1" in captured.out
    assert "Raw TF:" in captured.out
    assert "Weighted TF: 1 + log10(TF):" in captured.out
    assert "DF and IDF:" in captured.out
    assert "IDF = log10(2 / DF)" in captured.out
    assert "TF-IDF Matrix:" in captured.out
    assert "Document Vector Lengths:" in captured.out
    assert "Normalized TF-IDF Matrix:" in captured.out
    assert "0.4257" in captured.out
    assert "0.7071" in captured.out


def test_cli_missing_corpus_returns_nonzero_exit_code(
    tmp_path: Path,
    capsys: object,
) -> None:
    """Verify CLI returns non-zero exit code with error message for missing corpus."""
    from ir_system.cli import main

    exit_code = main(["--corpus", str(tmp_path / "missing_dir")])
    assert exit_code != 0

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Error:" in captured.err


def test_cli_subprocess_smoke_indexes_technology_corpus_and_exports(
    technology_corpus_dir: Path,
    tmp_path: Path,
) -> None:
    """Smoke test real PySpark CLI execution and Part A artifact export/reload."""
    export_dir = tmp_path / "part_a_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ir_system.cli",
            "--corpus",
            str(technology_corpus_dir),
            "--export-dir",
            str(export_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "Documents (N = 10):" in result.stdout
    assert "Positional Index (27 Terms):" in result.stdout
    assert "cloud | d6: 1; d9: 1, 3" in result.stdout
    assert "machine | d1: 1; d2: 1; d5: 1, 3, 9; d7: 4; d10: 1, 5" in result.stdout
    assert "Raw TF:" in result.stdout
    assert "Weighted TF: 1 + log10(TF):" in result.stdout
    assert "IDF = log10(10 / DF)" in result.stdout
    assert "TF-IDF Matrix:" in result.stdout
    assert "Document Vector Lengths:" in result.stdout
    assert "Normalized TF-IDF Matrix:" in result.stdout
    assert "0.4257" in result.stdout
    assert "1.8031" in result.stdout
    assert 'Phrase query: "machine learning"' in result.stdout
    assert "0.425721" in result.stdout
    assert "0.545341" in result.stdout
    assert "0.691834" in result.stdout
    assert "0.5113" in result.stdout
    assert "Ranking (highest similarity first): d1 > d5 > d10 > d2 > d7" in result.stdout
    assert "Ranking (highest similarity first): d5 > d3 > d8" in result.stdout
    assert "Ranking (highest similarity first): d1 > d10 > d2 > d7" in result.stdout

    report_path = export_dir / "report.md"
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Part A" in report_text
    assert "## Part B" in report_text
    assert "## Query Results" in report_text
    assert "0.425721" in report_text
    assert "d1 > d5 > d10 > d2 > d7" in report_text

    reloaded = load_positional_index(export_dir / "positional_index.json")
    assert reloaded.doc_count == 10
    assert len(reloaded.postings) == 27
    assert reloaded.postings["systems"] == {
        "d4": (5,),
        "d8": (9,),
        "d9": (7,),
        "d10": (10,),
    }


def test_cli_from_index_supports_adhoc_query_mode_and_export_roundtrip(
    tmp_path: Path,
    capsys: object,
) -> None:
    """Verify CLI --from-index with --query, --mode, --export-dir, and error handling."""
    from ir_system.cli import main

    sample = PositionalIndex(
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
    )
    exported = export_positional_index(sample, tmp_path / "seed")
    out_dir = tmp_path / "query_export"

    exit_code = main(
        [
            "--from-index",
            str(exported),
            "--query",
            "machine learning AND NOT data science",
            "--mode",
            "boolean",
            "--export-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert 'Boolean AND NOT: "machine learning AND NOT data science"' in captured.out
    assert "Ranking (highest similarity first): d1" in captured.out
    assert (out_dir / "positional_index.json").is_file()
    assert (out_dir / "report.md").is_file()
    assert 'Boolean AND NOT: "machine learning AND NOT data science"' in (out_dir / "report.md").read_text(
        encoding="utf-8"
    )

    bad_exit = main(
        [
            "--from-index",
            str(exported),
            "--query",
            "two words",
            "--mode",
            "term",
        ]
    )
    assert bad_exit == 1
    bad_captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Error:" in bad_captured.err

