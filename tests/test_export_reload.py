from __future__ import annotations

from pathlib import Path

import pytest

from ir_system.models import DocumentRecord, PositionalIndex
from ir_system.spark_indexer import export_positional_index, load_positional_index


def test_export_and_reload_positional_index_preserves_metadata_and_postings(
    tmp_path: Path,
) -> None:
    """Verify export_positional_index and load_positional_index round-trip losslessly."""
    original = PositionalIndex(
        documents=(
            DocumentRecord(doc_id="alpha", text="alpha beta", word_count=2),
            DocumentRecord(doc_id="doc_2", text="beta gamma alpha", word_count=3),
            DocumentRecord(doc_id="doc_10", text="alpha", word_count=1),
        ),
        postings={
            "alpha": {"alpha": (1,), "doc_2": (3,), "doc_10": (1,)},
            "beta": {"alpha": (2,), "doc_2": (1,)},
            "gamma": {"doc_2": (2,)},
        },
        warnings=("Skipped empty or whitespace-only file: empty.txt",),
        indexing_seconds=1.25,
        corpus_path="custom/corpus",
    )

    out_dir = tmp_path / "nested_output"
    exported_path = export_positional_index(original, out_dir)

    assert exported_path == out_dir / "positional_index.json"
    assert exported_path.is_file()

    reloaded_from_file = load_positional_index(exported_path)
    assert reloaded_from_file == original
    assert list(reloaded_from_file.postings["alpha"].keys()) == ["alpha", "doc_2", "doc_10"]

    reloaded_from_dir = load_positional_index(out_dir)
    assert reloaded_from_dir == original


def test_load_positional_index_missing_path_raises(tmp_path: Path) -> None:
    """Verify loading a missing PositionalIndex artifact raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_positional_index(tmp_path / "nonexistent.json")

