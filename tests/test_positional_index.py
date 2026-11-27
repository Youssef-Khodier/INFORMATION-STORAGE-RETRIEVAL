from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession
import pytest

from ir_system.models import PositionalIndex
from ir_system.spark_indexer import (
    build_positional_index,
    export_positional_index,
    load_positional_index,
)


EXPECTED_TECHNOLOGY_DOCUMENTS = [
    ("d1", "machine learning", 2),
    ("d2", "machine learning improves data processing", 5),
    ("d3", "data science tools support analysis", 5),
    ("d4", "artificial intelligence powers smart systems", 5),
    ("d5", "machine learning machine models support data science and machine learning", 10),
    ("d6", "cloud computing supports data storage", 5),
    ("d7", "artificial intelligence uses machine learning models", 6),
    ("d8", "data science explores data analytics through research computing systems", 9),
    ("d9", "cloud computing cloud storage supports data systems", 7),
    (
        "d10",
        "machine learning models improve machine learning applications research data systems",
        10,
    ),
]

EXPECTED_TECHNOLOGY_POSTINGS: dict[str, dict[str, tuple[int, ...]]] = {
    "analysis": {"d3": (5,)},
    "analytics": {"d8": (5,)},
    "and": {"d5": (8,)},
    "applications": {"d10": (7,)},
    "artificial": {"d4": (1,), "d7": (1,)},
    "cloud": {"d6": (1,), "d9": (1, 3)},
    "computing": {"d6": (2,), "d8": (8,), "d9": (2,)},
    "data": {
        "d2": (4,),
        "d3": (1,),
        "d5": (6,),
        "d6": (4,),
        "d8": (1, 4),
        "d9": (6,),
        "d10": (9,),
    },
    "explores": {"d8": (3,)},
    "improve": {"d10": (4,)},
    "improves": {"d2": (3,)},
    "intelligence": {"d4": (2,), "d7": (2,)},
    "learning": {"d1": (2,), "d2": (2,), "d5": (2, 10), "d7": (5,), "d10": (2, 6)},
    "machine": {"d1": (1,), "d2": (1,), "d5": (1, 3, 9), "d7": (4,), "d10": (1, 5)},
    "models": {"d5": (4,), "d7": (6,), "d10": (3,)},
    "powers": {"d4": (3,)},
    "processing": {"d2": (5,)},
    "research": {"d8": (7,), "d10": (8,)},
    "science": {"d3": (2,), "d5": (7,), "d8": (2,)},
    "smart": {"d4": (4,)},
    "storage": {"d6": (5,), "d9": (4,)},
    "support": {"d3": (4,), "d5": (5,)},
    "supports": {"d6": (3,), "d9": (5,)},
    "systems": {"d4": (5,), "d8": (9,), "d9": (7,), "d10": (10,)},
    "through": {"d8": (6,)},
    "tools": {"d3": (3,)},
    "uses": {"d7": (3,)},
}


def test_technology_corpus_documents_and_positional_index_match_reference(
    technology_index: PositionalIndex,
) -> None:
    """Verify all 10 Technology Documents and all 27 Positional Index Terms match PDF p. 11."""
    assert technology_index.doc_count == 10
    assert technology_index.warnings == ()

    actual_docs = [(d.doc_id, d.text, d.word_count) for d in technology_index.documents]
    assert actual_docs == EXPECTED_TECHNOLOGY_DOCUMENTS

    assert list(technology_index.postings.keys()) == list(EXPECTED_TECHNOLOGY_POSTINGS.keys())
    assert len(technology_index.postings) == 27

    for term, expected_doc_postings in EXPECTED_TECHNOLOGY_POSTINGS.items():
        actual_doc_postings = technology_index.postings[term]
        assert list(actual_doc_postings.keys()) == list(expected_doc_postings.keys())
        assert actual_doc_postings == expected_doc_postings


def test_dataset_flexibility_ignores_non_txt_warns_empty_and_preserves_tokens(
    spark_session: SparkSession,
    tmp_path: Path,
) -> None:
    """Verify arbitrary filenames, natural sort, non-.txt ignore, empty skip, and raw tokens."""
    (tmp_path / "doc_10.txt").write_text("The API-v2, running\tfast! fast!", encoding="utf-8")
    (tmp_path / "doc_2.txt").write_text("  AND   the   API-v2, ", encoding="utf-8")
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "whitespace.txt").write_text("   \n\t  ", encoding="utf-8")
    (tmp_path / "readme.md").write_text("should be ignored", encoding="utf-8")
    (tmp_path / "nested.txt").mkdir()

    index = build_positional_index(tmp_path, spark=spark_session)

    assert index.doc_count == 3
    assert index.doc_ids == ("alpha", "doc_2", "doc_10")

    docs_by_id = {doc.doc_id: doc for doc in index.documents}
    assert docs_by_id["doc_10"].text == "The API-v2, running\tfast! fast!"
    assert docs_by_id["doc_10"].word_count == 5
    assert docs_by_id["doc_2"].text == "  AND   the   API-v2, "
    assert docs_by_id["doc_2"].word_count == 3
    assert docs_by_id["alpha"].word_count == 1

    assert len(index.warnings) == 2
    assert any("empty.txt" in w for w in index.warnings)
    assert any("whitespace.txt" in w for w in index.warnings)

    assert index.terms == ("alpha", "and", "api-v2,", "fast!", "running", "the")
    assert index.postings["api-v2,"] == {"doc_2": (3,), "doc_10": (2,)}
    assert list(index.postings["the"].keys()) == ["doc_2", "doc_10"]
    assert index.postings["the"] == {"doc_2": (2,), "doc_10": (1,)}
    assert index.postings["fast!"] == {"doc_10": (4, 5)}
    assert index.postings["and"] == {"doc_2": (1,)}


def test_build_positional_index_rejects_missing_or_empty_corpus(
    spark_session: SparkSession,
    tmp_path: Path,
) -> None:
    """Verify clear errors when corpus path is missing or has no non-empty .txt files."""
    with pytest.raises(FileNotFoundError):
        build_positional_index(tmp_path / "missing", spark=spark_session)

    (tmp_path / "only_empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="No non-empty .txt documents"):
        build_positional_index(tmp_path, spark=spark_session)


def test_technology_index_export_reload_roundtrip(
    technology_index: PositionalIndex,
    tmp_path: Path,
) -> None:
    """Verify the Spark-built Technology PositionalIndex exports and reloads losslessly."""
    exported = export_positional_index(technology_index, tmp_path / "part_a")
    reloaded = load_positional_index(exported)
    assert reloaded == technology_index


def test_windows_junction_helper_is_deterministic_conditional_and_recovers_stale_links(
    tmp_path: Path,
) -> None:
    """Verify _ensure_safe_windows_dir skips safe paths, reuses deterministic junctions, and heals stale links."""
    import os
    from ir_system.spark_indexer import _ensure_safe_windows_dir

    safe_dir = tmp_path / "clean_dir"
    safe_dir.mkdir()
    assert _ensure_safe_windows_dir(safe_dir, "test_junc") == safe_dir.resolve()

    if os.name != "nt":
        return

    unsafe_dir = tmp_path / "dir with & space"
    unsafe_dir.mkdir()
    (unsafe_dir / "marker.txt").write_text("ok", encoding="utf-8")

    junction = _ensure_safe_windows_dir(unsafe_dir, "test_junc")
    try:
        assert junction != unsafe_dir.resolve()
        assert " " not in str(junction) and "&" not in str(junction)
        assert (junction / "marker.txt").read_text(encoding="utf-8") == "ok"
        assert _ensure_safe_windows_dir(unsafe_dir, "test_junc") == junction

        # Simulate a stale/dangling junction after target directory removal and recreation.
        (unsafe_dir / "marker.txt").unlink()
        unsafe_dir.rmdir()
        unsafe_dir.mkdir()
        (unsafe_dir / "marker.txt").write_text("recreated", encoding="utf-8")

        repaired = _ensure_safe_windows_dir(unsafe_dir, "test_junc")
        assert repaired == junction
        assert (repaired / "marker.txt").read_text(encoding="utf-8") == "recreated"
    finally:
        if os.path.lexists(junction):
            os.rmdir(junction)
