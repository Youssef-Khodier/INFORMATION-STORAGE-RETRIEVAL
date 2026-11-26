from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from ir_system.models import DocumentRecord, PositionalIndex, natural_sort_key


_DOCUMENT_SCHEMA = StructType(
    [
        StructField("doc_id", StringType(), False),
        StructField("text", StringType(), False),
    ]
)


def _needs_windows_junction(path: Path) -> bool:
    """Return True on Windows when a path contains cmd.exe-unsafe characters."""
    path_str = str(path)
    return os.name == "nt" and (" " in path_str or "&" in path_str)


def _ensure_safe_windows_dir(target_dir: Path, prefix: str) -> Path:
    """Return a space-free NTFS junction for target_dir when needed on Windows."""
    resolved = target_dir.resolve()
    if not _needs_windows_junction(resolved):
        return resolved

    temp_root = Path(tempfile.gettempdir())
    if _needs_windows_junction(temp_root):
        return resolved

    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    junction_path = temp_root / f"{prefix}_{digest}"
    if not junction_path.exists():
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction_path), str(resolved)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return junction_path


def _configure_pyspark_python() -> None:
    """Ensure PySpark worker, driver, and SPARK_HOME work cleanly on Windows."""
    python_path = Path(sys.executable)
    venv_dir = python_path.parent.parent
    if _needs_windows_junction(python_path) and (venv_dir / "pyvenv.cfg").is_file():
        safe_venv = _ensure_safe_windows_dir(venv_dir, "pyspark_venv")
        safe_python = str(safe_venv / python_path.parent.name / python_path.name)
    else:
        safe_python = sys.executable

    os.environ["PYSPARK_PYTHON"] = safe_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = safe_python

    spark_home = Path(os.environ.get("SPARK_HOME") or Path(pyspark.__file__).resolve().parent)
    if _needs_windows_junction(spark_home):
        os.environ["SPARK_HOME"] = str(_ensure_safe_windows_dir(spark_home, "pyspark_home"))


def create_spark_session(
    app_name: str = "TechnologyIRSystem",
    master: str = "local[*]",
) -> SparkSession:
    """Create or retrieve a configured local SparkSession without requiring winutils."""
    _configure_pyspark_python()
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _read_corpus_files(corpus_path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Read non-empty .txt files from corpus_path while recording skipped-file warnings."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_path}")
    if not corpus_path.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {corpus_path}")

    txt_files = sorted(
        (
            entry
            for entry in corpus_path.iterdir()
            if entry.is_file() and entry.suffix.lower() == ".txt"
        ),
        key=lambda path: natural_sort_key(path.stem),
    )

    raw_documents: list[tuple[str, str]] = []
    warnings: list[str] = []
    seen_doc_ids: set[str] = set()

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            warnings.append(f"Skipped empty or whitespace-only file: {file_path.name}")
            continue
        doc_id = file_path.stem
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate document ID '{doc_id}' in corpus directory: {corpus_path}")
        seen_doc_ids.add(doc_id)
        raw_documents.append((doc_id, text))

    if not raw_documents:
        raise ValueError(f"No non-empty .txt documents found in corpus directory: {corpus_path}")

    return raw_documents, warnings


def build_positional_index(
    corpus_dir: Path | str,
    spark: SparkSession | None = None,
    master: str = "local[*]",
) -> PositionalIndex:
    """Build a sorted PositionalIndex from .txt Documents using PySpark transformations."""
    corpus_path = Path(corpus_dir)
    raw_documents, warnings = _read_corpus_files(corpus_path)

    owns_spark = spark is None
    active_spark = spark if spark is not None else create_spark_session(master=master)
    start_time = time.perf_counter()

    try:
        docs_df = active_spark.createDataFrame(raw_documents, schema=_DOCUMENT_SCHEMA)
        tokens_df = (
            docs_df.select(
                F.col("doc_id"),
                F.posexplode(
                    F.split(
                        F.trim(F.regexp_replace(F.lower(F.col("text")), r"\s+", " ")),
                        " ",
                    )
                ).alias("zero_pos", "term"),
            )
            .filter(F.col("term") != "")
            .withColumn("position", F.col("zero_pos") + F.lit(1))
        )

        postings_rows = (
            tokens_df.groupBy("term", "doc_id")
            .agg(F.sort_array(F.collect_list("position")).alias("positions"))
            .orderBy("term", "doc_id")
            .collect()
        )
    finally:
        if owns_spark:
            active_spark.stop()

    elapsed_seconds = time.perf_counter() - start_time

    word_counts: dict[str, int] = {doc_id: 0 for doc_id, _ in raw_documents}
    grouped_postings: dict[str, dict[str, tuple[int, ...]]] = {}

    for row in postings_rows:
        term = str(row["term"])
        doc_id = str(row["doc_id"])
        positions = tuple(sorted(int(pos) for pos in row["positions"]))
        grouped_postings.setdefault(term, {})[doc_id] = positions
        word_counts[doc_id] += len(positions)

    sorted_postings: dict[str, dict[str, tuple[int, ...]]] = {
        term: {
            doc_id: grouped_postings[term][doc_id]
            for doc_id in sorted(grouped_postings[term].keys(), key=natural_sort_key)
        }
        for term in sorted(grouped_postings.keys())
    }

    document_records = tuple(
        DocumentRecord(
            doc_id=doc_id,
            text=text,
            word_count=word_counts[doc_id],
        )
        for doc_id, text in sorted(raw_documents, key=lambda item: natural_sort_key(item[0]))
    )

    return PositionalIndex(
        documents=document_records,
        postings=sorted_postings,
        warnings=tuple(warnings),
        indexing_seconds=elapsed_seconds,
        corpus_path=str(corpus_path),
    )


DEFAULT_INDEX_FILENAME = "positional_index.json"


def export_positional_index(
    index: PositionalIndex,
    output_dir: Path | str,
    filename: str = DEFAULT_INDEX_FILENAME,
) -> Path:
    """Serialize a PositionalIndex artifact to JSON inside output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / filename
    target.write_text(json.dumps(index.to_dict(), indent=2), encoding="utf-8")
    return target


def load_positional_index(index_path: Path | str) -> PositionalIndex:
    """Reload a serialized PositionalIndex artifact from a JSON file or directory."""
    candidate = Path(index_path)
    if candidate.is_dir():
        candidate = candidate / DEFAULT_INDEX_FILENAME
    if not candidate.is_file():
        raise FileNotFoundError(f"Positional index artifact not found: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return PositionalIndex.from_dict(payload)
