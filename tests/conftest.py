from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from pyspark.sql import SparkSession

from ir_system.models import PositionalIndex
from ir_system.spark_indexer import build_positional_index, create_spark_session


@pytest.fixture(scope="session")
def spark_session() -> Iterator[SparkSession]:
    """Provide a shared session-scoped SparkSession for real PySpark tests."""
    spark = create_spark_session(app_name="TechnologyIRTests", master="local[2]")
    try:
        yield spark
    finally:
        spark.stop()


@pytest.fixture(scope="session")
def technology_corpus_dir() -> Path:
    """Return the path to the Technology Version corpus directory."""
    return Path(__file__).resolve().parent.parent / "data" / "2. Technology"


@pytest.fixture(scope="session")
def technology_index(
    spark_session: SparkSession,
    technology_corpus_dir: Path,
) -> PositionalIndex:
    """Index the Technology Corpus once per test session using real PySpark."""
    return build_positional_index(technology_corpus_dir, spark=spark_session)

