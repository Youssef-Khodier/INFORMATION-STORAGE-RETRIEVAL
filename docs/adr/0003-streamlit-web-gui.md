# 0003 — Streamlit for the Bonus Web GUI

Status: accepted

We use Streamlit as the Web GUI framework for the Information Retrieval application instead of a custom `http.server` + HTML/CSS/JS stack or desktop `tkinter`.

## Context

The project coversheet awards +1.0 bonus mark for a clear, usable GUI supporting dataset import, query entry, and mode selection, while the project requirements ask us to justify any dependency beyond PySpark.

## Considered Options

- **Streamlit (chosen)**: Provides built-in searchable/scrollable data tables for large TF/TF-IDF matrices, native file/folder upload widgets, tabs, and Markdown download buttons with minimal UI boilerplate — letting us focus effort on PySpark indexing accuracy and clean retrieval presentation.
- **Python `http.server` + vanilla HTML/CSS/JS**: Zero extra Python dependencies, but requires custom HTTP routing, multipart upload handling, and manual table rendering/filtering.
- **Built-in `tkinter`**: Zero extra dependencies, but clunky for wide 10-column scrollable numeric matrices and hard-copy screenshots.

## Consequences

- `streamlit` (and its required runtime libraries such as `pandas`) is added as an explicit, justified GUI dependency in `pyproject.toml`, while the core Part A (`spark_indexer.py`), Part B (`retrieval.py`), and CLI remain completely independent of Streamlit.
- Because Streamlit re-runs its script on widget interactions, the PySpark `SparkSession` and indexed corpus must be cached so Spark runs only when importing/re-indexing a dataset, not on every query.
