# Technology Version PySpark Information Retrieval System

A modular **Information Storage & Retrieval** application (**Technology Version**) built with **PySpark 4.0.1 (`local[*]`)** for **Part A** (Positional Index construction) and pure Python for **Part B** (TF-IDF vector-space modeling, exact Phrase Queries, case-sensitive uppercase Boolean `AND` / `AND NOT` queries, and cosine Similarity Score ranking). Includes both a reproducible **CLI (`ir-system`)** and a **Streamlit Web GUI (`app.py`)** supporting **Dataset Flexibility** across arbitrary `.txt` corpora.

## Documentation & Architecture Links

- [Domain Glossary](GLOSSARY.md) — Canonical IR terms and formulas (`Positional Index`, `Raw TF`, `Weighted TF`, `DF`, `IDF`, `TF-IDF Weight`, `Document Vector Length`, `Normalized TF-IDF`, `Query Vector`, `Similarity Score`).
- [Architecture Decision Records (`docs/adr/`)](docs/adr/) — Key design decisions:
  - [ADR 0001 — PySpark for Part A Positional Indexing, Pure Python for Part B Retrieval](docs/adr/0001-pyspark-indexing-python-retrieval.md)
  - [ADR 0002 — Case-Sensitive Uppercase Boolean Operators](docs/adr/0002-case-sensitive-boolean-operators.md)
  - [ADR 0003 — Streamlit for the Bonus Web GUI](docs/adr/0003-streamlit-web-gui.md)
- [Part C Hard-Copy Submission Report](docs/PROJECT_REPORT.md) — Complete system architecture, Apache Spark Part A breakdown, mathematical formulas, dependency justification, Windows execution guide, live-generated Technology Version tables/query results, and answers to all five Part C conceptual questions.

---

## Quick-Start Setup (Windows / PowerShell)

### Prerequisites

- **Java 17 JDK** (verified with OpenJDK Temurin `17.0.16+8`)
- **Python `3.10.20`** (pinned in `.python-version`, managed automatically by `uv`)
- **PySpark `4.0.1`** (installed via `pyproject.toml`)
- **`uv`** package manager

### Install Dependencies

```powershell
uv sync
```

---

## CLI Usage (`ir-system`)

Run the default Technology Corpus (`data/2. Technology/`) and all four assigned queries (`"machine learning"`, `"data science"`, `"machine learning AND data science"`, `"machine learning AND NOT data science"`):

```powershell
uv run ir-system
```

Export the Part A Positional Index (`positional_index.json`) and complete Markdown report (`report.md`):

```powershell
uv run ir-system --export-dir output/part_c_capture
```

Run custom queries with explicit modes (`auto`, `term`, `phrase`, `boolean`):

```powershell
uv run ir-system --query "machine" --mode term
uv run ir-system --query "data science and machine learning" --mode phrase
uv run ir-system --query "machine learning AND NOT data science" --mode boolean
```

Reload a previously exported Part A `positional_index.json` without restarting Spark:

```powershell
uv run ir-system --from-index output/part_c_capture/positional_index.json --query "machine learning"
```

Index and query any custom folder of `.txt` documents (Dataset Flexibility):

```powershell
uv run ir-system --corpus "path\to\custom_corpus" --query "your search query" --mode auto
```

---

## Streamlit Web GUI

Launch the interactive Web GUI:

```powershell
uv run streamlit run app.py
```

Key GUI features:
- Auto-loads `data/2. Technology/` on startup using a cached `SparkSession`.
- Sidebar **Corpus Import** via local folder path or `.txt` file/folder upload, plus **Reset to Technology**.
- Four-mode query selector (`Auto`, `Term Search`, `Exact Phrase`, `Boolean (AND / AND NOT)`) and quick-fill buttons for all four assigned Technology queries.
- Interactive tabs for **Documents**, **Positional Index** (with live term filter), **Raw TF & Weighted TF**, **DF & IDF**, and **TF-IDF / Document Vector Lengths / Normalized TF-IDF**, plus one-click **Download Report (Markdown)**.

---

## Running the Verification Suite

Run the full automated test suite (Part A PySpark indexing, Part B PDF reference tables, Phrase & Boolean retrieval, edge cases, CLI export/reload, Dataset Flexibility, Streamlit `AppTest`, and documentation checks):

```powershell
uv run pytest -v
```
