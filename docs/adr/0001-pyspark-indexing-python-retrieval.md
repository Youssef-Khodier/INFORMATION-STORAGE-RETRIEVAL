# 0001 — PySpark for Part A Positional Indexing, Pure Python for Part B Retrieval

Status: accepted

We use PySpark (`local[*]`) exclusively for Part A (ingesting `.txt` documents, lowercasing, whitespace-tokenizing, assigning 1-based positions, and aggregating the sorted Positional Index) and implement Part B (Raw TF, Weighted TF, DF/IDF, TF-IDF, document lengths, Normalized TF-IDF, phrase/Boolean matching, cosine similarity, and ranking) in pure Python directly consuming Part A's Positional Index output.

## Considered Options

- **PySpark for Part A + Pure Python for Part B (chosen)**: Mirrors the project specification ("Part B: 1. Use the output of the Positional Index from Part A"), keeps Spark's distributed ingestion/indexing role crisp and easy to explain during evaluation, and makes interactive CLI/Web queries instantaneous without paying Spark job startup latency per query.
- **PySpark DataFrames/RDDs for both Part A and Part B**: Adds per-query Spark overhead and boilerplate matrix pivoting without improving accuracy or rubric coverage.

## Consequences

- Part A produces a clean, inspectable Positional Index structure (and optionally serialized artifact) that Part B consumes deterministically.
- The SparkSession only runs when loading/indexing a corpus, keeping query latency sub-millisecond in the Web UI and CLI.
