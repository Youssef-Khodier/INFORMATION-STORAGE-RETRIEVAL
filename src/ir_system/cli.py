from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from ir_system.formatting import (
    format_part_a_output,
    format_part_b_output,
    format_query_results,
)
from ir_system.report import export_markdown_report
from ir_system.retrieval import (
    DEFAULT_TECHNOLOGY_QUERIES,
    build_vector_space_model,
    normalize_query_mode,
    run_queries,
)
from ir_system.spark_indexer import (
    build_positional_index,
    export_positional_index,
    load_positional_index,
)


def _default_corpus_dir() -> Path:
    """Resolve the default Technology Corpus directory."""
    cwd_candidate = Path("data") / "2. Technology"
    if cwd_candidate.is_dir():
        return cwd_candidate
    return Path(__file__).resolve().parent.parent.parent / "data" / "2. Technology"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the IR system CLI."""
    parser = argparse.ArgumentParser(
        prog="ir-system",
        description="Technology Version PySpark Positional Indexer & IR CLI",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to a Corpus directory of .txt Documents (default: data/2. Technology).",
    )
    source_group.add_argument(
        "--from-index",
        type=Path,
        default=None,
        help="Reload a previously exported Part A Positional Index artifact instead of running Spark.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help=(
            "Query string to execute (repeatable). When omitted, runs the 4 assigned "
            "Technology queries."
        ),
    )
    parser.add_argument(
        "--mode",
        default="auto",
        help="Query parsing mode: auto, term, phrase, or boolean (default: auto).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional output directory to write positional_index.json and report.md.",
    )
    parser.add_argument(
        "--master",
        default="local[*]",
        help="Spark master URL for Part A indexing (default: local[*]).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Part A Positional Index, Part B Vector-Space, Retrieval, and Report CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        canonical_mode = normalize_query_mode(args.mode)
        if args.from_index is not None:
            index = load_positional_index(args.from_index)
        else:
            corpus_dir = args.corpus if args.corpus is not None else _default_corpus_dir()
            index = build_positional_index(corpus_dir, master=args.master)

        vector_space = build_vector_space_model(index)
        target_queries = tuple(args.query) if args.query is not None else DEFAULT_TECHNOLOGY_QUERIES
        effective_mode = canonical_mode if args.query is not None else "auto"
        query_results = run_queries(
            index,
            queries=target_queries,
            mode=effective_mode,
            model=vector_space,
        )

        print(format_part_a_output(index))
        print()
        print(format_part_b_output(vector_space))
        print()
        print(format_query_results(query_results))

        if args.export_dir is not None:
            exported_index_path = export_positional_index(index, args.export_dir)
            exported_report_path = export_markdown_report(
                index,
                vector_space,
                query_results,
                args.export_dir,
            )
            print(f"\nExported Positional Index to: {exported_index_path}")
            print(f"Exported Markdown Report to: {exported_report_path}")
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
