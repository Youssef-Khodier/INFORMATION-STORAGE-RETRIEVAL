from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from ir_system.formatting import format_part_a_output, format_part_b_output
from ir_system.retrieval import build_vector_space_model
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
        "--export-dir",
        type=Path,
        default=None,
        help="Optional output directory to write the Part A positional_index.json artifact.",
    )
    parser.add_argument(
        "--master",
        default="local[*]",
        help="Spark master URL for Part A indexing (default: local[*]).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Part A Positional Index and Part B Vector-Space CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.from_index is not None:
            index = load_positional_index(args.from_index)
        else:
            corpus_dir = args.corpus if args.corpus is not None else _default_corpus_dir()
            index = build_positional_index(corpus_dir, master=args.master)

        vector_space = build_vector_space_model(index)

        print(format_part_a_output(index))
        print()
        print(format_part_b_output(vector_space))

        if args.export_dir is not None:
            exported_path = export_positional_index(index, args.export_dir)
            print(f"\nExported Positional Index to: {exported_path}")
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

