from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


_NATURAL_CHUNK_RE = re.compile(r"(\d+)")


def natural_sort_key(text: str) -> tuple[tuple[int, int | str, str], ...]:
    """Return a type-safe natural sort key splitting numeric and text chunks."""
    parts: list[tuple[int, int | str, str]] = []
    for chunk in _NATURAL_CHUNK_RE.split(text):
        if not chunk:
            continue
        if chunk.isdigit():
            parts.append((0, int(chunk), chunk))
        else:
            parts.append((1, chunk.casefold(), chunk))
    return tuple(parts)


@dataclass(frozen=True)
class DocumentRecord:
    """Metadata and exact original text for a single Corpus Document."""

    doc_id: str
    text: str
    word_count: int


@dataclass(frozen=True)
class PositionalIndex:
    """Part A Positional Index artifact consumed by Part B retrieval and formatting."""

    documents: tuple[DocumentRecord, ...]
    postings: dict[str, dict[str, tuple[int, ...]]]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    indexing_seconds: float = 0.0
    corpus_path: str = ""

    @property
    def doc_count(self) -> int:
        """Return the active Corpus size N."""
        return len(self.documents)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        """Return naturally ordered Document IDs in the Corpus."""
        return tuple(doc.doc_id for doc in self.documents)

    @property
    def terms(self) -> tuple[str, ...]:
        """Return alphabetically sorted vocabulary Terms."""
        return tuple(self.postings.keys())

    @property
    def total_tokens(self) -> int:
        """Return total Token count across all Documents in the Corpus."""
        return sum(doc.word_count for doc in self.documents)

    def to_dict(self) -> dict[str, Any]:
        """Convert this PositionalIndex into a JSON-serializable dictionary."""
        return {
            "corpus_path": self.corpus_path,
            "indexing_seconds": self.indexing_seconds,
            "warnings": list(self.warnings),
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "word_count": doc.word_count,
                }
                for doc in self.documents
            ],
            "postings": {
                term: {doc_id: list(positions) for doc_id, positions in doc_map.items()}
                for term, doc_map in self.postings.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PositionalIndex:
        """Reconstruct a sorted PositionalIndex from a serialized dictionary."""
        documents = tuple(
            sorted(
                (
                    DocumentRecord(
                        doc_id=str(item["doc_id"]),
                        text=str(item["text"]),
                        word_count=int(item["word_count"]),
                    )
                    for item in payload.get("documents", [])
                ),
                key=lambda doc: natural_sort_key(doc.doc_id),
            )
        )
        raw_postings = payload.get("postings", {})
        sorted_postings: dict[str, dict[str, tuple[int, ...]]] = {}
        for term in sorted(raw_postings.keys()):
            doc_map = raw_postings[term]
            sorted_postings[str(term)] = {
                str(doc_id): tuple(sorted(int(pos) for pos in doc_map[doc_id]))
                for doc_id in sorted(doc_map.keys(), key=natural_sort_key)
            }
        return cls(
            documents=documents,
            postings=sorted_postings,
            warnings=tuple(str(w) for w in payload.get("warnings", [])),
            indexing_seconds=float(payload.get("indexing_seconds", 0.0)),
            corpus_path=str(payload.get("corpus_path", "")),
        )


def default_corpus_dir() -> Path:
    """Resolve the default Technology Version Corpus directory."""
    cwd_candidate = Path("data") / "2. Technology"
    if cwd_candidate.is_dir():
        return cwd_candidate.resolve()
    return (Path(__file__).resolve().parent.parent.parent / "data" / "2. Technology").resolve()


def is_technology_corpus(index: PositionalIndex, corpus_dir: Path | None = None) -> bool:
    """Return True when the active Corpus matches the default Technology Corpus."""
    tech_dir = default_corpus_dir()
    if corpus_dir is not None and corpus_dir.resolve() == tech_dir:
        return True
    if index.corpus_path:
        try:
            if Path(index.corpus_path).resolve() == tech_dir:
                return True
        except OSError:
            pass
    if not tech_dir.is_dir():
        return False

    tech_docs = tuple(
        (path.stem, text)
        for path in sorted(
            (
                entry
                for entry in tech_dir.iterdir()
                if entry.is_file() and entry.suffix.lower() == ".txt"
            ),
            key=lambda path: natural_sort_key(path.stem),
        )
        if (text := path.read_text(encoding="utf-8")).strip()
    )
    active_docs = tuple((doc.doc_id, doc.text) for doc in index.documents)
    return active_docs == tech_docs

