from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Sequence

from ir_system.models import PositionalIndex, natural_sort_key
from ir_system.vector_space import VectorSpaceModel, build_vector_space_model

QUERY_MODES: tuple[str, ...] = ("auto", "term", "phrase", "boolean")

_MODE_ALIASES: dict[str, str] = {
    "auto": "auto",
    "term": "term",
    "term search": "term",
    "term_search": "term",
    "single-term": "term",
    "single term": "term",
    "phrase": "phrase",
    "exact phrase": "phrase",
    "exact_phrase": "phrase",
    "phrase query": "phrase",
    "boolean": "boolean",
    "bool": "boolean",
}


def normalize_query_mode(mode: str) -> str:
    """Normalize a query mode name or UI label to one of auto/term/phrase/boolean."""
    normalized = " ".join(mode.strip().lower().split())
    canonical = _MODE_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unsupported query mode: {mode!r}. Expected one of: {', '.join(QUERY_MODES)}."
        )
    return canonical


@dataclass(frozen=True)
class QueryClause:
    """A single Phrase Query operand within a ParsedQuery."""

    tokens: tuple[str, ...]
    operator: str | None = None
    is_negated: bool = False


@dataclass(frozen=True)
class ParsedQuery:
    """Validated query representation split into positive and excluded Phrase Query clauses."""

    raw_query: str
    mode: str
    query_type: str
    clauses: tuple[QueryClause, ...]

    @property
    def positive_clauses(self) -> tuple[tuple[str, ...], ...]:
        """Return all Positive Clause token tuples in left-to-right order."""
        return tuple(clause.tokens for clause in self.clauses if not clause.is_negated)

    @property
    def excluded_clauses(self) -> tuple[tuple[str, ...], ...]:
        """Return all Excluded Clause token tuples in left-to-right order."""
        return tuple(clause.tokens for clause in self.clauses if clause.is_negated)


def _split_boolean_clauses(raw_tokens: list[str]) -> tuple[list[QueryClause], bool]:
    """Split raw whitespace tokens on case-sensitive uppercase AND NOT / AND operators."""
    clauses: list[QueryClause] = []
    current_tokens: list[str] = []
    current_operator: str | None = None
    saw_operator = False

    idx = 0
    while idx < len(raw_tokens):
        token = raw_tokens[idx]
        if token == "AND":
            if idx + 1 < len(raw_tokens) and raw_tokens[idx + 1] == "NOT":
                operator = "AND NOT"
                step = 2
            else:
                operator = "AND"
                step = 1

            saw_operator = True
            if not current_tokens:
                if current_operator is None:
                    raise ValueError(f"Malformed Boolean query: missing left clause before '{operator}'.")
                raise ValueError(
                    f"Malformed Boolean query: empty clause between '{current_operator}' and '{operator}'."
                )
            clauses.append(
                QueryClause(
                    tokens=tuple(tok.lower() for tok in current_tokens),
                    operator=current_operator,
                    is_negated=(current_operator == "AND NOT"),
                )
            )
            current_tokens = []
            current_operator = operator
            idx += step
            continue

        current_tokens.append(token)
        idx += 1

    if saw_operator and not current_tokens:
        raise ValueError(f"Malformed Boolean query: missing right clause after '{current_operator}'.")

    if current_tokens:
        clauses.append(
            QueryClause(
                tokens=tuple(tok.lower() for tok in current_tokens),
                operator=current_operator,
                is_negated=(current_operator == "AND NOT"),
            )
        )

    return clauses, saw_operator


def parse_query(raw_query: str, mode: str = "auto") -> ParsedQuery:
    """Parse and validate a user query string under the specified query mode."""
    canonical_mode = normalize_query_mode(mode)
    trimmed = raw_query.strip()
    if not trimmed:
        raise ValueError("Query must not be empty.")

    raw_tokens = trimmed.split()

    if canonical_mode == "phrase":
        literal_tokens = tuple(tok.lower() for tok in raw_tokens)
        return ParsedQuery(
            raw_query=trimmed,
            mode=canonical_mode,
            query_type="phrase",
            clauses=(QueryClause(tokens=literal_tokens, operator=None, is_negated=False),),
        )

    clauses, saw_operator = _split_boolean_clauses(raw_tokens)

    if canonical_mode == "term":
        if saw_operator:
            raise ValueError("Term Search mode does not allow Boolean operators (AND / AND NOT).")
        if len(clauses[0].tokens) != 1:
            raise ValueError("Term Search mode requires exactly one token.")
        return ParsedQuery(
            raw_query=trimmed,
            mode=canonical_mode,
            query_type="phrase",
            clauses=tuple(clauses),
        )

    if canonical_mode == "boolean":
        if not saw_operator:
            raise ValueError(
                "Boolean mode requires at least one uppercase Boolean operator (AND or AND NOT)."
            )
        return ParsedQuery(
            raw_query=trimmed,
            mode=canonical_mode,
            query_type="boolean",
            clauses=tuple(clauses),
        )

    return ParsedQuery(
        raw_query=trimmed,
        mode=canonical_mode,
        query_type="boolean" if saw_operator else "phrase",
        clauses=tuple(clauses),
    )


@dataclass(frozen=True)
class DocumentPhraseMatch:
    """Exact Phrase Query match positions and inclusive token ranges in one Document."""

    doc_id: str
    starting_positions: tuple[int, ...]
    token_ranges: tuple[tuple[int, int], ...]

    @property
    def formatted_positions(self) -> str:
        """Format 1-based starting Positions as comma-separated integers."""
        return ", ".join(str(pos) for pos in self.starting_positions)

    @property
    def formatted_ranges(self) -> str:
        """Format inclusive token ranges as 'start-end, ...'."""
        return ", ".join(f"{start}-{end}" for start, end in self.token_ranges)


@dataclass(frozen=True)
class ClauseMatchResult:
    """Positional matches for a single parsed Phrase Query clause."""

    clause: QueryClause
    matches: tuple[DocumentPhraseMatch, ...]

    @property
    def matched_doc_ids(self) -> tuple[str, ...]:
        """Return naturally ordered Document IDs matching this clause."""
        return tuple(match.doc_id for match in self.matches)


@dataclass(frozen=True)
class QueryTermWeight:
    """Weighting breakdown for one unique Term in a Query Vector."""

    term: str
    raw_tf: int
    weighted_tf: float
    idf: float
    tf_idf: float
    normalized_weight: float


@dataclass(frozen=True)
class RankedDocument:
    """A matched Document paired with its cosine Similarity Score and 1-based rank."""

    doc_id: str
    similarity: float
    rank: int


@dataclass(frozen=True)
class QueryResult:
    """Complete retrieval, scoring, and ranking output for a single query."""

    parsed_query: ParsedQuery
    clause_matches: tuple[ClauseMatchResult, ...]
    matched_doc_ids: tuple[str, ...]
    query_length: float
    term_weights: tuple[QueryTermWeight, ...]
    ranked_documents: tuple[RankedDocument, ...]

    @property
    def raw_query(self) -> str:
        """Return the trimmed query string."""
        return self.parsed_query.raw_query

    @property
    def phrase_matches(self) -> tuple[DocumentPhraseMatch, ...]:
        """Return DocumentPhraseMatch entries when the query is a single Phrase Query."""
        if len(self.clause_matches) == 1:
            return self.clause_matches[0].matches
        return ()

    @property
    def ranking_expression(self) -> str:
        """Return 'd1 > d5 > ...' ranking string, or empty string when no Documents match."""
        return " > ".join(item.doc_id for item in self.ranked_documents)


def match_phrase(index: PositionalIndex, phrase_tokens: Sequence[str]) -> tuple[DocumentPhraseMatch, ...]:
    """Find all Documents containing phrase_tokens contiguously in exact 1-based Position order."""
    if not phrase_tokens:
        return ()

    first_term = phrase_tokens[0]
    first_postings = index.postings.get(first_term)
    if not first_postings:
        return ()

    phrase_len = len(phrase_tokens)
    if phrase_len == 1:
        return tuple(
            DocumentPhraseMatch(
                doc_id=doc_id,
                starting_positions=positions,
                token_ranges=tuple((pos, pos) for pos in positions),
            )
            for doc_id, positions in first_postings.items()
        )

    subsequent_postings: list[dict[str, tuple[int, ...]]] = []
    for term in phrase_tokens[1:]:
        term_postings = index.postings.get(term)
        if not term_postings:
            return ()
        subsequent_postings.append(term_postings)

    matches: list[DocumentPhraseMatch] = []
    for doc_id in sorted(first_postings.keys(), key=natural_sort_key):
        if any(doc_id not in postings for postings in subsequent_postings):
            continue
        pos_sets = [set(postings[doc_id]) for postings in subsequent_postings]
        matched_starts = tuple(
            start_pos
            for start_pos in first_postings[doc_id]
            if all((start_pos + offset) in pos_sets[offset - 1] for offset in range(1, phrase_len))
        )
        if matched_starts:
            matches.append(
                DocumentPhraseMatch(
                    doc_id=doc_id,
                    starting_positions=matched_starts,
                    token_ranges=tuple((pos, pos + phrase_len - 1) for pos in matched_starts),
                )
            )
    return tuple(matches)


DEFAULT_TECHNOLOGY_QUERIES: tuple[str, ...] = (
    "machine learning",
    "data science",
    "machine learning AND data science",
    "machine learning AND NOT data science",
)


def _evaluate_boolean_matches(clause_matches: Sequence[ClauseMatchResult]) -> tuple[str, ...]:
    """Evaluate matched Document IDs left-to-right across Positive and Excluded Clauses."""
    active_docs = set(clause_matches[0].matched_doc_ids)
    for clause_result in clause_matches[1:]:
        clause_docs = set(clause_result.matched_doc_ids)
        if clause_result.clause.is_negated:
            active_docs -= clause_docs
        else:
            active_docs &= clause_docs
    return tuple(sorted(active_docs, key=natural_sort_key))


def _build_query_vector(
    parsed_query: ParsedQuery,
    model: VectorSpaceModel,
) -> tuple[float, tuple[QueryTermWeight, ...]]:
    """Build L2-normalized Query Vector weights strictly from Positive Clause Tokens."""
    positive_counts: Counter[str] = Counter()
    for clause_tokens in parsed_query.positive_clauses:
        for token in clause_tokens:
            positive_counts[token] += 1

    raw_weights: list[tuple[str, int, float, float, float]] = []
    squared_sum = 0.0
    for term, raw_tf in positive_counts.items():
        weighted_tf = 1.0 + math.log10(raw_tf) if raw_tf > 0 else 0.0
        idf = model.idf.get(term, 0.0)
        tf_idf = weighted_tf * idf
        squared_sum += tf_idf * tf_idf
        raw_weights.append((term, raw_tf, weighted_tf, idf, tf_idf))

    query_length = math.sqrt(squared_sum)
    term_weights = tuple(
        QueryTermWeight(
            term=term,
            raw_tf=raw_tf,
            weighted_tf=weighted_tf,
            idf=idf,
            tf_idf=tf_idf,
            normalized_weight=(tf_idf / query_length if query_length > 0.0 else 0.0),
        )
        for term, raw_tf, weighted_tf, idf, tf_idf in raw_weights
    )
    return query_length, term_weights


def _score_and_rank_documents(
    matched_doc_ids: Sequence[str],
    term_weights: Sequence[QueryTermWeight],
    model: VectorSpaceModel,
) -> tuple[RankedDocument, ...]:
    """Compute cosine Similarity Scores and rank descending with natural Document ID tie-breaking."""
    scored: list[tuple[str, float]] = []
    for doc_id in matched_doc_ids:
        similarity = 0.0
        for weight in term_weights:
            doc_norm = model.normalized_tf_idf.get(weight.term, {}).get(doc_id, 0.0)
            similarity += weight.normalized_weight * doc_norm
        scored.append((doc_id, similarity))

    scored.sort(key=lambda item: (-item[1], natural_sort_key(item[0])))
    return tuple(
        RankedDocument(doc_id=doc_id, similarity=similarity, rank=rank)
        for rank, (doc_id, similarity) in enumerate(scored, start=1)
    )


def execute_query(
    index: PositionalIndex,
    raw_query: str,
    mode: str = "auto",
    *,
    model: VectorSpaceModel | None = None,
) -> QueryResult:
    """Parse and execute a Phrase or Boolean Query against a PositionalIndex."""
    parsed = parse_query(raw_query, mode=mode)
    active_model = model if model is not None else build_vector_space_model(index)

    clause_matches = tuple(
        ClauseMatchResult(clause=clause, matches=match_phrase(index, clause.tokens))
        for clause in parsed.clauses
    )
    matched_doc_ids = _evaluate_boolean_matches(clause_matches)
    query_length, term_weights = _build_query_vector(parsed, active_model)
    ranked_documents = _score_and_rank_documents(matched_doc_ids, term_weights, active_model)

    return QueryResult(
        parsed_query=parsed,
        clause_matches=clause_matches,
        matched_doc_ids=matched_doc_ids,
        query_length=query_length,
        term_weights=term_weights,
        ranked_documents=ranked_documents,
    )


def run_queries(
    index: PositionalIndex,
    queries: Sequence[str] | None = None,
    mode: str = "auto",
    *,
    model: VectorSpaceModel | None = None,
) -> tuple[QueryResult, ...]:
    """Execute a sequence of queries against a single VectorSpaceModel instance."""
    active_model = model if model is not None else build_vector_space_model(index)
    target_queries = queries if queries is not None else DEFAULT_TECHNOLOGY_QUERIES
    return tuple(
        execute_query(index, query, mode=mode, model=active_model)
        for query in target_queries
    )


__all__ = [
    "ClauseMatchResult",
    "DEFAULT_TECHNOLOGY_QUERIES",
    "DocumentPhraseMatch",
    "ParsedQuery",
    "QUERY_MODES",
    "QueryClause",
    "QueryResult",
    "QueryTermWeight",
    "RankedDocument",
    "VectorSpaceModel",
    "build_vector_space_model",
    "execute_query",
    "match_phrase",
    "normalize_query_mode",
    "parse_query",
    "run_queries",
]
