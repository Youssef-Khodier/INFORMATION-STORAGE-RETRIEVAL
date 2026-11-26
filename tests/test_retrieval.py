from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession
import pytest

from ir_system.models import DocumentRecord, PositionalIndex
from ir_system.report import generate_markdown_report
from ir_system.retrieval import build_vector_space_model, execute_query, parse_query
from ir_system.spark_indexer import build_positional_index


def test_parse_query_modes_and_case_sensitive_boolean_operators() -> None:
    """Verify auto, term, phrase, and boolean parsing per ADR 0002 and Spec #1."""
    # Auto mode: phrase (multi-token and single-token) vs Boolean
    phrase_parsed = parse_query("  Machine   Learning  ", mode="auto")
    assert phrase_parsed.mode == "auto"
    assert phrase_parsed.query_type == "phrase"
    assert phrase_parsed.positive_clauses == (("machine", "learning"),)
    assert phrase_parsed.excluded_clauses == ()

    single_parsed = parse_query("and", mode="auto")
    assert single_parsed.query_type == "phrase"
    assert single_parsed.positive_clauses == (("and",),)
    assert single_parsed.excluded_clauses == ()

    lower_and_phrase = parse_query("data science and machine learning", mode="auto")
    assert lower_and_phrase.query_type == "phrase"
    assert lower_and_phrase.positive_clauses == (
        ("data", "science", "and", "machine", "learning"),
    )

    bool_and = parse_query("machine learning AND data science", mode="auto")
    assert bool_and.query_type == "boolean"
    assert bool_and.positive_clauses == (("machine", "learning"), ("data", "science"))
    assert bool_and.excluded_clauses == ()

    bool_and_not = parse_query("machine learning AND NOT data science", mode="auto")
    assert bool_and_not.query_type == "boolean"
    assert bool_and_not.positive_clauses == (("machine", "learning"),)
    assert bool_and_not.excluded_clauses == (("data", "science"),)

    # Chained left-to-right Boolean
    chained = parse_query("machine learning AND models AND NOT data science", mode="boolean")
    assert chained.mode == "boolean"
    assert chained.query_type == "boolean"
    assert [(c.operator, c.tokens, c.is_negated) for c in chained.clauses] == [
        (None, ("machine", "learning"), False),
        ("AND", ("models",), False),
        ("AND NOT", ("data", "science"), True),
    ]
    assert chained.positive_clauses == (("machine", "learning"), ("models",))
    assert chained.excluded_clauses == (("data", "science"),)

    # Term mode (including case-insensitive alias)
    term_parsed = parse_query("Learning", mode="Term Search")
    assert term_parsed.mode == "term"
    assert term_parsed.query_type == "phrase"
    assert term_parsed.positive_clauses == (("learning",),)

    with pytest.raises(ValueError, match="exactly one token"):
        parse_query("machine learning", mode="term")
    with pytest.raises(ValueError, match="Boolean"):
        parse_query("AND", mode="term")

    # Phrase mode: treats uppercase AND literally
    literal_phrase = parse_query("machine AND learning", mode="Exact Phrase")
    assert literal_phrase.mode == "phrase"
    assert literal_phrase.query_type == "phrase"
    assert literal_phrase.positive_clauses == (("machine", "and", "learning"),)
    assert literal_phrase.excluded_clauses == ()

    # Malformed Boolean & invalid mode / empty query
    for bad_query in ("", "   ", "AND data", "machine AND", "machine AND NOT", "machine AND  AND data"):
        with pytest.raises(ValueError):
            parse_query(bad_query, mode="auto")

    with pytest.raises(ValueError, match="Boolean operator"):
        parse_query("machine learning", mode="boolean")

    with pytest.raises(ValueError, match="Unsupported query mode"):
        parse_query("machine", mode="regex")


def test_phrase_matching_positions_ranges_single_term_and_lowercase_and(
    technology_index: PositionalIndex,
) -> None:
    """Verify exact contiguous phrase matching, 1-based positions, ranges, and literal 'and'."""
    ml_result = execute_query(technology_index, "machine learning")
    assert ml_result.matched_doc_ids == ("d1", "d2", "d5", "d7", "d10")
    assert [
        (m.doc_id, m.starting_positions, m.formatted_ranges)
        for m in ml_result.phrase_matches
    ] == [
        ("d1", (1,), "1-2"),
        ("d2", (1,), "1-2"),
        ("d5", (1, 9), "1-2, 9-10"),
        ("d7", (4,), "4-5"),
        ("d10", (1, 5), "1-2, 5-6"),
    ]

    ds_result = execute_query(technology_index, "data science")
    assert ds_result.matched_doc_ids == ("d3", "d5", "d8")
    assert [
        (m.doc_id, m.starting_positions, m.formatted_ranges)
        for m in ds_result.phrase_matches
    ] == [
        ("d3", (1,), "1-2"),
        ("d5", (6,), "6-7"),
        ("d8", (1,), "1-2"),
    ]

    single_result = execute_query(technology_index, "machine", mode="term")
    assert single_result.matched_doc_ids == ("d1", "d2", "d5", "d7", "d10")
    assert [
        (m.doc_id, m.starting_positions, m.formatted_ranges)
        for m in single_result.phrase_matches
    ] == [
        ("d1", (1,), "1-1"),
        ("d2", (1,), "1-1"),
        ("d5", (1, 3, 9), "1-1, 3-3, 9-9"),
        ("d7", (4,), "4-4"),
        ("d10", (1, 5), "1-1, 5-5"),
    ]

    lower_and = execute_query(technology_index, "and")
    assert lower_and.matched_doc_ids == ("d5",)
    assert [(m.doc_id, m.starting_positions, m.formatted_ranges) for m in lower_and.phrase_matches] == [
        ("d5", (8,), "8-8"),
    ]

    lower_phrase = execute_query(technology_index, "data science and machine learning")
    assert lower_phrase.matched_doc_ids == ("d5",)
    assert [
        (m.doc_id, m.starting_positions, m.formatted_ranges)
        for m in lower_phrase.phrase_matches
    ] == [("d5", (6,), "6-10")]


def test_technology_four_assigned_queries_and_chained_boolean_match_pdf_reference(
    technology_index: PositionalIndex,
) -> None:
    """Validate all 4 assigned Technology queries and chained Boolean against PDF pp. 14-15."""
    # 1. Phrase query: "machine learning"
    q1 = execute_query(technology_index, "machine learning")
    assert q1.query_length == pytest.approx(0.425721, abs=1e-6)
    assert [(w.term, w.weighted_tf, w.tf_idf, w.normalized_weight) for w in q1.term_weights] == [
        ("machine", pytest.approx(1.0, abs=1e-4), pytest.approx(0.3010, abs=1e-4), pytest.approx(0.7071, abs=1e-4)),
        ("learning", pytest.approx(1.0, abs=1e-4), pytest.approx(0.3010, abs=1e-4), pytest.approx(0.7071, abs=1e-4)),
    ]
    assert q1.ranking_expression == "d1 > d5 > d10 > d2 > d7"
    assert [(r.doc_id, round(r.similarity, 4), r.rank) for r in q1.ranked_documents] == [
        ("d1", 1.0000, 1),
        ("d5", 0.3809, 2),
        ("d10", 0.3072, 3),
        ("d2", 0.2867, 4),
        ("d7", 0.2730, 5),
    ]

    # 2. Phrase query: "data science"
    q2 = execute_query(technology_index, "data science")
    assert q2.query_length == pytest.approx(0.545341, abs=1e-6)
    assert [(w.term, w.weighted_tf, w.tf_idf, w.normalized_weight) for w in q2.term_weights] == [
        ("data", pytest.approx(1.0, abs=1e-4), pytest.approx(0.1549, abs=1e-4), pytest.approx(0.2840, abs=1e-4)),
        ("science", pytest.approx(1.0, abs=1e-4), pytest.approx(0.5229, abs=1e-4), pytest.approx(0.9588, abs=1e-4)),
    ]
    assert q2.ranking_expression == "d5 > d3 > d8"
    assert [(r.doc_id, round(r.similarity, 4), r.rank) for r in q2.ranked_documents] == [
        ("d5", 0.3513, 1),
        ("d3", 0.3267, 2),
        ("d8", 0.2715, 3),
    ]

    # 3. Boolean AND: "machine learning AND data science"
    q3 = execute_query(technology_index, "machine learning AND data science")
    assert q3.matched_doc_ids == ("d5",)
    assert q3.query_length == pytest.approx(0.691834, abs=1e-6)
    assert [(w.term, w.weighted_tf, w.tf_idf, w.normalized_weight) for w in q3.term_weights] == [
        ("machine", pytest.approx(1.0, abs=1e-4), pytest.approx(0.3010, abs=1e-4), pytest.approx(0.4351, abs=1e-4)),
        ("learning", pytest.approx(1.0, abs=1e-4), pytest.approx(0.3010, abs=1e-4), pytest.approx(0.4351, abs=1e-4)),
        ("data", pytest.approx(1.0, abs=1e-4), pytest.approx(0.1549, abs=1e-4), pytest.approx(0.2239, abs=1e-4)),
        ("science", pytest.approx(1.0, abs=1e-4), pytest.approx(0.5229, abs=1e-4), pytest.approx(0.7558, abs=1e-4)),
    ]
    assert q3.ranking_expression == "d5"
    assert [(r.doc_id, round(r.similarity, 4), r.rank) for r in q3.ranked_documents] == [
        ("d5", 0.5113, 1),
    ]

    # 4. Boolean AND NOT: "machine learning AND NOT data science"
    q4 = execute_query(technology_index, "machine learning AND NOT data science")
    assert q4.matched_doc_ids == ("d1", "d2", "d7", "d10")
    assert q4.query_length == pytest.approx(0.425721, abs=1e-6)
    assert [w.term for w in q4.term_weights] == ["machine", "learning"]
    assert q4.ranking_expression == "d1 > d10 > d2 > d7"
    assert [(r.doc_id, round(r.similarity, 4), r.rank) for r in q4.ranked_documents] == [
        ("d1", 1.0000, 1),
        ("d10", 0.3072, 2),
        ("d2", 0.2867, 3),
        ("d7", 0.2730, 4),
    ]

    # 5. Chained Boolean: "machine learning AND models AND NOT data science"
    q5 = execute_query(technology_index, "machine learning AND models AND NOT data science")
    assert q5.matched_doc_ids == ("d7", "d10")
    assert [w.term for w in q5.term_weights] == ["machine", "learning", "models"]
    assert len(q5.ranked_documents) == 2
    assert {r.doc_id for r in q5.ranked_documents} == {"d7", "d10"}


def test_retrieval_edge_cases_repeated_terms_oov_zero_match_universal_idf_and_tie_breaking() -> None:
    """Verify repeated Query Raw TF, OOV Terms, 0-match queries, DF=N safety, and natural ties."""
    index = PositionalIndex(
        documents=(
            DocumentRecord(doc_id="doc_2", text="common rare", word_count=2),
            DocumentRecord(doc_id="doc_10", text="common rare", word_count=2),
            DocumentRecord(doc_id="doc_11", text="common unique common", word_count=3),
        ),
        postings={
            "common": {"doc_2": (1,), "doc_10": (1,), "doc_11": (1, 3)},
            "rare": {"doc_2": (2,), "doc_10": (2,)},
            "unique": {"doc_11": (2,)},
        },
    )

    # Repeated positive-clause Terms: Query Raw TF > 1 (across phrase + AND)
    repeated = execute_query(index, "common rare AND common")
    assert repeated.matched_doc_ids == ("doc_2", "doc_10")
    weights_by_term = {w.term: w for w in repeated.term_weights}
    assert weights_by_term["common"].raw_tf == 2
    assert weights_by_term["common"].weighted_tf == pytest.approx(1.30103, abs=1e-5)
    assert weights_by_term["common"].idf == 0.0  # DF = N = 3
    assert weights_by_term["rare"].raw_tf == 1
    assert weights_by_term["rare"].weighted_tf == pytest.approx(1.0, abs=1e-12)
    # Natural-ID tie-breaking: doc_2 and doc_10 have identical text & similarity
    assert [r.doc_id for r in repeated.ranked_documents] == ["doc_2", "doc_10"]
    assert repeated.ranked_documents[0].similarity == pytest.approx(
        repeated.ranked_documents[1].similarity, abs=1e-12
    )

    # All-zero IDF (DF = N) query: no ZeroDivisionError, Similarity = 0.0, natural tie-break
    universal = execute_query(index, "common")
    assert universal.matched_doc_ids == ("doc_2", "doc_10", "doc_11")
    assert universal.query_length == 0.0
    assert universal.term_weights[0].normalized_weight == 0.0
    assert [(r.doc_id, r.similarity) for r in universal.ranked_documents] == [
        ("doc_2", 0.0),
        ("doc_10", 0.0),
        ("doc_11", 0.0),
    ]

    # OOV term in positive clause -> 0 matches, 0.0 query length, safe weights
    oov_pos = execute_query(index, "nonexistent")
    assert oov_pos.matched_doc_ids == ()
    assert oov_pos.query_length == 0.0
    assert oov_pos.term_weights[0].idf == 0.0
    assert oov_pos.term_weights[0].normalized_weight == 0.0
    assert oov_pos.ranked_documents == ()

    # OOV term in excluded clause -> ignored in Query Vector, keeps positive matches
    oov_excl = execute_query(index, "rare AND NOT nonexistent")
    assert oov_excl.matched_doc_ids == ("doc_2", "doc_10")
    assert [w.term for w in oov_excl.term_weights] == ["rare"]
    assert oov_excl.query_length > 0.0

    # Zero-match phrase (in-vocabulary terms, wrong order / non-contiguous)
    zero_match = execute_query(index, "rare common")
    assert zero_match.matched_doc_ids == ()
    assert zero_match.ranked_documents == ()
    assert zero_match.ranking_expression == ""
    assert len(zero_match.term_weights) == 2


def test_multi_domain_dataset_flexibility_end_to_end(
    spark_session: SparkSession,
    tmp_path: Path,
) -> None:
    """Verify full Spark indexing -> vector space -> phrase/Boolean retrieval -> Markdown report on arbitrary corpus."""
    (tmp_path / "doc_10.txt").write_text(
        "solar energy powers smart microgrid and clean energy", encoding="utf-8"
    )
    (tmp_path / "doc_2.txt").write_text("solar energy smart microgrid", encoding="utf-8")
    (tmp_path / "alpha.txt").write_text("wind turbine clean energy", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored non-txt", encoding="utf-8")

    index = build_positional_index(tmp_path, spark=spark_session)
    model = build_vector_space_model(index)

    phrase_res = execute_query(index, "solar energy", model=model)
    assert phrase_res.matched_doc_ids == ("doc_2", "doc_10")
    assert [(m.doc_id, m.formatted_ranges) for m in phrase_res.phrase_matches] == [
        ("doc_2", "1-2"),
        ("doc_10", "1-2"),
    ]
    assert phrase_res.ranking_expression == "doc_2 > doc_10"

    bool_res = execute_query(index, "solar energy AND NOT clean energy", model=model)
    assert bool_res.matched_doc_ids == ("doc_2",)
    assert bool_res.ranking_expression == "doc_2"

    lower_and_res = execute_query(index, "microgrid and clean energy", model=model)
    assert lower_and_res.matched_doc_ids == ("doc_10",)

    report = generate_markdown_report(
        index,
        model,
        (phrase_res, bool_res, lower_and_res),
        title="Renewable Energy IR Report",
    )
    assert "# Renewable Energy IR Report" in report
    assert "IDF = log10(3 / DF)" in report
    assert "empty.txt" in report
    assert "doc_2 > doc_10" in report
