from __future__ import annotations

from pathlib import Path

import pytest

from ir_system.models import DocumentRecord, PositionalIndex
from ir_system.spark_indexer import export_positional_index, load_positional_index


TECH_DOC_IDS = ("d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10")

EXPECTED_TECH_RAW_TF: dict[str, tuple[int, ...]] = {
    "analysis": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    "analytics": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    "and": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    "applications": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    "artificial": (0, 0, 0, 1, 0, 0, 1, 0, 0, 0),
    "cloud": (0, 0, 0, 0, 0, 1, 0, 0, 2, 0),
    "computing": (0, 0, 0, 0, 0, 1, 0, 1, 1, 0),
    "data": (0, 1, 1, 0, 1, 1, 0, 2, 1, 1),
    "explores": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    "improve": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    "improves": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    "intelligence": (0, 0, 0, 1, 0, 0, 1, 0, 0, 0),
    "learning": (1, 1, 0, 0, 2, 0, 1, 0, 0, 2),
    "machine": (1, 1, 0, 0, 3, 0, 1, 0, 0, 2),
    "models": (0, 0, 0, 0, 1, 0, 1, 0, 0, 1),
    "powers": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    "processing": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    "research": (0, 0, 0, 0, 0, 0, 0, 1, 0, 1),
    "science": (0, 0, 1, 0, 1, 0, 0, 1, 0, 0),
    "smart": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    "storage": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
    "support": (0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
    "supports": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
    "systems": (0, 0, 0, 1, 0, 0, 0, 1, 1, 1),
    "through": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    "tools": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    "uses": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
}

EXPECTED_TECH_WEIGHTED_TF: dict[str, tuple[float, ...]] = {
    "analysis": (0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0),
    "analytics": (0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0),
    "and": (0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0),
    "applications": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000),
    "artificial": (0, 0, 0, 1.0000, 0, 0, 1.0000, 0, 0, 0),
    "cloud": (0, 0, 0, 0, 0, 1.0000, 0, 0, 1.3010, 0),
    "computing": (0, 0, 0, 0, 0, 1.0000, 0, 1.0000, 1.0000, 0),
    "data": (0, 1.0000, 1.0000, 0, 1.0000, 1.0000, 0, 1.3010, 1.0000, 1.0000),
    "explores": (0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0),
    "improve": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000),
    "improves": (0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0),
    "intelligence": (0, 0, 0, 1.0000, 0, 0, 1.0000, 0, 0, 0),
    "learning": (1.0000, 1.0000, 0, 0, 1.3010, 0, 1.0000, 0, 0, 1.3010),
    "machine": (1.0000, 1.0000, 0, 0, 1.4771, 0, 1.0000, 0, 0, 1.3010),
    "models": (0, 0, 0, 0, 1.0000, 0, 1.0000, 0, 0, 1.0000),
    "powers": (0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0),
    "processing": (0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0),
    "research": (0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 1.0000),
    "science": (0, 0, 1.0000, 0, 1.0000, 0, 0, 1.0000, 0, 0),
    "smart": (0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0),
    "storage": (0, 0, 0, 0, 0, 1.0000, 0, 0, 1.0000, 0),
    "support": (0, 0, 1.0000, 0, 1.0000, 0, 0, 0, 0, 0),
    "supports": (0, 0, 0, 0, 0, 1.0000, 0, 0, 1.0000, 0),
    "systems": (0, 0, 0, 1.0000, 0, 0, 0, 1.0000, 1.0000, 1.0000),
    "through": (0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0),
    "tools": (0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0),
    "uses": (0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0),
}

EXPECTED_TECH_DF_IDF: dict[str, tuple[int, float]] = {
    "analysis": (1, 1.0000),
    "analytics": (1, 1.0000),
    "and": (1, 1.0000),
    "applications": (1, 1.0000),
    "artificial": (2, 0.6990),
    "cloud": (2, 0.6990),
    "computing": (3, 0.5229),
    "data": (7, 0.1549),
    "explores": (1, 1.0000),
    "improve": (1, 1.0000),
    "improves": (1, 1.0000),
    "intelligence": (2, 0.6990),
    "learning": (5, 0.3010),
    "machine": (5, 0.3010),
    "models": (3, 0.5229),
    "powers": (1, 1.0000),
    "processing": (1, 1.0000),
    "research": (2, 0.6990),
    "science": (3, 0.5229),
    "smart": (1, 1.0000),
    "storage": (2, 0.6990),
    "support": (2, 0.6990),
    "supports": (2, 0.6990),
    "systems": (4, 0.3979),
    "through": (1, 1.0000),
    "tools": (1, 1.0000),
    "uses": (1, 1.0000),
}

EXPECTED_TECH_TF_IDF: dict[str, tuple[float, ...]] = {
    "analysis": (0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0),
    "analytics": (0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0),
    "and": (0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0),
    "applications": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000),
    "artificial": (0, 0, 0, 0.6990, 0, 0, 0.6990, 0, 0, 0),
    "cloud": (0, 0, 0, 0, 0, 0.6990, 0, 0, 0.9094, 0),
    "computing": (0, 0, 0, 0, 0, 0.5229, 0, 0.5229, 0.5229, 0),
    "data": (0, 0.1549, 0.1549, 0, 0.1549, 0.1549, 0, 0.2015, 0.1549, 0.1549),
    "explores": (0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0),
    "improve": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000),
    "improves": (0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0),
    "intelligence": (0, 0, 0, 0.6990, 0, 0, 0.6990, 0, 0, 0),
    "learning": (0.3010, 0.3010, 0, 0, 0.3916, 0, 0.3010, 0, 0, 0.3916),
    "machine": (0.3010, 0.3010, 0, 0, 0.4447, 0, 0.3010, 0, 0, 0.3916),
    "models": (0, 0, 0, 0, 0.5229, 0, 0.5229, 0, 0, 0.5229),
    "powers": (0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0),
    "processing": (0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0),
    "research": (0, 0, 0, 0, 0, 0, 0, 0.6990, 0, 0.6990),
    "science": (0, 0, 0.5229, 0, 0.5229, 0, 0, 0.5229, 0, 0),
    "smart": (0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0),
    "storage": (0, 0, 0, 0, 0, 0.6990, 0, 0, 0.6990, 0),
    "support": (0, 0, 0.6990, 0, 0.6990, 0, 0, 0, 0, 0),
    "supports": (0, 0, 0, 0, 0, 0.6990, 0, 0, 0.6990, 0),
    "systems": (0, 0, 0, 0.3979, 0, 0, 0, 0.3979, 0.3979, 0.3979),
    "through": (0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0),
    "tools": (0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0),
    "uses": (0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0),
}

EXPECTED_TECH_DOC_LENGTHS: dict[str, float] = {
    "d1": 0.4257,
    "d2": 1.4850,
    "d3": 1.6691,
    "d4": 1.7707,
    "d5": 1.5526,
    "d6": 1.3278,
    "d7": 1.5594,
    "d8": 2.0577,
    "d9": 1.5033,
    "d10": 1.8031,
}

EXPECTED_TECH_NORMALIZED_TF_IDF: dict[str, tuple[float, ...]] = {
    "analysis": (0, 0, 0.5991, 0, 0, 0, 0, 0, 0, 0),
    "analytics": (0, 0, 0, 0, 0, 0, 0, 0.4860, 0, 0),
    "and": (0, 0, 0, 0, 0.6441, 0, 0, 0, 0, 0),
    "applications": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5546),
    "artificial": (0, 0, 0, 0.3947, 0, 0, 0.4482, 0, 0, 0),
    "cloud": (0, 0, 0, 0, 0, 0.5264, 0, 0, 0.6049, 0),
    "computing": (0, 0, 0, 0, 0, 0.3938, 0, 0.2541, 0.3478, 0),
    "data": (0, 0.1043, 0.0928, 0, 0.0998, 0.1167, 0, 0.0979, 0.1030, 0.0859),
    "explores": (0, 0, 0, 0, 0, 0, 0, 0.4860, 0, 0),
    "improve": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5546),
    "improves": (0, 0.6734, 0, 0, 0, 0, 0, 0, 0, 0),
    "intelligence": (0, 0, 0, 0.3947, 0, 0, 0.4482, 0, 0, 0),
    "learning": (0.7071, 0.2027, 0, 0, 0.2523, 0, 0.1930, 0, 0, 0.2172),
    "machine": (0.7071, 0.2027, 0, 0, 0.2864, 0, 0.1930, 0, 0, 0.2172),
    "models": (0, 0, 0, 0, 0.3368, 0, 0.3353, 0, 0, 0.2900),
    "powers": (0, 0, 0, 0.5647, 0, 0, 0, 0, 0, 0),
    "processing": (0, 0.6734, 0, 0, 0, 0, 0, 0, 0, 0),
    "research": (0, 0, 0, 0, 0, 0, 0, 0.3397, 0, 0.3877),
    "science": (0, 0, 0.3133, 0, 0.3368, 0, 0, 0.2541, 0, 0),
    "smart": (0, 0, 0, 0.5647, 0, 0, 0, 0, 0, 0),
    "storage": (0, 0, 0, 0, 0, 0.5264, 0, 0, 0.4650, 0),
    "support": (0, 0, 0.4188, 0, 0.4502, 0, 0, 0, 0, 0),
    "supports": (0, 0, 0, 0, 0, 0.5264, 0, 0, 0.4650, 0),
    "systems": (0, 0, 0, 0.2247, 0, 0, 0, 0.1934, 0.2647, 0.2207),
    "through": (0, 0, 0, 0, 0, 0, 0, 0.4860, 0, 0),
    "tools": (0, 0, 0.5991, 0, 0, 0, 0, 0, 0, 0),
    "uses": (0, 0, 0, 0, 0, 0, 0.6413, 0, 0, 0),
}


def test_raw_tf_and_weighted_tf_from_positional_index_dynamic_corpus() -> None:
    """Verify Raw TF and Weighted TF derive counts and log10 weights for any corpus size N."""
    from ir_system.retrieval import build_vector_space_model

    index = PositionalIndex(
        documents=(
            DocumentRecord(doc_id="alpha", text="spark spark python", word_count=3),
            DocumentRecord(doc_id="beta", text="python", word_count=1),
            DocumentRecord(doc_id="gamma", text="spark spark spark", word_count=3),
        ),
        postings={
            "python": {"alpha": (3,), "beta": (1,)},
            "spark": {"alpha": (1, 2), "gamma": (1, 2, 3)},
        },
    )

    model = build_vector_space_model(index)

    assert model.doc_count == 3
    assert model.doc_ids == ("alpha", "beta", "gamma")
    assert model.terms == ("python", "spark")

    assert model.raw_tf["python"] == {"alpha": 1, "beta": 1, "gamma": 0}
    assert model.raw_tf["spark"] == {"alpha": 2, "beta": 0, "gamma": 3}

    assert model.weighted_tf["python"]["alpha"] == pytest.approx(1.0, abs=1e-12)
    assert model.weighted_tf["python"]["beta"] == pytest.approx(1.0, abs=1e-12)
    assert model.weighted_tf["python"]["gamma"] == 0.0
    assert model.weighted_tf["spark"]["alpha"] == pytest.approx(1.3010299956639813, abs=1e-12)
    assert model.weighted_tf["spark"]["beta"] == 0.0
    assert model.weighted_tf["spark"]["gamma"] == pytest.approx(1.4771212547196624, abs=1e-12)


def test_df_and_idf_use_dynamic_corpus_size_and_handle_universal_terms() -> None:
    """Verify DF and base-10 IDF log10(N / DF) work dynamically for N != 10 and DF = N."""
    from ir_system.vector_space import build_vector_space_model

    index = PositionalIndex(
        documents=(
            DocumentRecord(doc_id="doc_1", text="common rare", word_count=2),
            DocumentRecord(doc_id="doc_2", text="common shared", word_count=2),
            DocumentRecord(doc_id="doc_3", text="common shared", word_count=2),
            DocumentRecord(doc_id="doc_4", text="common", word_count=1),
        ),
        postings={
            "common": {"doc_1": (1,), "doc_2": (1,), "doc_3": (1,), "doc_4": (1,)},
            "rare": {"doc_1": (2,)},
            "shared": {"doc_2": (2,), "doc_3": (2,)},
        },
    )

    model = build_vector_space_model(index)

    assert model.doc_count == 4
    assert model.df == {"common": 4, "rare": 1, "shared": 2}
    assert model.idf["common"] == 0.0
    assert model.idf["rare"] == pytest.approx(0.6020599913279624, abs=1e-12)
    assert model.idf["shared"] == pytest.approx(0.3010299956639812, abs=1e-12)


def test_tf_idf_doc_lengths_and_normalized_tf_idf_handle_zero_norm_safely() -> None:
    """Verify TF-IDF, L2 Document Vector Lengths, and zero-norm safe normalization."""
    from ir_system.vector_space import build_vector_space_model

    index = PositionalIndex(
        documents=(
            DocumentRecord(doc_id="doc_1", text="common unique", word_count=2),
            DocumentRecord(doc_id="doc_2", text="common", word_count=1),
        ),
        postings={
            "common": {"doc_1": (1,), "doc_2": (1,)},
            "unique": {"doc_1": (2,)},
        },
    )

    model = build_vector_space_model(index)

    assert model.tf_idf["common"] == {"doc_1": 0.0, "doc_2": 0.0}
    assert model.tf_idf["unique"]["doc_1"] == pytest.approx(0.3010299956639812, abs=1e-12)
    assert model.tf_idf["unique"]["doc_2"] == 0.0

    assert model.doc_lengths["doc_1"] == pytest.approx(0.3010299956639812, abs=1e-12)
    assert model.doc_lengths["doc_2"] == 0.0

    assert model.normalized_tf_idf["common"] == {"doc_1": 0.0, "doc_2": 0.0}
    assert model.normalized_tf_idf["unique"]["doc_1"] == pytest.approx(1.0, abs=1e-12)
    assert model.normalized_tf_idf["unique"]["doc_2"] == 0.0


def test_build_vector_space_model_rejects_empty_positional_index() -> None:
    """Verify empty PositionalIndex (N = 0) raises ValueError."""
    from ir_system.vector_space import build_vector_space_model

    empty_index = PositionalIndex(documents=(), postings={})
    with pytest.raises(ValueError, match="empty PositionalIndex"):
        build_vector_space_model(empty_index)


def test_technology_corpus_all_part_b_matrices_match_pdf_reference(
    technology_index: PositionalIndex,
) -> None:
    """Validate all 27x10 Part B matrices and 10 Document Vector Lengths against PDF pp. 11-13."""
    from ir_system.vector_space import VectorSpaceModel

    model = VectorSpaceModel.from_positional_index(technology_index)

    assert model.doc_count == 10
    assert model.doc_ids == TECH_DOC_IDS
    assert model.terms == tuple(EXPECTED_TECH_RAW_TF.keys())

    for term, expected_row in EXPECTED_TECH_RAW_TF.items():
        actual_row = tuple(model.raw_tf[term][doc_id] for doc_id in TECH_DOC_IDS)
        assert actual_row == expected_row, f"Raw TF mismatch for {term}"

    for term, expected_row in EXPECTED_TECH_WEIGHTED_TF.items():
        for idx, doc_id in enumerate(TECH_DOC_IDS):
            assert model.weighted_tf[term][doc_id] == pytest.approx(expected_row[idx], abs=1e-4), (
                f"Weighted TF mismatch for {term}, {doc_id}"
            )

    for term, (expected_df, expected_idf) in EXPECTED_TECH_DF_IDF.items():
        assert model.df[term] == expected_df, f"DF mismatch for {term}"
        assert model.idf[term] == pytest.approx(expected_idf, abs=1e-4), f"IDF mismatch for {term}"

    for term, expected_row in EXPECTED_TECH_TF_IDF.items():
        for idx, doc_id in enumerate(TECH_DOC_IDS):
            assert model.tf_idf[term][doc_id] == pytest.approx(expected_row[idx], abs=1e-4), (
                f"TF-IDF mismatch for {term}, {doc_id}"
            )

    for doc_id, expected_length in EXPECTED_TECH_DOC_LENGTHS.items():
        assert model.doc_lengths[doc_id] == pytest.approx(expected_length, abs=1e-4), (
            f"Document Vector Length mismatch for {doc_id}"
        )

    for term, expected_row in EXPECTED_TECH_NORMALIZED_TF_IDF.items():
        for idx, doc_id in enumerate(TECH_DOC_IDS):
            assert model.normalized_tf_idf[term][doc_id] == pytest.approx(
                expected_row[idx], abs=1e-4
            ), f"Normalized TF-IDF mismatch for {term}, {doc_id}"


def test_vector_space_model_matches_across_exported_and_reloaded_index(
    technology_index: PositionalIndex,
    tmp_path: Path,
) -> None:
    """Verify building Part B from a reloaded Part A artifact yields identical matrices."""
    from ir_system.vector_space import build_vector_space_model

    exported_path = export_positional_index(technology_index, tmp_path / "part_a")
    reloaded_index = load_positional_index(exported_path)

    from_memory = build_vector_space_model(technology_index)
    from_reloaded = build_vector_space_model(reloaded_index)

    assert from_reloaded == from_memory

