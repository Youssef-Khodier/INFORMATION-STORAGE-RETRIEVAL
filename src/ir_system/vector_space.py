from __future__ import annotations

from dataclasses import dataclass
import math

from ir_system.models import PositionalIndex, natural_sort_key


@dataclass(frozen=True)
class VectorSpaceModel:
    """Part B TF-IDF vector-space matrices derived from a Part A PositionalIndex."""

    doc_count: int
    doc_ids: tuple[str, ...]
    terms: tuple[str, ...]
    raw_tf: dict[str, dict[str, int]]
    weighted_tf: dict[str, dict[str, float]]
    df: dict[str, int]
    idf: dict[str, float]
    tf_idf: dict[str, dict[str, float]]
    doc_lengths: dict[str, float]
    normalized_tf_idf: dict[str, dict[str, float]]

    @classmethod
    def from_positional_index(cls, index: PositionalIndex) -> VectorSpaceModel:
        """Build a VectorSpaceModel directly from a Part A PositionalIndex."""
        return build_vector_space_model(index)


def build_vector_space_model(index: PositionalIndex) -> VectorSpaceModel:
    """Compute Part B vector-space weighting matrices strictly from a PositionalIndex."""
    doc_ids = tuple(sorted((doc.doc_id for doc in index.documents), key=natural_sort_key))
    doc_count = len(doc_ids)
    if doc_count == 0:
        raise ValueError("Cannot build VectorSpaceModel from an empty PositionalIndex (N = 0).")

    terms = tuple(sorted(index.postings.keys()))

    raw_tf: dict[str, dict[str, int]] = {}
    weighted_tf: dict[str, dict[str, float]] = {}
    df: dict[str, int] = {}
    idf: dict[str, float] = {}
    tf_idf: dict[str, dict[str, float]] = {}
    doc_squared_sums: dict[str, float] = {doc_id: 0.0 for doc_id in doc_ids}

    for term in terms:
        doc_postings = index.postings[term]
        term_raw_tf: dict[str, int] = {}
        term_weighted_tf: dict[str, float] = {}
        term_df = 0

        for doc_id in doc_ids:
            tf = len(doc_postings.get(doc_id, ()))
            if tf > 0:
                term_df += 1
            term_raw_tf[doc_id] = tf
            term_weighted_tf[doc_id] = 1.0 + math.log10(tf) if tf > 0 else 0.0

        term_idf = math.log10(doc_count / term_df) if term_df > 0 else 0.0
        term_tf_idf: dict[str, float] = {}
        for doc_id in doc_ids:
            weight = term_weighted_tf[doc_id] * term_idf
            term_tf_idf[doc_id] = weight
            doc_squared_sums[doc_id] += weight * weight

        raw_tf[term] = term_raw_tf
        weighted_tf[term] = term_weighted_tf
        df[term] = term_df
        idf[term] = term_idf
        tf_idf[term] = term_tf_idf

    doc_lengths = {
        doc_id: math.sqrt(doc_squared_sums[doc_id])
        for doc_id in doc_ids
    }
    normalized_tf_idf = {
        term: {
            doc_id: (
                tf_idf[term][doc_id] / doc_lengths[doc_id]
                if doc_lengths[doc_id] > 0.0
                else 0.0
            )
            for doc_id in doc_ids
        }
        for term in terms
    }

    return VectorSpaceModel(
        doc_count=doc_count,
        doc_ids=doc_ids,
        terms=terms,
        raw_tf=raw_tf,
        weighted_tf=weighted_tf,
        df=df,
        idf=idf,
        tf_idf=tf_idf,
        doc_lengths=doc_lengths,
        normalized_tf_idf=normalized_tf_idf,
    )
