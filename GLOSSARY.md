# Information Storage & Retrieval (Technology IR System)

A domain model for positional indexing, TF-IDF vector-space weighting, and exact phrase / Boolean retrieval over a corpus of plain-text documents.

## Corpus & Indexing

**Corpus**:
The active collection of `N` imported text documents being indexed and searched.
_Avoid_: Dataset folder, file list

**Document**:
A single plain-text file in the Corpus, identified by its filename stem (such as `d1` through `d10`).
_Avoid_: File, record, text

**Token**:
A single lowercased, whitespace-delimited unit extracted in order from a Document without stop-word removal or stemming.
_Avoid_: Word, lexeme

**Term**:
A distinct vocabulary entry in the Corpus formed by deduplicating Tokens across all Documents.
_Avoid_: Keyword, word

**Position**:
The 1-based ordinal index of a Token within a Document, counted from left to right.
_Avoid_: Offset, 0-based index

**Positional Index**:
The mapping from every unique Term (sorted alphabetically) to each Document containing it (sorted naturally) and the ordered list of 1-based Positions where that Term occurs.
_Avoid_: Inverted file, posting map

## Weighting & Vector Space

**Raw TF**:
The integer count of occurrences of a Term in a Document (or in a Query's positive clauses).
_Avoid_: Plain "TF" when ambiguous with Weighted TF

**Weighted TF**:
The log-scaled term frequency equal to `1 + log10(Raw TF)` when `Raw TF > 0`, and `0` otherwise.
_Avoid_: Log TF, sublinear TF

**DF (Document Frequency)**:
The count of distinct Documents in the Corpus in which a Term appears at least once.

**IDF (Inverse Document Frequency)**:
The base-10 collection rarity weight of a Term, defined as `log10(N / DF)`.

**TF-IDF Weight**:
The product `Weighted TF × IDF` representing a Term's unnormalized importance in a Document or Query.

**Document Vector Length**:
The Euclidean (L2) norm of a Document's TF-IDF Weight vector across all vocabulary Terms.
_Avoid_: Magnitude, norm alone

**Normalized TF-IDF**:
A Document's TF-IDF Weight divided by its Document Vector Length, yielding a unit-length Document vector.

## Querying & Ranking

**Phrase Query**:
One or more ordered Terms that match a Document only when they appear contiguously in exact sequence; a single-term search is a 1-token Phrase Query.
_Avoid_: Keyword query, substring match

**Boolean Query**:
A compound query combining Phrase Queries with uppercase `AND` (intersection of matching Document sets) or `AND NOT` (set difference keeping Documents that match the positive side while excluding Documents that match the negated side).

**Positive Clause**:
Any Phrase Query operand in a Query that is not preceded by `AND NOT`; only Positive Clauses contribute Terms to the Query Vector.

**Excluded Clause**:
A Phrase Query operand preceded by `AND NOT` that filters out matching Documents without contributing Terms to the Query Vector.

**Query Vector**:
The L2-normalized TF-IDF vector built from all Tokens in a Query's Positive Clauses using Weighted TF (`1 + log10(Query Raw TF)`) and collection IDF.

**Similarity Score**:
The cosine similarity between a Query Vector and a matched Document's Normalized TF-IDF vector, computed as their dot product.
_Avoid_: Relevance score, distance
