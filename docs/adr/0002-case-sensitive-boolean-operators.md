# 0002 — Case-Sensitive Uppercase Boolean Operators

Status: accepted

We treat only uppercase `AND` and `AND NOT` as Boolean operators in the query parser, while lowercase `and` / `not` are treated as ordinary searchable tokens.

## Context

The assigned Technology dataset contains the literal token `and` in `d5.txt` (`machine learning machine models support data science and machine learning`, position 8). Making Boolean operators case-insensitive would break literal searches for `and` or exact phrases containing `and` unless users wrapped every phrase in quotes.

## Consequences

- Queries like `machine learning AND data science` and `machine learning AND NOT data science` parse as Boolean queries.
- Queries like `and` or `data science and machine learning` match `d5` directly without quoting.
- Future contributors must not "normalize" the raw query string to lowercase before splitting on `AND` / `AND NOT`; lowercasing happens per clause *after* operator parsing.
