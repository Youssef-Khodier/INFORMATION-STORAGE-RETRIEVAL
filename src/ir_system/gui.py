from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import pandas as pd
from pyspark.sql import SparkSession
import streamlit as st

from ir_system.formatting import (
    clause_matches_table_data,
    df_idf_table_data,
    doc_lengths_table_data,
    documents_table_data,
    positional_index_table_data,
    query_heading_label,
    query_similarity_table_data,
    query_weights_table_data,
    term_doc_matrix_data,
)
from ir_system.models import PositionalIndex, natural_sort_key
from ir_system.report import generate_markdown_report
from ir_system.retrieval import DEFAULT_TECHNOLOGY_QUERIES, QueryResult, execute_query
from ir_system.spark_indexer import build_positional_index, create_spark_session
from ir_system.vector_space import VectorSpaceModel, build_vector_space_model


_QUERY_MODE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "Auto",
        "auto",
        "Auto detects uppercase AND / AND NOT as Boolean operators; otherwise evaluates as a Phrase Query.",
    ),
    (
        "Term Search",
        "term",
        "Requires a single whitespace-delimited Token without Boolean operators.",
    ),
    (
        "Exact Phrase",
        "phrase",
        "Evaluates the entire input as a contiguous Phrase Query (lowercase 'and' / 'not' are matched literally).",
    ),
    (
        "Boolean (AND / AND NOT)",
        "boolean",
        "Requires case-sensitive uppercase AND or AND NOT combining Phrase Query clauses.",
    ),
)

_MODE_LABEL_TO_CANONICAL: dict[str, str] = {
    label: canonical for label, canonical, _ in _QUERY_MODE_OPTIONS
}
_MODE_LABEL_TO_HINT: dict[str, str] = {
    label: hint for label, _, hint in _QUERY_MODE_OPTIONS
}

_SAMPLE_QUERY_MODES: dict[str, str] = {
    "machine learning": "Exact Phrase",
    "data science": "Exact Phrase",
    "machine learning AND data science": "Boolean (AND / AND NOT)",
    "machine learning AND NOT data science": "Boolean (AND / AND NOT)",
}


def default_corpus_dir() -> Path:
    """Resolve the default Technology Version Corpus directory."""
    cwd_candidate = Path("data") / "2. Technology"
    if cwd_candidate.is_dir():
        return cwd_candidate.resolve()
    return (Path(__file__).resolve().parent.parent.parent / "data" / "2. Technology").resolve()


@st.cache_resource(show_spinner="Starting Apache Spark session...")
def get_spark_session() -> SparkSession:
    """Return a process-wide cached SparkSession so Spark JVM never restarts on reruns."""
    return create_spark_session(app_name="TechnologyIRWebGUI", master="local[*]")


def compute_folder_fingerprint(corpus_dir: Path) -> str:
    """Compute a deterministic fingerprint for a Corpus directory and its .txt files."""
    resolved = corpus_dir.resolve()
    hasher = hashlib.sha256(str(resolved).encode("utf-8"))
    if resolved.is_dir():
        txt_files = sorted(
            (
                entry
                for entry in resolved.iterdir()
                if entry.is_file() and entry.suffix.lower() == ".txt"
            ),
            key=lambda path: natural_sort_key(path.name),
        )
        for file_path in txt_files:
            stat = file_path.stat()
            hasher.update(f"{file_path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    return hasher.hexdigest()


def compute_uploaded_fingerprint(uploaded_files: Sequence[Any]) -> str:
    """Compute a deterministic fingerprint over uploaded file names and contents."""
    hasher = hashlib.sha256()
    ordered = sorted(uploaded_files, key=lambda item: natural_sort_key(Path(item.name).name))
    for uploaded in ordered:
        safe_name = Path(uploaded.name).name
        payload = uploaded.getvalue()
        hasher.update(safe_name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(payload)
        hasher.update(b"\x00")
    return hasher.hexdigest()


def stage_uploaded_corpus(
    uploaded_files: Sequence[Any],
    *,
    previous_temp_dir: Path | None = None,
) -> tuple[Path, str, tuple[str, ...]]:
    """Write uploaded .txt files into a managed temporary directory and ignore non-.txt files."""
    if previous_temp_dir is not None and previous_temp_dir.exists():
        shutil.rmtree(previous_temp_dir, ignore_errors=True)

    fingerprint = compute_uploaded_fingerprint(uploaded_files)
    temp_dir = Path(tempfile.mkdtemp(prefix="ir_uploaded_corpus_"))
    warnings: list[str] = []
    seen_names: set[str] = set()
    staged_count = 0

    try:
        for uploaded in uploaded_files:
            safe_name = Path(uploaded.name).name
            if Path(safe_name).suffix.lower() != ".txt":
                warnings.append(f"Ignored non-.txt uploaded file: {safe_name}")
                continue
            if safe_name in seen_names:
                raise ValueError(f"Duplicate uploaded .txt filename: {safe_name}")
            seen_names.add(safe_name)
            (temp_dir / safe_name).write_bytes(uploaded.getvalue())
            staged_count += 1

        if staged_count == 0:
            raise ValueError("Uploaded selection contained no .txt documents.")
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return temp_dir, fingerprint, tuple(warnings)


def _table_to_dataframe(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """Convert formatting table (headers, rows) into a Pandas DataFrame."""
    return pd.DataFrame(list(rows), columns=list(headers))


def _load_corpus_into_state(
    corpus_dir: Path,
    source_label: str,
    *,
    fingerprint: str | None = None,
    extra_warnings: Sequence[str] = (),
) -> None:
    """Run PySpark Part A indexing and Part B model construction, caching results in session_state."""
    spark = get_spark_session()
    with st.spinner(f"Indexing Corpus via Apache Spark ({source_label})..."):
        index = build_positional_index(corpus_dir, spark=spark)
        model = build_vector_space_model(index)

    st.session_state["positional_index"] = index
    st.session_state["vector_space_model"] = model
    st.session_state["corpus_fingerprint"] = fingerprint or compute_folder_fingerprint(corpus_dir)
    st.session_state["corpus_source_label"] = source_label
    st.session_state["extra_warnings"] = tuple(extra_warnings)
    st.session_state["query_history"] = {}


def _ensure_initial_corpus() -> None:
    """Auto-load the default Technology Corpus once per session."""
    if "positional_index" not in st.session_state:
        tech_dir = default_corpus_dir()
        st.session_state.setdefault("corpus_folder_input", str(tech_dir))
        _load_corpus_into_state(tech_dir, "Default: data/2. Technology")


def _reset_to_technology_callback() -> None:
    """Prepare session state before widgets render to reset back to the Technology Corpus."""
    st.session_state["corpus_folder_input"] = str(default_corpus_dir())
    st.session_state["corpus_import_error"] = ""
    st.session_state["reset_requested"] = True


def _cleanup_managed_upload_dir() -> None:
    """Remove any previously staged uploaded Corpus temporary directory."""
    managed_dir = st.session_state.pop("managed_upload_dir", None)
    if isinstance(managed_dir, Path) and managed_dir.exists():
        shutil.rmtree(managed_dir, ignore_errors=True)


def _handle_folder_import(folder_input: str) -> None:
    """Validate and load a local Corpus folder when its fingerprint changed."""
    st.session_state["corpus_import_error"] = ""
    raw_folder = folder_input.strip()
    try:
        if not raw_folder:
            raise ValueError("Please enter a local Corpus folder path.")
        target_dir = Path(raw_folder)
        target_fp = compute_folder_fingerprint(target_dir)
        if not target_dir.is_dir() or target_fp != st.session_state.get("corpus_fingerprint"):
            _cleanup_managed_upload_dir()
            _load_corpus_into_state(
                target_dir,
                f"Folder: {target_dir.name or target_dir}",
                fingerprint=target_fp,
            )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        st.session_state["corpus_import_error"] = str(exc)


def _handle_uploaded_import(uploaded_files: Sequence[Any]) -> None:
    """Stage uploaded .txt files into a temp dir and index them when content changes."""
    upload_fp = compute_uploaded_fingerprint(uploaded_files)
    ignored_fps = {
        st.session_state.get("last_upload_fingerprint"),
        st.session_state.get("ignored_upload_fingerprint"),
    }
    if upload_fp in ignored_fps:
        return

    st.session_state["corpus_import_error"] = ""
    try:
        staged_dir, staged_fp, upload_warnings = stage_uploaded_corpus(
            uploaded_files,
            previous_temp_dir=st.session_state.get("managed_upload_dir"),
        )
        st.session_state["managed_upload_dir"] = staged_dir
        st.session_state["last_upload_fingerprint"] = staged_fp
        _load_corpus_into_state(
            staged_dir,
            f"Uploaded ({len(uploaded_files)} files)",
            fingerprint=staged_fp,
            extra_warnings=upload_warnings,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        st.session_state["corpus_import_error"] = str(exc)


def _render_sidebar() -> tuple[PositionalIndex, VectorSpaceModel]:
    """Render Corpus Import controls, summary metrics, and warnings in the sidebar."""
    with st.sidebar:
        st.header("Corpus Import")
        st.session_state.setdefault("corpus_folder_input", str(default_corpus_dir()))
        st.session_state.setdefault("corpus_import_error", "")

        folder_input = st.text_input("Local Corpus Folder Path", key="corpus_folder_input")
        col_load, col_reset = st.columns(2)
        load_clicked = col_load.button("Load Folder", key="load_folder_btn", width="stretch")
        col_reset.button(
            "Reset to Technology",
            key="reset_technology_btn",
            on_click=_reset_to_technology_callback,
            width="stretch",
        )

        uploaded_files = st.file_uploader(
            "Upload .txt Files or Folder",
            accept_multiple_files=True,
            type=None,
            key="corpus_uploader",
        )

        if st.session_state.pop("reset_requested", False):
            if uploaded_files:
                st.session_state["ignored_upload_fingerprint"] = compute_uploaded_fingerprint(
                    uploaded_files
                )
            _cleanup_managed_upload_dir()
            tech_dir = default_corpus_dir()
            if compute_folder_fingerprint(tech_dir) != st.session_state.get("corpus_fingerprint"):
                _load_corpus_into_state(tech_dir, "Default: data/2. Technology")
        elif load_clicked:
            _handle_folder_import(folder_input)
        elif uploaded_files:
            _handle_uploaded_import(uploaded_files)

        import_error = st.session_state.get("corpus_import_error", "")
        if import_error:
            st.error(import_error)

        index: PositionalIndex = st.session_state["positional_index"]
        model: VectorSpaceModel = st.session_state["vector_space_model"]

        st.divider()
        st.header("Corpus Summary")
        st.caption(st.session_state.get("corpus_source_label", ""))

        col1, col2 = st.columns(2)
        col1.metric("Documents (N)", str(index.doc_count))
        col2.metric("Vocabulary Terms", str(len(index.postings)))

        col3, col4 = st.columns(2)
        col3.metric("Total Tokens", str(index.total_tokens))
        col4.metric("Spark Indexing Time", f"{index.indexing_seconds:.2f}s")

        for warning in [*index.warnings, *st.session_state.get("extra_warnings", ())]:
            st.warning(warning)

    return index, model


def _apply_sample_query(sample_query: str, mode_label: str) -> None:
    """Populate the query input and mode selector from a quick-fill sample button."""
    st.session_state["query_text"] = sample_query
    st.session_state["query_mode"] = mode_label


def _render_query_result(result: QueryResult) -> None:
    """Render structured Phrase/Boolean matches, Query Vector weights, scores, and ranking."""
    st.markdown(f"#### {query_heading_label(result)}")

    if result.parsed_query.query_type == "phrase":
        headers, rows = clause_matches_table_data(result.clause_matches[0])
        if rows:
            st.dataframe(_table_to_dataframe(headers, rows), width="stretch", hide_index=True)
        else:
            st.info("No matching documents.")
    else:
        matched_str = ", ".join(result.matched_doc_ids) if result.matched_doc_ids else "None"
        st.markdown(f"**Matched documents:** `{matched_str}`")
        for clause_result in result.clause_matches:
            clause_phrase = " ".join(clause_result.clause.tokens)
            prefix = f"{clause_result.clause.operator} " if clause_result.clause.operator else ""
            st.markdown(f"**Clause {prefix}`{clause_phrase}`**")
            c_headers, c_rows = clause_matches_table_data(clause_result)
            if c_rows:
                st.dataframe(
                    _table_to_dataframe(c_headers, c_rows),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No matching documents for this clause.")

    st.markdown(f"**Query vector length:** `{result.query_length:.6f}`")
    st.dataframe(
        _table_to_dataframe(*query_weights_table_data(result)),
        width="stretch",
        hide_index=True,
    )

    s_headers, s_rows = query_similarity_table_data(result, preserve_ranking_order=True)
    if s_rows:
        st.dataframe(
            _table_to_dataframe(s_headers, s_rows),
            width="stretch",
            hide_index=True,
        )
        st.markdown(f"**Ranking (highest similarity first):** `{result.ranking_expression}`")
    else:
        st.info("No matching documents.")


def _render_query_panel(index: PositionalIndex, model: VectorSpaceModel) -> None:
    """Render the 4-mode query selector, quick-fill buttons, query input, and live results."""
    st.subheader("Query & Retrieval")

    st.session_state.setdefault("query_mode", "Auto")
    st.session_state.setdefault("query_text", "")
    st.session_state.setdefault("query_history", {})

    mode_labels = tuple(label for label, _, _ in _QUERY_MODE_OPTIONS)
    selected_mode = st.radio(
        "Query Mode",
        options=mode_labels,
        horizontal=True,
        key="query_mode",
    )
    st.caption(_MODE_LABEL_TO_HINT[selected_mode])

    query_text = st.text_input(
        "Enter Query",
        placeholder="e.g. machine learning AND data science",
        key="query_text",
    )

    st.caption("Quick-fill Technology sample queries:")
    cols = st.columns(len(DEFAULT_TECHNOLOGY_QUERIES))
    for idx, sample_query in enumerate(DEFAULT_TECHNOLOGY_QUERIES):
        sample_mode = _SAMPLE_QUERY_MODES.get(sample_query, "Auto")
        cols[idx].button(
            sample_query,
            key=f"quick_fill_{idx}",
            on_click=_apply_sample_query,
            args=(sample_query, sample_mode),
            width="stretch",
        )

    if not query_text.strip():
        st.info(
            "Enter a query above or click one of the quick-fill sample query buttons to execute live retrieval."
        )
        return

    canonical_mode = _MODE_LABEL_TO_CANONICAL[selected_mode]
    try:
        result = execute_query(index, query_text, mode=canonical_mode, model=model)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state["query_history"][(result.raw_query, canonical_mode)] = result
    _render_query_result(result)


def _render_corpus_tabs(
    index: PositionalIndex,
    model: VectorSpaceModel,
    query_history: Sequence[QueryResult],
) -> None:
    """Render Part A, Part B, and Markdown report download below the Query Panel."""
    report_md = generate_markdown_report(index, model, query_history)
    st.download_button(
        label="Download Report (Markdown)",
        data=report_md,
        file_name="report.md",
        mime="text/markdown",
        key="download_report_btn",
    )

    tab_docs, tab_pos, tab_tf, tab_df_idf, tab_tfidf = st.tabs(
        [
            "Documents",
            "Positional Index",
            "Raw TF & Weighted TF",
            "DF & IDF",
            "TF-IDF / Document Vector Lengths / Normalized TF-IDF",
        ]
    )

    with tab_docs:
        st.subheader(f"Documents (N = {index.doc_count})")
        st.dataframe(
            _table_to_dataframe(*documents_table_data(index)),
            width="stretch",
            hide_index=True,
        )

    with tab_pos:
        st.subheader(f"Positional Index ({len(index.postings)} Terms)")
        term_filter = st.text_input(
            "Filter Positional Index by Term",
            placeholder="Type a term or substring...",
            key="pos_index_filter",
        )
        pos_headers, all_pos_rows = positional_index_table_data(index)
        needle = term_filter.strip().lower()
        filtered_pos_rows = (
            [row for row in all_pos_rows if needle in row[0].lower()]
            if needle
            else all_pos_rows
        )
        st.caption(f"Showing {len(filtered_pos_rows)} of {len(all_pos_rows)} Terms")
        st.dataframe(
            _table_to_dataframe(pos_headers, filtered_pos_rows),
            width="stretch",
            hide_index=True,
        )

    with tab_tf:
        st.subheader("Raw TF")
        st.dataframe(
            _table_to_dataframe(*term_doc_matrix_data(model, model.raw_tf)),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Weighted TF (1 + log10(TF))")
        st.dataframe(
            _table_to_dataframe(*term_doc_matrix_data(model, model.weighted_tf)),
            width="stretch",
            hide_index=True,
        )

    with tab_df_idf:
        st.subheader("DF and IDF")
        st.dataframe(
            _table_to_dataframe(*df_idf_table_data(model)),
            width="stretch",
            hide_index=True,
        )

    with tab_tfidf:
        st.subheader("TF-IDF Weight Matrix")
        st.dataframe(
            _table_to_dataframe(*term_doc_matrix_data(model, model.tf_idf)),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Document Vector Lengths")
        st.dataframe(
            _table_to_dataframe(*doc_lengths_table_data(model)),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Normalized TF-IDF Matrix")
        st.dataframe(
            _table_to_dataframe(*term_doc_matrix_data(model, model.normalized_tf_idf)),
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    """Render the Technology IR System Streamlit Web GUI."""
    st.set_page_config(
        page_title="Technology IR System — PySpark & TF-IDF",
        layout="wide",
    )
    st.title("Technology Information Retrieval System")
    st.caption(
        "Part A: PySpark Positional Indexer | Part B: TF-IDF Vector-Space & Phrase/Boolean Retrieval"
    )

    _ensure_initial_corpus()

    index, model = _render_sidebar()
    _render_query_panel(index, model)

    query_history: dict[tuple[str, str], QueryResult] = st.session_state.get("query_history", {})
    _render_corpus_tabs(index, model, tuple(query_history.values()))


if __name__ == "__main__":
    main()
