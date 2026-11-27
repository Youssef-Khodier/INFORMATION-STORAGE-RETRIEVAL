from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_startup_autoloads_technology_corpus_and_renders_tabs() -> None:
    """AppTest startup auto-loads data/2. Technology via PySpark and renders all GUI tabs."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()

    assert not at.exception

    metric_values = {metric.label: metric.value for metric in at.metric}
    assert metric_values["Documents (N)"] == "10"
    assert metric_values["Vocabulary Terms"] == "27"
    assert metric_values["Total Tokens"] == "64"

    tab_labels = [tab.label for tab in at.tabs]
    assert tab_labels == [
        "Documents",
        "Positional Index",
        "Raw TF & Weighted TF",
        "DF & IDF",
        "TF-IDF / Document Vector Lengths / Normalized TF-IDF",
    ]

    assert len(at.dataframe) >= 7
    docs_df = at.dataframe[0].value
    assert list(docs_df.columns) == ["File", "Exact document text", "Word count"]
    assert len(docs_df) == 10
    assert list(docs_df["File"]) == [f"d{i}" for i in range(1, 11)]

    pos_df = at.dataframe[1].value
    assert list(pos_df.columns) == ["Term", "Document IDs and 1-based positions"]
    assert len(pos_df) == 27


def test_query_modes_quick_fill_and_live_execution() -> None:
    """Query panel supports 4 modes, validation errors, quick-fill buttons, and live outputs."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception

    # 1. Click quick-fill button for "machine learning"
    at.button(key="quick_fill_0").click().run()
    assert not at.exception
    assert at.radio(key="query_mode").value == "Exact Phrase"
    assert at.text_input(key="query_text").value == "machine learning"

    markdown_text = "\n".join(el.value for el in at.markdown)
    assert "0.425721" in markdown_text
    assert "d1 > d5 > d10 > d2 > d7" in markdown_text

    # 2. Click quick-fill button for "data science"
    at.button(key="quick_fill_1").click().run()
    assert not at.exception
    assert at.text_input(key="query_text").value == "data science"
    markdown_text = "\n".join(el.value for el in at.markdown)
    assert "0.545341" in markdown_text
    assert "d5 > d3 > d8" in markdown_text

    # 3. Click quick-fill button for "machine learning AND data science"
    at.button(key="quick_fill_2").click().run()
    assert not at.exception
    assert at.radio(key="query_mode").value == "Boolean (AND / AND NOT)"
    markdown_text = "\n".join(el.value for el in at.markdown)
    assert "0.691834" in markdown_text
    assert "0.5113" in "\n".join(df.value.to_string() for df in at.dataframe)

    # 4. Click quick-fill button for "machine learning AND NOT data science"
    at.button(key="quick_fill_3").click().run()
    assert not at.exception
    assert at.radio(key="query_mode").value == "Boolean (AND / AND NOT)"
    markdown_text = "\n".join(el.value for el in at.markdown)
    assert "0.425721" in markdown_text
    assert "d1 > d10 > d2 > d7" in markdown_text

    # 5. Mode validation: Term Search rejects multi-word query gracefully
    at.radio(key="query_mode").set_value("Term Search").run()
    at.text_input(key="query_text").set_value("machine learning").run()
    assert not at.exception
    assert len(at.error) == 1
    assert "one token" in at.error[0].value.lower()

    # 6. Term Search succeeds for single token
    at.text_input(key="query_text").set_value("neural").run()
    assert not at.exception
    assert len(at.error) == 0
    assert any("no matching documents" in el.value.lower() for el in [*at.info, *at.markdown])

    # 7. Boolean mode rejects phrase without uppercase operator
    at.radio(key="query_mode").set_value("Boolean (AND / AND NOT)").run()
    at.text_input(key="query_text").set_value("machine learning").run()
    assert not at.exception
    assert len(at.error) == 1

    # 8. Exact Phrase matches literal lowercase 'and' phrase in d5; Auto mode also works
    at.radio(key="query_mode").set_value("Exact Phrase").run()
    at.text_input(key="query_text").set_value("data science and machine learning").run()
    assert not at.exception
    assert len(at.error) == 0
    assert "d5" in "\n".join(el.value for el in at.markdown)

    at.radio(key="query_mode").set_value("Auto").run()
    at.text_input(key="query_text").set_value("machine learning AND data science").run()
    assert not at.exception
    assert len(at.error) == 0
    assert "0.691834" in "\n".join(el.value for el in at.markdown)


def test_positional_index_live_filter(tmp_path: Path) -> None:
    """Positional Index tab filters displayed Terms dynamically via search input."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception

    at.text_input(key="pos_index_filter").set_value("cloud").run()
    assert not at.exception

    pos_dfs = [
        df.value
        for df in at.dataframe
        if list(df.value.columns) == ["Term", "Document IDs and 1-based positions"]
    ]
    assert len(pos_dfs) == 1
    assert list(pos_dfs[0]["Term"]) == ["cloud"]

    at.text_input(key="pos_index_filter").set_value("").run()
    assert not at.exception
    pos_dfs = [
        df.value
        for df in at.dataframe
        if list(df.value.columns) == ["Term", "Document IDs and 1-based positions"]
    ]
    assert len(pos_dfs[0]) == 27


def test_sidebar_folder_import_error_handling_and_reset(tmp_path: Path) -> None:
    """Sidebar imports arbitrary .txt folders, preserves Corpus on errors, and resets cleanly."""
    custom_dir = tmp_path / "energy_corpus"
    custom_dir.mkdir()
    (custom_dir / "alpha.txt").write_text("solar energy grid", encoding="utf-8")
    (custom_dir / "beta.txt").write_text("wind energy grid storage", encoding="utf-8")
    (custom_dir / "empty.txt").write_text("   \n", encoding="utf-8")
    (custom_dir / "notes.md").write_text("ignored non-txt", encoding="utf-8")

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception

    # Invalid folder shows friendly error while keeping Technology Corpus intact
    missing_dir = tmp_path / "does_not_exist"
    at.text_input(key="corpus_folder_input").set_value(str(missing_dir)).run()
    at.button(key="load_folder_btn").click().run()
    assert not at.exception
    assert len(at.error) >= 1
    metric_values = {metric.label: metric.value for metric in at.metric}
    assert metric_values["Documents (N)"] == "10"

    # Load custom multi-domain folder
    at.text_input(key="corpus_folder_input").set_value(str(custom_dir)).run()
    at.button(key="load_folder_btn").click().run()
    assert not at.exception
    assert len(at.error) == 0

    metric_values = {metric.label: metric.value for metric in at.metric}
    assert metric_values["Documents (N)"] == "2"
    assert metric_values["Vocabulary Terms"] == "5"
    assert metric_values["Total Tokens"] == "7"
    assert any("empty.txt" in w.value for w in at.warning)

    # Execute query against custom corpus
    at.text_input(key="query_text").set_value("energy grid").run()
    assert not at.exception
    markdown_text = "\n".join(el.value for el in at.markdown)
    assert "alpha > beta" in markdown_text

    # Reset back to Technology Corpus
    at.button(key="reset_technology_btn").click().run()
    assert not at.exception
    metric_values = {metric.label: metric.value for metric in at.metric}
    assert metric_values["Documents (N)"] == "10"
    assert metric_values["Vocabulary Terms"] == "27"
    assert metric_values["Total Tokens"] == "64"


def test_stage_uploaded_corpus_filters_non_txt_and_manages_temp_dir(tmp_path: Path) -> None:
    """Uploaded file staging writes .txt files to a managed temp dir and ignores non-.txt."""
    from dataclasses import dataclass
    from ir_system.gui import stage_uploaded_corpus

    @dataclass
    class FakeUpload:
        name: str
        payload: bytes

        def getvalue(self) -> bytes:
            return self.payload

    uploads = [
        FakeUpload("folder/doc_a.txt", b"quantum computing"),
        FakeUpload("folder/doc_b.txt", b"quantum optics"),
        FakeUpload("folder/readme.pdf", b"%PDF-1.4"),
    ]
    staged_dir, fingerprint, warnings = stage_uploaded_corpus(uploads)
    try:
        assert staged_dir.is_dir()
        assert len(fingerprint) == 64
        assert (staged_dir / "doc_a.txt").read_text(encoding="utf-8") == "quantum computing"
        assert (staged_dir / "doc_b.txt").read_text(encoding="utf-8") == "quantum optics"
        assert not (staged_dir / "readme.pdf").exists()
        assert any("readme.pdf" in w for w in warnings)
    finally:
        import shutil

        shutil.rmtree(staged_dir, ignore_errors=True)


