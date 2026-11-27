from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_project_report_covers_all_part_c_deliverables() -> None:
    """Verify docs/PROJECT_REPORT.md covers architecture, Spark, math, live outputs, and 5 questions."""
    report_path = REPO_ROOT / "docs" / "PROJECT_REPORT.md"
    assert report_path.is_file(), "docs/PROJECT_REPORT.md must exist for Part C submission"

    content = report_path.read_text(encoding="utf-8")

    required_phrases = (
        "Part A",
        "Part B",
        "local[*]",
        "posexplode",
        "PYSPARK_PYTHON",
        "Positional Index",
        "Raw TF",
        "Weighted TF",
        "IDF",
        "TF-IDF Weight",
        "Document Vector Length",
        "Normalized TF-IDF",
        "Query Vector",
        "Similarity Score",
        "pyspark==4.0.1",
        "streamlit",
        "pytest",
        "uv run ir-system",
        "uv run streamlit run app.py",
        "uv run pytest -v",
        "machine learning",
        "data science",
        "machine learning AND data science",
        "machine learning AND NOT data science",
        "0.425721",
        "0.545341",
        "0.691834",
        "0.5113",
        "d1 > d5 > d10 > d2 > d7",
        "d5 > d3 > d8",
        "d1 > d10 > d2 > d7",
        "Question 1:",
        "Question 2:",
        "Question 3:",
        "Question 4:",
        "Question 5:",
    )
    for phrase in required_phrases:
        assert phrase in content, f"Missing expected Part C report content: {phrase!r}"

    for expected_term in (
        "analysis",
        "analytics",
        "and",
        "applications",
        "artificial",
        "cloud",
        "computing",
        "data",
        "explores",
        "improve",
        "improves",
        "intelligence",
        "learning",
        "machine",
        "models",
        "powers",
        "processing",
        "research",
        "science",
        "smart",
        "storage",
        "support",
        "supports",
        "systems",
        "through",
        "tools",
        "uses",
    ):
        assert f"| {expected_term} |" in content, f"Missing term row in report: {expected_term!r}"


def test_readme_documents_setup_usage_and_valid_relative_links() -> None:
    """Verify README.md covers setup/CLI/GUI/tests and all relative links resolve."""
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.is_file()
    content = readme_path.read_text(encoding="utf-8")

    for required in (
        "3.10.20",
        "4.0.1",
        "Java 17",
        "uv sync",
        "uv run ir-system",
        "uv run streamlit run app.py",
        "uv run pytest -v",
        "GLOSSARY.md",
        "docs/adr/",
        "docs/PROJECT_REPORT.md",
    ):
        assert required in content, f"Missing expected README content: {required!r}"

    relative_targets = [
        target
        for target in MARKDOWN_LINK_RE.findall(content)
        if not target.startswith(("http://", "https://", "#"))
    ]
    assert relative_targets, "README.md should include relative links to docs"
    for target in relative_targets:
        clean_target = target.split("#", 1)[0]
        resolved = (REPO_ROOT / clean_target).resolve()
        assert resolved.exists(), f"Broken relative link in README.md: {target}"
