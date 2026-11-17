"""Display-only QC numbering, continuing through data-structure findings."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _app(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", ModuleType("streamlit"))
    spec = importlib.util.spec_from_file_location("hdq_numbered_findings_app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _display(monkeypatch, app):
    headings = []
    stub = SimpleNamespace(
        container=lambda **kwargs: _Context(),
        expander=lambda *args, **kwargs: _Context(),
        columns=lambda count: [SimpleNamespace(metric=lambda *args, **kwargs: None) for _ in range(count)],
        markdown=lambda value, **kwargs: headings.append(value),
        dataframe=lambda *args, **kwargs: None,
        write=lambda *args, **kwargs: None,
        caption=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(app, "st", stub)
    return headings


def test_qc_findings_have_sequential_numbers_across_sections(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    headings = _display(monkeypatch, app)
    findings = [
        {"title": "Fractional counts", "explanation": "explanation", "evidence": [], "next_step": "review"},
        {"title": "Taxonomy levels", "type": "taxonomy_structure", "explanation": "explanation",
         "evidence": [], "next_step": "review"},
    ]
    monkeypatch.setattr(app, "baseline_issues", lambda *args, **kwargs: findings)
    monkeypatch.setattr(app, "header_findings", lambda *args, **kwargs: [
        {"title": "Identical columns", "explanation": "explanation", "next_step": "review",
         "header": "X", "columns": [2, 3], "matching_rows": 1},
    ])
    monkeypatch.setattr(app, "header_issues", lambda *args, **kwargs: [
        {"title": "Unnamed column", "explanation": "explanation", "next_step": "review", "rows": []},
    ])
    count = app.render_baseline("dataset_1", {"checks": {}}, source_path=tmp_path / "sample.csv")
    assert count == 2
    app.render_header_findings(tmp_path / "sample.csv", {}, start_index=count + 1)
    assert [heading for heading in headings if heading.startswith("**Finding ")] == [
        "**Finding 1 · Fractional counts**",
        "**Finding 2 · Taxonomy levels**",
        "**Finding 3 · Identical columns**",
        "**Finding 4 · Unnamed column**",
    ]


def test_numbering_resets_per_dataset_and_header_only_starts_at_one(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    headings = _display(monkeypatch, app)
    monkeypatch.setattr(app, "baseline_issues", lambda *args, **kwargs: [])
    monkeypatch.setattr(app, "header_findings", lambda *args, **kwargs: [])
    monkeypatch.setattr(app, "header_issues", lambda *args, **kwargs: [
        {"title": "Unnamed column", "explanation": "explanation", "next_step": "review", "rows": []},
    ])
    count = app.render_baseline("dataset_2", {"checks": {}}, source_path=tmp_path / "sample.csv")
    assert count == 0
    app.render_header_findings(tmp_path / "sample.csv", {}, start_index=count + 1)
    assert "**Finding 1 · Unnamed column**" in headings
