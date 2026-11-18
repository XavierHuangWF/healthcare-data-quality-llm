"""Taxonomy structure evidence should be opt-in, like the other finding cards."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class _Context:
    def __init__(self, events, kind):
        self.events = events
        self.kind = kind

    def __enter__(self):
        self.events.append(("enter", self.kind))
        return self

    def __exit__(self, *args):
        self.events.append(("exit", self.kind))
        return False


def _load_app(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", ModuleType("streamlit"))
    spec = importlib.util.spec_from_file_location("hdq_taxonomy_collapse_app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ui(monkeypatch, app):
    events = []
    stub = SimpleNamespace(
        container=lambda **kwargs: _Context(events, "container"),
        expander=lambda label, **kwargs: (events.append(("expander", label, kwargs)) or _Context(events, "expander")),
        markdown=lambda *args, **kwargs: events.append(("markdown", args)),
        write=lambda *args, **kwargs: events.append(("write", args)),
        caption=lambda *args, **kwargs: events.append(("caption", args)),
        dataframe=lambda frame, **kwargs: events.append(("dataframe", frame.copy(), kwargs)),
    )
    monkeypatch.setattr(app, "st", stub)
    return events


def test_taxonomy_table_is_collapsed_in_same_affected_rows_expander(monkeypatch):
    app = _load_app(monkeypatch)
    events = _ui(monkeypatch, app)
    evidence = [
        {"CSV row": 3, "Original classification (verbatim)": "a;b;c", "Levels found": 3, "Expected levels": 7},
        {"CSV row": 5, "Original classification (verbatim)": "a;b", "Levels found": 2, "Expected levels": 7},
    ]
    app.show_taxonomy_structure_finding({
        "title": "2 rows have an incorrect number of taxonomy levels",
        "explanation": "Only number of positions is assessed.",
        "evidence": evidence,
        "next_step": "Review the source.",
    }, finding_index=5)

    assert ("expander", "View affected rows (2)", {}) in events
    open_index = events.index(("enter", "expander"))
    close_index = events.index(("exit", "expander"))
    table_events = [(i, event) for i, event in enumerate(events) if event[0] == "dataframe"]
    assert len(table_events) == 1
    table_index, (_, frame, kwargs) = table_events[0]
    assert open_index < table_index < close_index
    assert frame.to_dict("records") == evidence
    assert kwargs == {"hide_index": True, "width": "stretch"}
    assert any(event[0] == "markdown" and "**Finding 5 ·" in event[1][0] for event in events)
    assert any(event[0] == "write" and "Review the source." in event[1] for event in events)
    assert any(event[0] == "markdown" and "Expected seven-rank order" in event[1][0] for event in events)
    assert any(event[0] == "write" and "Original classifications and taxonomy level counts" in event[1][0]
               and open_index < i < close_index for i, event in enumerate(events))


def test_taxonomy_with_no_evidence_does_not_show_empty_expander(monkeypatch):
    app = _load_app(monkeypatch)
    events = _ui(monkeypatch, app)
    app.show_taxonomy_structure_finding({
        "title": "Taxonomy structure", "explanation": "Inspect levels", "evidence": [], "next_step": "Review"
    })
    assert not any(event[0] == "expander" for event in events)
    assert not any(event[0] == "dataframe" for event in events)
