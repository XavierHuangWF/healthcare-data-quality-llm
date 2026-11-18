"""Full CSV rendering must be opt-in, without hiding the dataset summary."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_app(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", ModuleType("streamlit"))
    spec = importlib.util.spec_from_file_location("hdq_preview_test", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.metrics = []
        self.controls = []
        self.tables = []
        self.selection_calls = 0

    def subheader(self, text):
        self.heading = text

    def selectbox(self, label, options, **kwargs):
        self.selection_calls += 1
        return options[-1]

    def columns(self, count):
        assert count == 3
        return [SimpleNamespace(metric=lambda label, value: self.metrics.append((label, value))) for _ in range(count)]

    def button(self, label, **kwargs):
        self.controls.append((label, kwargs))
        return False

    def caption(self, text):
        self.caption_text = text

    def dataframe(self, value, **kwargs):
        self.tables.append((value.copy(), kwargs))

    def error(self, message):
        raise AssertionError(message)


def fixtures():
    uploads = [SimpleNamespace(name="one.csv"), SimpleNamespace(name="two.csv")]
    reports = [
        {"data_rows": 158, "total_columns": 106, "duplicate_headers": ["ABC99"]},
        {"data_rows": 10, "total_columns": 7, "duplicate_headers": []},
    ]
    return [Path("one.csv"), Path("two.csv")], uploads, reports, "testfingerprint"


def test_preview_is_collapsed_and_not_loaded_by_default(monkeypatch):
    app = load_app(monkeypatch)
    fake = FakeStreamlit()
    app.st = fake
    monkeypatch.setattr(app, "preview_csv", lambda *_: (_ for _ in ()).throw(AssertionError("CSV loaded while collapsed")))
    assert app.render_full_dataset_section(*fixtures())
    assert fake.heading == "Full dataset"
    assert fake.metrics == [("Records", 158), ("Columns", 106), ("Repeated column names", 1)]
    assert fake.selection_calls == 0
    assert fake.tables == []
    assert fake.controls[0][0] == "▶ View dataset preview"


def test_expand_renders_complete_table_and_collapse_preserves_selection(monkeypatch):
    app = load_app(monkeypatch)
    fake = FakeStreamlit()
    app.st = fake
    source = pd.DataFrame({"taxonomy": ["a", "b"], "count": [0, 1]})
    requested = []

    def preview(path):
        requested.append(path)
        return source.copy()

    monkeypatch.setattr(app, "preview_csv", preview)
    args = fixtures()
    assert app.render_full_dataset_section(*args)
    label, button_args = fake.controls[-1]
    button_args["on_click"](*button_args["args"])
    fake.metrics.clear()
    assert app.render_full_dataset_section(*args)
    assert requested == [Path("two.csv")]
    assert fake.metrics == [("Records", 10), ("Columns", 7), ("Repeated column names", 0)]
    table, opts = fake.tables[0]
    assert len(table) == 2
    assert list(table.index) == [2, 3]
    assert table.index.name == "CSV row"
    assert opts["height"] == 650
    assert fake.controls[-1][0] == "▼ Hide dataset preview"
    _, button_args = fake.controls[-1]
    button_args["on_click"](*button_args["args"])
    fake.metrics.clear()
    fake.tables.clear()
    assert app.render_full_dataset_section(*args)
    assert requested == [Path("two.csv")]
    assert fake.tables == []
    assert fake.metrics[0] == ("Records", 10)
