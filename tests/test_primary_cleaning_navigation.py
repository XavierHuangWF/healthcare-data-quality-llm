"""Navigation contract: page always opens; individual review cards wait for Agent B."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _app(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", ModuleType("streamlit"))
    spec = importlib.util.spec_from_file_location("hdq_nav_test_app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Column:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


class _Streamlit:
    def __init__(self):
        self.session_state = {}
        self.buttons = []
        self.layout = None
        self.markdown_calls = []

    def markdown(self, text, **kwargs):
        self.markdown_calls.append((text, kwargs))

    def columns(self, widths, **kwargs):
        self.layout = list(widths)
        return [_Column() for _ in widths]

    def button(self, label, **kwargs):
        self.buttons.append((label, kwargs))
        return False


def test_cleaning_page_is_open_while_individual_findings_wait(monkeypatch):
    app = _app(monkeypatch)
    streamlit = _Streamlit()
    monkeypatch.setattr(app, "st", streamlit)
    assert app._resolve_results_page("investigation-one", False) == "clean"
    app._render_results_navigation("clean", False)
    labels = [label for label, _ in streamlit.buttons]
    assert labels == [
        "Clean data with LM model", "Manage skills & memory",
        "Data quality findings", "Supporting evidence",
    ]
    assert streamlit.buttons[0][1]["disabled"] is False
    assert all(not options["disabled"] for _, options in streamlit.buttons[1:])
    assert streamlit.layout[0] > 3 * max(streamlit.layout[2:])
    assert streamlit.layout[1] < min(streamlit.layout[2:])  # deliberate visual gap
    app._select_results_page("clean")
    assert streamlit.session_state["hdq_results_nav_page"] == "clean"
    app._select_results_page("findings")
    assert app._resolve_results_page("investigation-one", False) == "findings"


def test_first_suggestion_unlocks_clean_as_default(monkeypatch):
    app = _app(monkeypatch)
    streamlit = _Streamlit()
    monkeypatch.setattr(app, "st", streamlit)
    assert app._resolve_results_page("investigation-one", False) == "clean"
    assert app._resolve_results_page("investigation-one", True) == "clean"
    app._render_results_navigation("clean", True)
    assert streamlit.buttons[0][1]["disabled"] is False
    assert streamlit.buttons[0][1]["type"] == "primary"
    app._select_results_page("evidence")
    assert app._resolve_results_page("investigation-one", True) == "evidence"
    assert app._resolve_results_page("new-investigation", False) == "clean"
    assert app._resolve_results_page("new-investigation", True) == "clean"


def test_switching_from_waiting_to_secondary_page_is_preserved(monkeypatch):
    app = _app(monkeypatch)
    streamlit = _Streamlit()
    monkeypatch.setattr(app, "st", streamlit)
    app._resolve_results_page("one", False)
    app._select_results_page("memory")
    assert app._resolve_results_page("one", True) == "memory"
    app._select_results_page("unrecognized")
    assert streamlit.session_state["hdq_results_nav_page"] == "memory"
    app._select_results_page("clean")
    assert streamlit.session_state["hdq_results_nav_page"] == "clean"
    assert app._resolve_results_page("one", False) == "clean"


def test_main_ui_only_renders_one_page_and_has_one_suggestion_gate():
    code = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'active_page == "clean"' in code
    assert 'active_page == "findings"' in code
    assert 'active_page == "evidence"' in code
    assert 'active_page == "memory"' in code
    assert "st.tabs(results_tab_labels)" not in code
    assert "st.fragment(run_every=\"3s\")(_unlock_results_when_suggestion_ready)()" in code
    assert "render_cleaning_workspace(paths, uploaded, current, cache_key)" in code
    assert 'st.subheader("Clean data with LM model")' in code


def test_only_a_ready_current_run_model_suggestion_unlocks(monkeypatch):
    app = _app(monkeypatch)
    from healthcare_data_quality.jobs import cleaning_jobs

    class Manager:
        entries = {}
        def snapshot(self, key):
            assert key == "current-run-key"
            return {"entries": self.entries}

    manager = Manager()
    monkeypatch.setattr(cleaning_jobs, "recommendation_manager", lambda: manager)
    monkeypatch.setattr(cleaning_jobs, "job_fingerprint", lambda *_args, **_kw: "current-run-key")
    audit = {"cleaning_recommendation_run_id": "fresh-run"}
    assert app._agent_b_first_suggestion_ready({}, "ctx") is False
    manager.entries = {"one": {"status": "ready", "origin": "review_memory", "suggestion": {"operations": [1]}}}
    assert not app._agent_b_first_suggestion_ready(audit, "ctx")
    manager.entries = {"one": {"status": "queued", "origin": "gemma", "suggestion": {"operations": [1]}}}
    assert not app._agent_b_first_suggestion_ready(audit, "ctx")
    manager.entries = {"one": {"status": "failed", "origin": "gemma", "suggestion": {"operations": [1]}}}
    assert not app._agent_b_first_suggestion_ready(audit, "ctx")
    manager.entries = {"one": {"status": "ready", "origin": "gemma", "suggestion": {"operations": [1]}}}
    assert app._agent_b_first_suggestion_ready(audit, "ctx")
