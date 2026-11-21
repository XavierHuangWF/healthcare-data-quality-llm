"""Terminal-status dismissals are independent, opt-in, and per investigation."""
from pathlib import Path
import ast
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from healthcare_data_quality.ui.exploration_banner import recommendation_is_complete


def test_only_terminal_agent_b_recommendations_are_dismissible():
    assert not recommendation_is_complete(None)
    assert not recommendation_is_complete({"entries": {}})
    assert not recommendation_is_complete({"entries": {"a": {"status": "pending"}}})
    assert not recommendation_is_complete({"entries": {"a": {"status": "running"}, "b": {"status": "ready"}}})
    assert recommendation_is_complete({"entries": {"a": {"status": "ready"}}})
    assert recommendation_is_complete({"entries": {"a": {"status": "ready"}, "b": {"status": "failed"}}})
    assert recommendation_is_complete({"entries": {"a": {"status": "failed"}}})


def test_close_buttons_are_real_streamlit_controls_not_embedded_html():
    source = (ROOT / "app.py").read_text()
    section = source[source.index("def render_live_investigation_status("):source.index("def render_pre_investigation_status(")]
    assert 'investigation_complete = not (registered_running or (live and live.get("status") == "running"))' in section
    assert 'recommendations_complete = recommendation_is_complete(recommendation_state)' in section
    assert 'if investigation_complete:' in section
    assert 'if recommendations_complete:' in section
    assert 'key="hdq_close_investigation"' in section
    assert 'key="hdq_close_recommendations"' in section
    assert 'on_click=_hide_completed_status' in section
    assert 'with st.container(key="hdq_investigation_status_card"):' in section
    assert 'with st.container(key="hdq_recommendation_status_card"):' in section
    idle = source[source.index("def render_pre_investigation_status("):source.index("def _toggle_full_dataset_preview(")]
    assert 'hdq_close_investigation' not in idle
    assert 'hdq_close_recommendations' not in idle


def test_dismissal_is_per_run_and_does_not_touch_saved_results(monkeypatch):
    # Execute only the two pure session-state helpers: this test does not
    # require Streamlit to be installed in the portable test environment.
    source = ast.parse((ROOT / "app.py").read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_hide_completed_status", "_status_card_visible"}]
    assert len(functions) == 2
    state = {}
    namespace = {"st": SimpleNamespace(session_state=state)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / "app.py"), "exec"), namespace)
    app = SimpleNamespace(**namespace)
    assert app._status_card_visible("investigation", "run-A")
    assert app._status_card_visible("recommendations", "run-A")
    app._hide_completed_status("investigation", "run-A")
    assert not app._status_card_visible("investigation", "run-A")
    assert app._status_card_visible("recommendations", "run-A")
    assert app._status_card_visible("investigation", "run-B")
    app._hide_completed_status("recommendations", "run-A")
    assert not app._status_card_visible("recommendations", "run-A")
    assert app._status_card_visible("recommendations", "run-B")
    assert set(state) == {"hdq_hidden_investigation_run", "hdq_hidden_recommendations_run"}
    with pytest.raises(ValueError, match="Unknown"):
        app._hide_completed_status("other", "run-A")


def test_close_controls_are_clickable_in_fixed_overlay():
    from healthcare_data_quality.ui.exploration_banner import sticky_banner_css
    css = sticky_banner_css()
    assert "pointer-events: none !important;" in css
    # Target the dedicated keyed containers, not Streamlit's variable internal
    # .stButton wrapper (which previously appeared below the floating cards).
    assert ".st-key-hdq_investigation_close_overlay" in css
    assert ".st-key-hdq_recommendation_close_overlay" in css
    assert "position: absolute !important;" in css
    assert "right: 9px !important;" in css
    assert "pointer-events: auto !important;" in css


def test_close_button_containers_are_nested_inside_their_own_cards():
    source = (ROOT / "app.py").read_text()
    live = source[source.index("def render_live_investigation_status("):source.index("def render_pre_investigation_status(")]
    pairs = (
        ("hdq_investigation_status_card", "hdq_investigation_close_overlay", "hdq_close_investigation"),
        ("hdq_recommendation_status_card", "hdq_recommendation_close_overlay", "hdq_close_recommendations"),
    )
    for card, overlay, button in pairs:
        start = live.index(f'with st.container(key="{card}"):')
        stop = live.find('with st.container(key="hdq_', start + 1)
        # The nested close overlay is positioned inside the named card.
        nested = live.index(f'with st.container(key="{overlay}"):', start)
        assert nested > start
        assert f'key="{button}"' in live[nested:]
    assert 'if investigation_complete:' in live
    assert 'if recommendations_complete:' in live
