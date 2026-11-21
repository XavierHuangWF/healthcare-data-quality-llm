"""Third floating status panel reports genuine Gemma B conversation state."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from healthcare_data_quality.ui.agent_b_banner import agent_b_status_snapshot, render_agent_b_banner_html
from healthcare_data_quality.ui.exploration_banner import render_recommendation_banner_html, sticky_banner_css


def test_agent_b_pre_run_is_waiting_and_has_no_false_progress():
    status = agent_b_status_snapshot(None, active_context=None, context_key=None)
    assert status == {"state": "waiting", "pending": 0, "discussed": 0, "plans": 0, "applied": 0}
    html = render_agent_b_banner_html(status)
    assert "Agent B — waiting for findings" in html
    assert "progressbar" not in html
    assert 'class="card is-active"' not in html


def test_agent_b_ready_with_new_results_but_no_conversation():
    status = agent_b_status_snapshot({}, active_context="audit:run-id", context_key="audit:run-id")
    assert status["state"] == "ready"
    html = render_agent_b_banner_html(status)
    assert "Agent B — ready for questions" in html
    assert "approval and Execute" in html


def test_agent_b_pending_and_messages_are_counted_per_finding():
    session = {
        "cleaning_pending_gemma_b": {"f1": {"job_id": "xyz"}},
        "cleaning_messages_by_finding": {
            "f1": [{"role": "user", "content": "Could we round this?"},
                   {"role": "assistant", "content": "Depends on protocol"}],
            "f2": [{"role": "user", "content": "What about blanks?"}],
            "f3": [{"role": "assistant", "content": "Not a user conversation"}],
        },
        "cleaning_plans_by_finding": {"f1": {"operation": "cells"}},
        "cleaning_history": [{"finding_id": "f2"}, {"finding_id": "f2"}],
    }
    status = agent_b_status_snapshot(session, active_context="audit:run-id", context_key="audit:run-id")
    assert status == {"state": "responding", "pending": 1, "discussed": 2, "plans": 1, "applied": 1}
    html = render_agent_b_banner_html(status)
    assert "Agent B — responding" in html
    assert "1 finding conversation in progress" in html
    assert 'class="card is-active"' in html
    assert "1 draft plan(s)" in html


def test_agent_b_follow_up_and_context_isolation():
    session = {"cleaning_messages_by_finding": {"old": [{"role": "user", "content": "Sensitive question"}]}}
    same = agent_b_status_snapshot(session, active_context="old", context_key="old")
    assert same["state"] == "follow_up"
    assert "available for follow-up" in render_agent_b_banner_html(same)
    other = agent_b_status_snapshot(session, active_context="old", context_key="new")
    assert other["state"] == "ready"
    assert other["discussed"] == 0
    assert "Sensitive question" not in render_agent_b_banner_html(other)


def test_agent_b_rejects_invalid_state_and_counts():
    with pytest.raises(ValueError, match="state"):
        render_agent_b_banner_html({"state": "working-forever"})
    with pytest.raises(ValueError, match="nonnegative"):
        render_agent_b_banner_html({"state": "responding", "pending": -1})
    with pytest.raises(ValueError, match="nonnegative"):
        render_agent_b_banner_html({"state": "ready", "applied": True})


def test_two_background_cards_are_mounted_and_agent_c_is_inline():
    app = (ROOT / "app.py").read_text()
    idle = app[app.index("def render_pre_investigation_status("):app.index("def _toggle_full_dataset_preview(")]
    live = app[app.index("def render_live_investigation_status("):app.index("def render_pre_investigation_status(")]
    assert 'render_recommendation_banner_html(None, idle=True)' in idle
    assert 'with st.container(key="hdq_recommendation_status_card"):' in live
    assert 'with st.container(key="hdq_investigation_status_card"):' in live
    assert 'with st.container(key="hdq_agent_b_status_card"):' not in live
    assert 'render_agent_b_banner_html(' not in idle
    assert 'key="hdq_close_investigation"' in live
    assert 'key="hdq_close_recommendations"' in live
    assert 'Agent C — Interactive cleaning' in app
    assert 'ready_ids = {' in app
    assert '.st-key-hdq_recommendation_status_card' in sticky_banner_css()
    assert 'Agent B — waiting for QC findings' in render_recommendation_banner_html(None, idle=True)
