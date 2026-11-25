"""Explicit No must invalidate both the old state and its browser render tree."""
from pathlib import Path
import pytest
from test_cleaning_review_cards import StubUI, _app


def test_decline_replaces_body_without_full_rerun(monkeypatch):
    ui = StubUI()
    app = _app(monkeypatch, ui)
    state = ui.session_state
    state.cleaning_plans_by_finding = {'f': {'operation': 'drop_rows'}, 'other': {'operation': 'replace_cells'}}
    state.cleaning_discussion_mode_by_finding = {}
    state.cleaning_review_statuses = {}
    state.cleaning_keep_open_by_finding = {'other': True}
    state.cleaning_pending_gemma_b = {'f': {'job_id': 'old'}, 'other': {'job_id': 'live'}}
    state.cleaning_completed_gemma = {'f': {'status': 'done'}}
    state.cleaning_history = [{'finding_id': 'other'}]
    keys = []
    monkeypatch.setattr(ui, 'container', lambda **kw: keys.append(kw['key']))
    calls = []
    def rerun(*, scope='app'):
        calls.append(scope)
        raise StopIteration('rerun')
    ui.rerun = rerun
    app._replaceable_review_body('f')
    app._decline_and_discuss('f')
    app._replaceable_review_body('f')
    assert keys[0] != keys[1]
    assert state.cleaning_keep_open_by_finding == {'other': True, 'f': True}
    assert state.cleaning_plans_by_finding == {'other': {'operation': 'replace_cells'}}
    assert state.cleaning_pending_gemma_b == {'other': {'job_id': 'live'}}
    assert state.cleaning_retired_chat_jobs == {'old'}
    assert not state.cleaning_completed_gemma
    assert app._waiting_for_new_discussion(state, 'f')
    app._flush_declined_review_tree()
    app._flush_declined_review_tree()
    assert calls == []
    assert len(state.cleaning_history) == 1
    # Callback replay does not create a second rejection or refresh loop.
    app._decline_and_discuss('f')
    app._flush_declined_review_tree()
    assert calls == []


def test_retired_result_never_reappears(monkeypatch):
    ui = StubUI(); app = _app(monkeypatch, ui)
    state = ui.session_state
    state.cleaning_pending_gemma_b = {}
    state.cleaning_retired_chat_jobs = {'old'}
    class Manager:
        def take(self, job_id):
            assert job_id == 'old'
            return {'status': 'done', 'result': {'operation': 'drop_rows'}}
    monkeypatch.setattr('healthcare_data_quality.jobs.interactive_chat_jobs.interactive_chat_manager', lambda: Manager())
    app._poll_agent_c_completion()
    assert state.cleaning_retired_chat_jobs == set()
    assert state.cleaning_completed_gemma == {}


def test_readiness_watcher_is_sibling_and_decline_branch_terminates():
    code = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    workspace = code.split('def render_cleaning_workspace(', 1)[1].split('\ndef ', 1)[0]
    assert 'st.fragment(run_every="3s")(_refresh_parent_for_new_agent_b_choices)' not in workspace
    assert 'st.fragment(run_every="5s", key="hdq_cleaning_review")(refresh_cleaning)()' in code
    assert 'st.fragment(run_every="3s")(_refresh_parent_for_new_agent_b_choices)()' not in code
    branch = workspace.split('# Exclusive discussion state after No.', 1)[1].split('# Multi-dataset checks can bind', 1)[0]
    assert '_render_review_chat_input(' in branch
    assert '_render_bottom_review_closure(' in branch
    assert branch.rstrip().endswith('continue')
    assert 'st.button(' not in branch
