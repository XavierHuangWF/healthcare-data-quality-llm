"""Regression tests: the clean-data page opens while each waiting card stays locked."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util

from healthcare_data_quality.ui.finding_wait_gate import (
    state_by_finding, is_locked, waiting_card_html, should_refresh_for_new_results,
    poll_finding_readiness,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_installer():
    spec = importlib.util.spec_from_file_location(
        "gate_installer", ROOT / 'tools' / 'apply_per_finding_suggestion_gate.py')
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    return installer


def _snapshot(states):
    return {'entries': {k: {'status': s, 'suggestion': 'advice'} for k, s in states.items()}}


def test_pending_cards_never_expose_expand_control():
    pending = {'entries': {'f1': {'status':'queued'},'f2': {'status':'running'},
                           'f3': {'status':'ready'},'f4': {'status':'failed'}}}
    assert state_by_finding(pending) == {
        'f1':'queued', 'f2':'running','f3':'ready','f4':'failed'}
    for fid in ('f1','f2','f4','unknown'):
        assert is_locked(finding_id=fid, snapshot=pending,
                         completed=False, active_conversation=False)
    assert not is_locked(finding_id='f3', snapshot=pending,
                         completed=False, active_conversation=False)
    assert not is_locked(finding_id='f1', snapshot=pending,
                         completed=True, active_conversation=False)
    assert not is_locked(finding_id='f1', snapshot=pending,
                         completed=False, active_conversation=True)


def test_history_marker_does_not_unlock_unassessed_finding():
    snapshot = {'entries': {'f': {'status':'ready', 'origin':'review_memory',
                                 'suggestion': 'historic'}}}
    assert is_locked(finding_id='f',snapshot=snapshot,
                     completed=False,active_conversation=False)


def test_runner_html_safe_and_only_styled_noninteractive_row():
    html = waiting_card_html('Finding 1', '<test & unsafe>', 'running')
    assert '>↻</span>' in html
    assert 'data-motion="spin"' in html
    assert '@keyframes hdq-finding-spin' in html
    assert 'prefers-reduced-motion' in html
    assert '&lt;test &amp; unsafe&gt;' in html
    assert '<test' not in html
    for forbidden in ('<button', '<details', '<summary', 'onclick=', 'tabindex='):
        assert forbidden not in html
    failed = waiting_card_html('Gemma check 1', 'Issue', 'failed')
    assert 'Suggestion failed' in failed and '🏃‍♂️' not in failed


def test_poller_only_refreshes_on_terminal_transition():
    before = _snapshot({'f1':'queued', 'f2':'running'})
    assert not should_refresh_for_new_results(before, _snapshot({'f1':'running','f2':'running'}))
    assert should_refresh_for_new_results(before,_snapshot({'f1':'ready','f2':'running'}))
    assert should_refresh_for_new_results(before,_snapshot({'f1':'queued','f2':'failed'}))
    class St:
        def __init__(self):
            self.session_state = {}
            self.calls = []
        def rerun(self, *, scope):
            self.calls.append(scope)
    st = St()
    latest = _snapshot({'f1':'ready','f2':'running'})
    with patch('healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager',
               return_value=SimpleNamespace(snapshot=lambda x: latest)):
        poll_finding_readiness(st, 'job-1', before)
        assert st.calls == ['app']
        st.calls.clear()
        st.session_state['cleaning_discussion_mode_by_finding'] = {'f2': 'alternative'}
        poll_finding_readiness(st,'job-1',before)
        assert not st.calls


def test_uploaded_project_page_and_findings_are_independently_gated():
    source = (ROOT/'app.py').read_text(encoding='utf-8')
    assert source.count('HDQ_PER_FINDING_READY_GATE_V2') == 1
    assert '_hdq_wait_row(st, label, finding["title"]' in source
    assert 'continue\n        with st.expander(f"{label}' in source
    assert 'disabled=False,' in source
    assert 'hdq_results_nav_unlocked"] = True' in source


def test_current_source_uses_one_shared_readiness_watcher():
    source = (ROOT/'app.py').read_text(encoding='utf-8')
    assert 'st.fragment(run_every="3s")(refresh_cleaning)' in source
    assert 'st.fragment(run_every="3s")(_render_live_agent_b_finding)' not in source
    assert "poll_finding_readiness)(st, job_id" not in source


def test_gate_does_not_unblock_a_stale_memory_only_entry():
    assert is_locked(finding_id='f', snapshot={"entries": {
        'f': {'origin': 'review_memory', 'status': 'ready'}}},
        completed=False, active_conversation=False)
