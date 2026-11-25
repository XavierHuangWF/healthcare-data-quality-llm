"""Regression: per-finding owner attribution and placement of terminal review action."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from healthcare_data_quality.ui.finding_wait_gate import (
    assigned_suggestion_agent,
    waiting_card_html,
)


@pytest.mark.parametrize(('entry', 'owner'), [
    ({'status': 'pending'}, None),
    ({'status': 'running', 'worker_role': 'agent_b'}, 'Agent B'),
    ({'status': 'running', 'worker_role': 'agent_c_idle_assist'}, 'Agent C'),
    ({'status': 'running', 'worker_role': 'agent_a'}, 'Agent A'),
    ({'status': 'ready', 'assisted_by_agent_c': True}, 'Agent C'),
    ({'status': 'ready', 'assisted_by_agent_c': False}, 'Agent B'),
    ({'status': 'failed', 'assisted_by_agent_c': True}, 'Agent C'),
    ({'status': 'running', 'worker_role': 'unrecognized'}, None),
])
def test_attribution_never_guesses_worker(entry, owner):
    assert assigned_suggestion_agent(entry) == owner


def test_pending_is_not_false_attributed_to_agent_b():
    html = waiting_card_html('Finding 1', 'Counts', 'pending', entry={'status': 'pending'})
    assert 'awaiting the next available suggestion worker' in html
    assert 'Agent B is preparing' not in html


def test_in_flight_worker_labels_follow_atomic_queue_claim():
    for role, expected in (('agent_b', 'Agent B'),
                           ('agent_c_idle_assist', 'Agent C'),
                           ('agent_a', 'Agent A')):
        html = waiting_card_html('Finding', 'Counts', 'running', entry={
            'status': 'running', 'worker_role': role})
        assert f'{expected} is preparing a suggestion' in html
        assert '🏃‍♂️' in html


def test_ready_and_failed_agent_c_owner_not_lost():
    assert assigned_suggestion_agent({'status': 'ready', 'assisted_by_agent_c': True}) == 'Agent C'
    html = waiting_card_html('Gemma check', 'Controls', 'failed', entry={
        'status': 'failed', 'assisted_by_agent_c': True,
        'error': 'private diagnostics must never appear in the title'})
    assert 'Agent C retry details' in html
    assert 'private diagnostics' not in html
    assert '🏃‍♂️' not in html


def test_universal_close_button_is_at_bottom_not_after_evidence():
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    parsed = ast.parse(source)
    funcs = {node.name: node for node in parsed.body
             if isinstance(node, ast.FunctionDef)}
    # One stable closure widget, rendered via a helper rather than once per branch.
    assert source.count("key=f'review_close_any_{fid}'") == 1
    assert '_render_bottom_review_closure' in funcs
    review_fun = next(f for name, f in funcs.items()
                      if name.startswith('render_cleaning_') and 'workspace' in name)
    # All main finding renderer invocation sites must appear below the Agent B
    # suggestions and chat controls, and the early visual-flag branch must also
    # call it after its own No — Discuss button.
    snippet = ast.get_source_segment(source, review_fun)
    assert snippet is not None
    # The fourth mutually exclusive path is the new discussion-only state.
    assert snippet.count('_render_bottom_review_closure(fid, finding, source, applied)') == 4
    assert (snippet.index('key=f\'flag_no_{fid}_')
            < snippet.index('_render_bottom_review_closure(fid, finding, source, applied)',
                            snippet.index('key=f\'flag_no_{fid}_')))
    assert (snippet.index('_render_live_agent_b_finding(')
            < snippet.rindex('_render_bottom_review_closure(fid, finding, source, applied)'))
    assert (snippet.rindex("'No — Discuss Another Idea'")
            < snippet.rindex('_render_bottom_review_closure(fid, finding, source, applied)'))


def test_special_prior_resolution_close_is_after_discuss():
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    parsed = ast.parse(source)
    fun = next(n for n in parsed.body if isinstance(n, ast.FunctionDef)
               and n.name == '_render_prior_approval_closure')
    section = ast.get_source_segment(source, fun)
    assert section is not None
    assert section.index("discuss = st.button('No — Discuss Another Idea'") < section.index(
        "finish = st.button(str(previous['label'])")


def test_worker_ownership_persists_from_ready_queue_to_display():
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    assert source.count('"_worker_role": (entry.get("worker_role") if entry.get("worker_role")') == 2
    assert 'assigned_suggestion_agent(entry' in source
    assert 'entry=_hdq_entry' in source
