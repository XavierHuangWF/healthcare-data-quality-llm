"""A verified staged finding is terminal UI state, not a new conversation."""
from types import SimpleNamespace

import pytest

from test_two_click_r_flow import FINDING, setup_review
from test_r_auto_preview_insertion_flow import _mock_r


def _finish_one(monkeypatch, tmp_path):
    app, ui, source, render = setup_review(monkeypatch, tmp_path)
    _mock_r(monkeypatch)
    render()
    ui.pressed.add('Approve & Stage in memory')
    with pytest.raises(StopIteration, match='rerun'):
        render()
    ui.pressed.clear()
    return app, ui, source, render


def test_successfully_staged_finding_auto_collapses_and_hides_actions(monkeypatch, tmp_path):
    app, ui, original, render = _finish_one(monkeypatch, tmp_path)
    fid = FINDING['id']
    recorded_expanders = []
    original_expander = ui.expander
    def track_expander(label, **kwargs):
        recorded_expanders.append((label, kwargs.get('expanded')))
        return original_expander(label, **kwargs)
    ui.expander = track_expander
    button_keys = []
    original_button = ui.button
    def track_button(label, *, key=None, **kwargs):
        button_keys.append(key)
        return original_button(label, key=key, **kwargs)
    ui.button = track_button
    forms = []
    original_form = ui.form
    def track_form(*args, **kwargs):
        forms.append(args)
        return original_form(*args, **kwargs)
    ui.form = track_form
    render()
    assert recorded_expanders[0][0].endswith('· Staged ✓')
    assert recorded_expanders[0][1] is False
    assert ui.session_state.cleaning_review_statuses[fid] == 'Applied and verified'
    assert not any(key and key.startswith(('review_yes_', 'review_no_', 'review_reject_',
                                             'review_no_action_', 'review_retry_r_'))
                   for key in button_keys)
    assert not forms
    assert ui.session_state.cleaning_r_previews_by_finding == {}
    assert ui.session_state.cleaning_completed_audits_by_finding[fid]['changes'][0]['after'] == 'NA'
    assert original.read_text(encoding='utf-8').endswith('a,0.2\n')


def test_completed_audit_can_show_r_code_without_reopening_editor(monkeypatch, tmp_path):
    app, ui, source, render = _finish_one(monkeypatch, tmp_path)
    from test_cleaning_review_cards import Context
    ui.chat_message = lambda *_args, **_kwargs: Context()
    ui.session_state.cleaning_messages_by_finding[FINDING['id']] = [
        {'role': 'user', 'content': 'Approve cleaning'},
        {'role': 'assistant', 'content': 'Structured plan prepared'},
    ]
    ui.checkbox = lambda label, **kwargs: label == 'Show completed finding audit'
    buttons = []
    original_button = ui.button
    def track_button(label, *, key=None, **kwargs):
        buttons.append(label)
        return original_button(label, key=key, **kwargs)
    ui.button = track_button
    render()
    assert len(ui.r_code) == 2
    assert 'clean_dataset <- function(df)' in ui.r_code[0]
    assert 'df[[col]][rows[i]] <- verified_changes$replacement[i]' in ui.r_code[1]
    assert 'Yes — prepare this action' not in buttons
    assert 'No — discuss another action' not in buttons
    assert 'Mark reviewed — no change required now' not in buttons
    assert source.read_text(encoding='utf-8').endswith('a,0.2\n')


def test_late_agent_c_response_cannot_reopen_staged_finding(monkeypatch, tmp_path):
    app, ui, source, render = _finish_one(monkeypatch, tmp_path)
    fid = FINDING['id']
    before = dict(ui.session_state.cleaning_review_statuses)
    app._record_gemma_b_response(fid, FINDING, source, {
        'operation': 'replace_cells',
        'reply': 'Another proposal',
        'dataset_id': 'dataset_1',
        'changes': [{'row': 2, 'column': 2, 'before': '0.2', 'after': '0'}],
    })
    assert ui.session_state.cleaning_review_statuses == before
    assert fid not in ui.session_state.cleaning_plans_by_finding
