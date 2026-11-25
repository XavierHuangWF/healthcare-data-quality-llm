"""Regression coverage for reviewer closure and superseded annotation projection."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from healthcare_data_quality.cleaning.active_visual_flags import split_active_visual_flags
from healthcare_data_quality.cleaning.review_closure import (
    requests_review_closure, closure_button_label,
)

APP = Path(__file__).resolve().parents[1] / 'app.py'


@pytest.mark.parametrize('message', [
    'closed', 'marked and closed', 'reviewed and closed', 'done',
    'close this finding', 'close review', 'finish session', 'pass this review',
    'mark this check reviewed', 'mark it as complete', 'no further changes needed',
])
def test_human_close_commands_are_terminal_intents(message):
    assert requests_review_closure(message)


@pytest.mark.parametrize('message', [
    'close review and delete row 5', 'close review, then rename column 1',
    'do nothing, but delete column 11', 'almost done', 'delete three columns',
    '', 'done and delete row 6', 'close these rows and flag them',
])
def test_compound_edits_are_not_closures(message):
    assert not requests_review_closure(message)


def test_close_labels_distinguish_approval_from_waiver():
    assert 'leave unchanged' in closure_button_label(has_edits=False, pending_proposal=False)
    assert 'decline proposal' in closure_button_label(has_edits=False, pending_proposal=True)
    assert 'keep approved edits' in closure_button_label(has_edits=True, pending_proposal=False)
    assert 'earlier approvals' in closure_button_label(
        has_edits=False, pending_proposal=False, previous_approval_proof=True)


def test_partial_supersession_never_highlights_deleted_coordinates():
    flags = [
        {'row': 2, 'column': 1, 'before': 'A'},
        {'row': 6, 'column': 1, 'before': 'B'},
        {'row': 7, 'column': 1, 'before': 'C'},
        {'row': 9, 'column': 4, 'before': 'D'},
    ]
    rows = [{'operation': 'drop_rows', 'changes': [{'row': 2}, {'row': 7}]},
            {'operation': 'merge_rows', 'changes': [{'row': 8, 'source_row': 9}]}]
    active, superseded = split_active_visual_flags(flags, rows, [])
    assert active == [flags[1]]
    assert superseded == [flags[0], flags[2], flags[3]]
    assert len(flags) == 4  # Preserve immutable historical audit evidence.


def test_column_deletion_supersedes_flags_even_on_surviving_rows():
    flags = [{'row': 6, 'column': 3}, {'row': 6, 'column': 4}]
    active, superseded = split_active_visual_flags(
        flags, [], [{'position': 4}])
    assert active == flags[:1]
    assert superseded == flags[1:]


def _load_function(name, ns):
    tree = ast.parse(APP.read_text(encoding='utf-8'))
    fun = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fun], type_ignores=[]), str(APP), 'exec'), ns)
    return ns[name]


class SessionState(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def test_close_with_prior_edit_retains_approval_and_clears_unapproved_plan(tmp_path):
    from healthcare_data_quality.cleaning.cleaning_authorization import file_sha256
    src = tmp_path / 'source.csv'
    src.write_text('taxon,A\nalpha,1\n')
    state = SessionState({
        'cleaning_approved_reviewer': 'Researcher',
        'cleaning_history': [{'finding_id': 'f1', 'operation': 'replace_cells'}],
        'cleaning_plans_by_finding': {'f1': {'operation': 'drop_rows'}},
        'cleaning_batch_queue_by_finding': {'f1': [{'operation': 'drop_rows'}]},
        'cleaning_discussion_mode_by_finding': {'f1': 'alternative'},
        'cleaning_pending_gemma_b': {}, 'cleaning_decisions': [],
        'cleaning_review_statuses': {},
    })
    func = _load_function('_complete_general_review_closure', {
        'st': SimpleNamespace(session_state=state),
        'Path': Path,
        '_complete_reviewer_no_change': lambda *args, **kwargs: pytest.fail('Existing edit must be retained'),
    })
    func({'id': 'f1', 'dataset_id': 'dataset_1'}, src)
    assert state['cleaning_history'][0]['operation'] == 'replace_cells'
    assert state['cleaning_review_statuses']['f1'] == 'Applied and verified'
    assert 'f1' not in state['cleaning_plans_by_finding']
    assert 'f1' not in state['cleaning_batch_queue_by_finding']
    assert state['cleaning_decisions'][0]['source_sha256'] == file_sha256(src)
    assert state['cleaning_decisions'][0]['new_edits'] == 0


def test_close_without_edits_is_named_review_not_auto_correction(tmp_path):
    src = tmp_path / 'source.csv'
    src.write_text('taxon,A\nalpha,-1\n')
    state = SessionState({
        'cleaning_approved_reviewer': 'Researcher',
        'cleaning_history': [], 'cleaning_pending_gemma_b': {},
        'cleaning_decisions': [], 'cleaning_review_statuses': {},
    })

    def record(finding, source, question, *, already_in_transcript=False):
        assert 'accepted unchanged' in question
        state['cleaning_review_statuses'][finding['id']] = 'Reviewed — no change'
        state['cleaning_decisions'].append({'finding_id': finding['id'], 'reviewer': 'Researcher'})

    func = _load_function('_complete_general_review_closure', {
        'st': SimpleNamespace(session_state=state),
        'Path': Path, '_complete_reviewer_no_change': record,
    })
    func({'id': 'g1'}, src)
    assert src.read_text().endswith('alpha,-1\n')
    assert state['cleaning_review_statuses']['g1'] == 'Reviewed — no change'
    assert state['cleaning_decisions'][-1]['review_outcome'] == 'reviewer_accepted_without_correction'


def test_pending_agent_c_cannot_be_closed_while_result_is_in_flight(tmp_path):
    src = tmp_path / 'source.csv'
    src.write_text('taxon,A\nalpha,-1\n')
    state = SessionState({'cleaning_approved_reviewer': 'Researcher',
                          'cleaning_pending_gemma_b': {'g1': {'job_id': 'pending'}}})
    func = _load_function('_complete_general_review_closure', {'st': SimpleNamespace(session_state=state), 'Path': Path})
    with pytest.raises(PermissionError, match='Wait for'):
        func({'id': 'g1'}, src)


def test_app_contains_one_universal_close_button_and_surviving_flag_projection():
    source = APP.read_text(encoding='utf-8')
    assert source.count('key=f\'review_close_any_{fid}\'') == 1
    assert 'cleaning_discussion_mode_by_finding.get(fid) not in' not in source[source.index('if st.button(close_label'):source.index('if st.button(close_label')+200]
    assert '_render_visual_flag_preview(active_flags, fid, phase="completed")' in source
    assert 'flag_preview_html(approved_visual)' not in source
    assert 'current_flags, _ = split_active_visual_flags(' in source


def test_audit_explicitly_calls_human_waiver_uncorrected(tmp_path):
    from hashlib import sha256
    from healthcare_data_quality.cleaning.reviewer_audit import build_reviewer_audit
    csv_bytes = b'taxon,A\nalpha,-1\n'
    sha = sha256(csv_bytes).hexdigest()
    decision = {'finding_id': 'f1', 'reviewer': 'Researcher',
                'decision': 'no_change_authorized_or_executed',
                'source_sha256': sha, 'new_edits': 0,
                'review_outcome': 'reviewer_accepted_without_correction'}
    audited = build_reviewer_audit(
        original_csv=csv_bytes, cleaned_csv=csv_bytes,
        original_filename='source.csv', history=[], decisions=[decision],
        review_statuses={'f1': 'Reviewed — no change'}, conversations={},
        approved_row_operations=[], approved_column_deletions=[],
        reviewer='Researcher', finding_titles={'f1': 'Negative counts'})
    outcomes = [r for r in audited['change_reconciliation']
                if r['operation'] == 'review_accepted_unchanged']
    assert len(outcomes) == 1
    assert outcomes[0]['outcome'] == 'Reviewed — accepted unchanged; not corrected'
    assert audited['cleaned_sha256'] == audited['original_sha256']
    with pytest.raises(PermissionError, match='source CSV'):
        build_reviewer_audit(
            original_csv=csv_bytes, cleaned_csv=csv_bytes,
            original_filename='source.csv', history=[],
            decisions=[{**decision, 'source_sha256': 'tampered'}],
            review_statuses={'f1': 'Reviewed — no change'}, conversations={},
            approved_row_operations=[], approved_column_deletions=[],
            reviewer='Researcher')


def test_close_rejects_finding_with_mismatched_source_hash(tmp_path):
    src = tmp_path / 'source.csv'
    src.write_text('taxon,A\nalpha,-1\n')
    state = SessionState({'cleaning_approved_reviewer': 'Researcher',
                          'cleaning_pending_gemma_b': {}})
    func = _load_function('_complete_general_review_closure', {
        'st': SimpleNamespace(session_state=state), 'Path': Path})
    with pytest.raises(PermissionError, match='Source CSV changed'):
        func({'id': 'g1', 'source_sha256': 'another source'}, src)


def test_prior_resolution_still_has_a_human_closure_when_model_fails():
    text = APP.read_text(encoding='utf-8')
    assert 'review_resolution_model_fallback_' in text
    assert "_complete_prior_approval_closure(finding, source, resolution)" in text
