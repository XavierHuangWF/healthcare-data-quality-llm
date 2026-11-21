"""Automatic isolated R previews and one final human authorization."""
import csv
from pathlib import Path
from types import SimpleNamespace
from io import StringIO

import pytest

from test_two_click_r_flow import setup_review, FINDING
from healthcare_data_quality.cleaning.finding_cleaning import validate_scoped_plan
from healthcare_data_quality.r_runtime.scoped_r_execution import (
    prepare_and_execute_scoped_r_preview,
    verify_prepared_r_insertion,
)


def _mock_r(monkeypatch, output=None):
    import healthcare_data_quality.r_runtime.scoped_r_execution as scoped
    calls = []
    def fake_run(r_code, input_csv, *, preserve_text=False):
        calls.append((r_code, input_csv, preserve_text))
        assert preserve_text is True
        return output if output is not None else 'kind,csv_row,csv_column,value\ncell,2,2,NA\n'
    monkeypatch.setattr(scoped, 'run_r_sandbox', fake_run)
    return calls


def test_preapproval_r_result_and_insertion_code_generated_without_working_csv_edit(monkeypatch, tmp_path):
    app, st, source, render = setup_review(monkeypatch, tmp_path)
    calls = _mock_r(monkeypatch)
    before = source.read_bytes()
    render()
    fid = FINDING['id']
    prepared = st.session_state.cleaning_r_previews_by_finding[fid]
    assert len(calls) == 1
    assert 'df$value <- df$replacement' in prepared['r_function']
    assert '"NA"' not in prepared['r_function']
    assert calls[0][1].splitlines()[0] == 'kind,csv_row,csv_column,value,replacement'
    assert 'for (col in unique(cols))' in prepared['insertion_r_function']
    assert 'df[[col]][rows[i]] <- verified_changes$replacement[i]' in prepared['insertion_r_function']
    assert 'df[[2]][[1]] <- "NA"' not in prepared['insertion_r_function']
    assert list(csv.reader(StringIO(prepared['insertion_changes_csv'])))[-1] == ['cell', '2', '2', '0.2', 'NA']
    assert prepared['r_result']['verified_output_csv'].splitlines()[-1] == 'cell,2,2,NA'
    assert prepared['insertion_plan']['changes'][0]['after'] == 'NA'
    assert st.session_state.cleaning_working_paths == {}
    assert source.read_bytes() == before
    assert st.session_state.cleaning_review_statuses[fid] == 'R output verified — insertion awaiting approval'
    assert not st.errors
    render()
    assert len(calls) == 1, 'rerenders must not run R again'


def test_approval_commits_only_exact_verified_r_values_and_preserves_original(monkeypatch, tmp_path):
    app, st, source, render = setup_review(monkeypatch, tmp_path)
    calls = _mock_r(monkeypatch)
    before = source.read_bytes()
    render()
    st.pressed.add('Approve & Stage in memory')
    with pytest.raises(StopIteration, match='rerun'):
        render()
    assert len(calls) == 1, 'R transformation is precomputed, not rerun after approval'
    assert source.read_bytes() == before
    assert st.session_state.cleaning_working_paths == {}
    assert st.session_state.cleaning_result is None
    assert not (tmp_path / 'cleaning-workspace').exists()
    data = list(csv.reader(StringIO(st.session_state.cleaning_staged_bytes_by_dataset['dataset_1'].decode('utf-8'))))
    assert data == [['Taxon', 'Count'], ['a', 'NA']]
    assert st.session_state.cleaning_review_statuses[FINDING['id']] == 'Applied and verified'
    assert st.session_state.cleaning_history[0]['changes'] == 1
    assert not st.errors


def test_missing_reviewer_never_commits_working_csv(monkeypatch, tmp_path):
    app, st, source, render = setup_review(monkeypatch, tmp_path)
    _mock_r(monkeypatch)
    render()
    st.pressed.add('Approve & Stage in memory')
    st.reviewer = ''
    st.session_state.cleaning_approved_reviewer = ''
    render()
    assert st.session_state.cleaning_working_paths == {}
    assert st.session_state.cleaning_result is None
    assert not st.session_state.cleaning_history  # no identity means no approval action


def test_r_output_with_unapproved_value_is_rejected(monkeypatch, tmp_path):
    app, st, source, render = setup_review(monkeypatch, tmp_path)
    _mock_r(monkeypatch, 'kind,csv_row,csv_column,value\ncell,2,2,zero\n')
    before = source.read_bytes()
    render()
    assert st.session_state.cleaning_working_paths == {}
    assert st.session_state.cleaning_r_previews_by_finding == {}
    assert 'Isolated R preview failed' in st.session_state.cleaning_r_errors_by_finding[FINDING['id']]
    assert source.read_bytes() == before


def test_source_change_after_preview_blocks_insertion(monkeypatch, tmp_path):
    app, st, source, render = setup_review(monkeypatch, tmp_path)
    _mock_r(monkeypatch)
    render()
    source.write_text('Taxon,Count\na,0.3\n', encoding='utf-8')
    st.pressed.add('Approve & Stage in memory')
    render()
    assert st.session_state.cleaning_working_paths == {}
    assert st.session_state.cleaning_result is None
    assert any('In-memory insertion failed' in e for e in st.errors)


def test_obsolete_yes_button_cannot_approve_an_unverified_suggestion(monkeypatch, tmp_path):
    # Agent B no longer exposes a prose-only "Yes" control. A source-backed
    # choice and one named human approval are required.
    app, st, source, render = setup_review(monkeypatch, tmp_path)
    original = source.read_bytes()
    fid = FINDING['id']
    st.session_state.cleaning_plans_by_finding.clear()
    st.session_state.cleaning_discussion_mode_by_finding.clear()
    queued = []
    monkeypatch.setattr(app, '_queue_agent_c_for_finding',
                        lambda *args, **kwargs: queued.append(args))
    st.buttons_on.add(f'review_yes_{fid}')  # old, unsupported widget key
    render()
    assert queued == []
    assert st.session_state.cleaning_plans_by_finding == {}
    assert st.session_state.cleaning_history == []
    assert source.read_bytes() == original

