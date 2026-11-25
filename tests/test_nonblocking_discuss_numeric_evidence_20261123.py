"""Regression coverage: no full-page bounce and source-based count comparisons."""
from __future__ import annotations

import csv
import json
from threading import Event

import pytest

from healthcare_data_quality.cleaning.numeric_context import (
    compare_evidenced_numeric_cells, display_numeric_comparisons,
)
from healthcare_data_quality.ui import async_review_feedback


def _finding():
    return {'id':'dataset_1:qc:3', 'dataset_id':'dataset_1',
            'source':'QC', 'title':'2 negative counts',
            'check_key':'find_invalid_counts', 'evidence': [
                {'CSV row':3, 'Column':2, 'Original value':'-679'},
                {'CSV row':4, 'Column':3, 'Original value':'-4133'},
            ]}


def test_numeric_comparison_uses_other_rows_and_never_changes_data(tmp_path):
    path = tmp_path / 'counts.csv'
    data = [['Taxon','AbC10','ABC105'],
            ['a','0','10'], ['b','-679','0'], ['c','2','-4133'],
            ['d','100','200'], ['e','3000','8000'], ['f','NA','']]
    with path.open('w',newline='') as f: csv.writer(f).writerows(data)
    original = path.read_bytes()
    comparison = compare_evidenced_numeric_cells(path, _finding())
    assert len(comparison) == 2
    first, second = comparison
    assert (first['CSV row'],first['Column'], first['Original']) == (3,2,'-679')
    assert first['Other zeros'] == 1
    assert first['Other positive observations'] == 3
    assert first['Positive maximum'] == 3000
    assert first['Candidate exceeds observed positive maximum'] is False
    assert first['Positive median'] == 100
    assert second['Other positive observations'] == 3
    assert second['Positive maximum'] == 8000
    assert second['Candidate exceeds observed positive maximum'] is False
    assert display_numeric_comparisons(comparison)[0]['Location'] == 'R3:C2 (AbC10)'
    assert path.read_bytes() == original


def test_numeric_context_cannot_guess_missing_targets(tmp_path):
    p=tmp_path/'tiny.csv'
    p.write_text('taxon,value\na,1\n',encoding='utf8')
    f=_finding()
    f['evidence']=[{'Column':2,'Original value':'-55'}]
    assert compare_evidenced_numeric_cells(p,f)==[]


def test_advisory_memory_fs_sync_does_not_block_click(monkeypatch):
    gate=Event()
    began=Event()
    completed=Event()
    def slow_write(*a,**k):
        began.set()
        assert gate.wait(3), 'Test timed out waiting for release'
        completed.set()
    monkeypatch.setattr('healthcare_data_quality.learning.adaptive_skill_memory.record_review',slow_write)
    try:
        async_review_feedback.record_decline_without_blocking(_finding())
        assert began.wait(2)
        assert not completed.is_set()
    finally:
        gate.set()
    assert completed.wait(3)


def test_agent_b_discuss_reruns_fragment_not_app(monkeypatch):
    from test_cleaning_review_cards import StubUI, _app
    st=StubUI()
    app=_app(monkeypatch,st)
    called=[]
    def rerun(*,scope=None):
        called.append(scope)
        raise StopIteration('rerun')
    st.rerun=rerun
    with pytest.raises(StopIteration,match='rerun'):
        app._rerun_parent_after_agent_b_choice('dataset_1:qc:3')
    assert called == ['fragment']
    assert st.session_state['cleaning_active_finding']=='dataset_1:qc:3'


def test_gemma_receives_column_evidence_instead_of_defaulting_to_absolute(monkeypatch):
    from healthcare_data_quality.cleaning import cleaning_workspace
    prompts=[]
    def fake(text, **kwargs):
        prompts.append(text)
        return json.dumps({'reply':'Other counts span 2 to 3000; absolute 679 is within that observed range, but not a verified correction.',
                           'operation':'none','dataset_id':'','changes':[],'clarification_options':[]})
    monkeypatch.setattr(cleaning_workspace,'query_local_model',fake)
    finding={'id':'d1', 'title':'2 negative counts',
             'numeric_column_comparisons':[{'CSV row':3,'Column':2,'Positive maximum':3000,
                 'Positive median':100, 'Other positive observations':3,
                 'Absolute-value candidate (not approved)':'679'}]}
    result=cleaning_workspace.ask_gemma_about_cleaning(
        'Are these values in the normal range?',[],[finding],[],
        selected_finding=finding)
    assert result['operation']=='none'
    assert 'not a verified correction' in result['reply']
    assert 'numeric_column_comparisons' in prompts[0]
    assert 'same' in prompts[0].lower()


def test_discuss_preserves_other_finding_open_states(monkeypatch):
    from test_cleaning_review_cards import StubUI, _app
    st=StubUI()
    app=_app(monkeypatch, st)
    st.session_state['cleaning_keep_open_by_finding']={'another_finding':True}
    def rerun(*,scope=None):
        assert scope == 'fragment'
        raise StopIteration('rerun')
    st.rerun=rerun
    with pytest.raises(StopIteration,match='rerun'):
        app._rerun_parent_after_agent_b_choice('dataset_1:qc:3')
    assert st.session_state['cleaning_keep_open_by_finding'] == {
        'another_finding':True, 'dataset_1:qc:3':True}
