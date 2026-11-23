"""Regression: reviewer intent beats generic rounding; one Yes is final approval."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from healthcare_data_quality.ui.human_choice_replay import (
    verified_approved_choices, prioritize_verified_human_choices,
)
from healthcare_data_quality.cleaning.explicit_columns import explicit_column_deletion
from healthcare_data_quality.ui.one_click_choice import execute_verified_choice
from healthcare_data_quality.learning.interaction_memory import save_reviewed_solution
from healthcare_data_quality.services.recommendation_options import source_backed_choices


def fixture_data(tmp_path):
    source = tmp_path / 'test.csv'
    source.write_text('taxonomy,x,y,z\nA,0.2,0.5,4\nB,1,2,3\n', encoding='utf8')
    finding = {'id':'dataset_1:qc:1', 'dataset_id':'dataset_1', 'title':'2 cells contain fractional count values',
               'check_key':'find_invalid_counts', 'source':'QC', 'actionable':True,
               'evidence':[{'CSV row':2,'Column':2, 'Original value':'0.2'},
                           {'CSV row':2,'Column':3, 'Original value':'0.5'}]}
    return source, finding


def test_memory_exact_decisions_first_not_incorrect_rounding(tmp_path):
    source, finding=fixture_data(tmp_path)
    saved=tmp_path/'reviews.jsonl'
    ops1=[{'operation':'replace_cells','changes':[
        {'row':2,'column':2,'before':'0.2','after':'2'}]},
        {'operation':'drop_column','changes':[{'position':3,'before':'y','after':''}]}]
    ops2=[{'operation':'replace_cells','changes':[
        {'row':2,'column':2,'before':'0.2','after':'2'},
        {'row':2,'column':3,'before':'0.5','after':'5'}]}]
    save_reviewed_solution(finding,source,operations=ops1,path=saved)
    save_reviewed_solution(finding,source,operations=ops2,path=saved)
    exact=verified_approved_choices(finding,source,path=saved)
    assert len(exact)==2
    assert {r['After'] for r in exact[0]['rows']}=={'2','5'}
    assert any(row['After']=='Delete column' for row in exact[1]['rows'])
    rounding={'label':'Round to nearest integer','operations':[
        {'operation':'replace_cells','changes':[
            {'row':2,'column':2,'before':'0.2','after':'0'},
            {'row':2,'column':3,'before':'0.5','after':'1'}]}]}
    # Replace the local memory lookup with the test's private memory file.
    from unittest.mock import patch
    with patch('healthcare_data_quality.ui.human_choice_replay.verified_approved_choices', return_value=exact):
        result=prioritize_verified_human_choices(finding,source,[rounding])
    assert len(result)==2
    assert not any('Round to nearest integer' in r['label'] for r in result)


def test_explicit_plural_whole_columns_not_cell_ambiguity(tmp_path):
    source,finding=fixture_data(tmp_path)
    response=explicit_column_deletion('deleted columns 2 and 3',finding,source)
    assert response and response['explicit_columns']==frozenset({2,3})
    assert [op['changes'][0]['position'] for op in response['operations']]==[2,3]
    assert explicit_column_deletion('do not delete columns 2 and 3',finding,source) is None
    assert explicit_column_deletion('clear cells in columns 2 and 3',finding,source) is None


def test_single_yes_atomically_approves_two_entire_column_deletions(tmp_path, monkeypatch):
    source,finding=fixture_data(tmp_path)
    selected=explicit_column_deletion('delete columns 2 and 3',finding,source)
    option={'label':'Delete entire columns 2 and 3','operations':selected['operations']}
    state={'cleaning_approved_reviewer':'Reviewer X', 'cleaning_review_statuses':{},
           'cleaning_history':[], 'cleaning_decisions':[],
           'cleaning_approved_columns_by_dataset':{}, 'cleaning_approved_rows_by_dataset':{},
           'cleaning_staged_bytes_by_dataset':{}}
    # Memory file is isolated from real project during the test.
    from healthcare_data_quality.learning import interaction_memory
    monkeypatch.setattr(interaction_memory,'_location',lambda path=None: tmp_path/'memory.jsonl')
    result=execute_verified_choice(state,finding,source,option)
    assert result['result']['export_columns']==2
    assert state['cleaning_review_statuses'][finding['id']]=='Applied and verified'
    assert len(state['cleaning_history'])==2
    assert [x['position'] for x in state['cleaning_approved_columns_by_dataset']['dataset_1']]==[2,3]
    assert source.read_text()=='taxonomy,x,y,z\nA,0.2,0.5,4\nB,1,2,3\n'
    with pytest.raises(PermissionError,match='already been reviewed'):
        execute_verified_choice(state,finding,source,option)


def test_no_reviewer_cannot_execute(tmp_path):
    source,finding=fixture_data(tmp_path)
    proposal=explicit_column_deletion('drop column 2',finding,source)
    with pytest.raises(PermissionError):
        execute_verified_choice({'cleaning_approved_reviewer':''},finding,source,
                                {'operations':proposal['operations']})


def test_actual_project_fractional_finding_can_delete_columns_7_and_11(tmp_path, monkeypatch):
    # The sample fixture from this repository has the same physical positions
    # shown in the Finding 1 UI. Never overwrite the fixture source itself.
    project_file = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'
    if not project_file.exists():
        pytest.skip('Optional example CSV was not distributed')
    source = tmp_path / 'source.csv'
    source.write_bytes(project_file.read_bytes())
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    finding = {'id': 'dataset_1:qc:1', 'dataset_id': 'dataset_1',
               'title': '2 cells contain fractional count values',
               'check_key': 'find_invalid_counts', 'source': 'QC', 'actionable': True,
               'evidence': [
                   {'CSV row': 11, 'Column': 7, 'Original value': rows[10][6]},
                   {'CSV row': 11, 'Column': 11, 'Original value': rows[10][10]},
               ]}
    response = explicit_column_deletion('deleted column 7 and 11', finding, source)
    assert response is not None
    assert [op['changes'][0]['position'] for op in response['operations']] == [7, 11]
    from healthcare_data_quality.learning import interaction_memory
    monkeypatch.setattr(interaction_memory, '_location', lambda path=None: tmp_path / 'learned.jsonl')
    state = {'cleaning_approved_reviewer': 'Xay', 'cleaning_review_statuses': {},
             'cleaning_history': [], 'cleaning_decisions': [],
             'cleaning_approved_columns_by_dataset': {}, 'cleaning_approved_rows_by_dataset': {},
             'cleaning_staged_bytes_by_dataset': {}}
    before=source.read_bytes()
    result = execute_verified_choice(state, finding, source, {'label': 'Delete columns 7 and 11',
                                                                'operations': response['operations']})
    assert result['result']['export_columns'] == len(rows[0]) - 2
    assert source.read_bytes() == before
    assert len(state['cleaning_history']) == 2
