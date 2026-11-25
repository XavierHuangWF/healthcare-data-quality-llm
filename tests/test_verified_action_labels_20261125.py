"""Labels must be generated from verified actions, not model/memory strings."""
from copy import deepcopy
from pathlib import Path
import csv
import pytest
from healthcare_data_quality.ui.clear_review_actions import final_action_label, option_action_label
from healthcare_data_quality.services.model_plan_review import option_fingerprint
from test_proposal_evidence_card_20261124 import CardUI, SRC
from test_cleaning_review_cards import _app


def taxonomy_options():
    with SRC.open(newline='',encoding='utf-8-sig') as f:
        before=list(csv.reader(f))[6][0]
    from healthcare_data_quality.cleaning.direct_actions import _standard_taxonomy_value
    after=_standard_taxonomy_value(before,dominant='unprefixed')
    assert after and '|' not in after
    outputs=[after,after.replace('; ',';'),_standard_taxonomy_value(before,dominant='prefixed')]
    return [{'label':before[:90], 'operations':[{'operation':'replace_cells','changes':[
        {'row':7,'column':1,'before':before,'after':a}]}],
        'rows':[{'Location':'R7:C1','Before':before,'After':a}]} for a in outputs]


@pytest.mark.parametrize('key',['inspect_taxonomy_structure','inspect_taxonomy_naming','gemma_check'])
def test_real_taxonomy_options_label_from_after(key):
    finding={'id':'dataset_1:qc:6','dataset_id':'dataset_1','check_key':key}
    options=taxonomy_options(); labels=[]
    for option in options:
        original=deepcopy(option); fingerprint=option_fingerprint(finding,SRC,option)
        label=option_action_label(option,finding)
        assert option==original and option_fingerprint(finding,SRC,option)==fingerprint
        assert label==final_action_label(option['operations'],finding)
        assert 'Bacteria' not in label and '|' not in label and len(label)<100
        labels.append(label)
    assert len(set(labels))==3
    assert 'Unprefixed' in labels[0] and 'with spaces' in labels[0]
    assert 'without spaces' in labels[1] and 'Fully prefixed' in labels[2]


def test_actual_agent_b_button_uses_verified_operation(monkeypatch):
    ui=CardUI(); app=_app(monkeypatch,ui)
    fid='dataset_1:qc:6'; f={'id':fid,'dataset_id':'dataset_1','title':'Inconsistent separators','check_key':'inspect_taxonomy_structure'}
    ui.session_state.update(cleaning_approved_reviewer='Reviewer',cleaning_sequential_active=fid,cleaning_review_catalog_snapshot=[f])
    options=taxonomy_options()
    app._render_live_agent_b_finding(fid,f,SRC,[],[],None,'dataset_1',prepared=('ready',{'suggestion':'Format correction'},options))
    buttons=[e for e in ui.events if e[1]=='button' and e[0]]
    assert [e[2] for e in buttons]==[option_action_label(o,f) for o in options]
    assert all(any(t[0]==b[0] and t[1]=='table' for t in ui.events) for b in buttons)


@pytest.mark.parametrize('plans,expected',[
    ([{'operation':'drop_column','changes':[{'position':2},{'position':3}]}],'Delete columns 2, 3'),
    ([{'operation':'drop_rows','changes':[{'row':16},{'row':24}]}],'Delete rows 16, 24'),
    ([{'operation':'rename_headers','changes':[{'position':1,'after':'Taxonomy'}]}],'Rename column 1'),
    ([{'operation':'merge_rows','changes':[{'row':11,'source_row':12}]}],'Merge row 12 into 11'),
    ([{'operation':'flag_cells','visual_scope':'rows','changes':[{'row':11,'column':1,'fill':False}]}],'Highlight 1 entire row (1 cell) · red/bold only'),
    ([{'operation':'drop_rows','changes':[{'row':7}]},{'operation':'drop_column','changes':[{'position':4}]}],'Delete row 7 + Delete column 4'),
])
def test_all_operations_ignore_stale_titles(plans,expected):
    assert option_action_label({'label':'a stale arbitrary value','operations':plans})==expected
