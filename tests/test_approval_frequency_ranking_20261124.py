from pathlib import Path
import json
import pytest
from healthcare_data_quality.learning import interaction_memory as mem
from healthcare_data_quality.ui.human_choice_replay import (
    action_key, approval_counts, rank_choices, verified_approved_choices,
    prioritize_verified_human_choices,
)
from healthcare_data_quality.services.recommendation_options import source_backed_choices
from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
from healthcare_data_quality.services.model_plan_review import option_fingerprint, reviewer_selectable_choices

@pytest.fixture
def case(tmp_path, monkeypatch):
    p = tmp_path/'data.csv'; p.write_text('taxonomy,A,B\nx,0.2,0.5\ny,1,2\n')
    log = tmp_path/'review.jsonl'
    monkeypatch.setattr(mem, '_location', lambda path=None: Path(path) if path else log)
    f = {'id':'dataset_1:qc:1','dataset_id':'dataset_1','check_key':'find_invalid_counts',
         'title':'2 cells contain fractional count values','evidence':[
             {'CSV row':2,'Column':2,'Original value':'0.2'},
             {'CSV row':2,'Column':3,'Original value':'0.5'}]}
    return p, f, log

def option(value):
    return {'label':f'Set value {value}', 'operations':[{'operation':'replace_cells','changes':[
        {'row':2,'column':2,'before':'0.2','after':str(value)}]}]}

def approve(p,f,log,o,n):
    for _ in range(n):
        mem.save_reviewed_solution(f,p,operations=o['operations'],path=log)

def test_frequency_before_recency_and_five_cap(case):
    p,f,log=case
    opts=[option(i) for i in range(1,8)]
    approve(p,f,log,opts[0],6); approve(p,f,log,opts[1],3); approve(p,f,log,opts[-1],1)
    ranked=rank_choices(f,p,list(reversed(opts))+[opts[0]])
    assert len(ranked)==5
    assert [o['label'] for o in ranked[:3]]==[opts[0]['label'],opts[1]['label'],opts[-1]['label']]
    assert [o['historical_approval_count'] for o in ranked[:3]]==[6,3,1]

def test_metadata_and_duplicate_memory_ids_not_extra_votes(case):
    p,f,log=case; o=option(2)
    original=mem.save_reviewed_solution(f,p,operations=o['operations'],path=log)
    with log.open('a') as stream: stream.write(json.dumps(original)+'\n')
    enriched=option(2); enriched['operations'][0]['changes'][0]['source_header']='A'
    approve(p,f,log,enriched,1)
    assert approval_counts(f,p)[action_key(o['operations'])]==2

def test_revoked_approval_not_replayed_but_count_is_historical(case):
    p,f,log=case; o=option(2); approve(p,f,log,o,4)
    mem.revoke_reviewed_solution(f,p,path=log)
    assert verified_approved_choices(f,p)==[]
    assert rank_choices(f,p,[o])[0]['historical_approval_count']==4

def test_no_cross_dataset_or_issue_vote_leak(case,tmp_path):
    p,f,log=case; o=option(2); approve(p,f,log,o,3)
    other=tmp_path/'other.csv'; other.write_text(p.read_text()+'z,3,4\n')
    assert not approval_counts(f,other)
    assert not approval_counts({**f,'title':'Other issue'},p)

def test_gemma_check_ranked_and_approval_gate_preserved(case):
    p,f,log=case; f={**f,'check_key':'gemma_check','id':'dataset_1:gemma:1'}
    opts=[option(2),option(3)]; approve(p,f,log,opts[1],4)
    ranked=rank_choices(f,p,scoped_review_choices(f,p,opts))
    assert ranked[0]['label']==opts[1]['label']
    reviews=[{'fingerprint':option_fingerprint(f,p,ranked[1]),'can_offer':True}]
    visible=reviewer_selectable_choices(f,p,ranked,reviews)
    assert [c['gemma_review_status'] for c in visible]==['not_assessed','supported']

def test_negative_counts_two_source_backed_options(tmp_path):
    p=tmp_path/'negative.csv';p.write_text('taxonomy,A\nx,-5\ny,3\n')
    f={'id':'dataset_1:qc:3','dataset_id':'dataset_1','check_key':'find_invalid_counts',
       'title':'1 cell contains negative count values','evidence':[{'CSV row':2,'Column':2,'Original value':'-5'}], 'total_evidence_records':1}
    opts=source_backed_choices(f,p,{})
    assert {o['operations'][0]['changes'][0]['after'] for o in opts}=={'5','NA'}

def test_five_clarifications_and_schema():
    from healthcare_data_quality.cleaning.cleaning_workspace import agent_c_clarification_options
    from healthcare_data_quality.ui.cleaning_review import RECOMMENDATION_SCHEMA
    labels=['Delete row','Keep row','Set to NA','Use absolute values','Enter values']
    assert agent_c_clarification_options('',labels)==labels
    assert RECOMMENDATION_SCHEMA['properties']['recommendations']['items']['properties']['choices']['maxItems']==5
