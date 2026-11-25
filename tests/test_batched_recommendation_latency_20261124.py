"""Batch automatic verification without weakening the review publication gate."""
import json
from threading import Event
from urllib.error import URLError
import pytest
from healthcare_data_quality.services import model_plan_review as mr, llm_client as llm
from healthcare_data_quality.ui import review_lookahead as la
from healthcare_data_quality.ui.human_choice_replay import memory_first_candidates
from test_automatic_final_actions_20261124 import SRC, render_state
from test_review_lookahead_20261124 import setup


def verdict(oid, *, supported=True):
    return {'option_id':oid,'assessment':'supported' if supported else 'reject',
        'finding_scope_matches':supported,'contains_unrequested_change':not supported,'reason':'Exact scope checked'}


def test_real_fractional_choices_reviewed_together(monkeypatch,tmp_path):
    from healthcare_data_quality.cleaning.finding_cleaning import build_cleaning_findings
    from healthcare_data_quality.investigation.baseline_audit import run_baseline_audit
    from healthcare_data_quality.learning import interaction_memory as mem
    origin=next(f for f in build_cleaning_findings({'baseline':run_baseline_audit(SRC)},[SRC],[SRC.name],[]) if 'fractional' in f['title'])
    log=tmp_path/'memory.jsonl'
    monkeypatch.setattr(mem,'_location',lambda path=None: log)
    mem.save_reviewed_solution(origin,SRC,operations=[{'operation':'replace_cells','changes':[
        {'row':11,'column':7,'before':'0.2','after':'2'}, {'row':11,'column':11,'before':'0.5','after':'5'}]}],path=log)
    mem.save_reviewed_solution(origin,SRC,operations=[{'operation':'drop_rows','changes':[{'row':11,'before':origin['evidence'][0]['Taxon / feature']}]}],path=log)
    opts=memory_first_candidates(origin,SRC)
    assert len(opts)>=2
    assert any(o['operations'][0]['operation']=='flag_cells' for o in opts)
    calls=[]
    def model(prompt, **kwargs):
        data=json.loads(prompt.split('DATA: ',1)[1]); calls.append(data)
        assert any(op['affected_count']==106 for c in data['candidates'] for op in c['proposed_action']['operations'] if op['operation']=='flag_cells')
        return json.dumps({'reviews':[verdict(c['option_id']) for c in reversed(data['candidates'])]})
    monkeypatch.setattr(mr,'query_local_model',model)
    reviews=mr.assess_exact_options(origin,SRC,opts,{})
    assert len(calls)==1 and len(reviews)==len(opts)
    assert [r['fingerprint'] for r in reviews]==[mr.option_fingerprint(origin,SRC,o) for o in opts]
    assert all(r['can_offer'] for r in reviews)
    app,ui=render_state(monkeypatch,origin,{'suggestion':'','choices':[],'gemma_option_reviews':reviews})
    phase,_,ready=app._prepare_initial_review_choices(origin['id'],origin,SRC,{})
    assert phase=='ready' and len(ready)==len(opts)
    assert not ui.session_state.cleaning_history


def test_invalid_batch_id_never_authorizes_other_option(monkeypatch):
    state,target,_=setup(monkeypatch); opts=memory_first_candidates(target,SRC)
    monkeypatch.setattr(mr,'query_local_model',lambda *a,**k:json.dumps({'reviews':[
        verdict('option_1'),verdict('option_1'),verdict('unknown'),verdict('option_2',supported=False)]}))
    reviews=mr.assess_review_batch(target,SRC,opts)
    assert len(reviews)==1
    assert reviews[0]['fingerprint']==mr.option_fingerprint(target,SRC,opts[1])
    assert not reviews[0]['can_offer']
    assert not any(o['gemma_review_status'] in {'supported','disputed'} for o in mr.reviewer_selectable_choices(target,SRC,opts,reviews))


def test_partial_batch_retries_only_missing_option(monkeypatch):
    state,target,_=setup(monkeypatch); opts=memory_first_candidates(target,SRC)
    monkeypatch.setattr(mr,'query_local_model',lambda *a,**k:json.dumps({'reviews':[verdict('option_1')]}))
    calls=[]
    def individual(f,s,o,**kw):
        fp=mr.option_fingerprint(f,s,o);calls.append(fp)
        return {'fingerprint':fp,'can_offer':True,'scope_matches':True,'has_extra_changes':False}
    monkeypatch.setattr(mr,'assess_review_option',individual)
    result=mr._assess_exact_options(target,SRC,opts,{})
    assert calls==[mr.option_fingerprint(target,SRC,opts[1])]
    assert len(result)==2 and all(r['can_offer'] for r in result)


def test_lookahead_sends_all_candidates_in_one_assessment(monkeypatch):
    state,target,_=setup(monkeypatch); calls=[]
    def assess(f,s,opts,c):
        calls.append(len(opts))
        return [{'fingerprint':mr.option_fingerprint(f,s,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in opts]
    monkeypatch.setattr(mr,'assess_exact_options',assess)
    result=la.prepare(state,target,SRC,Event())
    assert calls==[2] and result['ready_count']==2


def test_automatic_recommendation_budget_expires_during_reconnect(monkeypatch):
    clock=[0.]; attempts=[]
    monkeypatch.setattr(llm,'monotonic',lambda:clock[0])
    monkeypatch.setattr(llm,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(llm,'_GATE',llm._RequestGate())
    monkeypatch.setenv('HDQ_RECOMMENDATION_BUDGET_SECONDS','15')
    monkeypatch.setenv('HDQ_GEMMA_AUTO_RECONNECT','1')
    def unavailable(req,timeout):
        attempts.append(timeout);raise URLError('offline')
    monkeypatch.setattr(llm,'urlopen',unavailable)
    with llm.bounded_recommendation_review():
        with pytest.raises(TimeoutError,match='recommendation assessment timed out'):
            llm.query_local_model('short model request')
    assert clock[0]==15 and len(attempts)==3
    assert llm._GATE._active==0
    assert llm._REVIEW_DEADLINE.get() is None
