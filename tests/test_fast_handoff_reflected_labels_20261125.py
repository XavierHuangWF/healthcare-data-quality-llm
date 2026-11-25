"""Private draft reuse and safe reflected wording, using the real source CSV."""
import json
from copy import deepcopy
from hashlib import sha256
from threading import Event

from healthcare_data_quality.ui import review_lookahead as la
from healthcare_data_quality.services import model_plan_review as mr
from healthcare_data_quality.ui.clear_review_actions import action_label_variants, option_action_label
from test_review_lookahead_20261124 import setup, replace, record
from test_automatic_final_actions_20261124 import SRC, UI
from test_batched_recommendation_latency_20261124 import verdict
from test_cleaning_review_cards import _app


def test_real_finding_7_to_8_reuses_private_assessment(monkeypatch):
    state, _, _ = setup(monkeypatch)
    origin = next(f for f in state['cleaning_review_catalog_snapshot'] if f['check_key']=='inspect_taxonomy_naming')
    target = next(f for f in state['cleaning_review_catalog_snapshot'] if f['check_key']=='compare_repeated_headers')
    rec = record(state, target)
    assert rec['result']['ready_count']
    current = replace(state, origin, 6, 1, 'Bacteria; Actinobacteria; Actinobacteria; Actinomycetales; Actinomycetaceae; Mobiluncus; curtisii')
    assert la.promote(rec, current, target, SRC) is not None
    assert la.promote(rec, replace(state,origin, 6,105,'999'), target,SRC) is None
    assert la.promote(rec, replace(state,origin, 1,2,'ABC99'), target,SRC) is None
    # A new user instruction may change the policy even if data are unrelated.
    current['cleaning_messages_by_finding']={origin['id']:[{'role':'user','content':'Never delete duplicate columns'}]}
    assert la.promote(rec,current,target,SRC) is None


def test_unknown_or_malformed_column_dependency_stays_conservative(monkeypatch):
    state,_,origin=setup(monkeypatch)
    target=next(f for f in state['cleaning_review_catalog_snapshot'] if f['check_key']=='compare_repeated_headers')
    current=replace(state,origin,11,7,'2')
    assert not la.dependencies_unchanged(state,current,{**target,'evidence':[]},SRC,[])
    assert not la.dependencies_unchanged(state,current,{**target,'check_key':'gemma_exploration'},SRC,[])


def test_next_likely_structural_branch_precedes_third_normal_job(monkeypatch):
    from test_adaptive_prefetch_status_20261124 import branch_case
    state, origin, successor, choice=branch_case(monkeypatch)
    third=next(f for f in state['cleaning_review_catalog_snapshot'] if f['check_key']=='find_duplicate_taxa')
    rec=record(state,successor)
    ui=UI(); app=_app(monkeypatch,ui)
    ui.session_state.update(state)
    ui.session_state.update(cleaning_publication_epochs={origin['id']:{'published':True}},
        cleaning_lookahead_active_choices={'fid':origin['id'],'options':[choice]},
        cleaning_lookahead_records={successor['id']:rec})
    monkeypatch.setattr(app,'_finding_review_completed',lambda fid:False)
    calls=[]
    class Manager:
        def submit_role(self,role,fn,*args):
            calls.append((role,fn));return 'branch-job' if fn is la.prepare_conditional else 'ordinary-job'
    monkeypatch.setattr('healthcare_data_quality.jobs.interactive_chat_jobs.interactive_chat_manager',lambda:Manager())
    numbered=[('Finding 2',origin),('Finding 3',successor),('Finding 4',third)]
    app._schedule_lookahead(numbered,[SRC],origin['id'])
    assert calls==[('B',la.prepare_conditional)]
    assert third['id'] not in ui.session_state['cleaning_lookahead_records']
    # While branch running, don't queue another B job. Once finished, resume.
    app._schedule_lookahead(numbered,[SRC],origin['id'])
    assert len(calls)==1
    ui.session_state['cleaning_lookahead_branch'].pop('job_id')
    app._schedule_lookahead(numbered,[SRC],origin['id'])
    assert calls[-1]==('B',la.prepare)
    assert third['id'] in ui.session_state['cleaning_lookahead_records']


def test_gemma_reflects_in_same_batch_and_cannot_invent_action(monkeypatch):
    state,target,_=setup(monkeypatch)
    options=la._candidates(state,target,SRC,{})
    assert len(options)>=2
    seen=[]
    def model(prompt,**kw):
        data=json.loads(prompt.split('DATA: ',1)[1]); seen.append(data)
        assert 'Reflect on whether' in prompt
        entries=[]
        for i,c in enumerate(data['candidates']):
            variants=c['proposed_action']['safe_button_wordings']
            entries.append({**verdict(c['option_id']),'button_label': variants[-1] if i==0 else 'Delete all samples'})
        return json.dumps({'reviews':entries})
    monkeypatch.setattr(mr,'query_local_model',model)
    reviews=mr.assess_review_batch(target,SRC,options)
    ready=mr.reviewer_selectable_choices(target,SRC,options,reviews)
    assert len(seen)==1 and len(ready)==len(options)
    assert option_action_label(ready[0],target)==action_label_variants(options[0]['operations'],target)[-1]
    assert option_action_label(ready[1],target)==action_label_variants(options[1]['operations'],target)[0]
    for old,new in zip(options,ready):
        assert mr.option_fingerprint(target,SRC,old)==mr.option_fingerprint(target,SRC,new)


def test_reflected_wording_is_learned_from_approved_memory(monkeypatch,tmp_path):
    from healthcare_data_quality.learning import interaction_memory as mem
    from healthcare_data_quality.jobs.cleaning_jobs import reviewed_method_context
    from healthcare_data_quality.ui.one_click_choice import prepare_verified_choice_snapshot, commit_verified_choice_snapshot
    state,target,_=setup(monkeypatch)
    path=tmp_path/'memory.jsonl'
    monkeypatch.setattr(mem,'_location',lambda path=None:tmp_path/'memory.jsonl')
    choice=next(o for o in la._candidates(state,target,SRC,{}) if o['operations'][0]['operation']=='drop_rows')
    name=action_label_variants(choice['operations'],target)[-1]
    choice['gemma_button_label']=name
    commit_verified_choice_snapshot(state,target,SRC,choice,prepare_verified_choice_snapshot(state,target,SRC,choice))
    assert json.loads(path.read_text().splitlines()[-1])['button_label']==name
    assert any(m.get('previous_approved_button_wording')==name for m in reviewed_method_context(target,SRC))


def test_interactive_reflection_fallback_and_stale_scope(monkeypatch):
    ui=UI();app=_app(monkeypatch,ui)
    f={'id':'x','check_key':'gemma_check'}
    plan={'operation':'drop_column','changes':[{'position':2},{'position':3}]}
    ui.session_state['cleaning_memory_candidates']={'x':{'button_label':'Remove columns 2, 3'}}
    assert app._reflected_interactive_label([plan],f)=='Remove columns 2, 3'
    assert app._reflected_interactive_label([{**plan,'changes':[{'position':2}]}],f)=='Delete column 2'
    ui.session_state['cleaning_memory_candidates']['x']['button_label']='Rename all columns'
    assert app._reflected_interactive_label([plan],f)=='Delete columns 2, 3'
