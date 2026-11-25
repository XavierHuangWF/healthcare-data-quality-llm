from copy import deepcopy
from pathlib import Path
import pytest
from healthcare_data_quality.cleaning.one_click_visual_flag import apply_explicit_visual_flag_once
from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag
from healthcare_data_quality.cleaning.visual_recommendations import visual_option, verify_visual_option
from healthcare_data_quality.learning import interaction_memory as mem
from healthcare_data_quality.ui import review_lookahead as la
from healthcare_data_quality.ui.human_choice_replay import memory_first_candidates
from healthcare_data_quality.services import model_plan_review as mr
from test_review_lookahead_20261124 import setup, record, replace
from test_automatic_final_actions_20261124 import SRC, render_state


def isolate_log(monkeypatch, tmp_path):
    log = tmp_path/'approved.jsonl'
    monkeypatch.setattr(mem, '_location', lambda path=None: Path(path) if path else log)
    monkeypatch.setattr('healthcare_data_quality.learning.adaptive_skill_memory.record_review',lambda *a,**kw:None)
    return log


def test_annotation_memory_round_trip_frequency_and_style(monkeypatch,tmp_path):
    log=isolate_log(monkeypatch,tmp_path)
    source=tmp_path/'counts.csv';source.write_text('Taxon,A,B\nx,0.2,0.5\ny,5,9\n')
    f={'id':'f','dataset_id':'dataset_1','check_key':'find_invalid_counts',
       'title':'2 cells contain fractional count values','total_evidence_records':2,
       'evidence':[{'CSV row':2,'Column':2,'Original value':'0.2'},{'CSV row':2,'Column':3,'Original value':'0.5'}]}
    original=source.read_bytes()
    for _ in range(3):
        state={'cleaning_approved_reviewer':'Reviewer'}
        request='highlight this row; bold red font only'
        plan=requested_visual_flag(request,f,source)
        apply_explicit_visual_flag_once(state,f,source,request,previously_displayed=plan)
        assert state['cleaning_history'][0]['operation']=='flag_cells'
        assert state.get('cleaning_staged_bytes_by_dataset',{}).get('dataset_1',original)==original
        assert source.read_bytes()==original
    choices=memory_first_candidates(f,source)
    assert choices[0]['historical_approval_count']==3
    assert choices[0]['operations'][0]['operation']=='flag_cells'
    assert all(not c['fill'] for c in choices[0]['operations'][0]['changes'])
    assert len(choices[0]['operations'][0]['changes'])==3
    saved=mem.find_reviewed_solution(f,source,path=log)['record']
    assert 'Highlight' in mem.short_description(saved)
    assert all(x['Before']==x['After'] for x in mem.exact_change_rows(saved))
    assert all('font only' in x['Style'] for x in mem.exact_change_rows(saved))
    # The exact historical annotation is advice, never an authorization.
    assert saved['execution_authorized'] is False
    tampered=deepcopy(choices[0]);tampered['operations'][0]['changes'][0]['fill']=True
    assert verify_visual_option(f,source,tampered) is None


def test_familiar_annotation_available_before_chat_for_multiple_findings(monkeypatch):
    state,_,origin=setup(monkeypatch)
    for f in [origin, next(f for f in state['cleaning_review_catalog_snapshot'] if 'negative count' in f['title']),
              next(f for f in state['cleaning_review_catalog_snapshot'] if 'incorrect number of taxonomy levels' in f['title'])]:
        opts=memory_first_candidates(f,SRC)
        assert any(o['operations'][0]['operation']=='flag_cells' for o in opts)
        if 'taxonomy' not in f['title']:
            assert not any(c.get('after')=='NA' for o in opts for p in o['operations'] for c in p['changes'])
        assert len(opts)<=5


def test_real_annotation_commit_preserves_unaffected_prepared_missing_choices(monkeypatch,tmp_path):
    state,target,origin=setup(monkeypatch); isolate_log(monkeypatch,tmp_path)
    rec=record(state,target)
    current=deepcopy(state); request='highlight them row'
    current['cleaning_messages_by_finding']={origin['id']:[{'role':'user','content':'No — Discuss Another Idea'}, {'role':'user','content':request}]}
    plan=requested_visual_flag(request,origin,SRC)
    apply_explicit_visual_flag_once(current,origin,SRC,request,previously_displayed=plan)
    assert la.promote(rec,current,target,SRC) is not None
    # In-flight advice is still private; this is only a permission to finish.
    assert la.dependencies_unchanged(rec['snapshot'],current,target,SRC,allow_pending=True)
    # A genuinely new, broad instruction still forces fresh review.
    current['cleaning_messages_by_finding'][origin['id']].append({'role':'user','content':'Never delete rows in this experiment'})
    assert la.promote(rec,current,target,SRC) is None


def test_annotation_overlap_requires_new_review(monkeypatch,tmp_path):
    state,target,origin=setup(monkeypatch);isolate_log(monkeypatch,tmp_path)
    rec=record(state,target)
    current=deepcopy(state); request='highlight row 2'
    plan=requested_visual_flag(request,origin,SRC)
    apply_explicit_visual_flag_once(current,origin,SRC,request,previously_displayed=plan)
    assert la.promote(rec,current,target,SRC) is None


def test_inflight_unrelated_value_edit_can_finish_but_not_publish_unchecked(monkeypatch):
    state,target,origin=setup(monkeypatch)
    current=replace(state,origin,11,7,'2')
    assert not la.dependencies_unchanged(state,current,target,SRC)
    assert la.dependencies_unchanged(state,current,target,SRC,allow_pending=True)
    assert not la.dependencies_unchanged(state,replace(state,origin,11,7,''),target,SRC,allow_pending=True)


def test_initial_annotation_button_commits_without_agent_c_round_trip(monkeypatch,tmp_path):
    state,_,origin=setup(monkeypatch); isolate_log(monkeypatch,tmp_path)
    option=visual_option(origin,SRC,'highlight these rows')
    suggestion={'suggestion':'','choices':[],'gemma_option_reviews':[]}
    app,ui=render_state(monkeypatch,origin,suggestion)
    ui.session_state.update(state,cleaning_sequential_active=origin['id'])
    suggestion['_review_revision']=mr.dataset_review_context(ui.session_state,'dataset_1',origin['id'])['revision']
    fp=mr.option_fingerprint(origin,SRC,option)
    suggestion['gemma_option_reviews']=[{'fingerprint':fp,'can_offer':True,'scope_matches':True,'has_extra_changes':False}]
    reviewed={**option,'gemma_review_status':'supported'}
    def button(label,**kwargs): return label.startswith('Highlight')
    monkeypatch.setattr(ui,'button',button)
    monkeypatch.setattr(app,'_render_visual_flag_preview',lambda *a,**kw: True)
    monkeypatch.setattr(app,'_rerun_cleaning_fragment',lambda *a:None)
    def forbidden(*args,**kw): raise AssertionError('Annotation must not queue a value edit')
    monkeypatch.setattr(app,'_queue_local_choice',forbidden)
    app._render_live_agent_b_finding(origin['id'],origin,SRC,[],[SRC],None,'dataset_1',prepared=('ready',suggestion,[reviewed]))
    assert ui.session_state.cleaning_history[-1]['operation']=='flag_cells'
    assert ui.session_state.cleaning_review_statuses[origin['id']]=='Applied and verified'


def test_grouped_row_scope_is_not_limited_to_count_findings(monkeypatch,tmp_path):
    isolate_log(monkeypatch,tmp_path)
    source=tmp_path/'taxa.csv';source.write_text('Taxon,A,B\nx,2,3\nx,5,9\ny,1,0\n')
    f={'id':'dup','dataset_id':'dataset_1','check_key':'find_duplicate_taxa',
       'title':'1 taxon label appears in multiple rows','total_evidence_records':1,
       'evidence':[{'CSV rows':'2, 3','Taxon / feature':'x'}]}
    choice=next(o for o in memory_first_candidates(f,source) if o.get('visual_request'))
    assert {c['row'] for c in choice['operations'][0]['changes']}=={2,3}
    assert len(choice['operations'][0]['changes'])==6
    assert not visual_option(f,source,'highlight row 4')
