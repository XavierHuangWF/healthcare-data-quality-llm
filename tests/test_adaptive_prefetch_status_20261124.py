from copy import deepcopy
from hashlib import sha256
from threading import Event

import pytest

from healthcare_data_quality.ui import review_lookahead as la
from healthcare_data_quality.ui.finding_wait_gate import active_wait_status, waiting_card_html
from healthcare_data_quality.ui.one_click_choice import prepare_verified_choice_snapshot, commit_verified_choice_snapshot
from healthcare_data_quality.services import model_plan_review as mr
from test_review_lookahead_20261124 import setup, record, replace
from test_automatic_final_actions_20261124 import SRC, UI
from test_cleaning_review_cards import _app


def zero_check():
    return {'id':'gemma:1:check','dataset_id':'dataset_1','source':'Gemma check',
            'check_key':'gemma_check','evidence':[],'actionable':False,
            'gemma_check_details':{'verification':{'status':'source_cells_verified_and_output_reproduced'}}}


def test_verified_empty_check_does_not_restart_active_or_prepared_qc(monkeypatch):
    state, target, _ = setup(monkeypatch)
    rec = record(state, target)
    current = deepcopy(state)
    current['cleaning_review_catalog_snapshot'].append(zero_check())
    assert mr.dataset_review_context(state,'dataset_1',target['id'])['revision'] == mr.dataset_review_context(current,'dataset_1',target['id'])['revision']
    assert la.promote(rec,current,target,SRC) is not None
    current['cleaning_review_catalog_snapshot'][-1]['evidence'] = [{'CSV row':2,'Column':2}]
    assert mr.dataset_review_context(state,'dataset_1',target['id'])['revision'] != mr.dataset_review_context(current,'dataset_1',target['id'])['revision']
    assert la.promote(rec,current,target,SRC) is None


def test_negative_correction_outside_duplicate_rows_preserves_duplicate_draft(monkeypatch):
    state, _, _ = setup(monkeypatch)
    negative = next(f for f in state['cleaning_review_catalog_snapshot'] if 'negative count' in f['title'])
    duplicate = next(f for f in state['cleaning_review_catalog_snapshot'] if f['check_key']=='find_duplicate_taxa')
    rec = record(state, duplicate)
    assert rec['result']['ready_count']
    assert la.promote(rec,replace(state,negative,16,6,'679'),duplicate,SRC) is not None
    assert la.promote(rec,replace(state,negative,11,6,'999'),duplicate,SRC) is None
    assert la.promote(rec,replace(state,negative,16,1,'different taxon'),duplicate,SRC) is None


def branch_case(monkeypatch):
    state, missing, _ = setup(monkeypatch)
    next_f = next(f for f in state['cleaning_review_catalog_snapshot'] if 'negative count' in f['title'])
    option = next(o for o in la._candidates(state,missing,SRC,{}) if o['operations'][0]['operation']=='drop_rows')
    option.update(gemma_review_status='supported', historical_approval_count=32)
    monkeypatch.setattr('healthcare_data_quality.learning.interaction_memory.save_reviewed_solution',lambda *a,**kw:None)
    return state, missing, next_f, option


def test_real_six_row_deletion_conditional_branch_matches_only_real_approval(monkeypatch):
    state, missing, next_f, option = branch_case(monkeypatch)
    before = deepcopy(state); raw = SRC.read_bytes()
    out = la.prepare_conditional(state,missing,option,next_f,SRC,Event())
    assert state == before and SRC.read_bytes()==raw
    assert out['prepared']['ready_count'] >= 1
    rec = {'snapshot':out['predicted'],'finding':next_f,'source_sha256':sha256(raw).hexdigest(),'result':out['prepared']}
    assert not la.branch_selected(state,state,missing,option,SRC)
    assert la.promote(rec,state,next_f,SRC) is None
    actual = deepcopy(state)
    commit_verified_choice_snapshot(actual,missing,SRC,option,prepare_verified_choice_snapshot(actual,missing,SRC,option))
    assert la.branch_selected(state,actual,missing,option,SRC)
    assert la.promote(rec,actual,next_f,SRC) is not None
    current = deepcopy(actual)
    current['cleaning_messages_by_finding']={missing['id']:[{'role':'user','content':'Never delete any more rows'}]}
    assert not la.branch_selected(state,current,missing,option,SRC)
    assert la.promote(rec,current,next_f,SRC) is None
    current = deepcopy(actual)
    current['cleaning_history'][-1]['modified_positions'][0]['row']=99
    assert not la.branch_selected(state,current,missing,option,SRC)
    assert la.promote(rec,current,next_f,SRC) is None


def test_handoff_uses_matching_conditional_record_instead_of_stale_baseline(monkeypatch):
    state, missing, next_f, option = branch_case(monkeypatch)
    base = record(state,next_f)
    out=la.prepare_conditional(state,missing,option,next_f,SRC,Event())
    ui=UI(); app=_app(monkeypatch,ui)
    ui.session_state.update(deepcopy(state))
    commit_verified_choice_snapshot(ui.session_state,missing,SRC,option,prepare_verified_choice_snapshot(ui.session_state,missing,SRC,option))
    ui.session_state['cleaning_sequential_active']=next_f['id']
    ui.session_state['cleaning_lookahead_records']={next_f['id']:base}
    ui.session_state['cleaning_lookahead_branch']={'finding':next_f,'source_sha256':sha256(SRC.read_bytes()).hexdigest(),
        'snapshot':out['predicted'],'result':out['prepared'],'branch':{'base':state,'origin':missing,'choice':option}}
    monkeypatch.setattr(app,'_finding_review_completed',lambda fid:fid==missing['id'])
    monkeypatch.setattr('healthcare_data_quality.jobs.interactive_chat_jobs.interactive_chat_manager',lambda:object())
    app._collect_lookahead([('Finding 2',missing),('Finding 3',next_f)],[SRC],handoff=True,origin='Finding 2')
    assert ui.session_state['cleaning_lookahead_records'][next_f['id']].get('branch')
    assert not app._take_active_lookahead(next_f,SRC)
    assert ui.session_state['cleaning_recommendations'][next_f['id']]['gemma_option_reviews']


def test_wait_status_consistency_and_no_jogging_lock():
    assert active_wait_status('preparing',worker_phase='queued',revising=True)=='queued'
    assert active_wait_status('preparing',worker_phase='generating',model_stage='waiting_for_inference_slot')=='queued'
    assert active_wait_status('preparing',worker_phase='generating',revising=True)=='revision'
    assert active_wait_status('preparing',worker_phase='generating')=='running'
    assert active_wait_status('locked')=='recovering'
    assert active_wait_status('locked',exhausted=True)=='failed'
    assert active_wait_status('preparing',worker_phase='generating',model_stage='waiting_for_reconnect')=='recovering'
    for status, icon, motion in [('ordered','🔒','static'),('queued','◷','static'),('running','↻','spin'),('revision','↻','spin'),('failed','⚠','static')]:
        html=waiting_card_html('Finding 3','Count anomaly',status)
        assert '>'+icon+'</span>' in html
        assert f'data-motion="{motion}" role="img"' in html
        assert '🏃' not in html


def test_retired_speculative_job_cannot_dispatch_new_http(monkeypatch):
    from healthcare_data_quality.services import llm_client as client
    stop=Event(); stop.set()
    def forbidden(*args,**kwargs): raise AssertionError('Cancelled work reached HTTP')
    monkeypatch.setattr(client,'urlopen',forbidden)
    with client.cancel_speculative_requests(stop), pytest.raises(TimeoutError):
        client.query_local_model('test')
