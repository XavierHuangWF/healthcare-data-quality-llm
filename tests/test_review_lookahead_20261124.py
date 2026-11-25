"""Lookahead work is private and reusable only after dependency checks."""
from copy import deepcopy
from hashlib import sha256
from threading import Event
from healthcare_data_quality.ui import review_lookahead as la
from healthcare_data_quality.services import model_plan_review as mr
from test_automatic_final_actions_20261124 import SRC, isolate_memory, missing
from healthcare_data_quality.cleaning.finding_cleaning import build_cleaning_findings
from healthcare_data_quality.investigation.baseline_audit import run_baseline_audit
from healthcare_data_quality.cleaning.in_memory_cleaning import _records
import csv, io


def setup(monkeypatch):
    isolate_memory(monkeypatch)
    catalog = build_cleaning_findings({'baseline':run_baseline_audit(SRC)},[SRC],[SRC.name],[])
    state={'cleaning_approved_reviewer':'Reviewer','cleaning_review_catalog_snapshot':catalog}
    target=next(f for f in catalog if 'missing values' in f['title'])
    origin=next(f for f in catalog if 'fractional' in f['title'])
    monkeypatch.setattr(mr,'assess_exact_options',lambda finding,source,options,context:[
        {'fingerprint':mr.option_fingerprint(finding,source,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in options])
    return state,target,origin


def record(state,target):
    return {'snapshot':la.snapshot(state),'finding':deepcopy(target),'source_sha256':sha256(SRC.read_bytes()).hexdigest(),
            'result':la.prepare(la.snapshot(state),target,SRC,Event())}


def replace(state, origin, row, col, after):
    out=deepcopy(state)
    data=_records(SRC.read_bytes()); before=data[row-1][col-1]; data[row-1][col-1]=after
    stream=io.StringIO(newline='');csv.writer(stream).writerows(data)
    out['cleaning_staged_bytes_by_dataset']={'dataset_1':stream.getvalue().encode()}
    out['cleaning_history']=[{'dataset_id':'dataset_1','finding_id':origin['id'],'operation':'replace_cells',
        'approved_cell_changes':[{'row':row,'column':col,'before':before,'after':after}]}]
    out['cleaning_decisions']=[{'finding_id':origin['id'],'decision':'source_verified_choice_approved_once'}]
    return out


def test_real_missing_choices_reused_after_unrelated_fractional_correction(monkeypatch):
    state,target,origin=setup(monkeypatch); rec=record(state,target)
    assert rec['result']['ready_count']==2
    current=replace(state,origin,11,7,'2')
    result=la.promote(rec,current,target,SRC)
    assert result is not None
    assert result['suggestion']['_review_revision']==mr.dataset_review_context(current,'dataset_1',target['id'])['revision']
    assert len(result['suggestion']['gemma_option_reviews'])==2
    assert any(len(p['changes'])==618 for o in rec['result']['candidates'] for p in o['operations'])


def test_missingness_or_row_collateral_change_invalidates(monkeypatch):
    state,target,origin=setup(monkeypatch); rec=record(state,target)
    assert la.promote(rec,replace(state,origin,11,7,''),target,SRC) is None
    assert la.promote(rec,replace(state,origin,2,5,'1'),target,SRC) is None
    # A whole-row deletion option would remove the changed nonempty control cell too.
    assert la.promote(rec,replace(state,origin,2,2,'3'),target,SRC) is None


def test_new_instruction_and_structural_change_invalidate_even_if_scope_looks_same(monkeypatch):
    state,target,origin=setup(monkeypatch); rec=record(state,target)
    current=deepcopy(state); current['cleaning_messages_by_finding']={origin['id']:[{'role':'user','content':'Always preserve rows from now on'}]}
    assert la.promote(rec,current,target,SRC) is None
    current=deepcopy(state); current['cleaning_history']=[{'dataset_id':'dataset_1','finding_id':origin['id'],'operation':'drop_rows'}]
    assert la.promote(rec,current,target,SRC) is None


def test_unapproved_changes_and_unknown_dependencies_fail_closed(monkeypatch):
    state,target,origin=setup(monkeypatch); rec=record(state,target)
    current=replace(state,origin,11,7,'2'); current.pop('cleaning_history')
    assert la.promote(rec,current,target,SRC) is None
    check={**target,'check_key':'gemma_check'}
    assert not la.dependencies_unchanged(state,replace(state,origin,11,7,'2'),check,SRC,[])


def test_candidate_frequency_or_scope_change_requires_reassessment(monkeypatch):
    state,target,origin=setup(monkeypatch); rec=record(state,target)
    monkeypatch.setattr(la,'_signatures',lambda *args:['changed'])
    assert la.promote(rec,state,target,SRC) is None


def test_cancelled_worker_does_not_call_gemma(monkeypatch):
    state,target,origin=setup(monkeypatch)
    stop=Event();stop.set()
    def forbidden(*args,**kwargs): raise AssertionError('cancelled work reached Gemma')
    monkeypatch.setattr(mr,'assess_exact_options',forbidden)
    assert la.prepare(state,target,SRC,stop)=={'cancelled':True}


def test_same_column_statistics_force_revision_for_numeric_proposal(monkeypatch):
    state,_,origin=setup(monkeypatch)
    target=next(f for f in state['cleaning_review_catalog_snapshot'] if 'negative count' in f['title'])
    rec=record(state,target)
    # Fractional correction in column 7 does not change the negative finding's
    # affected columns 6/12 or their distributions.
    assert la.promote(rec,replace(state,origin,11,7,'2'),target,SRC) is not None
    # Changing another observation in column 6 invalidates the numeric context,
    # even though the two originally negative cells were not directly edited.
    assert la.promote(rec,replace(state,origin,11,6,'999'),target,SRC) is None


def test_handoff_retires_pending_and_ready_private_drafts(monkeypatch):
    from test_automatic_final_actions_20261124 import UI
    from test_cleaning_review_cards import _app
    state,target,origin=setup(monkeypatch)
    third=next(f for f in state['cleaning_review_catalog_snapshot'] if 'negative count' in f['title'])
    ui=UI();app=_app(monkeypatch,ui)
    second_record=record(state,target)
    stop=Event();pending={'snapshot':la.snapshot(state),'finding':deepcopy(third),'source_sha256':sha256(SRC.read_bytes()).hexdigest(),
        'job_id':'late-b','stop':stop}
    ui.session_state.update(deepcopy(state))
    ui.session_state['cleaning_messages_by_finding']={origin['id']:[{'role':'user','content':'Never delete rows from this experiment'}]}
    ui.session_state['cleaning_lookahead_records']={target['id']:second_record,third['id']:pending}
    monkeypatch.setattr(app,'_finding_review_completed',lambda fid:fid==origin['id'])
    class Manager:
        def take(self,job): return {'status':'running'}
    monkeypatch.setattr('healthcare_data_quality.jobs.interactive_chat_jobs.interactive_chat_manager',lambda:Manager())
    app._collect_lookahead([('Finding 1',origin),('Finding 2',target),('Finding 3',third)],[SRC],handoff=True,origin='Finding 1')
    assert ui.session_state['cleaning_lookahead_records']=={}
    assert stop.is_set()
    assert 'late-b' in ui.session_state['cleaning_retired_chat_jobs']
    assert ui.session_state['cleaning_lookahead_revising']=={target['id']:'Finding 1',third['id']:'Finding 1'}
    assert not app._take_active_lookahead(target,SRC)
    assert not ui.session_state.get('cleaning_recommendations')
