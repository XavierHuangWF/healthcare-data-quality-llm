"""The next review owns no controls until the prior named decision."""
from types import SimpleNamespace
from threading import Event
from hashlib import sha256
from pathlib import Path
from healthcare_data_quality.ui.ordered_review import sequential_active_review
from healthcare_data_quality.services import model_plan_review as review
from healthcare_data_quality.cleaning.sequential_context import projected_records, projected_review_context
from healthcare_data_quality.jobs.interactive_chat_jobs import InteractiveChatJobs
from test_automatic_final_actions_20261124 import UI, isolate_memory, missing, SRC
from test_cleaning_review_cards import _app
from test_responsive_review_workers_20261124 import take_when_done


def test_sequential_terminal_not_ready():
    numbered = [('Finding 1', {'id':'a'}), ('Finding 2', {'id':'b'}), ('Gemma check 1', {'id':'c'})]
    done = set()
    assert sequential_active_review(numbered, done.__contains__) == ('a','Finding 1')
    done.add('a')
    assert sequential_active_review(numbered, done.__contains__) == ('b','Finding 2')
    done.add('b')
    assert sequential_active_review(numbered, done.__contains__) == ('c','Gemma check 1')
    done.add('c')
    assert sequential_active_review(numbered, done.__contains__) == (None,None)


def test_actual_renderer_handoff_uses_private_draft_only_after_completion(monkeypatch, tmp_path):
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    from healthcare_data_quality.ui import review_lookahead as lookahead
    isolate_memory(monkeypatch)
    f=missing(); later={**f, 'id':'dataset_1:qc:99','title':f['title']+' (later)'}
    third={**f, 'id':'dataset_1:qc:100','title':f['title']+' (third)'}
    fourth={**f, 'id':'dataset_1:qc:101','title':f['title']+' (fourth)'}
    ui=UI(); app=_app(monkeypatch,ui)
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR',str(tmp_path/'checkpoint'))
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings',lambda *a,**k:[f,later,third,fourth])
    monkeypatch.setattr('healthcare_data_quality.learning.interaction_memory.save_reviewed_solution',lambda *a,**k:None)
    queued=[]
    def queue(fid, finding, source, suggestion, candidates):
        queued.append(fid); return False
    monkeypatch.setattr(app,'_queue_exact_option_recovery',queue)
    monkeypatch.setattr(app,'_queue_automatic_option_draft',lambda *a:False)
    class Manager:
        def __init__(self): self.jobs={}; self.submitted=[]
        def submit_role(self,role,fn,*args):
            assert role=='B'; job=str(len(self.submitted)); self.submitted.append(args[1]['id']); self.jobs[job]=(fn,args); return job
        def take(self,job):
            if job not in self.jobs: return {'status':'lost'}
            fn,args=self.jobs.pop(job); return {'status':'done','result':fn(*args)}
    manager=Manager()
    monkeypatch.setattr('healthcare_data_quality.jobs.interactive_chat_jobs.interactive_chat_manager',lambda:manager)
    monkeypatch.setattr(review,'assess_exact_options',lambda finding,source,options,context:[
        {'fingerprint':review.option_fingerprint(finding,source,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in options])
    ui.session_state.hdq_registered_reviewer='Reviewer'
    def render():
        ui.expanders.clear(); ui.rendered.clear(); ui.comments.clear()
        app.render_cleaning_workspace([SRC],[SimpleNamespace(name=SRC.name)],{},'sequential-test',[])
    render()
    assert queued == [f['id']]
    assert later['id'] not in ui.session_state.cleaning_recommendations
    opts=source_backed_choices(f,SRC,{})
    ui.session_state.cleaning_recommendations[f['id']]['gemma_option_reviews']=[
        {'fingerprint':review.option_fingerprint(f,SRC,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in opts]
    render()  # private second job
    render()  # collect second, start third
    render()  # collect third
    assert manager.submitted == [later['id'],third['id']]
    assert fourth['id'] not in ui.session_state.cleaning_lookahead_records
    assert later['id'] not in ui.session_state.cleaning_recommendations
    assert not any(t.startswith('Finding 2 ·') or t.startswith('Finding 3 ·') for t in ui.expanders)
    assert any(k.startswith('review_yes_'+f['id']) for _,k in ui.rendered)
    assert not any(k.startswith('review_yes_'+later['id']) or k.startswith('review_yes_'+third['id']) for _,k in ui.rendered)
    app._complete_reviewer_no_change(f,SRC,'Keep current data')
    render()
    assert ui.session_state.cleaning_sequential_active == later['id']
    assert queued == [f['id']]  # no new inference for unaffected second
    assert any(k.startswith('review_yes_'+later['id']) for _,k in ui.rendered)
    assert not any(k.startswith('review_yes_'+third['id']) for _,k in ui.rendered)
    assert manager.submitted[-1] == fourth['id']


def test_current_dataset_and_prior_human_decisions_are_grounded(tmp_path):
    source=tmp_path/'current.csv'; source.write_text('Taxon,Count\na,0\nb,3\nc,4\n')
    raw=source.read_bytes(); digest=sha256(raw).hexdigest()
    a={'id':'a','dataset_id':'dataset_1'}
    b={'id':'b','dataset_id':'dataset_1','check_key':'find_negative_counts','evidence':[{'CSV row':2,'Column':2,'Original value':'0'},{'CSV row':3,'Column':2,'Original value':'3'}]}
    state={'cleaning_review_catalog_snapshot':[a,b], 'cleaning_approved_reviewer':'Reviewer'}
    before=review.dataset_review_context(state,'dataset_1','b',source=source)
    state['cleaning_approved_rows_by_dataset']={'dataset_1':[{'finding_id':'a','operation':'drop_rows','reviewer':'Reviewer','source_sha256':digest,'changes':[{'row':2,'before':'a','after':''}]}]}
    state['cleaning_messages_by_finding']={'a':[{'role':'user','content':'Delete empty rows'}]}
    state['cleaning_decisions']=[{'finding_id':'a','decision':'approved','reviewer':'Reviewer'}]
    after=review.dataset_review_context(state,'dataset_1','b',source=source)
    assert after['revision']!=before['revision']
    assert after['current_dataset']['records']==2
    assert after['current_dataset']['evidence_status_counts']['Removal approved']==1
    assert after['previous_reviewer_turns'][0]['request']=='Delete empty rows'
    assert after['previous_decisions'][0]['finding_id']=='a'
    assert projected_records(state,'dataset_1',source)[1]==[1,3,4]
    assert source.read_bytes()==raw


def test_fixed_roles_never_borrow_each_other():
    mgr=InteractiveChatJobs(); entered, release=Event(),Event()
    def slow():
        entered.set(); assert release.wait(3); return {'ok':True}
    c=mgr.submit_role('C',slow)
    try:
        assert entered.wait(1)
        b=mgr.submit_role('B',lambda:{'ok':True})
        assert mgr.agent_for(b)=='B' and mgr.agent_for(c)=='C'
        assert take_when_done(mgr,b)['result']=={'ok':True}
        assert mgr.active_count('A')==0
    finally:
        release.set()
    assert take_when_done(mgr,c)['result']=={'ok':True}


def test_current_numeric_comparison_excludes_removed_rows(tmp_path):
    from healthcare_data_quality.cleaning.numeric_context import proposal_numeric_comparisons
    source=tmp_path/'numeric.csv'; source.write_text('Taxon,Count\na,-2\nb,100\nc,3\n')
    raw=source.read_bytes()
    state={'cleaning_approved_rows_by_dataset':{'dataset_1':[{'operation':'drop_rows','reviewer':'Reviewer','source_sha256':sha256(raw).hexdigest(),'changes':[{'row':3,'before':'b','after':''}]}]}}
    plan=[{'operation':'replace_cells','changes':[{'row':2,'column':2,'before':'-2','after':'2'}]}]
    result=proposal_numeric_comparisons(source,plan,projected=projected_records(state,'dataset_1',source))
    assert result[0]['Positive median']==3
    assert result[0]['Location']=='R2:C2 (Count)'


def test_inactive_finding_rejects_stale_submission(monkeypatch):
    import pytest
    ui=UI(); app=_app(monkeypatch,ui)
    ui.session_state['cleaning_sequential_active']='first'
    with pytest.raises(PermissionError): app._assert_sequential_owner('later')
