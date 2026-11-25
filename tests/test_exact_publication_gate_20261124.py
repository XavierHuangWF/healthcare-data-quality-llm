"""Draft readiness is not review readiness, including real missing-value scope."""
from healthcare_data_quality.ui.ordered_review import predecessor_blocker, publication_epoch
from healthcare_data_quality.services import model_plan_review as review
from test_automatic_final_actions_20261124 import SRC, missing, isolate_memory, render_state


def test_missing_cells_wait_for_exact_assessment_and_release_in_order(monkeypatch):
    isolate_memory(monkeypatch)
    f = missing(); fid = f['id']
    suggestion = {'suggestion':'', 'choices':[], 'gemma_option_reviews':[]}
    app, ui = render_state(monkeypatch, f, suggestion)
    state = ui.session_state
    queued = []
    def queue(fid, finding, source, draft, candidates):
        queued.extend(candidates)
        state.cleaning_pending_gemma_b[fid] = {'kind':'exact_option_recovery'}
        return True
    monkeypatch.setattr(app, '_queue_exact_option_recovery', queue)
    later = {'id':'later'}
    numbered = [('Finding 2',f),('Finding 3',later)]
    entries = {fid:{'status':'failed'}, 'later':{'status':'ready'}}
    phase, _, opts = app._prepare_initial_review_choices(fid,f,SRC,entries[fid])
    assert phase == 'preparing' and not opts and len(queued)==2
    assert not ui.rendered and not ui.comments  # preparation does not emit UI
    assert predecessor_blocker(numbered,'later',entries,state,lambda _:False,publication={}) == 'Finding 2'
    assert publication_epoch(state,fid,False)==(0,False)
    assert app._prepare_initial_review_choices(fid,f,SRC,entries[fid])[0]=='preparing'
    assert len(queued)==2 # no repeated queue while in flight
    state.cleaning_pending_gemma_b.clear()
    suggestion['gemma_option_reviews'] = [{'fingerprint': review.option_fingerprint(f,SRC,o),
        'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in queued]
    prepared = app._prepare_initial_review_choices(fid,f,SRC,entries[fid])
    assert prepared[0]=='ready' and len(prepared[2])==2
    assert publication_epoch(state,fid,True)==(1,True)
    assert predecessor_blocker(numbered,'later',entries,state,lambda _:False,publication={fid:'ready'}) is None
    app._render_live_agent_b_finding(fid,f,SRC,[],[SRC],None,'dataset_1',prepared=prepared)
    assert len([key for label,key in ui.rendered if key.startswith('review_yes_')])==2
    assert any('618' in label for label,key in ui.rendered)
    assert publication_epoch(state,fid,True)==(1,False)
    assert not state.cleaning_history


def test_cached_draft_cannot_bypass_running_recommendation(monkeypatch):
    f=missing(); app,ui=render_state(monkeypatch,f,{'suggestion':'draft'})
    assert app._prepare_initial_review_choices(f['id'],f,SRC,{'status':'running'}) == ('preparing',None,[])
    assert not ui.rendered


def test_terminal_failure_stays_locked_without_any_controls(monkeypatch):
    isolate_memory(monkeypatch)
    f=missing(); app,ui=render_state(monkeypatch,f,{'suggestion':'','gemma_option_reviews':[]})
    monkeypatch.setattr(app,'_queue_exact_option_recovery',lambda *a:False)
    monkeypatch.setattr(app,'_queue_automatic_option_draft',lambda *a:False)
    prepared=app._prepare_initial_review_choices(f['id'],f,SRC,{'status':'failed'})
    assert prepared[0]=='locked' and not prepared[2]
    app._render_live_agent_b_finding(f['id'],f,SRC,[],[SRC],None,'dataset_1',prepared=prepared)
    assert not ui.rendered
    assert predecessor_blocker([('Finding 2',f),('Finding 3',{'id':'next'})], 'next', {}, ui.session_state, lambda _:False, publication={f['id']:'locked'}) == 'Finding 2'


def test_revision_reopens_once_and_does_not_depend_on_worker_status():
    state={}; fid='f2'
    assert publication_epoch(state,fid,True)==(1,True)
    assert publication_epoch(state,fid,True)==(1,False)
    assert publication_epoch(state,fid,False)==(1,False)
    assert publication_epoch(state,fid,False)==(1,False)
    assert publication_epoch(state,fid,True)==(2,True)
    assert publication_epoch(state,fid,True)==(2,False)


def test_empty_review_retries_with_backoff_and_never_releases_successor():
    from healthcare_data_quality.ui.ordered_review import recovery_due
    state = {}; fid='f2'; identity=('source','revision',0)
    assert not recovery_due(state,fid,identity,0)
    assert not recovery_due(state,fid,identity,29)
    assert recovery_due(state,fid,identity,30)
    assert not recovery_due(state,fid,identity,31)
    assert recovery_due(state,fid,identity,90)
    assert recovery_due(state,fid,identity,210)
    assert not recovery_due(state,fid,identity,330)
    assert state['cleaning_empty_review_recovery'][fid]['exhausted']
    assert not recovery_due(state,fid,identity,500)
    assert not recovery_due(state,fid,('source','new-revision',0),500)
    assert recovery_due(state,fid,('source','new-revision',0),530)


def test_ready_card_loses_all_options_and_relocks(monkeypatch):
    isolate_memory(monkeypatch)
    f=missing(); fid=f['id']; suggestion={'suggestion':'','gemma_option_reviews':[]}
    app,ui=render_state(monkeypatch,f,suggestion)
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    opts=source_backed_choices(f,SRC,suggestion)
    suggestion['gemma_option_reviews']=[{'fingerprint':review.option_fingerprint(f,SRC,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in opts]
    monkeypatch.setattr(app,'_queue_exact_option_recovery',lambda *a:False)
    monkeypatch.setattr(app,'_queue_automatic_option_draft',lambda *a:False)
    assert app._prepare_initial_review_choices(fid,f,SRC,{'status':'failed'})[0]=='ready'
    assert publication_epoch(ui.session_state,fid,True)==(1,True)
    suggestion['gemma_option_reviews']=[]
    prepared=app._prepare_initial_review_choices(fid,f,SRC,{'status':'failed'})
    assert prepared[0]=='locked' and not prepared[2]
    assert publication_epoch(ui.session_state,fid,False)==(1,False)
    app._render_live_agent_b_finding(fid,f,SRC,[],[SRC],None,'dataset_1',prepared=prepared)
    assert not ui.rendered
    assert publication_epoch(ui.session_state,fid,True)==(2,True)


def test_missing_value_assessment_sees_full_source_scope(monkeypatch):
    import json
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    isolate_memory(monkeypatch)
    f=missing(); f={**f,'evidence':f['evidence'][:32]}
    options=source_backed_choices(f,SRC,{})
    observed=[]
    def model(prompt, **kwargs):
        payload=json.loads(prompt.split('DATA: ',1)[1]); observed.append(payload)
        return '{"assessment":"supported","finding_scope_matches":true,"contains_unrequested_change":false,"reason":"Verified complete scope."}'
    monkeypatch.setattr(review,'query_local_model',model)
    for option in options:
        assert review.assess_review_option(f,SRC,option)['can_offer']
    assert len(observed)==2
    for item in observed:
        scope=item['finding']['verified_complete_missing_scope']
        assert scope['empty_cells']==618 and scope['affected_rows']==6
        assert [x['row'] for x in scope['first_rows']]==[2,3,5,7,15,20]
        assert item['finding']['affected_records']==618


def test_verified_zero_finding_check_can_offer_meaningful_keep_action(monkeypatch):
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    f={'id':'gemma:2:check','source':'Gemma check','dataset_id':'dataset_1',
       'check_key':'gemma_check','actionable':False,'title':'Check for anomalies','evidence':[],
       'gemma_check_details':{'result':{'findings':[]},'verification':{'status':'source_cells_verified_and_output_reproduced'}}}
    opts=source_backed_choices(f,SRC,{})
    assert len(opts)==1 and opts[0]['decision']=='no_change' and opts[0]['rows']
    f['gemma_check_details']['verification']['status']='unverified'
    assert not source_backed_choices(f,SRC,{})


def test_parent_never_mounts_empty_finding_or_later_ready_card(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from test_cleaning_review_cards import _app
    from test_automatic_final_actions_20261124 import UI
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    isolate_memory(monkeypatch)
    f=missing(); fid=f['id']; later={**f,'id':'dataset_1:qc:99','title':f['title']+' (later)'}
    # Keep a separate finding identity; both candidates still require exact proof.
    entries={fid:{'status':'ready','suggestion':'','choices':[],'gemma_option_reviews':[]},
             later['id']:{'status':'ready','suggestion':'','choices':[],'gemma_option_reviews':[]}}
    for item in (f,later):
        opts=source_backed_choices(item,SRC,{})
        if item is later:
            entries[item['id']]['gemma_option_reviews']=[{'fingerprint':review.option_fingerprint(item,SRC,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in opts]
    class Manager:
        def snapshot(self,key): return {'entries':entries}
    ui=UI(); app=_app(monkeypatch,ui)
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR',str(tmp_path/'checkpoint'))
    monkeypatch.setattr('healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager',lambda:Manager())
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings',lambda *a,**k:[f,later])
    monkeypatch.setattr(app,'_queue_exact_option_recovery',lambda *a:False)
    monkeypatch.setattr(app,'_queue_automatic_option_draft',lambda *a:False)
    ui.session_state.hdq_registered_reviewer='Reviewer'
    def render():
        ui.expanders.clear(); ui.rendered.clear(); ui.comments.clear()
        app.render_cleaning_workspace([SRC],[SimpleNamespace(name=SRC.name)],{},'strict-order-test',[])
    render()
    assert not ui.expanders
    assert not any(key.startswith(('review_yes_','review_no_','review_close_')) for _,key in ui.rendered)
    opts=source_backed_choices(f,SRC,{})
    ui.session_state.cleaning_recommendations[fid]['gemma_option_reviews']=[{'fingerprint':review.option_fingerprint(f,SRC,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in opts]
    render()
    assert any(title.startswith('Finding 1 ·') for title in ui.expanders)
    assert len([key for _,key in ui.rendered if key.startswith('review_yes_'+fid)])==2
    # A later render loses the current proof. Both cards must leave the UI.
    assert not any(title.startswith('Finding 2 ·') for title in ui.expanders)
    ui.session_state.cleaning_recommendations[fid]['gemma_option_reviews']=[]
    render()
    assert not ui.expanders
    assert not any(key.startswith(('review_yes_','review_no_','review_close_')) for _,key in ui.rendered)
