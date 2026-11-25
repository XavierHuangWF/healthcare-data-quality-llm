"""Regression: prior approvals are a source-bound candidate, not a chat hint."""
from pathlib import Path
import json
from healthcare_data_quality.learning import interaction_memory as mem
from healthcare_data_quality.ui.human_choice_replay import memory_first_candidates
from healthcare_data_quality.services import model_plan_review as review
from healthcare_data_quality.jobs.cleaning_jobs import CleaningRecommendationJobs
from test_automatic_final_actions_20261124 import missing, SRC, render_state


def _proof(finding, source, option, **kwargs):
    return {'fingerprint': review.option_fingerprint(finding, source, option),
            'review_policy': review.EXACT_REVIEW_POLICY_VERSION,
            'can_offer': True, 'scope_matches': True, 'has_extra_changes': False,
            'decision': 'supported'}


def _local_memory(monkeypatch, tmp_path, *, seed_history=False):
    log = tmp_path / 'memory.jsonl'
    monkeypatch.setattr(mem, '_location', lambda path=None: Path(path) if path else log)
    if seed_history:
        # Reproduce the observed historical frequency against the bundled CSV,
        # without depending on the user's mutable runtime memory file.
        f = missing()
        for option in memory_first_candidates(f, SRC):
            kind = option['operations'][0]['operation']
            count = 32 if kind == 'drop_rows' else 2
            for _ in range(count):
                mem.save_reviewed_solution(f, SRC, operations=option['operations'], path=log)
    return log


def test_bundled_csv_history_prioritized_and_no_redundant_draft(monkeypatch, tmp_path):
    _local_memory(monkeypatch, tmp_path, seed_history=True)
    f = missing()
    calls = []
    def unexpected_draft(item):
        raise AssertionError('Known verified choices must not depend on drafting')
    def assess(finding, source, option, **kwargs):
        calls.append(option)
        return _proof(finding, source, option)
    jobs = CleaningRecommendationJobs(tmp_path/'jobs', recommender=unexpected_draft, choice_assessor=assess)
    key = jobs.start('memory-first-real', [f], source_paths_by_finding={f['id']:SRC})
    worker = jobs._workers.get(key)
    if worker is not None:
        worker.join(5)
        assert not worker.is_alive()
    entry = jobs.snapshot(key)['entries'][f['id']]
    assert entry['status'] == 'ready'
    assert len(calls) == 2
    assert calls[0]['historical_approval_count'] == 32
    assert calls[0]['operations'][0]['operation'] == 'drop_rows'
    assert [x['row'] for x in calls[0]['operations'][0]['changes']] == [2,3,5,7,15,20]
    assert 'Other measurements' in calls[0]['impact']
    assert calls[1]['historical_approval_count'] == 2
    assert len(calls[1]['operations'][0]['changes']) == 618
    # Persisted recommendation metadata does not contain copied raw memory operations.
    assert entry['choices'] == []
    app, ui = render_state(monkeypatch, f, {})
    prepared = app._prepare_initial_review_choices(f['id'], f, SRC, entry)
    assert prepared[0] == 'ready'
    assert [x['historical_approval_count'] for x in prepared[2]] == [32,2]
    app._render_live_agent_b_finding(f['id'], f, SRC, [], [SRC], None, 'dataset_1', prepared=prepared)
    labels = [label for label,key in ui.rendered if key.startswith('review_yes_')]
    assert len(labels) == 2 and labels[0].startswith('Delete')
    assert not ui.session_state.cleaning_history


def test_model_failure_still_assesses_host_candidates(monkeypatch, tmp_path):
    _local_memory(monkeypatch, tmp_path)
    f = missing(); drafted=[]; assessed=[]
    def fail(item):
        drafted.append(item['id']); raise RuntimeError('draft unavailable')
    def assess(finding, source, option, **kwargs):
        assessed.append(option); return _proof(finding, source, option)
    jobs = CleaningRecommendationJobs(tmp_path/'jobs', recommender=fail, choice_assessor=assess)
    key = jobs.start('memory-first-failure', [f], source_paths_by_finding={f['id']:SRC})
    worker = jobs._workers.get(key)
    if worker is not None:
        worker.join(5)
        assert not worker.is_alive()
    assert jobs.snapshot(key)['entries'][f['id']]['status'] == 'ready'
    assert len(drafted)==2 and len(assessed)==2


def test_recovery_receives_real_approval_count_not_untrusted_choice_field(monkeypatch,tmp_path):
    _local_memory(monkeypatch,tmp_path,seed_history=True)
    f=missing(); options=memory_first_candidates(f,SRC)
    options[0]['historical_approval_count']=99999
    prompts=[]
    def model(prompt, **kwargs):
        data = json.loads(prompt.split('DATA: ',1)[1])
        prompts.append(data)
        return json.dumps({'reviews':[{'option_id':c['option_id'], 'assessment':'supported',
                           'finding_scope_matches':True,'contains_unrequested_change':False,
                           'reason':'Scope verified'} for c in data['candidates']]})
    monkeypatch.setattr(review,'query_local_model',model)
    reviews=review.assess_exact_options(f,SRC,options,{})
    assert len(reviews)==2 and all(x['can_offer'] for x in reviews)
    assert len(prompts)==1
    assert prompts[0]['candidates'][0]['verified_historical_preference']['exact_source_and_issue_approval_count']==32
    assert prompts[0]['shared_context']['relevant_approved_methods']
    assert prompts[0]['candidates'][1]['verified_historical_preference']['exact_source_and_issue_approval_count']==2


def test_old_rejection_reappraised_but_never_executes_without_new_proof(monkeypatch,tmp_path):
    _local_memory(monkeypatch,tmp_path,seed_history=True)
    f=missing(); opts=memory_first_candidates(f,SRC)
    bad={**_proof(f,SRC,opts[0]),'review_policy':2,'can_offer':False,'scope_matches':False}
    suggestion={'suggestion':'','choices':[],'gemma_option_reviews':[bad,_proof(f,SRC,opts[1])]}
    app,ui=render_state(monkeypatch,f,suggestion)
    queued=[]
    def queue(fid, finding, source, draft, choices):
        queued.extend(choices); return True
    monkeypatch.setattr(app,'_queue_exact_option_recovery',queue)
    phase,_,options=app._prepare_initial_review_choices(f['id'],f,SRC,{'status':'failed'})
    assert phase=='preparing' and not options and len(queued)==1
    assert queued[0]['historical_approval_count']==32
    assert not ui.rendered and not ui.session_state.cleaning_history
    # A fresh scope rejection remains a real block; no execution bypass.
    draft=ui.session_state.cleaning_recommendations[f['id']]
    draft['gemma_option_reviews'].append({**bad,'review_policy':review.EXACT_REVIEW_POLICY_VERSION})
    queued.clear()
    monkeypatch.setattr(app,'_queue_automatic_option_draft',lambda *a:False)
    phase,_,options=app._prepare_initial_review_choices(f['id'],f,SRC,{'status':'failed'})
    assert phase=='ready' and len(options)==1 and not queued
    assert options[0]['operations'][0]['operation']=='replace_cells'


def test_same_pipeline_for_other_findings_and_gemma_checks(monkeypatch,tmp_path):
    log=_local_memory(monkeypatch,tmp_path)
    p=tmp_path/'counts.csv';p.write_text('Taxon,A,Rep_Blank\nx,-5,0\ny,3,0\n')
    f={'id':'negative','dataset_id':'dataset_1','check_key':'find_invalid_counts',
       'title':'1 cell contains negative count values','total_evidence_records':1,
       'evidence':[{'CSV row':2,'Column':2,'Original value':'-5'}]}
    opts=memory_first_candidates(f,p)
    assert not any(c.get('after') == 'NA' for o in opts for op in o['operations'] for c in op['changes'])
    # A human can explicitly approve an NA substitution. Automatic drafting
    # must not invent it merely to fill the option list.
    na=[{'operation':'replace_cells','changes':[{'row':2,'column':2,'before':'-5','after':'NA'}]}]
    for _ in range(3): mem.save_reviewed_solution(f,p,operations=na,path=log)
    assert memory_first_candidates(f,p)[0]['historical_approval_count']==3
    g={'id':'check','dataset_id':'dataset_1','check_key':'gemma_check','title':'Blank control Rep_Blank',
       'evidence':[],'gemma_check_details':{'proposal':{'hypothesis':'Blank control Rep_Blank'}}}
    deletion=next(o for o in memory_first_candidates(g,p) if o.get('operations'))
    mem.save_reviewed_solution(g,p,operations=deletion['operations'],path=log)
    ranked=memory_first_candidates(g,p)
    assert ranked[0]['operations'][0]['operation']=='drop_column'
    assert ranked[0]['historical_approval_count']==1
    assert ranked[1]['decision']=='no_change'
