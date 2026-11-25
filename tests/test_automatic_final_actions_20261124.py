"""Final-action controls: no manual assessment or draft-selection workflow."""
from pathlib import Path
from healthcare_data_quality.services import model_plan_review as review
from healthcare_data_quality.services import recommendation_options as ro
from healthcare_data_quality.cleaning.finding_cleaning import build_cleaning_findings
from healthcare_data_quality.investigation.baseline_audit import run_baseline_audit
from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
from test_cleaning_review_cards import StubUI, _app

SRC = Path(__file__).resolve().parents[1]/'data'/'dirtyL7PVDforXav.csv'


def missing():
    catalog = build_cleaning_findings({'baseline': run_baseline_audit(SRC)}, [SRC], [SRC.name], [])
    return next(f for f in catalog if '618 cells contain missing values' in f['title'])


def isolate_memory(monkeypatch):
    monkeypatch.setattr(ro, '_matching_approved_memory_choice', lambda *a, **k: None)
    monkeypatch.setattr('healthcare_data_quality.ui.human_choice_replay.verified_approved_choices', lambda *a, **k: [])
    monkeypatch.setattr('healthcare_data_quality.ui.human_choice_replay.approval_counts', lambda *a, **k: __import__('collections').Counter())


def test_complete_missing_scope_suppresses_accidental_sample_duplicate(monkeypatch):
    isolate_memory(monkeypatch)
    f = missing(); f = {**f, 'evidence': f['evidence'][:32]}
    cells = [{'row': e['CSV row'], 'column': e['Column'], 'before': e['Original value'], 'after':'NA'} for e in f['evidence'][:10]]
    candidate = {'label':'Mark sample as NA','operations':[{'operation':'replace_cells','changes':cells}]}
    opts = ro.source_backed_choices(f, SRC, {'choices':[candidate], 'suggestion':"Set to NA — row 2, column 5: '' -> 'NA'"})
    na = [o for o in opts if o['operations'][0]['operation']=='replace_cells']
    assert len(na)==1 and len(na[0]['operations'][0]['changes'])==618
    assert any(o['operations'][0]['operation']=='drop_rows' for o in opts)


def test_transient_failure_retries_exact_scope_once(monkeypatch):
    isolate_memory(monkeypatch)
    f=missing(); opt=ro.source_backed_choices(f,SRC,{})[0]; calls=[]
    def assess(finding, source, option, **kwargs):
        calls.append(review.option_fingerprint(finding,source,option))
        if len(calls)==1: raise TimeoutError('transient')
        return {'fingerprint': calls[-1], 'can_offer':True, 'scope_matches':True, 'has_extra_changes':False}
    monkeypatch.setattr(review,'assess_review_option',assess)
    result=review.assess_exact_options(f,SRC,[opt],{})
    assert len(calls)==2 and calls[0]==calls[1] and result[0]['can_offer']


def test_terminal_scope_rejection_is_not_retried(monkeypatch):
    isolate_memory(monkeypatch)
    f=missing(); opt=ro.source_backed_choices(f,SRC,{})[0]; calls=[]
    def assess(finding, source, option, **kwargs):
        calls.append(option)
        return {'fingerprint':review.option_fingerprint(finding,source,option),'can_offer':False,'scope_matches':False,'has_extra_changes':True,'review_policy':review.EXACT_REVIEW_POLICY_VERSION}
    monkeypatch.setattr(review,'assess_review_option',assess)
    result=review.assess_exact_options(f,SRC,[opt],{})
    assert len(calls)==1 and not result[0]['can_offer']
    assert review.reviewer_selectable_choices(f,SRC,[opt],result)==[]


class UI(StubUI):
    def __init__(self):
        super().__init__(); self.rendered=[]
    def button(self,label,*,key=None,**kwargs):
        self.rendered.append((str(label),str(key))); return False


def render_state(monkeypatch, f, suggestion):
    ui=UI(); app=_app(monkeypatch,ui)
    ui.session_state.update(cleaning_approved_reviewer='Reviewer',cleaning_recommendations={f['id']:suggestion},
        cleaning_plans_by_finding={},cleaning_batch_queue_by_finding={},cleaning_pending_gemma_b={},
        cleaning_discussion_mode_by_finding={},cleaning_history=[],cleaning_review_statuses={})
    return app,ui


def test_no_manual_button_or_unassessed_preview_then_final_choices(monkeypatch):
    isolate_memory(monkeypatch)
    f=missing(); suggestion={'suggestion':'','choices':[],'gemma_option_reviews':[]}
    app,ui=render_state(monkeypatch,f,suggestion)
    scheduled=[]
    def queue(fid,finding,source,sug,candidates):
        scheduled.extend(candidates); return True
    monkeypatch.setattr(app,'_queue_exact_option_recovery',queue)
    app._render_live_agent_b_finding(f['id'],f,SRC,[],[SRC],None,'dataset_1')
    assert len(scheduled)==2
    assert all('Reassess' not in label and 'Find alternatives' not in label for label,key in ui.rendered)
    assert not any(key.startswith('review_yes_') for label,key in ui.rendered)
    assert not any(isinstance(x,tuple) and x[0]=='table' for x in ui.comments)
    # Deliver valid reviews; no preliminary click is needed for either final action.
    suggestion['gemma_option_reviews']=[{'fingerprint':review.option_fingerprint(f,SRC,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in scheduled]
    ui.rendered.clear();ui.comments.clear()
    app._render_live_agent_b_finding(f['id'],f,SRC,[],[SRC],None,'dataset_1')
    labels=[label for label,key in ui.rendered if key.startswith('review_yes_')]
    assert len(labels)==2 and any('618' in label for label in labels) and any('Delete' in label for label in labels)
    assert not ui.session_state.cleaning_history


def test_recovery_exhausted_never_shows_unverified_action(monkeypatch):
    isolate_memory(monkeypatch)
    f=missing(); suggestion={'suggestion':'','choices':[],'gemma_option_reviews':[]}
    app,ui=render_state(monkeypatch,f,suggestion)
    monkeypatch.setattr(app,'_queue_exact_option_recovery',lambda *a,**k:False)
    monkeypatch.setattr(app,'_queue_automatic_option_draft',lambda *a,**k:False)
    app._render_live_agent_b_finding(f['id'],f,SRC,[],[SRC],None,'dataset_1')
    assert not ui.rendered  # an empty initial recommendation remains locked
    assert not any(isinstance(x,tuple) and x[0]=='table' for x in ui.comments)


def test_gemma_check_automatic_review_then_two_actions(monkeypatch):
    isolate_memory(monkeypatch)
    f={'id':'gemma:1:check','dataset_id':'dataset_1','source':'Gemma check','check_key':'gemma_check',
       'title':'Blank controls Rep_Blank, P7_Blank, P6_Blank should be zero','evidence':[],
       'gemma_check_details':{'proposal':{'hypothesis':'Blank controls Rep_Blank, P7_Blank, P6_Blank should be zero'}}}
    suggestion={'suggestion':'','gemma_option_reviews':[]}
    app,ui=render_state(monkeypatch,f,suggestion)
    scheduled=[]
    def queue(fid,finding,source,sug,candidates):
        scheduled.extend(candidates); return True
    monkeypatch.setattr(app,'_queue_exact_option_recovery',queue)
    app._render_live_agent_b_finding(f['id'],f,SRC,[],[SRC],None,'dataset_1')
    assert len(scheduled)==2
    assert not any(key.startswith('review_yes_') for label,key in ui.rendered)
    suggestion['gemma_option_reviews']=[{'fingerprint':review.option_fingerprint(f,SRC,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in scheduled]
    ui.rendered.clear();ui.comments.clear()
    app._render_live_agent_b_finding(f['id'],f,SRC,[],[SRC],None,'dataset_1')
    labels=[label for label,key in ui.rendered if key.startswith('review_yes_')]
    assert len(labels)==2 and any('Keep' in label for label in labels) and any('Delete' in label for label in labels)
    assert not ui.session_state.cleaning_history


def test_automatic_draft_once_and_decline_blocks_new_drafts(monkeypatch):
    f=missing(); app,ui=render_state(monkeypatch,f,{})
    submitted=[]
    monkeypatch.setattr(app,'_submit_review_worker',lambda *a,**k: submitted.append((a,k)) or True)
    assert app._queue_automatic_option_draft(f['id'],f,SRC,{})
    assert not app._queue_automatic_option_draft(f['id'],f,SRC,{})
    assert len(submitted)==1 and submitted[0][1]['regenerate']
    ui.session_state.cleaning_auto_draft_attempts={}
    ui.session_state.cleaning_discussion_mode_by_finding[f['id']]='alternative'
    ui.session_state.cleaning_messages_by_finding={f['id']:[{'role':'user','content':'No — Discuss Another Idea','reviewer_declined_all':True}]}
    assert not app._queue_automatic_option_draft(f['id'],f,SRC,{})
    assert len(submitted)==1
