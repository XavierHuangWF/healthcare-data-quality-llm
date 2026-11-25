from pathlib import Path
from types import SimpleNamespace
import io
import shutil
import pytest
from openpyxl import load_workbook
from healthcare_data_quality.cleaning.visual_flags import contextual_visual_request, requested_visual_flag, row_flag_preview
from healthcare_data_quality.cleaning.one_click_visual_flag import apply_explicit_visual_flag_once
from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download
from healthcare_data_quality.cleaning.final_export_integrity import verify_final_csv, verify_final_excel
from healthcare_data_quality.services.model_plan_review import dataset_review_context, review_visual_proposal
from test_cleaning_review_cards import StubUI, _app
from test_explicit_row_highlight_and_review_controls_20261124 import _finding, ROOT

@pytest.fixture
def source(tmp_path):
    p = tmp_path / 'dirtyL7PVDforXav.csv'
    shutil.copyfile(ROOT / 'data' / p.name, p)
    return p

def test_context_scope_and_font_only_export(source):
    first = contextual_visual_request('highlight this row')
    second = contextual_visual_request('only make font red and bold', first)
    third = contextual_visual_request('i meant whole row', second)
    fourth = contextual_visual_request('highlight whole row', third)
    for request in [second, third, fourth]:
        plan = requested_visual_flag(request, _finding(), source)
        assert plan['scope'] == 'rows'
        assert len(plan['flags']) == 106
        assert all(f['fill'] is False for f in plan['flags'])
    assert contextual_visual_request('replace 0.2 with 2', fourth) is None
    frame = row_flag_preview(plan['flags'], source).data
    assert frame.shape == (2,108) and frame.iloc[0, 2:].tolist() == frame.iloc[1, 2:].tolist()
    original = source.read_bytes()
    state = {'cleaning_approved_reviewer':'Researcher'}
    apply_explicit_visual_flag_once(state, _finding(), source, fourth, previously_displayed=plan)
    _, xlsx = build_styled_excel_download(original, original_csv=original, original_filename=source.name,
        approved_flags=plan['flags'], approved_row_operations=[], approved_column_deletions=[])
    workbook = load_workbook(io.BytesIO(xlsx))
    assert all(workbook.active.cell(11, c).font.bold and workbook.active.cell(11,c).font.color.rgb == 'FFDC2626'
               and workbook.active.cell(11,c).fill.patternType is None for c in range(1,107))
    verify_final_csv(source, None, original, history=state['cleaning_history'],
        review_statuses=state['cleaning_review_statuses'], finding_titles={_finding()['id']:_finding()['title']},
        approved_rows=[], approved_columns=[], review_decisions=state['cleaning_decisions'])
    verify_final_excel(xlsx, original, original_csv=original, approved_flags=plan['flags'], approved_rows=[], approved_columns=[])


def test_context_tracks_other_findings_and_actual_approved_flag_coordinates():
    state = {'cleaning_review_catalog_snapshot': [_finding(), {'id':'other','dataset_id':'dataset_1','title':'Another issue','evidence':[{'CSV row':11,'Column':7}]}]}
    before = dataset_review_context(state,'dataset_1',_finding()['id'])
    assert before['findings'][0]['id'] == 'other'
    state['cleaning_history'] = [{'dataset_id':'dataset_1','finding_id':'other','operation':'flag_cells','flags':[{'row':11,'column':7,'before':'0.2'}]}]
    after = dataset_review_context(state,'dataset_1',_finding()['id'])
    assert before['revision'] != after['revision']
    assert after['approved_actions'][0]['flag_columns'] == [7]
    state['cleaning_history'][0]['flags'][0]['column'] = 11
    assert dataset_review_context(state,'dataset_1',_finding()['id'])['revision'] != after['revision']


def test_async_gemma_review_and_exact_single_button_tree(monkeypatch,source):
    from healthcare_data_quality.services import model_plan_review
    from healthcare_data_quality.jobs.interactive_chat_jobs import interactive_chat_manager
    class UI(StubUI):
        def __init__(self):
            super().__init__(); self.labels=[]
        def chat_message(self,*a,**kw): return self
        def chat_input(self,*a,**kw): return None
        def button(self,label,*,key=None,**kwargs):
            self.labels.append((label,key)); return False
    ui=UI(); app=_app(monkeypatch,ui); finding=_finding(); fid=finding['id']; uploads=[SimpleNamespace(name=source.name)]
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings',lambda *a,**kw:[finding])
    monkeypatch.setattr(app,'_background_agents_idle_for_run',lambda *a:(False,False))
    manager=interactive_chat_manager(); jobs={}; counter=[0]
    def submit(action,*args,**kw):
        counter[0]+=1; key=str(counter[0]); jobs[key]=(action,args); return key,'C'
    monkeypatch.setattr(manager,'submit_routed',submit)
    monkeypatch.setattr(manager,'take',lambda k:{'status':'running'})
    monkeypatch.setattr(model_plan_review,'query_local_model',lambda *a,**kw:'{"assessment":"supported","finding_scope_matches":true,"contains_unrequested_change":false,"reason":"Exact row and style match the request."}')
    ui.session_state.hdq_registered_reviewer='Researcher'
    def render(): app.render_cleaning_workspace([source],uploads,{},'contextual-flag-test',[])
    render(); state=ui.session_state; state.cleaning_discussion_mode_by_finding[fid]='alternative'
    def request(text):
        app._queue_agent_c_for_finding(finding,source,uploads,[source],text)
        assert not state.cleaning_pending_visual_flags_by_finding.get(fid)
        ui.labels.clear(); render()
        assert not any(k and k.startswith('flag_apply_verified_') for _,k in ui.labels)
        key=state.cleaning_pending_gemma_b[fid]['job_id']; fn,args=jobs[key]
        state.cleaning_completed_gemma={fid:{'status':'done','result':fn(*args)}}
        ui.labels.clear(); render()
        assert len([k for _,k in ui.labels if k and k.startswith('flag_apply_verified_')]) == 1
        assert len([k for _,k in ui.labels if k and k.startswith('flag_no_')]) == 1
        assert len([k for _,k in ui.labels if k and k.startswith('review_close_any_')]) == 1
        assert not any(k and k.startswith('review_clarify_yes_') for _,k in ui.labels)
        assert not state.cleaning_history
    request('highlight this row')
    request('only make font red and bold')
    request('i meant whole row')
    assert len(state.cleaning_pending_visual_flags_by_finding[fid]['flags']) == 106
    assert all(f['fill'] is False for f in state.cleaning_pending_visual_flags_by_finding[fid]['flags'])


def test_rejected_or_stale_gemma_review_cannot_publish_action(monkeypatch, source):
    from healthcare_data_quality.services import model_plan_review
    from healthcare_data_quality.jobs.interactive_chat_jobs import interactive_chat_manager
    from test_explicit_row_highlight_and_review_controls_20261124 import _mock_visual_worker
    ui = StubUI(); app = _app(monkeypatch, ui)
    fid = _finding()['id']; ds = _finding()['dataset_id']; state = ui.session_state
    state.update({'cleaning_pending_gemma_b':{}, 'cleaning_messages_by_finding':{fid:[{'role':'user','content':'highlight this row'}]},
        'cleaning_review_statuses':{}, 'cleaning_history':[], 'cleaning_discussion_mode_by_finding':{},
        'cleaning_pending_visual_flags_by_finding':{}, 'cleaning_control_generation_by_finding':{},
        'cleaning_review_catalog_snapshot':[_finding()]})
    finish = _mock_visual_worker(monkeypatch, app)
    plan = {**requested_visual_flag('highlight this row', _finding(), source), 'request':'highlight this row'}
    app._submit_visual_review(_finding(), source, plan, 'highlight this row')
    old_job = state.cleaning_pending_gemma_b[fid]['job_id']
    finish(state, fid)
    # A different finding's new approval changes the review revision even when CSV bytes stay identical.
    state.cleaning_history.append({'dataset_id':ds,'finding_id':'finding5','operation':'flag_cells', 'flags':[{'row':10,'column':1,'before':'other'}]})
    app._sync_gemma_conversations(_finding(), source)
    assert fid not in state.cleaning_pending_visual_flags_by_finding
    assert state.cleaning_pending_gemma_b[fid]['job_id'] != old_job
    monkeypatch.setattr(model_plan_review,'query_local_model',lambda *a,**kw:'{"assessment":"reject","finding_scope_matches":false,"contains_unrequested_change":true,"reason":"The request is ambiguous."}')
    finish(state, fid); app._sync_gemma_conversations(_finding(),source)
    assert fid not in state.cleaning_pending_visual_flags_by_finding
    assert fid not in state.cleaning_pending_gemma_b
    assert 'Annotation review did not complete' in state.cleaning_messages_by_finding[fid][-1]['content']
    assert not any(e.get('finding_id') == fid for e in state.cleaning_history)


def test_cross_finding_annotation_conflicts_and_reopen_restores_earlier_style(source):
    from healthcare_data_quality.cleaning.proposal_conflicts import proposal_conflicts
    from healthcare_data_quality.cleaning.visual_flags import visual_flags_from_history
    flag = requested_visual_flag('highlight these cells', _finding(), source)['flags'][0]
    old = {'operation':'flag_cells','dataset_id':'dataset_1','finding_id':'finding1','reviewer':'Researcher','flags':[flag]}
    new = {**old,'finding_id':'finding5','flags':[{**flag,'fill':False}]}
    assert visual_flags_from_history([old,new])['dataset_1'][0]['fill'] is False
    restored = visual_flags_from_history([old])['dataset_1'][0]
    assert restored['fill'] is True and restored['finding_id'] == 'finding1'
    op = {'operation':'replace_cells','changes':[{'row':flag['row'],'column':flag['column'],'before':flag['before'],'after':'2'}]}
    issues = proposal_conflicts([op],source,source.read_bytes(),approved_flags=[flag])
    assert not issues  # A coordinate annotation is not a value lock.
    assert not proposal_conflicts([op],source,source.read_bytes(),approved_flags=[])
