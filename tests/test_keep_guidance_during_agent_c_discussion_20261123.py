"""Regression coverage: No/Discuss retains rejected context internally, never in the UI."""
from pathlib import Path

from healthcare_data_quality.ui.discussion_reference import (
    capture_discussion_reference, should_show_discussion_reference,
)

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def test_captured_context_preserves_explanation_and_before_after():
    original = {"suggestion": "Delete three blank control columns"}
    options = [{"label": "Delete columns 2, 3, 4", "rows": [
        {"Location": "Column 2", "Before": "Rep_Blank", "After": "Delete column"},
    ], "operations": [{"operation": "drop_column", "column": 2}], "gemma_token": "private"}]
    saved = capture_discussion_reference(original, options)
    assert saved["summary"] == "Delete three blank control columns"
    assert saved["choices"][0]["rows"][0] == {
        "Location": "Column 2", "Before": "Rep_Blank", "After": "Delete column"
    }
    assert "operations" not in saved["choices"][0]
    assert "gemma_token" not in saved["choices"][0]
    options[0]["rows"][0]["After"] = "Altered later"
    assert saved["choices"][0]["rows"][0]["After"] == "Delete column"


def test_no_phase_displays_rejected_reference():
    context = {"summary": "Keep earlier guidance", "choices": []}
    assert not should_show_discussion_reference("alternative", False, context)
    assert not should_show_discussion_reference("alternative", True, context)
    assert not should_show_discussion_reference("accepted", False, context)
    assert not should_show_discussion_reference(None, False, context)
    assert not should_show_discussion_reference("alternative", False, None)


def test_application_captures_context_after_verified_scope_filters():
    start = APP.index("def _render_live_agent_b_finding(")
    stop = APP.index("def _expand_ready_finding_without_rerun", start)
    section = APP[start:stop]
    assert section.index("reviewer_selectable_choices(") < section.index("filter_conflicting_choices(")
    assert section.index("filter_conflicting_choices(") < section.index("capture_discussion_reference(suggestion, options)")


def test_discussion_reference_does_not_generate_yes_no_or_auto_submit():
    section = APP.split("def _render_prior_agent_b_guidance(", 1)[1].split(
        "def _render_live_agent_b_finding(", 1)[0]
    assert "st.button(" not in section
    assert "_queue_agent_c_for_finding(" not in section
    assert 'st.dataframe(pd.DataFrame(choice["rows"])' not in section


def test_discuss_retains_capture_and_reset_is_scoped_to_investigation():
    method = APP.split("def _open_agent_c_alternative_chat(", 1)[1].split("def _recommendation_preview", 1)[0]
    assert "cleaning_prior_guidance_by_finding" not in method
    assert "cleaning_prior_guidance_by_finding = {}" in APP
    assert '("cleaning_prior_guidance_by_finding", {})' in APP
    rendering = APP.split("def render_cleaning_workspace(",1)[1]
    assert "_render_prior_agent_b_guidance(fid, completed=completed)" not in rendering


def test_rejected_reference_panel_is_hidden_during_processing(monkeypatch):
    from test_cleaning_review_cards import StubUI, _app

    class UI(StubUI):
        def __init__(self):
            super().__init__()
            self.headings = []
            self.controls = []
        def markdown(self, text, **kwargs):
            self.headings.append(str(text))
        def write(self, text, **kwargs):
            self.headings.append(str(text))
        def button(self, text, **kwargs):
            self.controls.append(text)
            return False

    ui = UI()
    app = _app(monkeypatch, ui)
    fid = "gemma-check-1"
    ui.session_state.cleaning_discussion_mode_by_finding = {fid: "alternative"}
    ui.session_state.cleaning_prior_guidance_by_finding = {fid: capture_discussion_reference(
        {"suggestion": "Remove three blank controls"}, [{
            "label": "Delete columns 2, 3 and 4",
            "rows": [{"Location": "Column 2", "Before": "Rep_Blank", "After": "Delete column"}]
        }]
    )}
    app._render_prior_agent_b_guidance(fid)
    assert not ui.headings and not ui.comments and not ui.controls
    assert ui.session_state.cleaning_prior_guidance_by_finding[fid]['summary'] == 'Remove three blank controls'
    ui.headings.clear()
    ui.comments.clear()
    app._render_prior_agent_b_guidance(fid, completed=True)
    assert not ui.headings and not ui.comments


def test_zero_values_in_before_after_not_lost():
    saved = capture_discussion_reference({"suggestion": "Set to zero"}, [{
        "label": "Set zero", "rows": [{"Location": "R2:C3", "Before": 0, "After": 0}]
    }])
    assert saved["choices"][0]["rows"][0]["Before"] == "0"
    assert saved["choices"][0]["rows"][0]["After"] == "0"


def test_no_retires_all_choices_and_preserves_conversation_across_findings(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from test_cleaning_review_cards import _app
    from test_interactive_two_agent_chat import ChatStub
    from test_explicit_row_highlight_and_review_controls_20261124 import _finding, ROOT, _mock_visual_worker
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    from healthcare_data_quality.services.model_plan_review import option_fingerprint
    from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
    import shutil
    import pytest

    class UI(ChatStub):
        def __init__(self):
            super().__init__()
            self.events = []
            self.callbacks = {}
        def button(self, label, *, key=None, **kwargs):
            self.events.append(('button', label, key, kwargs.get('disabled', False)))
            if kwargs.get('on_click'):
                self.callbacks[key] = (kwargs['on_click'], kwargs.get('args', ()))
            return super().button(label, key=key, **kwargs)
        def chat_input(self, *args, **kwargs):
            self.events.append(('chat', kwargs['key']))
            return super().chat_input(*args, **kwargs)
        def dataframe(self, data, *args, **kwargs):
            self.events.append(('table', getattr(data, 'data', data)))
        def markdown(self, text, **kwargs):
            self.events.append(('markdown', text))
        def caption(self, text, **kwargs):
            self.events.append(('caption', text))

    source = tmp_path / 'dirtyL7PVDforXav.csv'
    shutil.copyfile(ROOT / 'data' / source.name, source)
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'checkpoint'))
    ui = UI(); app = _app(monkeypatch, ui)
    finding = _finding(); fid = finding['id']
    other = {**finding, 'id': 'gemma:second:check', 'source': 'Gemma check'}
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings', lambda *a, **k: [finding, other])
    monkeypatch.setattr(app, '_background_agents_idle_for_run', lambda *a: (False, False))
    monkeypatch.setattr(app, '_prior_approval_closure', lambda *a: None)
    # Both cards are ready, with a verified initial proposal.
    suggestion = {'suggestion': 'Mark fractional counts as NA'}
    opts = scoped_review_choices(finding, source, source_backed_choices(finding, source, suggestion))
    assert opts
    reviews = [{'fingerprint': option_fingerprint(finding, source, o), 'can_offer': True, 'reason': 'Exact scope verified'} for o in opts]
    entry = {'status': 'ready', **suggestion, 'gemma_option_reviews': reviews}
    class Ready:
        def snapshot(self, *a): return {'entries': {fid: entry, other['id']: {'status':'failed','error':'test'}}}
    monkeypatch.setattr('healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager', lambda: Ready())
    ui.session_state.hdq_registered_reviewer = 'Researcher'
    uploads = [SimpleNamespace(name=source.name)]
    def render():
        ui.events.clear(); ui.callbacks.clear(); ui.visible.clear()
        app.render_cleaning_workspace([source], uploads, {}, 'exclusive-review-regression', [])
    def buttons_for(card):
        return [e for e in ui.events if e[0]=='button' and e[2] and card in e[2]]
    render(); state = ui.session_state
    initial = buttons_for(fid)
    no = next(e for e in initial if e[1]=='No — Discuss Another Idea')
    assert initial[-1][1].startswith('Mark reviewed')
    yes = next(e for e in initial if e[2].startswith('review_yes_'))
    i = ui.events.index(yes)
    assert any(e[0]=='table' for e in ui.events[:i])
    # Exercise the real pre-render callback, not a synthetic No click after the old UI has painted.
    cb, args = ui.callbacks[no[2]]; cb(*args)
    assert state.cleaning_messages_by_finding[fid][-1]['reviewer_declined_all']
    assert state.cleaning_prior_guidance_by_finding[fid]['choices']
    render()
    assert [e[1] for e in buttons_for(fid)] == ['Mark reviewed — leave unchanged']
    assert any(e[0]=='chat' and fid in e[1] for e in ui.events)
    assert ('user', 'No — Discuss Another Idea') in ui.visible
    assert not any('Previous Agent B guidance' in str(e) for e in ui.events)
    assert not state.cleaning_pending_gemma_b
    assert other['id'] not in state.cleaning_discussion_mode_by_finding
    # Late B refresh cannot change the rejection state or restore its old option.
    render()
    assert [e[1] for e in buttons_for(fid)] == ['Mark reviewed — leave unchanged']
    finish = _mock_visual_worker(monkeypatch, app)
    app._queue_agent_c_for_finding(finding, source, uploads, [source], 'highlight this row')
    render()
    assert len(buttons_for(fid)) == 1 and buttons_for(fid)[0][3]  # only disabled footer while processing
    finish(state, fid); render()
    active = buttons_for(fid)
    assert len(active)==3 and active[0][1].startswith('Highlight')
    assert active[1][1]=='No — Discuss Another Idea' and active[-1][1].startswith('Mark reviewed')
    assert not any(e[0]=='chat' and fid in e[1] for e in ui.events)
    assert ('user','highlight this row') in ui.visible and ('user','No — Discuss Another Idea') in ui.visible
    assert any(e[0]=='table' and hasattr(e[1],'shape') and e[1].shape==(2,108) for e in ui.events[:ui.events.index(active[0])])
    # Declining the newly reviewed annotation preserves the conversation but clears its draft.
    cb, args=ui.callbacks[active[1][2]]; cb(*args)
    render()
    assert [e[1] for e in buttons_for(fid)]==['Mark reviewed — leave unchanged']
    assert ('user','highlight this row') in ui.visible
    assert not state.cleaning_pending_visual_flags_by_finding.get(fid)
    assert not state.cleaning_history


def test_pending_response_has_priority_over_stale_plan(monkeypatch):
    from test_cleaning_review_cards import StubUI, _app
    app = _app(monkeypatch, StubUI())
    assert app._agent_c_control_phase(mode='alternative', transcript=[],
        plan={'operation':'drop_rows'}, queued_actions=[], pending=True) == ('processing', [])


def test_model_prompt_keeps_rejection_and_latest_request_outside_truncated_evidence(monkeypatch):
    import json
    from healthcare_data_quality.cleaning import cleaning_workspace
    prompts = []
    def fake(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({'reply':'Please specify a replacement value.', 'operation':'none',
                           'dataset_id':'','changes':[], 'clarification_options':[]})
    monkeypatch.setattr(cleaning_workspace, 'query_local_model', fake)
    finding = {'finding_id':'f1', 'dataset_id':'dataset_1', 'evidence':[{'text':'x'*9000}],
               'reviewer_discussion_instruction':'All prior options rejected',
               'rejected_suggestions_context':{'summary':'Mark 2 counts NA'},
               'rejected_discussion_proposals':[{'operation':'drop_rows'}]}
    cleaning_workspace.ask_gemma_about_cleaning('Explain a different approach', [], [finding], [],
                                                selected_finding=finding, force_model=True)
    assert prompts
    prompt = prompts[0]
    assert 'ALL earlier unapproved choices were declined' in prompt
    assert 'REJECTED HISTORICAL OPTIONS' in prompt and 'Mark 2 counts NA' in prompt
    assert 'Latest reviewer request (current intent, not approval)' in prompt
    assert 'Explain a different approach' in prompt
