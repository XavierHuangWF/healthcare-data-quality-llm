"""Regression for row 11 fractional counts: styling is not a value replacement."""
from pathlib import Path
from types import SimpleNamespace
import io
import csv
import shutil
import pytest
from openpyxl import load_workbook
from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag
from healthcare_data_quality.cleaning.one_click_visual_flag import apply_explicit_visual_flag_once
from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download
from test_cleaning_review_cards import StubUI, _app

ROOT = Path(__file__).resolve().parents[1]


def _finding():
    return {'id': 'dataset_1:qc:1', 'dataset_id': 'dataset_1',
            'title': '2 cells contain fractional count values', 'source': 'QC',
            'name': 'dirtyL7PVDforXav.csv', 'check_key': 'find_invalid_counts',
            'method': 'Review counts', 'evidence': [
                {'CSV row': 11, 'Column': 7, 'Original value': '0.2'},
                {'CSV row': 11, 'Column': 11, 'Original value': '0.5'}]}


@pytest.fixture
def source(tmp_path):
    target = tmp_path / 'dirtyL7PVDforXav.csv'
    shutil.copyfile(ROOT / 'data' / target.name, target)
    return target


def test_real_source_entire_row_and_xlsx(source):
    original = source.read_bytes()
    plan = requested_visual_flag('highlight this row', _finding(), source)
    records = list(csv.reader(io.StringIO(original.decode('utf-8-sig'))))
    assert plan['scope'] == 'rows'
    assert [(f['row'], f['column'], f['before']) for f in plan['flags']] == [(11, c, v) for c, v in enumerate(records[10], 1)]
    assert len(plan['flags']) == 106
    assert records[10][6] == '0.2' and records[10][10] == '0.5'
    state = {'cleaning_approved_reviewer': 'Researcher'}
    result = apply_explicit_visual_flag_once(state, _finding(), source, 'highlight this row', previously_displayed=plan)
    assert result['approved'] == 106
    _, xlsx = build_styled_excel_download(original, original_csv=original,
        original_filename=source.name, approved_flags=plan['flags'],
        approved_row_operations=[], approved_column_deletions=[])
    sheet = load_workbook(io.BytesIO(xlsx)).active
    assert all(sheet.cell(11, c).font.bold and sheet.cell(11, c).font.color.rgb == 'FFDC2626' for c in range(1, 107))
    assert not sheet.cell(10, 7).font.bold
    assert sheet.cell(11, 7).value == 0.2 and sheet.cell(11, 11).value == 0.5
    assert source.read_bytes() == original


def test_explicit_taxonomy_and_cells_have_distinct_scopes(source):
    for request, expected in [('highlight the taxonomy of this row', [1]),
                              ('highlight these cells', [7, 11]),
                              ('highlight row 11 columns 7 and 11', [7, 11])]:
        plan = requested_visual_flag(request, _finding(), source)
        assert [f['column'] for f in plan['flags']] == expected
    bad = {**_finding(), 'evidence': [{'CSV row': 11, 'Column': 7}, {'CSV row': 12, 'Column': 7}]}
    with pytest.raises(ValueError, match='requested 1 rows'):
        requested_visual_flag('highlight this row', bad, source)
    with pytest.raises(ValueError, match='separately'):
        requested_visual_flag('highlight this row and replace 0.2', _finding(), source)
    with pytest.raises(ValueError, match='does not exist'):
        requested_visual_flag('highlight row 11 column 200', _finding(), source)


def test_stale_deleted_target_cannot_be_approved(source):
    request = 'highlight this row'
    plan = requested_visual_flag(request, _finding(), source)
    state = {'cleaning_approved_reviewer': 'Researcher', 'cleaning_approved_columns_by_dataset': {'dataset_1': [{'position': 11}]}}
    with pytest.raises(PermissionError, match='removed'):
        apply_explicit_visual_flag_once(state, _finding(), source, request, previously_displayed=plan)
    assert not state.get('cleaning_history')


def _mock_visual_worker(monkeypatch, app):
    from healthcare_data_quality.jobs.interactive_chat_jobs import interactive_chat_manager
    from healthcare_data_quality.services import model_plan_review
    jobs = {}
    manager = interactive_chat_manager()
    monkeypatch.setattr(app, '_background_agents_idle_for_run', lambda *a: (False, False))
    monkeypatch.setattr(model_plan_review, 'query_local_model', lambda *a, **kw:
        '{"assessment":"supported","finding_scope_matches":true,"contains_unrequested_change":false,"reason":"Exact requested scope."}')
    def submit(action, *args, **kwargs):
        key = str(len(jobs)+1)
        jobs[key] = (action, args)
        return key, 'C'
    monkeypatch.setattr(manager, 'submit_routed', submit)
    monkeypatch.setattr(manager, 'take', lambda *a: {'status':'running'})
    def finish(state, fid):
        key = state.cleaning_pending_gemma_b[fid]['job_id']
        action, args = jobs[key]
        state.setdefault('cleaning_completed_gemma', {})[fid] = {'status':'done', 'result':action(*args)}
    return finish


def test_direct_row_highlight_requires_gemma_before_single_control_set(monkeypatch, source):
    class TrackingUI(StubUI):
        def __init__(self):
            super().__init__()
            self.rendered_buttons = []
        def chat_message(self, *args, **kwargs):
            return self
        def button(self, label, *, key=None, **kwargs):
            self.rendered_buttons.append((label, key))
            return super().button(label, key=key, **kwargs)
    st = TrackingUI()
    app = _app(monkeypatch, st)
    finding = _finding()
    fid = finding['id']
    upload = [SimpleNamespace(name=source.name)]
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings', lambda *a, **k: [finding])
    st.session_state.hdq_registered_reviewer = 'Researcher'
    app.render_cleaning_workspace([source], upload, {}, 'row-highlight-regression', [])
    state = st.session_state
    state.cleaning_discussion_mode_by_finding[fid] = 'alternative'
    state.cleaning_plans_by_finding[fid] = {'operation': 'replace_cells'}
    finish = _mock_visual_worker(monkeypatch, app)
    app._queue_agent_c_for_finding(finding, source, upload, [source], 'highlight this row')
    assert not state.cleaning_plans_by_finding.get(fid)
    assert not state.cleaning_pending_visual_flags_by_finding.get(fid)
    assert len(state.cleaning_pending_gemma_b[fid]['plan']['flags']) == 106
    finish(state, fid)
    st.rendered_buttons.clear()
    app.render_cleaning_workspace([source], upload, {}, 'row-highlight-regression', [])
    buttons = st.rendered_buttons
    assert len([k for label,k in buttons if k and k.startswith('flag_apply_verified_')]) == 1
    assert len([k for label,k in buttons if k and k.startswith('flag_no_')]) == 1
    assert len([k for label,k in buttons if k and k.startswith('review_close_any_')]) == 1
    assert not any(label.startswith('Apply 2 verified cell changes') for label,k in buttons)
    assert not state.cleaning_history


def test_recover_old_failure_without_reviving_replacement(monkeypatch, source):
    st = StubUI()
    app = _app(monkeypatch, st)
    fid = _finding()['id']
    st.session_state.update({'cleaning_messages_by_finding': {fid: [
        {'role': 'user', 'content': 'highlight this row'},
        {'role': 'assistant', 'content': 'The requested bulk action was not prepared: Gemma did not support the exact requested plan'},
        {'role': 'user', 'content': 'Prepare the exact structured operation(s)', 'automatic': True},
        {'role': 'assistant', 'content': '1. Replace 2 cells'}]},
        'cleaning_plans_by_finding': {fid: {'operation': 'replace_cells'}},
        'cleaning_discussion_mode_by_finding': {fid:'alternative'},
        'cleaning_review_statuses': {}, 'cleaning_history': []})
    _mock_visual_worker(monkeypatch, app)
    app._recover_legacy_failed_visual_request(fid, _finding(), source)
    assert not st.session_state.cleaning_plans_by_finding.get(fid)
    assert not st.session_state.get('cleaning_pending_visual_flags_by_finding', {}).get(fid)
    assert len(st.session_state.cleaning_pending_gemma_b[fid]['plan']['flags']) == 106
    st.session_state.cleaning_pending_gemma_b.pop(fid)
    app._recover_legacy_failed_visual_request(fid, _finding(), source)
    assert not st.session_state.cleaning_pending_gemma_b.get(fid)


def test_row_highlight_chat_submit_uses_fragment_and_native_preview(monkeypatch, source):
    from test_interactive_two_agent_chat import ChatStub

    class TrackingChat(ChatStub):
        def __init__(self):
            super().__init__()
            self.rendered_buttons = []
            self.tables = []
            self.rerun_scopes = []
            self.html_previews = []
        def rerun(self, *, scope=None):
            self.rerun_scopes.append(scope)
            raise StopIteration('rerun')
        def button(self, label, *, key=None, **kwargs):
            self.rendered_buttons.append((label, key, kwargs))
            return super().button(label, key=key, **kwargs)
        def dataframe(self, data, *args, **kwargs):
            self.tables.append(getattr(data, 'data', data))
        def markdown(self, body, **kwargs):
            if isinstance(body, str) and '<table' in body:
                self.html_previews.append(body)

    st = TrackingChat()
    app = _app(monkeypatch, st)
    finding = _finding()
    fid = finding['id']
    upload = [SimpleNamespace(name=source.name)]
    raw = source.read_bytes()
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings', lambda *a, **k: [finding])
    st.session_state.hdq_registered_reviewer = 'Researcher'
    def render():
        app.render_cleaning_workspace([source], upload, {}, 'row-highlight-chat-submit', [])
    render()
    state = st.session_state
    state.cleaning_discussion_mode_by_finding[fid] = 'alternative'
    finish = _mock_visual_worker(monkeypatch, app)
    st.submitted_text[f'review_message_{fid}'] = 'highlight this row'
    st.sent_buttons.add('Send to Agent C')
    with pytest.raises(StopIteration, match='rerun'):
        render()
    assert st.rerun_scopes == ['fragment']
    assert len(state.cleaning_pending_gemma_b[fid]['plan']['flags']) == 106
    assert not state.cleaning_history
    finish(state, fid)
    st.tables.clear()
    st.rendered_buttons.clear()
    render()
    tables = [t for t in st.tables if hasattr(t, 'columns') and list(t.columns[:2]) == ['CSV row', 'Preview']]
    assert len(tables) == 1 and tables[0].shape == (2,108)
    assert tables[0].iloc[0,2:].equals(tables[0].iloc[1,2:])
    assert not st.html_previews
    approvals = [(label, kwargs) for label, key, kwargs in st.rendered_buttons if key and key.startswith('flag_apply_verified_')]
    assert len(approvals) == 1 and not approvals[0][1]['disabled']
    assert sum(bool(key and key.startswith('flag_no_')) for _, key, _ in st.rendered_buttons) == 1
    assert sum(bool(key and key.startswith('review_close_any_')) for _, key, _ in st.rendered_buttons) == 1
    assert not any(label.startswith('Apply 2 verified cell changes') for label, _, _ in st.rendered_buttons)
    assert source.read_bytes() == raw

    # A render failure must not create an approval with an invisible preview.
    monkeypatch.setattr(app, '_render_visual_flag_preview', lambda *a, **k: False)
    st.rendered_buttons.clear()
    render()
    approvals = [kwargs for _, key, kwargs in st.rendered_buttons if key and key.startswith('flag_apply_verified_')]
    assert len(approvals) == 1 and approvals[0]['disabled']
    assert not state.cleaning_history


def test_annotation_preview_pagination_preserves_all_targets(monkeypatch):
    class PagedUI(StubUI):
        def selectbox(self, *args, **kwargs):
            assert kwargs['options'] == [0, 1]
            return 1
        def dataframe(self, data, *args, **kwargs):
            self.frame = data.data
    st = PagedUI()
    app = _app(monkeypatch, st)
    flags = [{'row': 11, 'column': c, 'before': '001' if c == 201 else '', 'style': 'bold_red'} for c in range(1, 207)]
    assert app._render_visual_flag_preview(flags, 'f', phase='pending')
    assert len(flags) == 206
    assert st.frame['Location'].tolist() == [f'R11:C{c}' for c in range(201, 207)]
    assert st.frame['Before'].tolist() == ['001', '', '', '', '', '']
    assert st.frame['After'].equals(st.frame['Before'])


def test_annotation_preview_native_failure_is_recoverable(monkeypatch):
    st = StubUI()
    app = _app(monkeypatch, st)
    def fail(*args, **kwargs):
        raise RuntimeError('table renderer unavailable')
    st.dataframe = fail
    assert not app._render_visual_flag_preview([{'row': 11, 'column': 7, 'before': '0.2'}], 'f', phase='pending')
    assert any('table renderer unavailable' in str(comment) for comment in st.comments)


def test_background_recommendation_poll_does_not_remount_highlight_preview(monkeypatch):
    st = StubUI()
    app = _app(monkeypatch, st)
    st.session_state.cleaning_pending_visual_flags_by_finding = {'f1': {'flags': [{'row': 11, 'column': 7}]}}
    class ReadyAgentB:
        def snapshot(self, job_id):
            return {'entries': {'f2': {'status': 'ready'}}}
    monkeypatch.setattr('healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager', lambda: ReadyAgentB())
    def unexpected_rerun(*args, **kwargs):
        pytest.fail('Background recommendation must not remount a pending visual preview')
    st.rerun = unexpected_rerun
    app._refresh_parent_for_new_agent_b_choices('job-1', frozenset())
