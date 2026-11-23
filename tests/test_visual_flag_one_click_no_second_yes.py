"""Regression: prior flag plans never require a second Yes button."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from healthcare_data_quality.cleaning.one_click_visual_flag import apply_explicit_visual_flag_once
from test_cleaning_review_cards import StubUI, _app


def _finding():
    return {
        'id': 'dataset_1:qc:5', 'dataset_id': 'dataset_1',
        'title': 'Four rows have incorrect taxonomy depth', 'source': 'QC', 'name': 'data.csv',
        'check_key': 'inspect_taxonomy_structure', 'method': 'Review taxonomy',
        'evidence': [{'CSV row': row} for row in (2, 3, 4, 5)],
    }


def _source(tmp_path):
    p = tmp_path / 'data.csv'
    p.write_text('Taxon,Count\na,1\nb,2\nc,3\nd,4\n', encoding='utf-8')
    return p


def test_explicit_flag_is_one_click_with_provenance_and_no_mutation(tmp_path):
    source = _source(tmp_path)
    before = source.read_bytes()
    state = {
        'cleaning_approved_reviewer': 'Researcher A',
        'cleaning_review_statuses': {},
        'cleaning_messages_by_finding': {},
    }
    from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag
    command = 'flag these four rows taxonomy bold and red'
    preview = requested_visual_flag(command, _finding(), source)
    with pytest.raises(PermissionError, match='Before.*After'):
        apply_explicit_visual_flag_once(state, _finding(), source, command)
    result = apply_explicit_visual_flag_once(
        state, _finding(), source, command, previously_displayed=preview)
    assert result['approved'] == 4
    assert state['cleaning_review_statuses'][_finding()['id']] == 'Applied and verified'
    assert len(state['cleaning_history']) == 1
    assert len(state['cleaning_visual_flags_by_dataset']['dataset_1']) == 4
    assert state['cleaning_history'][0]['reviewer'] == 'Researcher A'
    assert source.read_bytes() == before
    with pytest.raises(PermissionError, match='already completed'):
        apply_explicit_visual_flag_once(
            state, _finding(), source, 'flag these four rows taxonomy bold and red')
    assert len(state['cleaning_history']) == 1


def test_flag_requires_named_human_and_correct_original_source(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(PermissionError, match='reviewer'):
        apply_explicit_visual_flag_once({}, _finding(), source, 'flag all rows')
    from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag
    request = 'flag these four rows taxonomy bold and red'
    preview = requested_visual_flag(request, _finding(), source)
    source.write_text('Taxon,Count\na,1\nb,2\nc,3\nd,999\n', encoding='utf-8')
    state = {'cleaning_approved_reviewer': 'R'}
    with pytest.raises(PermissionError, match='Displayed Before/After'):
        apply_explicit_visual_flag_once(state, _finding(), source, request,
                                         previously_displayed=preview)
    assert not state.get('cleaning_history')


def test_real_agent_c_direct_flag_request_skips_second_form(monkeypatch, tmp_path):
    source = _source(tmp_path)
    st = StubUI()
    app = _app(monkeypatch, st)
    fid = _finding()['id']
    st.session_state.cleaning_approved_reviewer = 'Researcher A'
    st.session_state.cleaning_recommendations = {fid: {'suggestion': 'Flag taxonomy'}}
    st.session_state.cleaning_discussion_mode_by_finding = {fid: 'alternative'}
    st.session_state.cleaning_pending_gemma_b = {}
    st.session_state.cleaning_review_statuses = {}
    st.session_state.cleaning_messages_by_finding = {fid: []}
    st.session_state.cleaning_plans_by_finding = {}
    st.session_state.cleaning_batch_queue_by_finding = {}
    st.session_state.cleaning_batch_total_by_finding = {}
    st.session_state.cleaning_compound_previews_by_finding = {}
    st.session_state.cleaning_completed_gemma = {}
    st.session_state.cleaning_staged_bytes_by_dataset = {}
    st.session_state.cleaning_history = []
    st.session_state.cleaning_decisions = []
    st.session_state.cleaning_visual_flags_by_dataset = {}
    st.session_state.cleaning_pending_visual_flags_by_finding = {}
    st.session_state.cleaning_r_previews_by_finding = {}
    st.session_state.cleaning_r_errors_by_finding = {}
    st.session_state.cleaning_r_diagnostics_by_finding = {}
    st.session_state.cleaning_r_selfchecks_by_finding = {}
    st.session_state.cleaning_explicit_row_deletions_by_finding = {}
    st.session_state.cleaning_explicit_column_deletions_by_finding = {}
    st.session_state.cleaning_memory_candidates = {}
    st.session_state.cleaning_ui_generation_by_finding = {}
    original = source.read_bytes()

    def prepare(finding, source, uploaded, dataset_paths, question, *, automatic=False):
        state = st.session_state
        thread = state.cleaning_messages_by_finding[fid]
        thread.append({'role': 'user', 'content': question})
        return fid, thread, [], {}, []

    monkeypatch.setattr(app, '_prepare_finding_discussion', prepare)
    app._queue_agent_c_for_finding(_finding(), source, [SimpleNamespace(name=source.name)],
                                   [source], 'flag these four rows taxonomy bold and red')
    # Chat entry is NOT a visually informed approval; no styling or review
    # may complete before exact BEFORE/AFTER is displayed.
    assert st.session_state.cleaning_review_statuses[fid] == 'Visual flag preview ready'
    assert not st.session_state.cleaning_history
    assert st.session_state.cleaning_pending_visual_flags_by_finding[fid]['flags']
    assert not st.session_state.cleaning_plans_by_finding
    assert source.read_bytes() == original
    assert 'Before' in st.session_state.cleaning_messages_by_finding[fid][-1]['content']


def test_no_visual_flag_yes_approve_form_in_live_app():
    code = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    scope = code.split('# A chat instruction may express a visual-flag intent', 1)[1].split(
        '# Even a zero-issue Gemma negative-control check', 1)[0]
    assert 'review_visual_flag_' not in scope
    assert 'Yes — Approve & Execute' not in scope
    assert 'apply_explicit_visual_flag_once' in scope


def test_legacy_already_selected_flag_renders_completed_without_second_yes(monkeypatch, tmp_path):
    """An older pending flag is a recorded user instruction, not another review."""
    source = _source(tmp_path)
    finding = _finding()
    fid = finding['id']
    request = 'flag these four rows taxonomy bold and red'

    class FormTrackingUI(StubUI):
        def __init__(self):
            super().__init__()
            self.approval_forms = []
        def form(self, key, **kwargs):
            self.approval_forms.append(str(key))
            return super().form(key, **kwargs)

    st = FormTrackingUI()
    app = _app(monkeypatch, st)
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings',
                        lambda *args, **kwargs: [finding])
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.allowed_positions',
                        lambda *args, **kwargs: (set(), set()))
    class Ready:
        def snapshot(self, job_id):
            return {'entries': {fid: {'status': 'ready', 'suggestion': 'Flag taxonomy'}}}
    monkeypatch.setattr('healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager', lambda: Ready())
    from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag
    st.session_state.hdq_registered_reviewer = 'Researcher A'
    app.render_cleaning_workspace([source], [SimpleNamespace(name=source.name)], {}, 'test-flag-legacy', [])
    st.session_state.cleaning_messages_by_finding[fid] = [
        {'role': 'user', 'content': request},
        {'role': 'assistant', 'content': 'Flag 4 taxonomy names bold red. Values unchanged; review below.'},
    ]
    st.session_state.cleaning_pending_visual_flags_by_finding[fid] = {
        **requested_visual_flag(request, finding, source), 'request': request,
    }
    st.session_state.cleaning_discussion_mode_by_finding[fid] = 'alternative'
    # Without the reviewer clicking the displayed button, the legacy draft
    # stays pending rather than being silently applied during page render.
    app.render_cleaning_workspace([source], [SimpleNamespace(name=source.name)], {},
                                  'test-flag-legacy', [])
    assert st.session_state.cleaning_review_statuses[fid] != 'Applied and verified'
    assert not st.session_state.cleaning_history
    assert st.session_state.cleaning_pending_visual_flags_by_finding[fid]
    st.buttons_on.add(f'flag_apply_verified_{fid}_0')
    with pytest.raises(StopIteration, match='rerun'):
        app.render_cleaning_workspace([source], [SimpleNamespace(name=source.name)], {},
                                      'test-flag-legacy', [])
    assert st.session_state.cleaning_review_statuses[fid] == 'Applied and verified'
    assert len(st.session_state.cleaning_history) == 1
    assert not st.session_state.cleaning_pending_visual_flags_by_finding
    assert not any('review_visual_flag_' in key for key in st.approval_forms)


def test_actual_finding_5_flags_survive_in_excel_without_changing_csv():
    """Use the bundled real source that produced the user's screenshot."""
    import io
    from openpyxl import load_workbook
    from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download

    source = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'
    original = source.read_bytes()
    finding = {**_finding(), 'evidence': [
        {'CSV row': row} for row in (3, 5, 15, 20)
    ]}
    state = {'cleaning_approved_reviewer': 'Researcher A'}
    from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag
    command = 'flag these four rows make the taxonomy bold and red'
    preview = requested_visual_flag(command, finding, source)
    result = apply_explicit_visual_flag_once(
        state, finding, source, command, previously_displayed=preview)
    assert result['approved'] == 4
    _, book_bytes = build_styled_excel_download(
        original, original_csv=original, original_filename=source.name,
        approved_flags=state['cleaning_visual_flags_by_dataset']['dataset_1'],
        approved_row_operations=[], approved_column_deletions=[],
    )
    workbook = load_workbook(io.BytesIO(book_bytes))
    sheet = workbook['Cleaned dataset']
    for row in (3, 5, 15, 20):
        assert sheet.cell(row, 1).font.bold is True
        assert sheet.cell(row, 1).font.color.rgb == 'FFDC2626'
    assert source.read_bytes() == original


def test_automatic_model_draft_cannot_self_authorize_visual_flag(monkeypatch, tmp_path):
    source = _source(tmp_path)
    st = StubUI()
    app = _app(monkeypatch, st)
    finding = _finding()
    fid = finding['id']
    st.session_state.cleaning_approved_reviewer = 'Researcher A'
    st.session_state.cleaning_recommendations = {fid: {'suggestion': 'Flag taxonomy'}}
    st.session_state.cleaning_discussion_mode_by_finding = {fid: 'alternative'}
    st.session_state.cleaning_pending_gemma_b = {}
    st.session_state.cleaning_review_statuses = {}
    st.session_state.cleaning_messages_by_finding = {fid: []}
    st.session_state.cleaning_plans_by_finding = {}
    st.session_state.cleaning_batch_queue_by_finding = {}
    st.session_state.cleaning_batch_total_by_finding = {}
    st.session_state.cleaning_compound_previews_by_finding = {}
    st.session_state.cleaning_completed_gemma = {}
    st.session_state.cleaning_history = []
    st.session_state.cleaning_decisions = []
    st.session_state.cleaning_pending_visual_flags_by_finding = {}
    st.session_state.cleaning_r_previews_by_finding = {}
    st.session_state.cleaning_r_errors_by_finding = {}
    st.session_state.cleaning_r_diagnostics_by_finding = {}
    st.session_state.cleaning_r_selfchecks_by_finding = {}
    st.session_state.cleaning_explicit_row_deletions_by_finding = {}
    st.session_state.cleaning_explicit_column_deletions_by_finding = {}
    st.session_state.cleaning_memory_candidates = {}
    st.session_state.cleaning_ui_generation_by_finding = {}

    def prepare(finding, source, uploaded, dataset_paths, question, *, automatic=False):
        thread = st.session_state.cleaning_messages_by_finding[fid]
        thread.append({'role': 'user', 'content': question, 'automatic': automatic})
        return fid, thread, [], {}, []

    monkeypatch.setattr(app, '_prepare_finding_discussion', prepare)
    app._queue_agent_c_for_finding(finding, source, [SimpleNamespace(name=source.name)],
                                   [source], 'flag these four rows bold red', automatic=True)
    assert not st.session_state.cleaning_history
    assert st.session_state.cleaning_pending_visual_flags_by_finding[fid]['flags']
    assert st.session_state.cleaning_review_statuses[fid] == 'Visual flag proposal only'
