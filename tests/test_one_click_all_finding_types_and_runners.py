"""The same final authorization contract applies to QC findings and Gemma checks."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_cleaning_review_cards import StubUI, _app, _catalog


def test_each_queued_qc_finding_and_gemma_check_has_a_runner_not_an_expander(
    monkeypatch, tmp_path,
):
    file = tmp_path / 'data.csv'
    file.write_text('Taxon,Count\na,0.2\n', encoding='utf-8')

    class HTMLUI(StubUI):
        def __init__(self):
            super().__init__()
            self.html = []
        def markdown(self, value, **kwargs):
            if 'hdq-finding-wait' in str(value):
                self.html.append(value)

    st = HTMLUI()
    app = _app(monkeypatch, st)
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings',
                        lambda *args, **kwargs: _catalog())
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.allowed_positions',
                        lambda *args: (set(), set()))
    app.render_cleaning_workspace([file], [SimpleNamespace(name=file.name)], {}, 'test-lock', [])
    assert not st.expanders
    assert len(st.html) == 3  # two QC findings + one Gemma check
    assert all('🏃‍♂️' in html and 'Waiting for Agent B' in html for html in st.html)
    assert any('Gemma check 1' in html for html in st.html)


@pytest.mark.parametrize('kind', ['qc', 'gemma_check'])
def test_verified_option_click_is_final_and_never_prepares_another_yes(
    monkeypatch, tmp_path, kind,
):
    source = tmp_path / 'data.csv'
    source.write_text('Taxon,Count\na,0.2\n', encoding='utf-8')
    finding = {
        'id': 'dataset_1:qc:1' if kind == 'qc' else 'dataset_1:gemma:1',
        'dataset_id': 'dataset_1', 'source': 'QC' if kind == 'qc' else 'Gemma',
        'check_key': 'find_invalid_counts' if kind == 'qc' else 'gemma_check',
        'title': 'Counts reviewed', 'evidence': [{'CSV row': 2, 'Column': 2, 'Original value': '0.2'}], 'actionable': True,
    }
    option = {'label': 'Replace 0.2 with 2', 'operations': [
        {'operation': 'replace_cells', 'changes': [
            {'row': 2, 'column': 2, 'before': '0.2', 'after': '2'}]}],
        'rows': [{'Location': 'R2:C2', 'Before': '0.2', 'After': '2'}],
    }
    st = StubUI()
    app = _app(monkeypatch, st)
    monkeypatch.setattr(
        'healthcare_data_quality.services.recommendation_options.source_backed_choices',
        lambda finding, source, suggestion: [option],
    )
    monkeypatch.setattr(
        'healthcare_data_quality.ui.human_choice_replay.prioritize_verified_human_choices',
        lambda finding, source, choices: choices,
    )
    from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
    from healthcare_data_quality.services.model_plan_review import option_fingerprint
    reviewed = scoped_review_choices(finding, source, [option])
    st.session_state.cleaning_recommendations = {
        finding['id']: {'suggestion': 'Replace count', 'memory_assessed': True,
                        'gemma_option_reviews': [{'fingerprint':option_fingerprint(finding,source,c), 'can_offer': True, 'reason':'Exact scope reviewed'} for c in reviewed]}
    }
    for key in ('cleaning_plans_by_finding', 'cleaning_batch_queue_by_finding',
                'cleaning_pending_gemma_b', 'cleaning_discussion_mode_by_finding',
                'cleaning_review_statuses'):
        st.session_state[key] = {}
    st.session_state.cleaning_history = []
    st.session_state.cleaning_approved_reviewer = 'Named Reviewer'
    executed = []

    def apply(selected, original, action):
        executed.append(action)
        st.session_state.cleaning_pending_gemma_b[selected['id']] = {'kind':'local_choice'}
        return True

    monkeypatch.setattr(app, '_queue_local_choice', apply)
    st.buttons_on.add(f"review_yes_{finding['id']}_0_option_{option_fingerprint(finding, source, reviewed[0])[:20]}")
    with pytest.raises(StopIteration, match='rerun'):
        app._render_live_agent_b_finding(finding['id'], finding, source, [], [source],
                                          None, 'dataset_1')
    assert len(executed) == 1
    assert executed[0]['operations'][0]['operation'] == 'replace_cells'
    assert executed[0]['operations'][0]['changes'] == option['operations'][0]['changes']
    assert executed[0]['operations'][0]['dataset_id'] == finding['dataset_id']
    assert executed[0]['finding_id'] == finding['id']
    assert st.session_state.cleaning_plans_by_finding == {}
    assert st.session_state.cleaning_history == []
    assert st.session_state.cleaning_pending_gemma_b[finding['id']]['kind'] == 'local_choice'
    assert source.read_bytes() == b'Taxon,Count\na,0.2\n'


def test_no_second_prepare_step_remains_in_the_active_source():
    root = Path(__file__).resolve().parents[1]
    code = (root / 'app.py').read_text(encoding='utf-8')
    block = code.split('def _render_live_agent_b_finding(', 1)[1].split(
        'def _', 1)[0]
    assert '_queue_local_choice(' in block
    assert 'execute_verified_choice(' not in block
    assert '_record_gemma_b_response(' not in block
    assert '"Yes — Prepare Changes"' not in code
    assert 'HDQ_PER_FINDING_READY_GATE_V2' in code
