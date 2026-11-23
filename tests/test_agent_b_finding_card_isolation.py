"""Guard against Agent B controls crossing adjacent findings.

A parent expander owns each finding's controls; independent polling fragments
must not append a prior finding's Yes/No buttons underneath another one.
"""
from __future__ import annotations

import csv
from pathlib import Path

from healthcare_data_quality.cleaning.finding_cleaning import validate_scoped_plan
from healthcare_data_quality.services.recommendation_options import source_backed_choices

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'dirtyL7PVDforXav.csv'


def _duplicate_finding():
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    return {
        'id': 'dataset_1:qc:4', 'dataset_id': 'dataset_1',
        'source': 'QC', 'title': 'Duplicate taxon names',
        'check_key': 'find_duplicate_taxa', 'actionable': True,
        'evidence': [
            {'Taxon / feature': rows[8][0], 'CSV rows': '9, 11'},
            {'Taxon / feature': rows[9][0], 'CSV rows': '10, 12'},
        ],
    }


def _taxonomy_finding():
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    return {
        'id': 'dataset_1:qc:5', 'dataset_id': 'dataset_1',
        'source': 'QC', 'title': 'Short taxonomy labels',
        'check_key': 'inspect_taxonomy_structure', 'actionable': True,
        'total_evidence_records': 4,
        'evidence': [
            {'CSV row': pos, 'Original classification (verbatim)': rows[pos-1][0],
             'Levels found': len(rows[pos-1][0].split(';')), 'Expected levels': 7}
            for pos in (3, 5, 15, 20)
        ],
    }


def test_merge_proposal_is_source_scoped_to_finding_4():
    finding = _duplicate_finding()
    choices = source_backed_choices(
        finding, SOURCE, {'suggestion':
            'Merge 1 row group — CSV rows 10 + 12 — sum counts; retain first row'})
    assert len(choices) == 1
    assert choices[0]['label'] == 'Merge rows 10 + 12 (sum counts)'
    assert [item['Location'] for item in choices[0]['rows']] == ['Rows 10 + 12']
    assert 'remove row 12' in choices[0]['rows'][0]['After']
    checked = validate_scoped_plan(choices[0]['operations'][0], finding, SOURCE)
    assert checked['changes'][0]['source_row'] == 12


def test_taxonomy_padding_cannot_leak_into_duplicate_finding():
    duplicate = _duplicate_finding()
    taxonomy = _taxonomy_finding()
    pad = {'suggestion': 'Pad rows 3, 5, 15 and 20 with NA to seven levels'}
    assert source_backed_choices(duplicate, SOURCE, pad) == []
    options = source_backed_choices(taxonomy, SOURCE, pad)
    assert len(options) == 1
    assert {c['row'] for c in options[0]['operations'][0]['changes']} == {3, 5, 15, 20}
    assert source_backed_choices(taxonomy, SOURCE, {
        'suggestion': 'Merge 1 row group — CSV rows 10 + 12 — sum counts'}) == []


def test_duplicate_finding_rejects_model_merge_other_rows():
    finding = _duplicate_finding()
    for suggestion in (
        'Merge CSV rows 3 + 5 — sum counts',
        'Merge CSV rows 9 + 10 — sum counts',
        'Merge CSV rows 10 + 999 — sum counts',
        'Merge CSV rows 9 + 11 — keep first, delete second', # no explicit sum
    ):
        assert source_backed_choices(finding, SOURCE, {'suggestion': suggestion}) == []


def test_parent_owns_all_finding_controls_and_single_poller_has_no_buttons():
    code = (ROOT / 'app.py').read_text(encoding='utf-8')
    parent = code.split('def render_cleaning_workspace(', 1)[1].split('def _', 1)[0]  # marker only
    assert 'st.fragment(run_every="3s")(_render_live_agent_b_finding)' not in code
    assert '_render_live_agent_b_finding(\n                    fid, finding, source' in code
    refresher = code.split('def _refresh_parent_for_new_agent_b_choices(', 1)[1].split(
        'def _render_live_agent_b_finding(', 1)[0]
    assert 'st.button(' not in refresher
    assert 'st.dataframe(' not in refresher
    assert 'cleaning_discussion_mode_by_finding' in refresher
    assert 'st.rerun(scope="app")' in refresher


def test_rendered_button_sets_are_isolated_per_finding(monkeypatch):
    from test_cleaning_review_cards import StubUI, _app

    class Buttons(StubUI):
        def __init__(self):
            super().__init__()
            self.rendered = []
        def button(self, label, *, key=None, **kwargs):
            self.rendered.append((str(label), str(key)))
            return False

    st = Buttons()
    app = _app(monkeypatch, st)
    duplicate = _duplicate_finding()
    taxonomy = _taxonomy_finding()
    st.session_state.cleaning_recommendations = {
        duplicate['id']: {'suggestion': 'Merge 1 row group — CSV rows 10 + 12 — sum counts; retain first row'},
        taxonomy['id']: {'suggestion': 'Pad 4 rows with NA to reach 7 taxonomy levels'},
    }
    st.session_state.cleaning_plans_by_finding = {}
    st.session_state.cleaning_batch_queue_by_finding = {}
    st.session_state.cleaning_pending_gemma_b = {}
    st.session_state.cleaning_discussion_mode_by_finding = {}
    st.session_state.cleaning_history = []
    st.session_state.cleaning_review_statuses = {}

    app._render_live_agent_b_finding(duplicate['id'], duplicate, SOURCE, [], [SOURCE], None, 'dataset_1')
    duplicate_buttons = list(st.rendered)
    assert [x[0] for x in duplicate_buttons] == [
        'Yes — Merge rows 10 + 12 (sum counts)', 'No — Discuss Another Idea',
    ]
    assert all(duplicate['id'] in key for _, key in duplicate_buttons)
    assert any('Rows 10 + 12' == item['Location'] for kind, frame in st.comments
               if kind == 'table' for item in frame.to_dict('records'))
    st.rendered.clear()
    st.comments.clear()

    app._render_live_agent_b_finding(taxonomy['id'], taxonomy, SOURCE, [], [SOURCE], None, 'dataset_1')
    taxonomy_buttons = list(st.rendered)
    assert [x[0] for x in taxonomy_buttons] == [
        'Yes — Append NA to reach 7 levels', 'No — Discuss Another Idea',
    ]
    assert all(taxonomy['id'] in key for _, key in taxonomy_buttons)
    assert all(duplicate['id'] not in key for _, key in taxonomy_buttons)
    assert all('Rows 10 + 12' != item['Location'] for kind, frame in st.comments
               if kind == 'table' for item in frame.to_dict('records'))
