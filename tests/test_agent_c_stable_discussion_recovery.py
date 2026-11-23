"""Regressions for Findings 6, 8, 9 and the repeatedly flashing Agent C card."""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from healthcare_data_quality.cleaning.taxonomy_review_explanation import taxonomy_level_answer
from healthcare_data_quality.ui.recommendation_failure_recovery import (
    recover_identical_column_suggestion,
)
from healthcare_data_quality.services.recommendation_options import source_backed_choices

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'dirtyL7PVDforXav.csv'


def _source_rows():
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.reader(handle))


def _finding_6():
    rows = _source_rows()
    return {
        'id': 'dataset_1:qc:6', 'dataset_id': 'dataset_1',
        'check_key': 'inspect_taxonomy_structure',
        'evidence': [{'CSV row': 7, 'Original classification': rows[6][0],
                      'Problem': 'Uses | instead of ;'}],
    }


def _finding_8():
    return {
        'id': 'dataset_1:qc:8', 'dataset_id': 'dataset_1',
        'check_key': 'compare_repeated_headers',
        'evidence': [{'Column A': 105, 'Column B': 106}],
        'title': 'Columns 105 and 106 contain identical values',
    }


def test_finding6_taxonomy_inspection_answers_rank_count_not_approval():
    original = SOURCE.read_bytes()
    answer = taxonomy_level_answer(
        "Verify the original classifications level also; it doesn't look right",
        _finding_6(), SOURCE,
    )
    assert 'Row 7: 7/7 taxonomic positions' in answer
    assert 'k/p/c/o/f/g/s' in answer
    assert 'uses | rather than ;' in answer
    assert 'not the biological accuracy' in answer
    assert 'No changes were applied' in answer
    assert SOURCE.read_bytes() == original


def test_finding6_noninspection_question_falls_through_to_agent_c():
    assert taxonomy_level_answer('please fix it', _finding_6(), SOURCE) is None


def test_taxonomy_inspection_requires_verbatim_evidence_match():
    finding = _finding_6()
    finding['evidence'][0]['Original classification'] = 'not actually in source'
    import pytest
    with pytest.raises(ValueError, match='disagrees'):
        taxonomy_level_answer('Check the classification levels', finding, SOURCE)


def test_failed_finding8_recovers_only_source_verified_duplicate():
    initial = {'status': 'failed', 'error': 'Timeout while requesting model'}
    proposal = recover_identical_column_suggestion(_finding_8(), SOURCE, initial)
    assert proposal and proposal['status'] == 'ready'
    assert proposal['origin'] == 'source_verified_fallback'
    assert proposal['agent_b_error'] == initial['error']
    choices = source_backed_choices(_finding_8(), SOURCE, proposal)
    assert len(choices) == 1
    assert choices[0]['rows'] == [
        {'Location': 'Column 106', 'Before': 'ABC99', 'After': 'Delete column'}
    ]
    assert initial['status'] == 'failed'  # backend diagnostics remain unchanged


def test_fallback_rejects_different_values_even_when_header_matches(tmp_path):
    source = tmp_path / 'different.csv'
    source.write_text('taxon,ABC99,ABC99\na,1,2\n', encoding='utf-8')
    finding = {'id': 'f', 'dataset_id': 'dataset_1',
               'check_key': 'compare_repeated_headers',
               'evidence': [{'Column A': 2, 'Column B': 3}]}
    assert recover_identical_column_suggestion(finding, source, {'status': 'failed'}) is None
    assert recover_identical_column_suggestion(finding, source, {'status': 'ready'}) is None
    assert recover_identical_column_suggestion({**finding, 'check_key': 'inspect_headers'},
                                               source, {'status': 'failed'}) is None


def test_agent_c_poll_restarts_app_only_once_for_same_completed_request(monkeypatch):
    from test_cleaning_review_cards import StubUI, _app
    ui = StubUI()
    app = _app(monkeypatch, ui)
    class State(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
    ui.session_state = State(
        cleaning_pending_gemma_b={'finding9': {'job_id': 'job1'}},
        cleaning_completed_gemma={},
    )
    count = []
    def rerun(*args, **kwargs):
        count.append((args, kwargs))
    monkeypatch.setattr(ui, 'rerun', rerun, raising=False)
    class Worker:
        def __init__(self): self.takes = 0
        def take(self, job_id):
            self.takes += 1
            return {'status': 'done', 'result': {'reply': 'Source checked'}}
        def agent_for(self, job_id): return 'C'
    worker = Worker()
    import healthcare_data_quality.jobs.interactive_chat_jobs as job_mod
    monkeypatch.setattr(job_mod, 'interactive_chat_manager', lambda: worker)
    app._poll_agent_c_completion()
    app._poll_agent_c_completion()
    assert len(count) == 1
    assert worker.takes == 1
    assert ui.session_state['cleaning_completed_gemma']['finding9']['status'] == 'done'


def test_pending_discussion_keeps_chat_widget_identity_in_source():
    code = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert '"discussion", "processing"}' in code
    assert 'disabled=b_pending)' in code
    assert 'if newly_ready:\n        st.rerun()' in code
    assert 'cleaning_agent_c_completion_notified' in code
