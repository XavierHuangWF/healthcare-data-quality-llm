"""Agent C's partial example is expanded only for an accepted full-finding action."""
import csv
from types import SimpleNamespace


def test_accepted_all_evidenced_missing_cells_become_one_complete_plan(tmp_path, monkeypatch):
    from test_cleaning_review_cards import _app, StubUI
    ui = StubUI()
    app = _app(monkeypatch, ui)

    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    source = tmp_path / 'test.csv'
    with source.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['taxon', 'count'])
        writer.writerows([(str(i), '') for i in range(618)])
    finding = {
        'id': 'dataset_1:qc:2', 'dataset_id': 'dataset_1',
        'check_key': 'find_invalid_counts', 'actionable': True,
        'evidence': [{'CSV row': i + 2, 'Column': 2} for i in range(618)],
    }
    state = State(
        cleaning_history=[], cleaning_messages_by_finding={},
        cleaning_review_statuses={}, cleaning_plans_by_finding={},
        cleaning_recommendations={finding['id']: {
            'suggestion': 'Set all 618 empty cells in the count matrix to NA.',
        }},
        cleaning_discussion_mode_by_finding={finding['id']: 'accepted'},
        cleaning_staged_bytes_by_dataset={},
    )
    ui.session_state = state
    response = {
        'reply': 'I prepared the first 32 exact replacements.',
        'operation': 'replace_cells', 'dataset_id': 'dataset_1',
        'changes': [{'row': i+2, 'column': 2, 'before': '', 'after': 'NA'}
                    for i in range(32)],
    }
    app._record_gemma_b_response(finding['id'], finding, source, response)
    validated = state.cleaning_plans_by_finding[finding['id']]
    assert len(validated['changes']) == 618
    assert len(state.cleaning_messages_by_finding[finding['id']]) == 1
    assert 'Replace 618 cells' in state.cleaning_messages_by_finding[finding['id']][0]['content']
