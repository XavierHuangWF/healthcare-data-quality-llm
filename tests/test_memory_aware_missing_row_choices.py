"""Reviewer-facing choices include complete, source-verified row removal."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from healthcare_data_quality.services.recommendation_options import (
    source_backed_choices, _source_verified_missing_rows,
)
from healthcare_data_quality.validation.count_validation import find_invalid_counts


def _source(tmp_path, body: str) -> Path:
    path = tmp_path / 'source.csv'
    path.write_text(body, encoding='utf-8')
    return path


def _finding(path: Path, *, sample: bool = False, title: str | None = None) -> dict:
    all_missing = [issue for issue in find_invalid_counts(path)
                   if issue['issue_type'] == 'missing']
    observed = all_missing[:1] if sample else all_missing
    evidence = [{'CSV row': item['row'], 'Column': item['column'],
                 'Original value': item['original_value']}
                for item in observed]
    return {'id': 'dataset_1:qc:2', 'dataset_id': 'dataset_1',
            'check_key': 'find_invalid_counts',
            'title': title or f'{len(all_missing)} cells contain missing values',
            'evidence': evidence,
            'total_evidence_records': len(all_missing), 'actionable': True}


def _fill_choices():
    return [{'label': 'Set empty cells to NA',
             'rule': 'replace_evidenced_cells', 'replacement': 'NA'},
            {'label': 'Set empty cells to 0',
             'rule': 'replace_evidenced_cells', 'replacement': '0'}]


def test_delete_rows_offered_in_addition_to_two_fill_choices(tmp_path):
    source = _source(tmp_path, 'Taxon,A,B,C\na,,2,3\nb,0,1,2\nc,1,,8\n')
    finding = _finding(source)
    recommendation = {'choices': _fill_choices(), 'suggestion': 'Set two blank counts to NA'}
    choices = source_backed_choices(finding, source, recommendation)
    assert [x['label'] for x in choices] == ['Set empty cells to NA',
                                              'Set empty cells to 0',
                                              'Delete 2 rows containing empty cells']
    deletion = choices[2]
    assert deletion['operations'][0]['operation'] == 'drop_rows'
    assert [x['row'] for x in deletion['operations'][0]['changes']] == [2, 4]
    assert [row['Location'] for row in deletion['rows']] == ['Row 2', 'Row 4']
    assert [row['After'] for row in deletion['rows']] == ['Delete row'] * 2
    assert 'Other measurements' in deletion['impact']
    assert all('approval' not in str(x).casefold() for x in choices)


def test_related_approved_deletion_method_promotes_source_verified_choice(tmp_path):
    source = _source(tmp_path, 'Taxon,A,B\na,,2\nb,0,\nc,3,5\n')
    choices = source_backed_choices(_finding(source), source, {
        'choices': _fill_choices(),
        'memory_context': [{'operation_types': ['drop_rows'], 'issue_relation': 'same',
                            'same_original_csv': True}],
    })
    assert choices[0]['operations'][0]['operation'] == 'drop_rows'
    assert [x['row'] for x in choices[0]['operations'][0]['changes']] == [2, 3]
    assert 'prior approved' in choices[0]['memory_note'].casefold()
    assert len(choices) == 3


def test_cached_memory_marker_promotes_even_when_no_detailed_context(tmp_path):
    source = _source(tmp_path, 'Taxon,A\na,\nb,1\n')
    choices = source_backed_choices(_finding(source), source, {
        'choices': _fill_choices(), 'approved_row_deletion_method_seen': True,
    })
    assert choices[0]['operations'][0]['operation'] == 'drop_rows'


def test_source_exact_even_when_evidence_is_sampled(tmp_path):
    source = _source(tmp_path, 'Taxon,A,B\na,,\nb,0,\nc,3,5\n')
    finding = _finding(source, sample=True)
    choices = source_backed_choices(finding, source, {'choices': []})
    assert [c['row'] for c in choices[0]['operations'][0]['changes']] == [2, 3]
    assert len(choices) == 1


@pytest.mark.parametrize('mismatch', ['count', 'before', 'duplicate', 'wrong_category', 'wrong_check'])
def test_reject_stale_or_misclassified_evidence(tmp_path, mismatch):
    source = _source(tmp_path, 'Taxon,A,B\na,,1\nb,3,\n')
    f = _finding(source)
    if mismatch == 'count':
        f['total_evidence_records'] = 999
    elif mismatch == 'before':
        f['evidence'][0]['Original value'] = '999'
    elif mismatch == 'duplicate':
        f['evidence'][1] = dict(f['evidence'][0])
    elif mismatch == 'wrong_category':
        f['title'] = '2 cells contain negative count values'
    else:
        f['check_key'] = 'find_duplicate_taxa'
    assert _source_verified_missing_rows(f, list(csv.reader(source.open()))) == []
    choices = source_backed_choices(f, source, {'choices': []})
    assert not any(x['operations'][0]['operation'] == 'drop_rows' for x in choices)


def test_valid_zeros_and_taxonomy_na_are_not_blank_targets(tmp_path):
    source = _source(tmp_path, 'Taxon,A,B\nNA;NA,0,1\na,1,\nb,2,3\n')
    choices = source_backed_choices(_finding(source), source, {'choices': []})
    assert [c['row'] for c in choices[0]['operations'][0]['changes']] == [3]


def test_no_deletion_for_unrelated_findings(tmp_path):
    source = _source(tmp_path, 'Taxon,A\na,\nb,1\n')
    f = _finding(source)
    f['title'] = '1 cell contains fractional values'
    assert not source_backed_choices(f, source, {'choices': []})


def test_real_dirty_dataset_has_six_deletable_rows():
    source = Path(__file__).parents[1] / 'data' / 'dirtyL7PVDforXav.csv'
    assert source.exists()
    finding = _finding(source)
    assert finding['total_evidence_records'] == 618
    choices = source_backed_choices(finding, source, {'choices': _fill_choices()})
    deletion = next(c for c in choices if c['operations'][0]['operation'] == 'drop_rows')
    assert [c['row'] for c in deletion['operations'][0]['changes']] == [2, 3, 5, 7, 15, 20]
    assert len(deletion['rows']) == 6


def test_never_offer_deleting_every_data_row(tmp_path):
    source = _source(tmp_path, 'Taxon,A\na,\nb,\n')
    assert not source_backed_choices(_finding(source), source, {'choices': []})


def test_click_request_is_approved_scope_not_model_invented_rows():
    from healthcare_data_quality.cleaning.finding_cleaning import (
        explicitly_requested_row_deletions, validate_scoped_plan,
        propose_bulk_blank_row_deletion,
    )
    source = Path(__file__).parents[1] / 'data' / 'dirtyL7PVDforXav.csv'
    finding = _finding(source)
    recommendation = {'choices': _fill_choices()}
    deletion = next(x for x in source_backed_choices(finding, source, recommendation)
                    if x['operations'][0]['operation'] == 'drop_rows')
    op = deletion['operations'][0]
    submitted = 'Delete CSV rows '+', '.join(str(c['row']) for c in op['changes'])
    user_ids = explicitly_requested_row_deletions(submitted)
    assert user_ids == frozenset({2, 3, 5, 7, 15, 20})
    checked = validate_scoped_plan(op, finding, source, user_requested_rows=user_ids)
    assert [c['row'] for c in checked['changes']] == [2, 3, 5, 7, 15, 20]
    # The same natural-language instruction already has a direct host path.
    manual = propose_bulk_blank_row_deletion(
        'delete the rows if there are any empty cells', finding, source)
    assert [c['row'] for c in manual['changes']] == [2, 3, 5, 7, 15, 20]


def test_rejected_memory_method_still_available_but_not_prioritized(tmp_path):
    source = _source(tmp_path, 'Taxon,A\na,\nb,1\n')
    result = source_backed_choices(_finding(source), source, {
        'choices': _fill_choices(),
        'approved_row_deletion_method_seen': True,
        'memory_assessment': {'decision': 'rejected', 'reason': 'Too much information loss'},
    })
    assert result[0]['label'] == 'Set empty cells to NA'
    assert result[-1]['operations'][0]['operation'] == 'drop_rows'
    assert 'memory_note' not in result[-1]
