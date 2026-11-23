"""No cross-finding buttons; no invisible or duplicate flag authorization."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from healthcare_data_quality.ui.finding_choice_integrity import (
    finding_completed, scoped_review_choices,
)
from healthcare_data_quality.cleaning.one_click_visual_flag import (
    apply_explicit_visual_flag_once,
)
from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'dirtyL7PVDforXav.csv'


def duplicate_finding():
    return {
        'id': 'dataset_1:qc:4', 'dataset_id': 'dataset_1',
        'source': 'QC', 'title': 'Duplicate taxa', 'check_key': 'find_duplicate_taxa',
        'evidence': [
            {'Taxon / feature': 'Bacteria;Actinobacteria;Actinobacteria;Bifidobacteriales;Bifidobacteriaceae;Bifidobacterium; breve',
             'CSV rows': '9, 11'},
            {'Taxon / feature': 'Bacteria;Actinobacteria;Actinobacteria;Bifidobacteriales;Bifidobacteriaceae;Bifidobacterium; dentium',
             'CSV rows': '10, 12'},
        ],
    }


def test_duplicate_taxon_card_never_renders_taxonomy_padding_from_other_finding():
    finding = duplicate_finding()
    # Reproduces the screenshot's Finding 4 incorrectly showing the Finding 5
    # 'pad row 3/5 with NA' choices after row deletion buttons.
    foreign = {'label': "Pad row 3 with NA", 'operations': [
        {'operation': 'replace_cells', 'changes': [{
            'row': 3, 'column': 1,
            'before': 'Bacteria;Actinobacteria;Actinobacteria;Actinomycetales;Actinomycetaceae;Arcanobacterium',
            'after': 'Bacteria;Actinobacteria;Actinobacteria;Actinomycetales;Actinomycetaceae;Arcanobacterium; NA',
        }]}
    ]}
    result = scoped_review_choices(finding, SOURCE, [foreign])
    assert result == []


def test_verified_duplicate_taxon_row_deletion_stays_available_and_preview_cannot_be_forged():
    import csv
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.reader(f))
    finding = duplicate_finding()
    choice = {'label': 'Delete row 12', 'operations': [{
        'operation': 'drop_rows', 'changes': [{'row': 12, 'before': raw[11][0]}]
    }], 'rows': [{'Location': 'Row 12', 'Before': 'WRONG', 'After': 'Delete row'}]}
    result = scoped_review_choices(finding, SOURCE, [choice, copy.deepcopy(choice)])
    assert len(result) == 1
    assert result[0]['finding_id'] == finding['id']
    assert result[0]['dataset_id'] == finding['dataset_id']
    assert result[0]['rows'][0]['Before'] == raw[11][0]
    assert result[0]['rows'][0]['After'] == 'Delete row'


def test_other_finding_id_or_dataset_id_cannot_render_in_current_card():
    finding = duplicate_finding()
    valid = {'label': 'Delete row 12', 'operations': [{
        'operation': 'drop_rows', 'changes': [{'row': 12}]}
    ]}
    foreign = {**valid, 'finding_id': 'dataset_1:qc:5'}
    other_data = {**valid, 'dataset_id': 'dataset_2'}
    assert scoped_review_choices(finding, SOURCE, [foreign, other_data]) == []


def test_any_completed_review_blocks_resurrection_from_late_worker():
    fid = 'dataset_1:qc:3'
    assert finding_completed({'cleaning_review_statuses': {fid: 'Applied and verified'}}, fid)
    assert finding_completed({'cleaning_history': [{'finding_id': fid}]}, fid)
    assert not finding_completed({'cleaning_review_statuses': {fid: 'Discussion open'}}, fid)


def test_explicit_flag_chat_requires_displayed_preview_and_single_click():
    source = SOURCE
    finding = {'id': 'dataset_1:qc:3', 'dataset_id': 'dataset_1',
               'title': 'Negative counts', 'check_key': 'find_invalid_counts',
               'evidence': [{'CSV row': 16, 'Column': 6}, {'CSV row': 24, 'Column': 12}]}
    before = source.read_bytes()
    request = 'flag these two rows taxonomy bold and red'
    state = {'cleaning_approved_reviewer': 'Scientist'}
    with pytest.raises(PermissionError, match='Before.*After'):
        apply_explicit_visual_flag_once(state, finding, source, request)
    assert not state.get('cleaning_history')
    preview = requested_visual_flag(request, finding, source)
    assert [flag['row'] for flag in preview['flags']] == [16, 24]
    result = apply_explicit_visual_flag_once(state, finding, source, request,
                                              previously_displayed=preview)
    assert result['approved'] == 2
    assert state['cleaning_review_statuses'][finding['id']] == 'Applied and verified'
    with pytest.raises(PermissionError, match='already completed'):
        apply_explicit_visual_flag_once(state, finding, source, request,
                                         previously_displayed=preview)
    assert source.read_bytes() == before
    assert len(state['cleaning_history']) == 1


def test_preview_cannot_be_replayed_under_another_finding():
    source = SOURCE
    first = {'id': 'dataset_1:qc:3', 'dataset_id': 'dataset_1', 'title': 'Negative',
             'evidence': [{'CSV row': 16}]}
    second = {'id': 'dataset_1:qc:5', 'dataset_id': 'dataset_1', 'title': 'Taxonomy',
              'evidence': [{'CSV row': 16}]}
    request = 'flag row 16 taxonomy bold red'
    preview = requested_visual_flag(request, first, source)
    with pytest.raises(PermissionError, match='Displayed Before/After'):
        apply_explicit_visual_flag_once({'cleaning_approved_reviewer': 'Scientist'},
                                        second, source, request,
                                        previously_displayed=preview)


def test_duplicate_finding_discards_foreign_clarification_buttons():
    from healthcare_data_quality.ui.finding_choice_integrity import scoped_clarification_choices
    finding = duplicate_finding()
    labels = [
        'Bacteria;Actinobacteria;Actinobacteria;Actinomycetales',
        'Pad row 3 with NA',
        'Pad row 5 with NA',
        'Delete row 12',
        'Delete row 3',
        'Delete row 12',
    ]
    assert scoped_clarification_choices(finding, labels) == ['Delete row 12']
