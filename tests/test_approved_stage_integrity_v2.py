"""Regression tests: a checkpoint is reconstructed from explicit human approvals."""
import csv
from io import StringIO
from pathlib import Path

import pytest

from healthcare_data_quality.cleaning import cleaning_stage_checkpoint as checkpoint
from healthcare_data_quality.cleaning.in_memory_cleaning import sha256_bytes


def _source(tmp_path):
    path = tmp_path / 'original.csv'
    path.write_text('id,measurement,comment\na,1,x\nb,2,y\n', encoding='utf-8')
    return path


def _edited(original: bytes, edits):
    data = checkpoint._records(original)
    for (row, col), value in edits.items():
        data[row-1][col-1] = value
    out = StringIO(newline='')
    csv.writer(out).writerows(data)
    return out.getvalue().encode('utf-8')


def _event(source, staged, *, fid='f1', row=2, col=2, before='1', after='10'):
    return {
        'dataset_id': 'dataset_1', 'finding_id': fid, 'finding': 'Bad measurement',
        'operation': 'replace_cells', 'changes': 1, 'already_resolved': 0,
        'reviewer': 'researcher',
        'modified_positions': [{'csv_row': row, 'column': col}],
        'approved_cell_changes': [{'row': row, 'column': col, 'before': before, 'after': after}],
        'source_sha256': sha256_bytes(source.read_bytes()),
        'staged_sha256': sha256_bytes(staged),
    }


def _save(source, staged, events, *, statuses=None, audits=None):
    return checkpoint.save_approved_stage(
        [source], {'dataset_1': staged}, {}, {}, events,
        statuses or {'f1': 'Applied and verified'}, 'researcher',
        {'f1': 'Bad measurement'}, completed_audits=audits,
    )


def test_exact_approval_roundtrip_and_r_audit_persist(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    staged = _edited(source.read_bytes(), {(2, 2): '10'})
    audit = {'f1': {'r_function': 'clean_dataset <- function(df) df',
                    'insertion_r_function': 'apply_changes <- function(df) df',
                    'changes': [{'row': 2, 'column': 2, 'before': '1', 'after': '10'}]}}
    _save(source, staged, [_event(source, staged)], audits=audit)
    loaded = checkpoint.load_approved_stage([source])
    result = checkpoint.verify_and_reconstruct(loaded, [source], {'f1': 'Bad measurement'})
    assert checkpoint._records(result['cleaning_staged_bytes_by_dataset']['dataset_1']) == checkpoint._records(staged)
    assert result['cleaning_completed_audits_by_finding'] == audit
    assert source.read_text(encoding='utf-8').splitlines()[-1] == 'b,2,y'


def test_rogue_extra_cell_rejected_even_with_matching_source_hash(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    rogue = _edited(source.read_bytes(), {(2, 2): '10', (3, 3): 'tampered'})
    with pytest.raises(PermissionError, match='Staged cells differ'):
        _save(source, rogue, [_event(source, rogue)])


def test_changed_approved_value_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    rogue = _edited(source.read_bytes(), {(2, 2): '999'})
    with pytest.raises(PermissionError, match='Staged cells differ'):
        _save(source, rogue, [_event(source, rogue)])


def test_legacy_history_without_exact_values_cannot_be_saved(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    staged = _edited(source.read_bytes(), {(2, 2): '10'})
    event = _event(source, staged)
    event.pop('approved_cell_changes')
    with pytest.raises(PermissionError, match='Approved values are missing'):
        _save(source, staged, [event])


def test_forged_completed_status_does_not_restore(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    with pytest.raises(PermissionError, match='status lacks'):
        _save(source, source.read_bytes(), [])


def test_multiple_approved_events_replay_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    first = _edited(source.read_bytes(), {(2, 2): '10'})
    both = _edited(source.read_bytes(), {(2, 2): '10', (3, 2): '20'})
    e1 = _event(source, first)
    e2 = _event(source, both, row=3, before='2', after='20')
    _save(source, both, [e1, e2])
    result = checkpoint.verify_and_reconstruct(checkpoint.load_approved_stage([source]),
                                               [source], {'f1': 'Bad measurement'})
    assert checkpoint._records(result['cleaning_staged_bytes_by_dataset']['dataset_1'])[2][1] == '20'


def test_restoration_rejects_tampered_payload_even_if_hashes_recomputed(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    staged = _edited(source.read_bytes(), {(2, 2): '10'})
    _save(source, staged, [_event(source, staged)])
    saved = checkpoint.load_approved_stage([source])
    saved['datasets']['dataset_1']['changes'][0]['after'] = '999'
    changed = checkpoint._records(_edited(source.read_bytes(), {(2, 2): '999'}))
    saved['datasets']['dataset_1']['stage_record_sha256'] = checkpoint._record_hash(changed)
    with pytest.raises(PermissionError, match='Staged cells differ'):
        checkpoint.verify_and_reconstruct(saved, [source], {'f1': 'Bad measurement'})


def test_column_delete_without_matching_history_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    original = source.read_bytes()
    entry = {'position': 3, 'before': 'comment', 'reviewer': 'researcher',
             'source_sha256': sha256_bytes(original), 'finding_id': 'f1'}
    with pytest.raises(PermissionError, match='matching approval history'):
        checkpoint.save_approved_stage(
            [source], {}, {'dataset_1': [entry]}, {}, [],
            {'f1': 'Approved column deletion for final export'},
            'researcher', {'f1': 'Bad measurement'})


def test_column_delete_matching_history_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    original = source.read_bytes()
    digest = sha256_bytes(original)
    entry = {'position': 3, 'before': 'comment', 'reviewer': 'researcher',
             'source_sha256': digest, 'finding_id': 'f1'}
    event = {'dataset_id': 'dataset_1', 'finding_id': 'f1', 'operation': 'drop_column',
             'changes': 1, 'modified_positions': [{'column': 3}], 'reviewer': 'researcher',
             'source_sha256': digest, 'staged_sha256': digest}
    checkpoint.save_approved_stage(
        [source], {}, {'dataset_1': [entry]}, {}, [event],
        {'f1': 'Approved column deletion for final export'},
        'researcher', {'f1': 'Bad measurement'})
    restored = checkpoint.verify_and_reconstruct(checkpoint.load_approved_stage([source]),
                                                  [source], {'f1': 'Bad measurement'})
    assert restored['cleaning_approved_columns_by_dataset']['dataset_1'] == [entry]


def test_no_change_review_can_be_saved_without_approval_event(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    checkpoint.save_approved_stage(
        [source], {}, {}, {}, [], {'f1': 'Reviewed — no change'},
        'researcher', {'f1': 'Bad measurement'})
    restored = checkpoint.verify_and_reconstruct(checkpoint.load_approved_stage([source]),
                                                 [source], {'f1': 'Bad measurement'})
    assert restored['cleaning_review_statuses']['f1'] == 'Reviewed — no change'


def test_old_checkpoint_version_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = _source(tmp_path)
    location = checkpoint._location([source])
    location.parent.mkdir()
    location.write_text('{"version":1,"source_hashes":[]}')
    with pytest.raises(ValueError, match='Legacy checkpoint'):
        checkpoint.load_approved_stage([source])
