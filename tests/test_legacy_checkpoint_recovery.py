"""Legacy V1 checkpoints are viewable and archived, never implicitly applied."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from healthcare_data_quality import cleaning_stage_checkpoint as cp


def _legacy(tmp_path: Path, monkeypatch, *, edited=True):
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path / 'recovery'))
    source = tmp_path / 'source.csv'
    source.write_text('id,measure\na,1\nb,2\n', encoding='utf-8')
    original = source.read_bytes()
    rows = cp._records(original)
    changes = []
    if edited:
        changes = [{'row': 2, 'column': 2, 'before': '1', 'after': '10'}]
        rows[1][1] = '10'
    data = {
        'version': 1,
        'source_hashes': [cp.sha256_bytes(original)],
        'datasets': {'dataset_1': {
            'source_sha256': cp.sha256_bytes(original),
            'stage_record_sha256': cp._record_hash(rows),
            'changes': changes,
            'approved_rows': [], 'approved_columns': [],
        }},
        'history': [{'finding_id': 'f1', 'dataset_id': 'dataset_1',
                     'source_sha256': cp.sha256_bytes(original),
                     'operation': 'replace_cells', 'changes': 1,
                     'modified_positions': [{'csv_row': 2, 'column': 2}]}] if edited else [],
        'statuses': {'f1': 'Applied and verified'} if edited else {'f1': 'Reviewed — no change'},
        'titles': {'f1': 'Suspect measurement'},
        'reviewer': 'A reviewer',
    }
    path = cp._location([source])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return source, path


def test_legacy_checkpoint_cannot_be_restored_even_when_valid(tmp_path, monkeypatch):
    source, path = _legacy(tmp_path, monkeypatch)
    with pytest.raises(cp.LegacyCheckpointError, match='approval evidence'):
        cp.load_approved_stage([source])
    with pytest.raises(ValueError, match='Invalid approved-stage'):
        cp.verify_and_reconstruct(json.loads(path.read_text()), [source],
                                  {'f1': 'Suspect measurement'})
    preview = cp.inspect_legacy_stage([source], {'f1': 'Suspect measurement'})
    assert preview['cell_changes'] == [
        {'dataset': 'dataset_1', 'csv_row': 2, 'column': 2, 'before': '1', 'after': '10'}
    ]
    assert 'cleaning_staged_bytes_by_dataset' not in preview
    assert 'cleaning_review_statuses' not in preview
    assert source.read_bytes() == b'id,measure\na,1\nb,2\n'


def test_archive_preserves_legacy_exact_bytes_and_opens_new_review(tmp_path, monkeypatch):
    source, path = _legacy(tmp_path, monkeypatch)
    preview = cp.inspect_legacy_stage([source], {'f1': 'Suspect measurement'})
    original = path.read_bytes()
    backup = cp.archive_legacy_stage([source], expected_digest=preview['checkpoint_digest'])
    assert backup.is_file() and backup.read_bytes() == original
    assert backup.name.endswith('.json') and '.legacy-v1-' in backup.name
    assert not path.exists()
    assert cp.load_approved_stage([source]) is None
    # A new V2 approval can be saved at the original location; the V1 backup remains.
    cp.save_approved_stage([source], {}, {}, {}, [],
                           {'f1': 'Reviewed — no change'}, 'fresh reviewer',
                           {'f1': 'Suspect measurement'})
    assert cp.load_approved_stage([source])['version'] == 2
    assert backup.read_bytes() == original


def test_archive_refuses_changed_checkpoint_without_losing_it(tmp_path, monkeypatch):
    source, path = _legacy(tmp_path, monkeypatch)
    preview = cp.inspect_legacy_stage([source], {'f1': 'Suspect measurement'})
    content = json.loads(path.read_text())
    content['reviewer'] = 'altered after preview'
    path.write_text(json.dumps(content))
    with pytest.raises(PermissionError, match='changed since preview'):
        cp.archive_legacy_stage([source], expected_digest=preview['checkpoint_digest'])
    assert path.exists()


def test_tampered_legacy_delta_cannot_be_previewed_or_restored(tmp_path, monkeypatch):
    source, path = _legacy(tmp_path, monkeypatch)
    content = json.loads(path.read_text())
    content['datasets']['dataset_1']['changes'][0]['after'] = '999'
    path.write_text(json.dumps(content))
    with pytest.raises(PermissionError, match='checksum'):
        cp.inspect_legacy_stage([source], {'f1': 'Suspect measurement'})
    with pytest.raises(cp.LegacyCheckpointError):
        cp.load_approved_stage([source])
    # The archived file retains tampered bytes for forensics but is never staged.
    payload = path.read_bytes()
    backup = cp.archive_legacy_stage([source])
    assert backup.read_bytes() == payload


def test_changed_source_rejects_legacy_preview_and_archive(tmp_path, monkeypatch):
    source, path = _legacy(tmp_path, monkeypatch)
    content = json.loads(path.read_text())
    content['source_hashes'] = ['different']
    path.write_text(json.dumps(content))
    with pytest.raises(PermissionError, match='source hashes'):
        cp.inspect_legacy_stage([source], {'f1': 'Suspect measurement'})
    with pytest.raises(PermissionError, match='source bytes'):
        cp.archive_legacy_stage([source])
    assert path.exists()


def test_archive_never_moves_current_v2_checkpoint(tmp_path, monkeypatch):
    source, path = _legacy(tmp_path, monkeypatch, edited=False)
    data = json.loads(path.read_text())
    data['version'] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(PermissionError, match='no longer V1'):
        cp.archive_legacy_stage([source])
    assert path.exists()


def test_legacy_ui_has_explicit_action_and_no_implicit_staging():
    app = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    assert 'except LegacyCheckpointError:' in app
    assert 'inspect_legacy_stage(source_paths, finding_titles)' in app
    assert 'Archive old checkpoint and start new review' in app
    assert 'Review saved changes (read-only)' in app
    assert 'archive_legacy_stage(' in app
