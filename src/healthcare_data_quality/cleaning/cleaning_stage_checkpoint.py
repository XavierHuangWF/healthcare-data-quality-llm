"""Private, source-bound recovery for previously approved *session* cleaning state.

Only the modified cells, approved structural operations, and review metadata
are persisted. Original CSV contents and unmodified cells are never copied.
Recovery is opt-in and rechecks every source hash, edited coordinate, and final
export before making anything visible as staged in a new Streamlit session.
"""
from __future__ import annotations

import csv
from hashlib import sha256
from collections import Counter
from io import StringIO
import json
import os
from pathlib import Path
import threading
from uuid import uuid4

from .in_memory_cleaning import (
    _records, sha256_bytes, apply_approved_row_operations,
    apply_approved_column_deletions,
)

_VERSION = 2
_LOCK = threading.RLock()
_MAX_BYTES = 12_000_000
_BASE = Path.home() / '.local' / 'share' / 'healthcare-data-quality-llm' / 'approved-stages'
_APPROVED = {
    'Applied and verified', 'Resolved by earlier approved edit',
    'Approved column deletion for final export',
    'Approved row changes for final export',
    'Partially approved — remaining action rejected',
    'Reviewed — no change',
}


def _directory() -> Path:
    return Path(os.environ.get('HDQ_APPROVED_STAGE_DIR', str(_BASE)))


def _record_hash(rows: list[list[str]]) -> str:
    return sha256(json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def _identity(sources: list[Path]) -> tuple[str, list[str]]:
    hashes = [sha256_bytes(Path(p).read_bytes()) for p in sources]
    fingerprint = sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()
    return fingerprint, hashes


def _location(sources: list[Path]) -> Path:
    return _directory() / (_identity(sources)[0] + '.json')


def _reviewed(status: str) -> bool:
    return status in _APPROVED


def _approval_ledger_matches_stage(
    sources: list[Path], staged: dict, columns: dict, rows: dict,
    history: list[dict], statuses: dict, titles: dict[str, str],
) -> None:
    """Replay EXACT approved before/after values; structural events must match approvals.

    A source hash and an affected-coordinate list by themselves do NOT prove
    which replacement values a human reviewed. Missing value-level proof
    fails closed, including for legacy in-session history.
    """
    if not isinstance(history, list):
        raise ValueError('Invalid approval ledger')
    expected_columns, actual_columns = Counter(), Counter()
    expected_rows, actual_rows = Counter(), Counter()
    for idx, source in enumerate(sources, 1):
        ds_id = f'dataset_{idx}'
        original = Path(source).read_bytes()
        source_hash = sha256_bytes(original)
        initial = _records(original)
        current = _records(staged.get(ds_id, original))
        if len(initial) != len(current) or any(len(a) != len(b) for a, b in zip(initial, current)):
            raise PermissionError('Unapproved change in staged CSV dimensions')
        replay = [list(r) for r in initial]
        ledger = [event for event in history if event.get('dataset_id') == ds_id]
        for event in ledger:
            fid, op = event.get('finding_id'), event.get('operation')
            reviewer = event.get('reviewer')
            if (fid not in titles or event.get('source_sha256') != source_hash
                    or not isinstance(reviewer, str) or not reviewer.strip()):
                raise PermissionError('Approval event is not bound to this finding/source/reviewer')
            positions = event.get('modified_positions')
            if not isinstance(positions, list) or any(not isinstance(p, dict) for p in positions):
                raise PermissionError('Approval event lacks an exact position ledger')
            if op in {'replace_cells', 'rename_headers'}:
                if type(event.get('changes')) is not int or event['changes'] < 0:
                    raise PermissionError('Invalid number of approved cell changes')
                changes = event.get('approved_cell_changes')
                if changes is None and event.get('changes') == 0:
                    changes = []
                if not isinstance(changes, list) or len(changes) != event.get('changes'):
                    raise PermissionError('Approved values are missing; legacy history cannot be restored safely')
                seen = set()
                for delta in changes:
                    if not isinstance(delta, dict):
                        raise PermissionError('Invalid approved value-level change')
                    r, c = delta.get('row'), delta.get('column')
                    if (type(r) is not int or type(c) is not int
                            or not (1 if op == 'rename_headers' else 2) <= r <= len(initial)
                            or not 1 <= c <= len(initial[0]) or (r, c) in seen
                            or not isinstance(delta.get('before'), str)
                            or not isinstance(delta.get('after'), str)
                            or delta['before'] == delta['after']
                            or replay[r - 1][c - 1] != delta['before']):
                        raise PermissionError('Approved cell change does not match prior staged state')
                    seen.add((r, c))
                    replay[r - 1][c - 1] = delta['after']
                if Counter((p.get('csv_row'), p.get('column')) for p in positions) != Counter(
                    (d['row'], d['column']) for d in changes
                ):
                    raise PermissionError('Approved cell position list differs from exact approved edits')
            elif op == 'flag_cells':
                # A style-only annotation must never account for a CSV delta.
                # Verify its exact original coordinates and values from the
                # same immutable source while leaving `replay` unchanged.
                flags = event.get('flags')
                if (positions != [] or not isinstance(flags, list)
                        or not flags or len(flags) > 50
                        or event.get('changes') != len(flags)):
                    raise PermissionError('Visual flag approval lacks exact source-bound targets')
                unique = set()
                for flag in flags:
                    if not isinstance(flag, dict):
                        raise PermissionError('Malformed visual flag target')
                    r, c = flag.get('row'), flag.get('column')
                    if (type(r) is not int or type(c) is not int or c != 1
                            or r < 2 or r > len(initial)
                            or (r, c) in unique or flag.get('style') != 'bold_red'
                            or flag.get('before') != initial[r - 1][c - 1]):
                        raise PermissionError('Visual flag target does not match the original CSV')
                    unique.add((r, c))
            elif op == 'drop_column':
                if len(positions) != 1 or type(positions[0].get('column')) is not int:
                    raise PermissionError('Invalid column approval event')
                actual_columns[(ds_id, fid, reviewer, positions[0]['column'])] += 1
            elif op in {'drop_rows', 'merge_rows'}:
                if (not positions or event.get('changes') != len(positions)
                        or any(type(p.get('row')) is not int for p in positions)):
                    raise PermissionError('Invalid row approval event')
                actual_rows[(ds_id, fid, reviewer, op,
                             tuple((p['row'], p.get('source_row')) for p in positions))] += 1
            else:
                raise PermissionError('Unexpected operation in approval history')
        if replay != current:
            raise PermissionError('Staged cells differ from exact approved values in the history')
        # Last event is the last commit to this dataset. This catches a
        # changed working version even if no cell coordinates were involved.
        if ledger and (ledger[-1].get('revision_reconciled_sha256')
                       or ledger[-1].get('staged_sha256')) != sha256_bytes(staged.get(ds_id, original)):
            raise PermissionError('Last approval or verified revision does not match staged CSV bytes')
        for item in columns.get(ds_id, []):
            expected_columns[(ds_id, item.get('finding_id'), item.get('reviewer'),
                              item.get('position'))] += 1
        for item in rows.get(ds_id, []):
            expected_rows[(ds_id, item.get('finding_id'), item.get('reviewer'),
                           item.get('operation'), tuple(
                               (c.get('row'), c.get('source_row')) for c in item.get('changes', [])))] += 1
    if expected_columns != actual_columns or expected_rows != actual_rows:
        raise PermissionError('Export-only row or column operations lack matching approvals')
    for fid, status in statuses.items():
        if not _reviewed(status) or fid not in titles:
            raise PermissionError('Review status does not belong to a known finding')
        if status != 'Reviewed — no change' and not any(
                e.get('finding_id') == fid for e in history):
            raise PermissionError('Completed finding status lacks an approved operation')


def _checked_audits(audits: dict, statuses: dict) -> dict:
    """Copy only serializable audit artifacts for completed findings."""
    if not isinstance(audits, dict):
        raise ValueError('Invalid saved R audit artifacts')
    clean = {}
    for fid, entry in audits.items():
        if fid not in statuses or not isinstance(entry, dict):
            continue
        if not all(isinstance(entry.get(k, ''), str)
                   for k in ('r_function', 'insertion_r_function')):
            raise ValueError('Invalid completed R code')
        if not isinstance(entry.get('changes', []), list):
            raise ValueError('Invalid completed R change ledger')
        clean[fid] = {k: entry[k] for k in
                      ('r_function', 'insertion_r_function', 'changes') if k in entry}
    return clean


def save_approved_stage(
    sources: list[Path], staged: dict, columns: dict, rows: dict,
    history: list[dict], statuses: dict, reviewer: str, titles: dict[str, str],
    *, completed_audits: dict | None = None,
) -> Path | None:
    """Persist only completed researcher decisions; no original CSV is saved.

    Unreviewed recommendations and Gemma conversations are never checkpointed.
    """
    safe_statuses = {k: v for k, v in statuses.items()
                     if isinstance(k, str) and isinstance(v, str) and _reviewed(v)
                     and k in titles}
    if not history and not safe_statuses:
        return None
    _, hashes = _identity(sources)
    datasets = {}
    for idx, source in enumerate(sources, 1):
        ds_id = f'dataset_{idx}'
        original = Path(source).read_bytes()
        current = staged.get(ds_id, original)
        before, after = _records(original), _records(current)
        if len(before) != len(after) or any(len(a) != len(b) for a, b in zip(before, after)):
            raise ValueError('Staged CSV shape is inconsistent with the original')
        changes = [
            {'row': ri, 'column': ci, 'before': x, 'after': y}
            for ri, (a, b) in enumerate(zip(before, after), 1)
            for ci, (x, y) in enumerate(zip(a, b), 1)
            if x != y
        ]
        col_ops = columns.get(ds_id, [])
        row_ops = rows.get(ds_id, [])
        # Recheck the same immutable-source export guards as the main app.
        exported_rows = apply_approved_row_operations(original, current, row_ops)
        apply_approved_column_deletions(
            original, exported_rows, col_ops,
            approved_row_operations=row_ops or None,
            staged_before_rows=current if row_ops else None,
        )
        # CSV cell edits can be serialized with a different line ending from
        # the uploaded source (e.g. csv.writer's CRLF on a LF input). Preserve
        # only the staged serialization dialect, never unapproved data bytes.
        if b'\r\n' in current:
            stage_newline = 'crlf'
        elif b'\n' in current:
            stage_newline = 'lf'
        elif b'\r' in current:
            stage_newline = 'cr'
        else:
            stage_newline = 'none'
        datasets[ds_id] = {
            'stage_newline': stage_newline,
            'source_sha256': hashes[idx - 1],
            'stage_record_sha256': _record_hash(after),
            'changes': changes,
            'approved_rows': row_ops,
            'approved_columns': col_ops,
        }
    allowed_ids = set(titles)
    relevant_history = [event for event in history
                        if isinstance(event, dict) and event.get('finding_id') in allowed_ids]
    # Do not infer missing approvals: every staged change requires a review event.
    if any((v['changes'] or v['approved_rows'] or v['approved_columns']) for v in datasets.values()) \
            and not relevant_history:
        raise PermissionError('Edited stage has no matching approval history')
    for event in relevant_history:
        ds_id = event.get('dataset_id')
        if ds_id not in datasets or event.get('source_sha256') != datasets[ds_id]['source_sha256']:
            raise PermissionError('Approval history does not match the original dataset')
    _approval_ledger_matches_stage(
        sources, staged, columns, rows, relevant_history, safe_statuses, titles)
    audits = _checked_audits(completed_audits or {}, safe_statuses)
    checkpoint = {
        'version': _VERSION,
        'source_hashes': hashes,
        'datasets': datasets,
        'history': relevant_history,
        'audits': audits,
        'audits_sha256': sha256(json.dumps(audits, ensure_ascii=False, sort_keys=True,
                                          separators=(',', ':')).encode('utf-8')).hexdigest(),
        'statuses': safe_statuses,
        'titles': {k: titles[k] for k in safe_statuses},
        'reviewer': str(reviewer or ''),
    }
    data = json.dumps(checkpoint, ensure_ascii=False, separators=(',', ':'), default=str).encode('utf-8')
    if len(data) > _MAX_BYTES:
        raise ValueError('Approved-stage checkpoint exceeds the safe size limit')
    destination = _location(sources)
    with _LOCK:
        if destination.parent.is_symlink() or destination.is_symlink():
            raise OSError('Refusing to write a recovery checkpoint through a symbolic link')
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        if destination.exists() and destination.read_bytes() == data:
            return destination
        temporary = destination.parent / ('.' + uuid4().hex + '.tmp')
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return destination


class LegacyCheckpointError(ValueError):
    """An old, source-bound record may be inspected but never auto-restored."""


def inspect_legacy_stage(sources: list[Path], finding_titles: dict[str, str]) -> dict:
    """Verify a V1 checkpoint for *read-only* review, never authorize its edits.

    The old format contains before/after cell deltas, but lacks value-level
    proof that those were the values approved. No staged bytes or trusted
    completion states are returned to the caller.
    """
    path = _location(sources)
    if path.parent.is_symlink() or path.is_symlink():
        raise OSError('Refusing to inspect a checkpoint through a symbolic link')
    if not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError('Legacy checkpoint is missing or exceeds the size limit')
    with _LOCK:
        raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get('version') != 1:
        raise ValueError('Not a V1 checkpoint')
    _, hashes = _identity(sources)
    if payload.get('source_hashes') != hashes:
        raise PermissionError('Legacy checkpoint source hashes do not match the current datasets')
    dataset_records = payload.get('datasets')
    if not isinstance(dataset_records, dict) or len(dataset_records) != len(sources):
        raise ValueError('Legacy checkpoint dataset count does not match')
    deltas = []
    structural = {}
    for index, source in enumerate(sources, 1):
        ds_id = f'dataset_{index}'
        record = dataset_records.get(ds_id)
        original = source.read_bytes()
        if not isinstance(record, dict) or record.get('source_sha256') != sha256_bytes(original):
            raise PermissionError('Legacy checkpoint contains a mismatched source dataset')
        base_rows = _records(original)
        if not base_rows or not base_rows[0]:
            raise ValueError('Cannot inspect empty legacy source')
        after_rows = [list(row) for row in base_rows]
        changes = record.get('changes')
        if not isinstance(changes, list):
            raise ValueError('Legacy checkpoint changes are malformed')
        seen = set()
        for change in changes:
            if not isinstance(change, dict):
                raise ValueError('Malformed legacy edit')
            r, c = change.get('row'), change.get('column')
            if (type(r) is not int or type(c) is not int
                    or not 1 <= r <= len(base_rows)
                    or not 1 <= c <= len(base_rows[r-1])
                    or (r, c) in seen
                    or not isinstance(change.get('before'), str)
                    or not isinstance(change.get('after'), str)
                    or base_rows[r-1][c-1] != change['before']):
                raise PermissionError('Legacy edit cannot be matched to its original cell')
            seen.add((r, c))
            after_rows[r-1][c-1] = change['after']
            deltas.append({'dataset': ds_id, 'csv_row': r, 'column': c,
                           'before': change['before'], 'after': change['after']})
        if record.get('stage_record_sha256') != _record_hash(after_rows):
            raise PermissionError('Legacy edits failed their saved checksum')
        row_ops = record.get('approved_rows')
        col_ops = record.get('approved_columns')
        if not isinstance(row_ops, list) or not isinstance(col_ops, list):
            raise ValueError('Malformed legacy structural operations')
        # Validate the actual transformation without importing it as approved.
        if changes:
            buf = StringIO(newline='')
            csv.writer(buf).writerows(after_rows)
            staged = (b'\xef\xbb\xbf' if original.startswith(b'\xef\xbb\xbf') else b'') + buf.getvalue().encode('utf-8')
        else:
            staged = original
        exported_rows = apply_approved_row_operations(original, staged, row_ops)
        apply_approved_column_deletions(
            original, exported_rows, col_ops,
            approved_row_operations=row_ops or None,
            staged_before_rows=staged if row_ops else None,
        )
        structural[ds_id] = {'rows': row_ops, 'columns': col_ops}
    statuses = payload.get('statuses')
    if not isinstance(statuses, dict):
        raise ValueError('Legacy checkpoint statuses are malformed')
    if any(not isinstance(fid, str) or not isinstance(status, str) for fid, status in statuses.items()):
        raise ValueError('Legacy checkpoint contains invalid review status data')
    history = payload.get('history')
    if not isinstance(history, list):
        raise ValueError('Legacy checkpoint history is malformed')
    return {
        'cell_changes': deltas,
        'structural_operations': structural,
        'saved_statuses': statuses,
        'history_count': len(history),
        'source_hashes': hashes,
        'checkpoint_digest': sha256(raw).hexdigest(),
        'backup_json': raw,
        'unmatched_findings': sorted(set(statuses).difference(finding_titles)),
    }


def archive_legacy_stage(sources: list[Path], *, expected_digest: str | None = None) -> Path:
    """Atomically preserve the V1 checkpoint elsewhere before a new review.

    Never erase it, import any edits, or overwrite a newer V2 checkpoint.
    """
    path = _location(sources)
    if path.parent.is_symlink() or path.is_symlink():
        raise OSError('Refusing to archive a checkpoint through a symbolic link')
    with _LOCK:
        if not path.is_file() or path.stat().st_size > _MAX_BYTES:
            raise ValueError('Legacy checkpoint is missing or too large')
        raw = path.read_bytes()
        if expected_digest is not None and sha256(raw).hexdigest() != expected_digest:
            raise PermissionError('Checkpoint changed since preview; refresh before archiving')
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get('version') != 1:
            raise PermissionError('Checkpoint is no longer V1; refusing to move it')
        if payload.get('source_hashes') != _identity(sources)[1]:
            raise PermissionError('Checkpoint is not for these source bytes')
        destination = path.with_name(path.stem + '.legacy-v1-' + uuid4().hex + '.json')
        os.replace(path, destination)
        return destination


def invalidate_approved_stage(sources: list[Path]) -> None:
    """Remove obsolete recovery data when every approval has been withdrawn.

    Source files are never touched. This prevents another session from
    restoring a cleaning action the researcher explicitly reversed.
    """
    location = _location(sources)
    with _LOCK:
        if location.parent.is_symlink() or location.is_symlink():
            raise OSError('Refusing to remove recovery through a symbolic link')
        location.unlink(missing_ok=True)


def load_approved_stage(sources: list[Path]) -> dict | None:
    """Locate matching history; this does not authorize or restore edits."""
    location = _location(sources)
    if location.parent.is_symlink() or location.is_symlink():
        raise OSError('Refusing to read a recovery checkpoint through a symbolic link')
    if not location.is_file():
        return None
    if location.stat().st_size > _MAX_BYTES:
        raise ValueError('Approved-stage checkpoint is unexpectedly large')
    with _LOCK:
        data = json.loads(location.read_text(encoding='utf-8'))
    if isinstance(data, dict) and data.get('version') == 1:
        raise LegacyCheckpointError('Legacy checkpoint lacks exact approval evidence; re-review old changes')
    if not isinstance(data, dict) or data.get('version') != _VERSION:
        raise ValueError('Unsupported or invalid approved-stage checkpoint version')
    if data.get('source_hashes') != _identity(sources)[1]:
        raise PermissionError('Saved edits belong to different original CSV bytes')
    return data


def verify_and_reconstruct(checkpoint: dict, sources: list[Path],
                           finding_titles: dict[str, str]) -> dict:
    """Verify every edit and reconstructed export BEFORE the user restores it."""
    if not isinstance(checkpoint, dict) or checkpoint.get('version') != _VERSION:
        raise ValueError('Invalid approved-stage checkpoint')
    if checkpoint.get('source_hashes') != _identity(sources)[1]:
        raise PermissionError('Original CSV has changed since approval')
    statuses = checkpoint.get('statuses')
    titles = checkpoint.get('titles')
    history = checkpoint.get('history')
    if not isinstance(statuses, dict) or not isinstance(titles, dict) or not isinstance(history, list):
        raise ValueError('Incomplete review checkpoint')
    if any(not _reviewed(v) or finding_titles.get(fid) != titles.get(fid)
           for fid, v in statuses.items()):
        raise PermissionError('Saved findings do not match this investigation')
    audits = checkpoint.get('audits')
    if not isinstance(audits, dict) or checkpoint.get('audits_sha256') != sha256(
            json.dumps(audits, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':')).encode('utf-8')).hexdigest():
        raise PermissionError('Saved R audit artifacts failed integrity verification')
    if _checked_audits(audits, statuses) != audits:
        raise PermissionError('Saved R audit artifacts reference a non-completed finding')
    dataset_records = checkpoint.get('datasets')
    if not isinstance(dataset_records, dict) or len(dataset_records) != len(sources):
        raise ValueError('Checkpoint dataset count does not match')
    staged = {}
    cols, rows = {}, {}
    for idx, source in enumerate(sources, 1):
        ds_id = f'dataset_{idx}'
        original = Path(source).read_bytes()
        record = dataset_records.get(ds_id)
        if not isinstance(record, dict) or record.get('source_sha256') != sha256_bytes(original):
            raise PermissionError('A source dataset is no longer identical to the approved one')
        original_rows = _records(original)
        restored_rows = [list(row) for row in original_rows]
        seen = set()
        changes = record.get('changes')
        if not isinstance(changes, list):
            raise ValueError('Invalid saved cell changes')
        for change in changes:
            if not isinstance(change, dict):
                raise ValueError('Invalid saved change')
            r, c = change.get('row'), change.get('column')
            if (type(r) is not int or type(c) is not int
                    or not 1 <= r <= len(restored_rows)
                    or not 1 <= c <= len(restored_rows[0])
                    or (r, c) in seen
                    or original_rows[r-1][c-1] != change.get('before')
                    or not isinstance(change.get('after'), str)):
                raise PermissionError('A saved cell edit is stale or invalid')
            seen.add((r, c))
            restored_rows[r-1][c-1] = change['after']
        if record.get('stage_record_sha256') != _record_hash(restored_rows):
            raise PermissionError('Recovered cell edits failed the recorded checksum')
        if changes:
            # An older checkpoint without newline metadata defaults to the
            # uploaded source's line ending; its event hash must still match.
            stage_newline = record.get('stage_newline')
            if stage_newline is None:
                stage_newline = ('crlf' if b'\r\n' in original else
                                 'lf' if b'\n' in original else
                                 'cr' if b'\r' in original else 'none')
            if stage_newline not in {'crlf', 'lf', 'cr', 'none'}:
                raise PermissionError('Invalid saved CSV serialization dialect')
            newline = {'crlf': '\r\n', 'lf': '\n',
                       'cr': '\r', 'none': '\n'}[stage_newline]
            buffer = StringIO(newline='')
            csv.writer(buffer, lineterminator=newline).writerows(restored_rows)
            content = (b'\xef\xbb\xbf' if original.startswith(b'\xef\xbb\xbf') else b'') + buffer.getvalue().encode('utf-8')
        else:
            content = original
        if _records(content) != restored_rows:
            raise AssertionError('Reconstructed in-memory CSV differs from verified cells')
        col_ops = record.get('approved_columns')
        row_ops = record.get('approved_rows')
        if not isinstance(col_ops, list) or not isinstance(row_ops, list):
            raise ValueError('Invalid approved structural operations')
        exported_rows = apply_approved_row_operations(original, content, row_ops)
        apply_approved_column_deletions(
            original, exported_rows, col_ops,
            approved_row_operations=row_ops or None,
            staged_before_rows=content if row_ops else None,
        )
        staged[ds_id], cols[ds_id], rows[ds_id] = content, col_ops, row_ops
    for event in history:
        if not isinstance(event, dict):
            raise ValueError('Invalid saved approval history')
        ds_id = event.get('dataset_id')
        if (ds_id not in dataset_records
                or event.get('source_sha256') != dataset_records[ds_id]['source_sha256']
                or event.get('finding_id') not in finding_titles):
            raise PermissionError('Saved approval cannot be matched to this investigation')
    _approval_ledger_matches_stage(sources, staged, cols, rows, history, statuses, finding_titles)
    visual_flags = {}
    for event in history:
        if event.get('operation') != 'flag_cells':
            continue
        ds_id = event['dataset_id']
        flagged = visual_flags.setdefault(ds_id, [])
        flagged.extend({**flag, 'finding_id': event['finding_id'],
                        'reviewer': event['reviewer']}
                       for flag in event['flags'])
    return {
        'cleaning_visual_flags_by_dataset': visual_flags,
        'cleaning_completed_audits_by_finding': audits,
        'cleaning_staged_bytes_by_dataset': staged,
        'cleaning_approved_columns_by_dataset': cols,
        'cleaning_approved_rows_by_dataset': rows,
        'cleaning_history': history,
        'cleaning_review_statuses': statuses,
        'cleaning_approved_reviewer': str(checkpoint.get('reviewer') or ''),
    }


def checkpoint_counts(checkpoint: dict) -> tuple[int, int]:
    """Number of approved operations and completed no-change reviews."""
    return (len(checkpoint.get('history') or []),
            sum(v == 'Reviewed — no change' for v in (checkpoint.get('statuses') or {}).values()))
