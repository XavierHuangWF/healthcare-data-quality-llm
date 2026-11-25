"""Verify zero-edit review closure when earlier named approvals removed the targets.

A completed status by itself is not authorization. The closure is only valid
when the reviewer explicitly closed the finding, the original source matches,
and all referenced rows are covered by already-replayed approval events.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from .in_memory_cleaning import _records


def verified_prior_review_closures(
    sources: list[Path], approved_rows: dict, history: list[dict],
    statuses: dict, decisions: list[dict] | None,
) -> list[dict]:
    """Return checked minimal closure proofs; reject missing/ambiguous proofs."""
    if not isinstance(decisions, list):
        decisions = []
    terminal = {fid for fid, status in statuses.items()
                if status == 'Resolved by earlier approved edit'
                and not any(e.get('finding_id') == fid for e in history)}
    proofs = []
    for fid in sorted(terminal):
        candidates = [d for d in decisions if isinstance(d, dict)
                      and d.get('finding_id') == fid
                      and d.get('decision') == 'resolved_by_prior_approvals']
        if not candidates:
            raise PermissionError(
                f"Finding {fid} was closed as resolved by earlier work but lacks "
                "a named, source-verified closure record")
        # Some earlier UI versions duplicated identical close events on rerun.
        # Duplicates do not create extra authorization; contradictory recorded
        # closures must never be silently resolved in favor of the newest one.
        important = ('source_sha256', 'dataset_id', 'reviewer', 'original_csv_rows',
                     'earlier_approved_findings', 'new_edits')
        if any(any(candidate.get(k) != candidates[-1].get(k) for k in important)
               for candidate in candidates[:-1]):
            raise PermissionError(f'Finding {fid} has conflicting closure records')
        proof = candidates[-1]
        ds_id = proof.get('dataset_id')
        if ds_id is None:
            # Legacy session decision records have no dataset_id. The source
            # evidence is still unambiguous only for the one-dataset workflow.
            if len(sources) != 1:
                raise PermissionError('Cross-finding review is missing dataset identity')
            ds_id = 'dataset_1'
        if not isinstance(ds_id, str) or not ds_id.startswith('dataset_'):
            raise PermissionError('Invalid cross-finding dataset identity')
        try:
            i = int(ds_id.rsplit('_', 1)[1]) - 1
        except (ValueError, IndexError):
            raise PermissionError('Invalid cross-finding dataset identity') from None
        if not 0 <= i < len(sources):
            raise PermissionError('Cross-finding review targets another dataset')
        raw = Path(sources[i]).read_bytes()
        original = _records(raw)
        digest = sha256(raw).hexdigest()
        requested = proof.get('original_csv_rows')
        referenced_fids = proof.get('earlier_approved_findings')
        reviewer = proof.get('reviewer')
        if (proof.get('source_sha256') != digest or
            not isinstance(reviewer, str) or not reviewer.strip() or
            proof.get('new_edits') != 0 or
            not isinstance(requested, list) or not requested or
            any(type(r) is not int or not 2 <= r <= len(original) for r in requested) or
            len(set(requested)) != len(requested) or
            not isinstance(referenced_fids, list) or not referenced_fids or
            any(not isinstance(name, str) or not name for name in referenced_fids) or
            len(set(referenced_fids)) != len(referenced_fids)):
            raise PermissionError(f'Finding {fid} has an invalid source-bound closure record')
        approvals = [a for a in approved_rows.get(ds_id, []) if isinstance(a, dict)]
        covered = {}
        for approval in approvals:
            if (approval.get('source_sha256') != digest
                    or approval.get('finding_id') not in referenced_fids
                    or not str(approval.get('reviewer') or '').strip()
                    or approval.get('operation') not in {'drop_rows', 'merge_rows'}):
                continue
            for change in approval.get('changes') or []:
                if not isinstance(change, dict):
                    continue
                row = (change.get('row') if approval['operation'] == 'drop_rows'
                       else change.get('source_row'))
                if type(row) is int:
                    covered[row] = approval['finding_id']
        if not set(requested).issubset(covered):
            raise PermissionError(f'Finding {fid} references missing/revoked approved deletions')
        if set(covered[r] for r in requested) != set(referenced_fids):
            raise PermissionError(f'Finding {fid} references unrelated approved findings')
        # Ensure these structural operations have already passed ledger replay.
        if not all(any(e.get('dataset_id') == ds_id
                           and e.get('finding_id') == name
                           and e.get('operation') in {'drop_rows', 'merge_rows'}
                           for e in history) for name in referenced_fids):
            raise PermissionError(f'Finding {fid} references deletion without ledger evidence')
        proofs.append({**proof, 'dataset_id': ds_id})
    return proofs
