"""Independent, human-readable reconciliation of original and approved CSV.

Display rows are generated from immutable original and computed final CSV bytes;
Gemma prose is never used to assert execution. Unexpected differences fail closed.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from .in_memory_cleaning import _records
from .styled_export import _exported_positions

DETAIL_HEADERS = [
    'Finding / check', 'Reviewer', 'Action', 'Original coordinate',
    'Original column', 'Original source value', 'Approved Before → After',
    'Actual final value', 'Final coordinate', 'Outcome', 'Match / evidence',
    'Approval group / record',
]
OVERVIEW_HEADERS = [
    'Finding / check', 'Review status', 'Reviewer', 'Approved actions',
    'Detailed audit rows', 'Chat turns', 'Result',
]


def independently_reconcile(*, original_csv: bytes, final_csv: bytes,
                            history: list[dict], review_statuses: dict,
                            findings: dict[str, str], approved_rows: list[dict],
                            approved_columns: list[dict],
                            discussions: dict | None = None) -> dict:
    original, final = _records(original_csv), _records(final_csv)
    row_map, col_map = _exported_positions(original, final, approved_rows, approved_columns)
    edits = defaultdict(list)
    merged_rows = set()
    for approval in approved_rows:
        if approval.get('operation') == 'merge_rows':
            for item in approval.get('changes') or []:
                if type(item.get('row')) is int:
                    merged_rows.add(item['row'])
    for event in history:
        if event.get('dataset_id') != 'dataset_1':
            continue
        if event.get('operation') in ('replace_cells', 'rename_headers'):
            for change in event.get('approved_cell_changes') or []:
                r, c = change.get('row'), change.get('column')
                if type(r) is int and type(c) is int:
                    edits[(r, c)].append((event, change))
    changes, unchanged = [], 0
    # This comparison covers *every* surviving physical cell, including header.
    # The strict pre-existing approval-ledger verifier independently replays the
    # history as well; this reconciler additionally refuses unexplained drift.
    for orig_r, final_r in row_map.items():
        for orig_c, final_c in col_map.items():
            before = original[orig_r-1][orig_c-1]
            after = final[final_r-1][final_c-1]
            if before == after:
                unchanged += 1
                continue
            approvals = edits.get((orig_r, orig_c)) or []
            merge = orig_r in merged_rows and orig_c > 1
            if not approvals and not merge:
                raise PermissionError(
                    f'Unapproved CSV value change at source R{orig_r}:C{orig_c}; '
                    'final download is blocked')
            if approvals:
                event, change = approvals[-1]
                who = str(event.get('reviewer') or '').strip()
                if not who:
                    raise PermissionError('CSV changes have no named reviewer')
                # Last approved value must match exactly. Never allow a final
                # value to differ merely because *some* edit was approved.
                # Explicit approved row merges can subsequently change counts;
                # their ledger replay is independently checked by the exporter.
                matches_edit = str(change.get('after', '')) == after
                if not matches_edit and not merge:
                    raise PermissionError(
                        f'Unapproved CSV value change at source R{orig_r}:C{orig_c}: '
                        f'last approved value {change.get("after")!r}, final {after!r}')
                approval_text = str(change.get('before', '')) + ' → ' + str(change.get('after', ''))
                proof = ('Exact final value agrees with last approved edit' if
                         matches_edit else
                         'Followed by approved row merge; see source row ledger')
                fid = str(event.get('finding_id') or '')
                kind = str(event.get('operation') or '')
                group = str(event.get('approval_group') or '')
            else:
                who, fid, kind, group = '', '', 'merge_rows', ''
                approval_text = 'Source counts → approved merged counts'
                proof = 'Changed by approved row merge; see source row ledger'
                for approval in approved_rows:
                    if approval.get('operation') == 'merge_rows' and any(
                            item.get('row') == orig_r for item in approval.get('changes') or []):
                        fid = str(approval.get('finding_id') or '')
                        who = str(approval.get('reviewer') or '')
                        break
            changes.append([
                findings.get(fid, fid), who, kind, f'R{orig_r}:C{orig_c}',
                original[0][orig_c-1] or '(blank header)', before,
                approval_text, after, f'R{final_r}:C{final_c}',
                'Changed in final file', proof, group,
            ])
    for orig_r in sorted(set(range(2, len(original)+1)) - set(row_map)):
        related = [a for a in approved_rows
                   if any(item.get('row') == orig_r or
                          item.get('source_row') == orig_r
                          for item in a.get('changes') or [])]
        if not related:
            raise PermissionError(f'Row {orig_r} deleted without approval ledger entry')
        approval = related[-1]
        fid = str(approval.get('finding_id') or '')
        whole = json.dumps(original[orig_r-1], ensure_ascii=False)
        changes.append([
            findings.get(fid, fid), str(approval.get('reviewer') or ''),
            str(approval.get('operation') or 'drop_rows'), f'Row {orig_r}',
            '(complete original CSV row)', whole,
            'Approved row deletion', '(absent)', '(deleted)',
            'Deleted from final file',
            'Original row retained verbatim in this audit; full CSV is in the ZIP', '',
        ])
    for orig_c in sorted(set(range(1, len(original[0])+1)) - set(col_map)):
        related = [a for a in approved_columns if a.get('position') == orig_c]
        if not related:
            raise PermissionError(f'Column {orig_c} deleted without approval ledger entry')
        approval = related[-1]
        fid = str(approval.get('finding_id') or '')
        column_values = json.dumps([row[orig_c-1] for row in original], ensure_ascii=False)
        changes.append([
            findings.get(fid, fid), str(approval.get('reviewer') or ''),
            'drop_column', f'Column {orig_c}', original[0][orig_c-1],
            column_values, 'Approved entire-column deletion', '(absent)', '(deleted)',
            'Deleted from final file',
            'Original entire column retained in this audit; full CSV is in the ZIP', '',
        ])
    counts = defaultdict(lambda: {'changes': 0, 'actions': set(), 'reviewer': set()})
    for event in history:
        if event.get('dataset_id') != 'dataset_1':
            continue
        fid = str(event.get('finding_id') or '')
        counts[fid]['actions'].add(str(event.get('approval_group') or id(event)))
        if event.get('reviewer'):
            counts[fid]['reviewer'].add(str(event['reviewer']))
    for entry in changes:
        for fid, title in findings.items():
            if title == entry[0]:
                counts[fid]['changes'] += 1
                break
    chat_counts = {fid: sum(m.get('role') in {'user','assistant'}
                            for m in items if isinstance(m, dict))
                   for fid, items in (discussions or {}).items()}
    summary = []
    for fid, title in findings.items():
        status = str(review_statuses.get(fid) or 'Not reviewed')
        info = counts[fid]
        result = ('Kept unchanged' if status == 'Reviewed — no change'
                  else 'Approved; see source-to-final checks' if info['actions']
                  else 'Pending — export should be locked')
        summary.append([title, status, ', '.join(sorted(info['reviewer'])),
                        len(info['actions']), info['changes'], chat_counts.get(fid, 0), result])
    digest = hashlib.sha256(final_csv).hexdigest()
    return {'summary': summary, 'details': changes,
            'original_rows': len(original)-1, 'original_columns': len(original[0]),
            'final_rows': len(final)-1, 'final_columns': len(final[0]),
            'surviving_unchanged_cells': unchanged,
            'surviving_changed_cells': sum(c[9] == 'Changed in final file' for c in changes),
            'deleted_rows': len(original)-len(row_map),
            'deleted_columns': len(original[0])-len(col_map),
            'final_sha256': digest}
