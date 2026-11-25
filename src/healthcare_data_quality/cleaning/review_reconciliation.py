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
                            discussions: dict | None = None, reviewer: str = "") -> dict:
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
    # Formatting is an approved action even when it changes no CSV values.
    # Preserve each historical flag in the source-to-final proof, with the
    # later structural approval that superseded it (if any).
    row_owners = {ch.get('row') if a.get('operation') == 'drop_rows' else ch.get('source_row'): a.get('finding_id')
                  for a in approved_rows for ch in a.get('changes', [])}
    col_owners = {a.get('position'): a.get('finding_id') for a in approved_columns}
    flag_events = [e for e in history if e.get('dataset_id') == 'dataset_1' and e.get('operation') == 'flag_cells']
    latest = {(f['row'],f['column']): (i,j) for i,e in enumerate(flag_events) for j,f in enumerate(e.get('flags', []))}
    for i,event in enumerate(flag_events):
        fid = str(event.get('finding_id') or '')
        for j,flag in enumerate(event.get('flags') or []):
            r,c=flag['row'],flag['column']
            if flag.get('before') != original[r-1][c-1] or flag.get('style') != 'bold_red':
                raise PermissionError('Annotation audit differs from original evidence')
            survives = r in row_map and c in col_map
            owners = [findings.get(str(x),str(x)) for x in dict.fromkeys([row_owners.get(r),col_owners.get(c)]) if x]
            if not survives:
                outcome, proof = 'Superseded by approved deletion', 'Deleted by: ' + '; '.join(owners)
            elif latest[(r,c)] != (i,j):
                outcome, proof = 'Superseded by later annotation', 'Later approved style replaces this annotation'
            else:
                outcome, proof = 'Annotation survives', 'Verify bold/red and approved fill in actual workbook'
            changes.append([findings.get(fid,fid),str(event.get('reviewer') or ''),'flag_cells',f'R{r}:C{c}',original[0][c-1],flag['before'],
                'Bold red + yellow highlight' if flag.get('fill',True) else 'Bold red font only',
                final[row_map[r]-1][col_map[c]-1] if survives else '(absent)',
                f'R{row_map[r]}:C{col_map[c]}' if survives else '(deleted)',outcome,proof,str(event.get('approval_group') or '')])
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
                  else 'Resolved by earlier approvals; no new edits' if status == 'Resolved by earlier approved edit'
                  else 'Approved; see source-to-final checks' if info['actions']
                  else 'Pending — export should be locked')
        summary.append([title, status, ', '.join(sorted(info['reviewer'])) or (reviewer if status in {'Resolved by earlier approved edit', 'Reviewed — no change'} else ''),
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
