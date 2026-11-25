"""Reconcile historical visual flags with the surviving approved CSV structure.

Deleted original physical rows/columns remain in audit history only. They
must never appear as active red/bold annotations in review previews or Excel.
"""
from __future__ import annotations


def split_active_visual_flags(flags: list[dict], approved_rows: list[dict],
                              approved_columns: list[dict]) -> tuple[list[dict], list[dict]]:
    deleted_rows: set[int] = set()
    deleted_columns: set[int] = set()
    for event in approved_rows or []:
        if not isinstance(event, dict):
            continue
        kind = event.get('operation')
        for item in event.get('changes') or []:
            if not isinstance(item, dict):
                continue
            row = item.get('source_row') if kind == 'merge_rows' else item.get('row') if kind == 'drop_rows' else None
            if type(row) is int:
                deleted_rows.add(row)
    for event in approved_columns or []:
        if isinstance(event, dict) and type(event.get('position')) is int:
            deleted_columns.add(event['position'])
    active, superseded = [], []
    for flag in flags or []:
        if not isinstance(flag, dict):
            continue
        if flag.get('row') in deleted_rows or flag.get('column') in deleted_columns:
            superseded.append(flag)
        else:
            active.append(flag)
    return active, superseded
