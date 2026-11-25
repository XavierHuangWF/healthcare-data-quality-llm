"""Detect overlapping proposed edits against ALL previously approved actions.

Source coordinates are immutable physical CSV coordinates. A later finding
must not silently reapply a deletion or overwrite a surviving approved edit.
The reviewer may reopen an earlier finding to revise the dependency.
"""
from __future__ import annotations

from pathlib import Path


def _row_deletions(approved_rows: list[dict]) -> set[int]:
    deleted = set()
    for op in approved_rows or []:
        kind = op.get('operation')
        for change in op.get('changes') or []:
            if kind == 'drop_rows':
                deleted.add(change.get('row'))
            elif kind == 'merge_rows':
                deleted.add(change.get('source_row'))
    return deleted


def proposal_conflicts(operations: list[dict], source: Path, staged: bytes,
                       approved_rows: list[dict] | None = None,
                       approved_columns: list[dict] | None = None, *,
                       approved_flags: list[dict] | None = None) -> list[str]:
    """Return specific conflicts; no mutation or plan rewriting is performed."""
    from healthcare_data_quality.cleaning.in_memory_cleaning import _records
    original = _records(Path(source).read_bytes())
    working = _records(staged)
    if (len(original) != len(working) or
            any(len(left) != len(right) for left, right in zip(original, working))):
        return ['Working dataset no longer retains original physical coordinates']
    removed_rows = _row_deletions(approved_rows or [])
    removed_columns = {p.get('position') for p in approved_columns or []}
    issues = []
    # Visual annotations are coordinate-bound styling, not approved values.
    # They persist on surviving cells; actual earlier value edits still conflict.
    for plan in operations or []:
        kind = plan.get('operation')
        for change in plan.get('changes') or []:
            row = change.get('row')
            col = change.get('column') if kind == 'replace_cells' else change.get('position')
            if kind == 'merge_rows':
                if row in removed_rows or change.get('source_row') in removed_rows:
                    issues.append('A duplicate-row merge references an already removed row')
                continue
            if kind == 'drop_rows':
                if row in removed_rows:
                    issues.append(f'Row {row} was already approved for deletion')
                continue
            if kind == 'drop_column':
                if col in removed_columns:
                    issues.append(f'Column {col} was already approved for deletion')
                continue
            if kind in {'replace_cells', 'rename_headers'}:
                target_row = row if kind == 'replace_cells' else 1
                if target_row in removed_rows or col in removed_columns:
                    issues.append(f'R{target_row}:C{col} is removed by an earlier approval')
                elif (type(target_row) is int and type(col) is int and
                      1 <= target_row <= len(original) and
                      1 <= col <= len(original[0]) and
                      working[target_row-1][col-1] != original[target_row-1][col-1]):
                    issues.append(f'R{target_row}:C{col} was already changed by an earlier approval')
    return list(dict.fromkeys(issues))


def filter_conflicting_choices(choices: list[dict], *, state, finding: dict,
                               source: Path) -> tuple[list[dict], list[str]]:
    ds_id = finding['dataset_id']
    valid, blocked = [], []
    current = state.get('cleaning_staged_bytes_by_dataset', {}).get(ds_id, Path(source).read_bytes())
    for choice in choices:
        if choice.get('decision') == 'no_change':
            valid.append(choice)
            continue
        if any(op.get('operation') == 'flag_cells' for op in choice.get('operations', [])):
            from healthcare_data_quality.cleaning.visual_recommendations import visual_targets_current
            if visual_targets_current(state, finding, source, choice):
                valid.append(choice)
            else:
                blocked.append('Annotation targets changed or were removed')
            continue
        conflicts = proposal_conflicts(
            choice.get('operations') or [], source, current,
            state.get('cleaning_approved_rows_by_dataset', {}).get(ds_id, []),
            state.get('cleaning_approved_columns_by_dataset', {}).get(ds_id, []),
            approved_flags=state.get('cleaning_visual_flags_by_dataset', {}).get(ds_id, []),
        )
        if conflicts:
            blocked.extend(conflicts)
        else:
            valid.append(choice)
    return valid, list(dict.fromkeys(blocked))
