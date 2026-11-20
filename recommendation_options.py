"""Ground Agent B's proposed controls in immutable CSV evidence.

A recommendation is never authorization. Only exact source-backed proposals
are displayed as before/after choices; Agent C and the independent approval
validator still own the staging of any chosen operation.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

OPERATIONS = {'replace_cells', 'drop_rows', 'drop_column', 'rename_headers'}
_ROW_COL = re.compile(
    r'row\s+(\d+)\s*,?\s*column\s+(\d+)\s*:\s*[\'\"]([^\'\"]*)[\'\"]\s*(?:→|->|to)\s*[\'\"]([^\'\"]*)[\'\"]',
    re.IGNORECASE,
)


def _rows(source: Path) -> list[list[str]]:
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        result = list(csv.reader(stream))
    if len(result) < 2 or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise ValueError('The original source is not a rectangular CSV')
    return result


def _verified(operations: list[dict], rows: list[list[str]], finding: dict) -> list[dict] | None:
    """No hallucinated coordinates, stale before values, or source-independent edits."""
    if not isinstance(operations, list) or not 1 <= len(operations) <= 16:
        return None
    result, seen = [], set()
    evidence = [e for e in (finding.get('evidence') or []) if isinstance(e, dict)]
    allowed_rows = {e.get('CSV row') for e in evidence}
    for e in evidence:
        group = e.get('CSV rows') or e.get('csv_rows')
        if isinstance(group, str):
            allowed_rows.update(int(num) for num in re.findall(r'(?<!\d)\d+(?!\d)', group))
        elif isinstance(group, (list, tuple)):
            allowed_rows.update(num for num in group if type(num) is int)
    for e in evidence:
        group = e.get('CSV rows') or e.get('csv_rows')
        if isinstance(group, str):
            allowed_rows.update(int(num) for num in re.findall(r'(?<!\d)\d+(?!\d)', group))
        elif isinstance(group, (list, tuple)):
            allowed_rows.update(num for num in group if type(num) is int)
    allowed_cells = {(e.get('CSV row'), e.get('Column')) for e in evidence}
    allowed_cols = {v for e in evidence for v in (e.get('Column'), e.get('Column A'), e.get('Column B'))}
    if finding.get('check_key') == 'gemma_check':
        allowed_cols.update(i for i, h in enumerate(rows[0], 1) if 'blank' in h.casefold())
    for operation in operations:
        if not isinstance(operation, dict) or operation.get('operation') not in OPERATIONS:
            return None
        kind = operation['operation']
        changes = operation.get('changes')
        if not isinstance(changes, list) or not 1 <= len(changes) <= 3000:
            return None
        bound = []
        for change in changes:
            if not isinstance(change, dict):
                return None
            row = change.get('row')
            col = change.get('column') if kind == 'replace_cells' else change.get('position')
            if type(col) is not int or not 1 <= col <= len(rows[0]):
                if kind != 'drop_rows':
                    return None
            if kind == 'drop_rows':
                if type(row) is not int or not 2 <= row <= len(rows):
                    return None
                if row not in allowed_rows:
                    return None
                coord, before, after = ('row', row), rows[row-1][0], 'Delete row'
                bound.append({'row': row, 'before': before, 'after': ''})
            elif kind in {'rename_headers', 'drop_column'}:
                if col not in allowed_cols:
                    return None
                coord = ('column', col)
                before = rows[0][col-1]
                after = ('' if kind == 'drop_column' else change.get('after'))
                if kind == 'rename_headers' and (not isinstance(after, str) or not after.strip()):
                    return None
                if kind == 'drop_column' and col == 1:
                    return None
                bound.append({'position': col, 'before': before, 'after': after})
            else:
                if type(row) is not int or not 2 <= row <= len(rows):
                    return None
                if (row, col) not in allowed_cells:
                    return None
                coord = ('cell', row, col)
                before = rows[row-1][col-1]
                after = change.get('after')
                if not isinstance(after, str) or len(after) > 1500 or after == before:
                    return None
                bound.append({'row': row, 'column': col, 'before': before, 'after': after})
            # Even where we can recalculate Before, reject mismatches rather
            # than silently rewriting Gemma's alleged source evidence.
            if 'before' in change and change['before'] != before:
                return None
            if coord in seen:
                return None
            seen.add(coord)
        result.append({'operation': kind, 'dataset_id': finding.get('dataset_id', ''),
                       'changes': bound})
    return result


def _display(operations: list[dict]) -> list[dict]:
    output = []
    for op in operations:
        for change in op['changes']:
            kind = op['operation']
            if kind == 'replace_cells':
                location = f"R{change['row']} · C{change['column']}"
                after = change['after']
            elif kind == 'drop_rows':
                location, after = f"Row {change['row']}", 'Delete row'
            else:
                location = f"Column {change['position']}"
                after = 'Delete column' if kind == 'drop_column' else change['after']
            output.append({'Location': location, 'Before': change['before'], 'After': after})
    return output


def _replacements_from_evidence(finding: dict, rows: list[list[str]], after: str) -> list[dict]:
    evidence = finding.get('evidence') or []
    changes = []
    if len(evidence) != finding.get('total_evidence_records', len(evidence)):
        return []  # Evidence is only a sample; do not misrepresent coverage.
    for cell in evidence:
        if not isinstance(cell, dict):
            return []
        row, col = cell.get('CSV row'), cell.get('Column')
        if type(row) is not int or type(col) is not int or not (2 <= row <= len(rows) and 1 <= col <= len(rows[0])):
            return []
        if 'Original value' in cell and str(cell['Original value']) != rows[row-1][col-1]:
            return []
        if rows[row-1][col-1] != after:
            changes.append({'row': row, 'column': col, 'before': rows[row-1][col-1], 'after': after})
    return changes


def source_backed_choices(finding: dict, source: Path | None,
                          suggestion: dict) -> list[dict]:
    """Select 1–3 model-proposed controls only when exact CSV proof is available.

    The fallback can ground simple textual replacements. A model's unverified
    title or label is never itself proof that a proposed edit is safe.
    """
    if source is None:
        return []
    try:
        rows = _rows(source)
    except (OSError, ValueError, UnicodeError, csv.Error):
        return []
    output = []
    candidates = suggestion.get('choices') or []
    if not isinstance(candidates, list):
        candidates = []
    for candidate in candidates[:3]:
        if not isinstance(candidate, dict):
            continue
        label = ' '.join(str(candidate.get('label') or '').split())[:90]
        if not label:
            continue
        operations = candidate.get('operations')
        if not operations and candidate.get('operation') in OPERATIONS:
            operations = [candidate]
        # A bounded rule lets the model choose an approach without enumerating
        # hundreds of missing cells. Source evidence, not Gemma, fixes targets.
        if not operations and candidate.get('rule') == 'replace_evidenced_cells':
            replacement = candidate.get('replacement')
            if isinstance(replacement, str):
                changes = _replacements_from_evidence(finding, rows, replacement)
                if changes:
                    operations = [{'operation': 'replace_cells', 'changes': changes}]
        grounded = _verified(operations, rows, finding)
        if grounded:
            output.append({'label': label, 'operations': grounded,
                           'rows': _display(grounded)})
    if output:
        return output

    # Older cached recommendation prose: recover source-exact replacements.
    advice = str(suggestion.get('suggestion') or '')
    matches = _ROW_COL.findall(advice)
    if matches:
        changes = [{'row': int(r), 'column': int(c), 'before': old, 'after': new}
                   for r, c, old, new in matches]
        checked = _verified([{'operation': 'replace_cells', 'changes': changes}], rows, finding)
        if checked:
            return [{'label': 'Use suggested replacements', 'operations': checked,
                     'rows': _display(checked)}]
    # Narrow fallback for a fully evidenced, explicit constant replacement.
    if re.search(r'\b(?:replace|set|fill)\b', advice, flags=re.I):
        m = re.search(r'(?:with|to)\s+[\'\"]?(NA|0|1)[\'\"]?\b', advice, flags=re.I)
        if m:
            changes = _replacements_from_evidence(finding, rows, m.group(1).upper())
            if changes:
                checked = _verified([{'operation': 'replace_cells', 'changes': changes}], rows, finding)
                if checked:
                    return [{'label': 'Use suggested replacement', 'operations': checked,
                             'rows': _display(checked)}]
    return []
