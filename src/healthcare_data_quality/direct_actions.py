"""Small, evidence-only cleaning suggestions for unambiguous format defects.

These are drafts, not approvals. The standard finding validator, R check and
single named human approval remain mandatory. Never use these transforms to
invent taxonomy levels or modify biological identities.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

PREFIXES = ('k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__')


def _source_rows(source: Path) -> list[list[str]]:
    with Path(source).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle, strict=True))
    if len(rows) < 2 or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('Source CSV is not rectangular')
    return rows


def _majority_style(rows: list[list[str]]) -> str | None:
    full, unprefixed = 0, 0
    for row in rows[1:]:
        parts = [v.strip() for v in row[0].split(';')]
        if len(parts) != 7:
            continue
        meaningful = [(i, value) for i, value in enumerate(parts) if value and value.upper() != 'NA']
        if not meaningful:
            continue
        prefixed = sum(value.casefold().startswith(PREFIXES[i]) for i, value in meaningful)
        if prefixed == len(meaningful):
            full += 1
        elif prefixed == 0:
            unprefixed += 1
    return ("prefixed" if full > unprefixed else "unprefixed" if unprefixed > full else None)


def draft_format_correction(finding: dict, source: Path) -> dict | None:
    """Draft exact original-cell replacements, or decline ambiguous cases.

    The source and finding evidence must agree exactly. Only delimiter fixes or
    adding missing rank prefixes to a predominantly prefixed dataset qualify.
    """
    key = finding.get('check_key')
    if key not in {'inspect_taxonomy_structure', 'inspect_taxonomy_naming'}:
        return None
    evidence = finding.get('evidence') or []
    rows = _source_rows(source)
    dominant = _majority_style(rows) if key == 'inspect_taxonomy_naming' else None
    if key == 'inspect_taxonomy_naming' and dominant is None:
        return None
    changes = []
    for item in evidence:
        if not isinstance(item, dict):
            return None
        row_id = item.get('CSV row')
        if type(row_id) is not int or not 2 <= row_id <= len(rows):
            return None
        before = rows[row_id - 1][0]
        if key == 'inspect_taxonomy_structure':
            if item.get('Problem') != 'Uses | instead of ;' or item.get('Original classification') != before:
                return None
            if '|' not in before:
                return None
            after = before.replace('|', ';')
        else:
            if item.get('Original classification') != before:
                return None
            # Mixed separators can be normalized in the same exact edit;
            # the separate delimiter finding independently verifies row 7.
            parts = [piece.strip() for piece in re.split(r'[;|]', before)]
            if len(parts) != 7:
                return None
            corrected = []
            for index, part in enumerate(parts):
                if not part or part.upper() == 'NA':
                    corrected.append(part)
                elif part.casefold().startswith(PREFIXES[index]):
                    corrected.append(part if dominant == 'prefixed' else part[len(PREFIXES[index]):])
                elif any(part.casefold().startswith(prefix) for prefix in PREFIXES):
                    # A wrong-rank prefix needs scientific review, not blind normalization.
                    return None
                else:
                    corrected.append(PREFIXES[index] + part if dominant == 'prefixed' else part)
            after = ';'.join(corrected)
        if after == before:
            continue
        changes.append({'row': row_id, 'column': 1, 'before': before, 'after': after})
    if not changes:
        return None
    return {
        'reply': f'{len(changes)} taxonomy format correction(s) ready for review.',
        'operation': 'replace_cells', 'dataset_id': finding['dataset_id'],
        'changes': changes, 'clarification_options': [],
    }


def draft_identical_column_removal(finding: dict, source: Path) -> dict | None:
    """Suggest removing ONLY a fully identical duplicate physical column.

    A shared name alone is insufficient. Every value in all source records must
    match, and the exact evidence pair must agree with the physical positions.
    """
    if finding.get('check_key') != 'compare_repeated_headers':
        return None
    rows = _source_rows(source)
    changes = []
    seen = set()
    for item in finding.get('evidence') or []:
        if not isinstance(item, dict):
            return None
        left, right = item.get('Column A'), item.get('Column B')
        if (type(left) is not int or type(right) is not int
                or not 1 <= left < right <= len(rows[0]) or right in seen):
            return None
        if rows[0][left - 1] != rows[0][right - 1]:
            return None
        if any(row[left - 1] != row[right - 1] for row in rows):
            return None
        seen.add(right)
        changes.append({'operation': 'drop_column', 'dataset_id': finding['dataset_id'],
                        'changes': [{'position': right, 'before': rows[0][right - 1], 'after': ''}]})
    if not changes:
        return None
    return {'reply': f'{len(changes)} identical duplicate column(s) can be removed.',
            'operation': 'none', 'dataset_id': '', 'changes': [],
            'operations': changes, 'clarification_options': []}


def same_taxonomy_identity(*values: str) -> bool:
    """Check equality after ONLY the two known presentation normalizations.

    This is safe solely for distinguishing benign separator/rank-prefix edits
    from any change to taxon identity. It does NOT establish biological truth.
    """
    comparable = []
    for value in values:
        if not isinstance(value, str):
            return False
        parts = [part.strip() for part in re.split(r'[;|]', value)]
        if len(parts) != 7 or not all(parts):
            return False
        names = []
        for index, part in enumerate(parts):
            if part.casefold().startswith(PREFIXES[index]):
                part = part[len(PREFIXES[index]):].strip()
            elif any(part.casefold().startswith(prefix) for prefix in PREFIXES):
                return False
            names.append(part)
        comparable.append(tuple(names))
    return len(comparable) >= 2 and len(set(comparable)) == 1


def already_resolved_format_issue(finding: dict, source: Path, staged: bytes) -> bool:
    """Detect a format issue already fixed by ANOTHER explicitly approved edit.

    Never mark a scientific anomaly resolved by a formatting comparison alone.
    Only physical source positions in the original finding are inspected.
    """
    key = finding.get('check_key')
    if key not in {'inspect_taxonomy_structure', 'inspect_taxonomy_naming'}:
        return False
    originals = _source_rows(source)
    try:
        import io
        current = list(csv.reader(io.StringIO(staged.decode('utf-8-sig'), newline='')))
    except (TypeError, UnicodeError, csv.Error, ValueError):
        return False
    if len(originals) != len(current) or any(len(a) != len(b) for a, b in zip(originals, current)):
        return False
    majority = _majority_style(originals) if key == 'inspect_taxonomy_naming' else None
    if key == 'inspect_taxonomy_naming' and majority is None:
        return False
    checks = 0
    for item in finding.get('evidence') or []:
        row = item.get('CSV row') if isinstance(item, dict) else None
        if type(row) is not int or not 2 <= row <= len(originals):
            return False
        before, now = originals[row-1][0], current[row-1][0]
        if before == now or not same_taxonomy_identity(before, now):
            return False
        if key == 'inspect_taxonomy_structure':
            if item.get('Problem') != 'Uses | instead of ;' or '|' in now:
                return False
        else:
            parts = [v.strip() for v in now.split(';')]
            if len(parts) != 7:
                return False
            meaningful = [(i, v) for i, v in enumerate(parts) if v and v.upper() != 'NA']
            if not meaningful:
                return False
            prefixed = sum(v.casefold().startswith(PREFIXES[i]) for i, v in meaningful)
            if majority == 'unprefixed' and prefixed != 0:
                return False
            if majority == 'prefixed' and prefixed != len(meaningful):
                return False
        checks += 1
    return checks > 0
