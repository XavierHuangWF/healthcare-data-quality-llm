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


def taxonomy_convention_evidence(source: Path) -> dict:
    """Report dataset conventions, never assert biological taxon validity."""
    records = _source_rows(source)
    source_values = [row[0] for row in records[1:]]
    full = [item for item in source_values if len(item.split(';')) == 7]
    unprefixed = sum(not any(piece.strip().casefold().startswith(PREFIXES)
                             for piece in value.split(';')) for value in full)
    prefixed = sum(all(piece.strip().casefold().startswith(PREFIXES[index])
                       for index, piece in enumerate(value.split(';'))
                       if piece.strip().upper() != 'NA')
                   for value in full)
    return {
        'data_rows': len(source_values),
        'semicolon_rows': sum('|' not in value and ';' in value for value in source_values),
        'seven_rank_semicolon_rows': len(full),
        'unprefixed_seven_rank_rows': unprefixed,
        'prefixed_seven_rank_rows': prefixed,
        'majority_style': _majority_style(records),
    }


def _standard_taxonomy_value(before: str, *, dominant: str) -> str | None:
    """Change only known delimiters/rank prefixes, never scientific names.

    Preserve strain annotations, NA placeholders and the exact token sequence.
    Refuse an incorrect prefix at any position rather than "correcting" identity.
    """
    parts = [part.strip() for part in re.split(r'[;|]', before)]
    if len(parts) != 7 or any(not item for item in parts):
        return None
    names = []
    for index, part in enumerate(parts):
        if part.upper() == 'NA':
            names.append(part)
        elif part.casefold().startswith(PREFIXES[index]):
            names.append(part[len(PREFIXES[index]):].strip())
        elif any(part.casefold().startswith(p) for p in PREFIXES) or re.match(r'^[a-z]{1,3}__', part, re.I):
            # Unknown/misplaced rank prefixes are ambiguous, not a safe rename.
            return None
        else:
            names.append(part)
    if dominant == 'unprefixed':
        return '; '.join(names)
    if dominant == 'prefixed':
        return '; '.join(token if token.upper() == 'NA' else PREFIXES[i] + token
                         for i, token in enumerate(names))
    return None


def draft_format_correction(finding: dict, source: Path) -> dict | None:
    """Offer the *complete* dataset-convention correction for an evidenced row.

    A separator inconsistency may coexist with a naming inconsistency. Checking
    only the first defect would leave an unnecessarily malformed taxonomy.
    Read the whole CSV, preserve the biological tokens, and report both changes.
    """
    key = finding.get('check_key')
    if key not in {'inspect_taxonomy_structure', 'inspect_taxonomy_naming'}:
        return None
    evidence = finding.get('evidence') or []
    rows = _source_rows(source)
    observed = taxonomy_convention_evidence(source)
    dominant = observed['majority_style']
    if dominant is None:
        return None
    changes = []
    for item in evidence:
        if not isinstance(item, dict):
            return None
        row_id = item.get('CSV row')
        if type(row_id) is not int or not 2 <= row_id <= len(rows):
            return None
        before = rows[row_id - 1][0]
        expected = item.get('Original classification') or item.get('Original classification (verbatim)')
        if expected != before:
            return None
        if key == 'inspect_taxonomy_structure' and (
                item.get('Problem') != 'Uses | instead of ;' or '|' not in before):
            return None
        after = _standard_taxonomy_value(before, dominant=dominant)
        if after is None:
            return None
        # All named taxon ranks, including suffixes, must be byte-identical
        # after removing only verified presentation punctuation.
        if not same_taxonomy_identity(before, after):
            return None
        if after != before:
            changes.append({'row': row_id, 'column': 1,
                            'before': before, 'after': after})
    if not changes:
        return None
    return {
        'reply': ('Normalize the separators AND rank-prefix naming to match '
                  'the observed dataset convention. Taxon names and identifiers '
                  'are preserved; biological identity is not independently verified.'),
        'reason': (f"{observed['semicolon_rows']}/{observed['data_rows']} rows use semicolons; "
                   f"{observed['unprefixed_seven_rank_rows']} of the "
                   f"{observed['seven_rank_semicolon_rows']} seven-rank semicolon rows "
                   'use unprefixed naming. Do not infer biological truth from frequency.'),
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


# Explicit, exact reviewer requests belong to a deterministic drafting path.
# This is NOT an AI inference of missing biological taxonomy ranks.
_TAXONOMY_PAD_COMMAND = re.compile(
    r"\b(?:pad|append|fill|complete|extend|add)\b", re.I,
)
_TAXONOMY_SEVEN_LEVELS = re.compile(
    r"\b(?:7|seven)\s*(?:[- ]\s*)?(?:levels?|ranks?|positions?)\b", re.I,
)


def propose_explicit_taxonomy_padding(message: str, finding: dict,
                                      source: Path) -> dict | None:
    """Draft exact seven-position NA padding for a clearly scoped reviewer ask.

    Match *all* rows evidenced by the selected finding against the immutable
    CSV, including their original text and reported rank count. Only records
    with 1–6 nonempty ranks and unambiguous semicolon separators qualify.
    Anything else falls through for discussion; no fabricated taxon labels.
    """
    if (finding.get("check_key") != "inspect_taxonomy_structure"
            or not isinstance(message, str) or len(message) > 4000
            or not _TAXONOMY_PAD_COMMAND.search(message)
            or not _TAXONOMY_SEVEN_LEVELS.search(message)
            or re.search(r"\b(?:do not|don't|never|without|instead of)\b", message, re.I)
            or '?' in message):
        return None
    from healthcare_data_quality.services.recommendation_options import (
        _taxonomy_na_padding,
    )
    try:
        records = _source_rows(source)
    except (OSError, ValueError, UnicodeError, csv.Error):
        return None
    changes = _taxonomy_na_padding(finding, records, "pad missing taxonomy ranks with NA")
    if not changes:
        return None
    return {
        "reply": f"Pad {len(changes)} taxonomy cells to seven levels using NA; review the exact changes below.",
        "operation": "replace_cells", "dataset_id": finding["dataset_id"],
        "changes": changes, "clarification_options": [],
    }


_COLUMN_DELETE_INTENT = re.compile(
    r"\b(?:delete|deleted|drop|remove)\s+(?:(?:the|these|entire|whole|csv)\s+)*columns?\s+",
    re.I,
)


def propose_explicit_column_deletions(message: str, finding: dict, source: Path) -> dict | None:
    """One verified proposal and ONE named approval for all named source columns.

    A literal reviewer command such as 'delete columns 7 and 11' is not a
    request to clear cells, and it requires no model clarification. Numbered
    positions always refer to immutable ORIGINAL CSV columns.
    """
    if (not isinstance(message, str) or len(message) > 500
            or '?' in message or re.search(r"\b(?:don't|do not|never|without|instead of)\b", message, re.I)):
        return None
    match = _COLUMN_DELETE_INTENT.search(message)
    if not match:
        return None
    suffix = message[match.end():].strip().rstrip('.! ')
    if not re.fullmatch(r"(?:\d+\s*(?:,|&|and|\+)\s*)*\d+", suffix, re.I):
        return None
    columns = [int(value) for value in re.findall(r"\d+", suffix)]
    if not 1 <= len(columns) <= 16 or len(set(columns)) != len(columns):
        raise ValueError("Select 1–16 different columns for deletion")
    records = _source_rows(source)
    if any(col < 2 or col > len(records[0]) for col in columns):
        raise ValueError("Column positions must exist; column 1 is the taxonomy identifier")
    if len(columns) >= len(records[0]):
        raise ValueError("Do not delete every source column")
    plans = [{"operation": "drop_column", "dataset_id": finding["dataset_id"],
              "changes": [{"position": col, "before": records[0][col-1], "after": ""}]}
             for col in columns]
    return {"reply": f"Delete entire original CSV columns {', '.join(map(str, columns))}; review all changes below.",
            "operation": "none", "dataset_id": "", "changes": [],
            "operations": plans, "clarification_options": [],
            "explicit_columns": frozenset(columns)}
