"""Fail-closed boundaries between separately rendered findings/checks.

Recommendation workers can finish after a reviewer has moved on.  This module
prevents a stale, mismatched or cross-finding proposal from becoming a green
button.  It does not authorize or execute any change.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TERMINAL_REVIEW = frozenset({
    'Applied and verified', 'Resolved by earlier approved edit',
    'Approved column deletion for final export',
    'Approved row changes for final export', 'Reviewed — no change',
})


def finding_completed(state: dict, finding_id: str) -> bool:
    """A completed result is never rendered again as an active proposal."""
    if state.get('cleaning_review_statuses', {}).get(finding_id) in TERMINAL_REVIEW:
        return True
    return any(isinstance(item, dict) and item.get('finding_id') == finding_id
               for item in state.get('cleaning_history', []))


def scoped_review_choices(finding: dict, source: Path | None,
                          proposals: list[dict]) -> list[dict]:
    """Independently verify every button against THIS finding's actual evidence.

    A different finding may involve the same CSV, but its operations must never
    leak into this card (e.g. taxonomy padding buttons under duplicate taxa).
    """
    if source is None or not isinstance(proposals, list):
        return []
    from healthcare_data_quality.services.recommendation_options import (
        _display, _rows, _verified,
    )
    try:
        records = _rows(Path(source))
    except (OSError, UnicodeError, ValueError):
        return []
    fid, dataset_id = finding.get('id'), finding.get('dataset_id')
    check_key = finding.get('check_key')
    result, fingerprints = [], set()
    for option in proposals:
        if not isinstance(option, dict):
            continue
        if (option.get('finding_id', fid) != fid or
                option.get('dataset_id', dataset_id) != dataset_id):
            continue
        if option.get('decision') == 'no_change':
            if check_key not in {'inspect_headers', 'gemma_check'}:
                continue
            signature = ('no_change', fid)
            if signature not in fingerprints:
                result.append(option)
                fingerprints.add(signature)
            continue
        operations = option.get('operations')
        if not isinstance(operations, list) or not operations:
            continue
        kinds = {op.get('operation') for op in operations if isinstance(op, dict)}
        # An issue-specific recommendation cannot morph into an unrelated
        # correction merely because model/memory targets happen to exist.
        if check_key == 'find_duplicate_taxa' and not kinds <= {'drop_rows', 'merge_rows'}:
            continue
        if check_key in {'inspect_taxonomy_structure', 'inspect_taxonomy_naming'} and (
                kinds != {'replace_cells'} or any(
                    ch.get('column') != 1 for op in operations for ch in op.get('changes', [])
                )):
            continue
        if check_key == 'compare_repeated_headers' and kinds != {'drop_column'}:
            continue
        if check_key == 'inspect_headers' and kinds != {'rename_headers'}:
            continue
        verified = _verified(operations, records, finding)
        if verified is None:
            continue
        # Model-supplied or stale display rows cannot differ from the validated
        # operations. Render the exact current-source before/after instead.
        display = _display(verified)
        signature = json.dumps(verified, sort_keys=True, ensure_ascii=False)
        if signature in fingerprints:
            continue
        fingerprints.add(signature)
        result.append({**option, 'operations': verified, 'rows': display,
                       'finding_id': fid, 'dataset_id': dataset_id})
    return result


def scoped_clarification_choices(finding: dict, labels: list[str]) -> list[str]:
    """Suppress stale/foreign Agent C *questions* from another finding.

    These controls only clarify scope and are never approval actions.  A raw
    taxonomy string is a data value, not a user-facing option label.
    """
    evidence = [e for e in (finding.get('evidence') or []) if isinstance(e, dict)]
    allowed_rows = {e.get('CSV row') for e in evidence if type(e.get('CSV row')) is int}
    for item in evidence:
        group = item.get('CSV rows')
        if isinstance(group, str):
            allowed_rows.update(map(int, re.findall(r'(?<!\d)\d+(?!\d)', group)))
    clean, seen = [], set()
    for label in labels if isinstance(labels, list) else []:
        if not isinstance(label, str):
            continue
        label = ' '.join(label.split())
        if not 3 <= len(label) <= 110 or label.count(';') > 1:
            continue
        key = label.casefold()
        if key in seen:
            continue
        check_key = str(finding.get('check_key') or '')
        # Prevent a model's carried-over clarification from turning into
        # apparently valid buttons beneath an unrelated finding/check.
        if check_key in {'inspect_taxonomy_structure', 'inspect_taxonomy_naming'}:
            if any(phrase in key for phrase in (
                    'absolute value', 'negative count', 'fractional count',
                    'numeric replacement', 'specific replacement value',
                    'set count', 'delete column',
                    'clear cell', 'replace measurement')):
                continue
        elif check_key == 'find_duplicate_taxa':
            if any(phrase in key for phrase in (
                    'absolute value', 'specific replacement value', 'rename header')):
                continue
        elif check_key == 'inspect_headers':
            if any(phrase in key for phrase in (
                    'absolute value', 'delete row', 'drop row', 'taxonomy padding',
                    'fill missing value')):
                continue
        elif check_key in {'find_invalid_counts', 'find_negative_counts'}:
            if any(phrase in key for phrase in (
                    'taxonomy rank', 'taxonomy prefix', 'rename header',
                    'normalize separator', 'pad taxonomy')):
                continue
        if check_key == 'find_duplicate_taxa':
            # Do not offer taxonomy-level padding in a duplicate-row review.
            if re.search(r'\b(?:pad|append|na|missing rank|standardize taxonomy)\b', key):
                continue
            mentioned = [int(x) for x in re.findall(r'\brows?\s*(\d+)\b', key)]
            if mentioned and not set(mentioned).issubset(allowed_rows):
                continue
        seen.add(key)
        clean.append(label)
    return clean
