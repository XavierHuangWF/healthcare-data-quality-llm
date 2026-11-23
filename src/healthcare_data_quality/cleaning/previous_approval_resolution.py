"""Prove a finding's *complete* source evidence was already removed by named approvals.

This is a review-only decision: never generate a replacement or repeat deletions.
A missing/partial evidence set cannot establish completion and fails closed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from healthcare_data_quality.cleaning.in_memory_cleaning import _records


def source_verified_prior_row_resolution(
    finding: dict, source: Path, approved_rows: list[dict],
) -> dict | None:
    """Return a source-bound closure proof iff ALL evidenced targets were deleted.

    Original physical CSV row coordinates are used throughout. Accept neither
    unverified model prose nor approval records from another uploaded source.
    The result never authorizes an edit or assumes that a sample proves a full
    finding. A count in the title has to match the underlying evidence.
    """
    records = _records(Path(source).read_bytes())
    fingerprint = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    evidence = finding.get('evidence') or []
    if not evidence or not isinstance(evidence, list):
        return None
    targets: set[int] = set()
    for item in evidence:
        if not isinstance(item, dict):
            return None
        row = item.get('CSV row', item.get('csv_row'))
        group = item.get('CSV rows', item.get('csv_rows'))
        if type(row) is int:
            row_ids = {row}
        elif isinstance(group, str) and re.fullmatch(r'\s*\d+(?:\s*,\s*\d+)*\s*', group):
            row_ids = {int(v.strip()) for v in group.split(',')}
        else:
            return None
        if not row_ids or any(not 2 <= n <= len(records) for n in row_ids):
            return None
        targets.update(row_ids)
    # A finding may show only representative examples. Require complete
    # evidence, not a coincidence where those examples happened to be deleted.
    title = str(finding.get('title') or '')
    prefix = re.match(r'^\s*(\d+)\s+(rows?|cells?|taxon labels?)\b', title, re.I)
    if prefix:
        count, unit = int(prefix.group(1)), prefix.group(2).lower()
        actual = len(evidence) if unit.startswith(('cell', 'taxon')) else len(targets)
        if actual != count:
            return None
    elif not finding.get('evidence_registry'):
        return None
    removed: dict[int, dict] = {}
    for entry in approved_rows or []:
        if (not isinstance(entry, dict)
                or entry.get('source_sha256') != fingerprint
                or not str(entry.get('reviewer') or '').strip()
                or entry.get('operation') not in {'drop_rows', 'merge_rows'}):
            continue
        for change in entry.get('changes') or []:
            if not isinstance(change, dict):
                continue
            row = (change.get('row') if entry['operation'] == 'drop_rows'
                   else change.get('source_row'))
            if type(row) is not int or not 2 <= row <= len(records):
                continue
            if 'before' in change and str(change['before']) != records[row-1][0]:
                continue
            removed[row] = {'finding_id': entry.get('finding_id'),
                            'reviewer': entry['reviewer']}
    if not targets.issubset(removed):
        return None
    rows = sorted(targets)
    evidence_ids = sorted({str(removed[r]['finding_id']) for r in rows})
    digest = hashlib.sha256(json.dumps({
        'source_sha256': fingerprint, 'finding_id': finding.get('id'),
        'resolved_rows': rows, 'approved_by': evidence_ids,
    }, sort_keys=True).encode()).hexdigest()
    return {'finding_id': finding.get('id'), 'dataset_id': finding.get('dataset_id'),
            'source_sha256': fingerprint, 'rows': rows,
            'earlier_finding_ids': evidence_ids, 'fingerprint': digest}


def entirely_superseded_operations(plans: list[dict], resolution: dict) -> bool:
    """True only for duplicate *row-scoped* work entirely inside removed rows.

    A prior deletion can retire the same row edit, never a new unrelated
    column deletion, header rename, merge into a survivor, or a visual flag.
    """
    deleted = set(resolution.get('rows') or [])
    if not deleted or not plans:
        return False
    inspected = 0
    for plan in plans:
        if not isinstance(plan, dict) or plan.get('operation') not in {'drop_rows', 'replace_cells'}:
            return False
        changes = plan.get('changes') or []
        if not changes:
            return False
        for change in changes:
            row = change.get('row') if isinstance(change, dict) else None
            if type(row) is not int or row not in deleted:
                return False
            inspected += 1
    return inspected > 0


def requests_close_resolved_finding(text: str) -> bool:
    """Recognize only an entire, unmistakable reviewer close command.

    Do not interpret 'ignore it and delete another row' as a final no-edit
    review, or use model prose as the reviewer's instruction.
    """
    if not isinstance(text, str) or len(text) > 140:
        return False
    normalized = re.sub(r'\s+', ' ', text.strip()).strip(' .!').casefold()
    return bool(re.fullmatch(
        r'(?:ignore (?:it|this(?: finding)?)|'
        r'(?:it is |it\'s )?completed|'
        r'(?:please )?close (?:this |the )?finding|'
        r'(?:please )?mark (?:this |the )?finding (?:as )?(?:reviewed|complete|completed)|'
        r'(?:mark (?:it|this) (?:as )?(?:reviewed|complete|completed))|'
        r'no (?:further|more|additional) changes? (?:are )?(?:needed|required))',
        normalized,
    ))
