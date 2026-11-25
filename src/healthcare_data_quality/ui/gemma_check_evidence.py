"""Readable, source-linked evidence summary for completed zero-finding Gemma tests.

Never transform a zero-candidate result into a claim that all the underlying
cells were checked, or that the dataset is scientifically clean.
"""
from __future__ import annotations


def check_evidence_rows(finding: dict) -> list[dict[str, str]]:
    """Present what Gemma actually proposed, executed and concluded."""
    details = finding.get('gemma_check_details') or {}
    def display(value, unknown='Not recorded in this investigation'):
        value = str(value or '').strip()
        return value if value else unknown

    proposal = details.get('proposal') or {}
    result = details.get('result') or {}
    verification = details.get('verification') or {}
    status = str(verification.get('status') or '').strip()
    if status == 'source_cells_verified_and_output_reproduced':
        verified = 'Read-only sandbox output reproduced and source cells verified'
    elif status:
        verified = f'Verification state: {status}'
    else:
        verified = 'Verification details not recorded; do not infer full coverage'
    return [
        {'Check': 'Hypothesis / target', 'Recorded evidence': display(proposal.get('hypothesis'))},
        {'Check': 'How it was tested', 'Recorded evidence': display(proposal.get('method'))},
        {'Check': 'Observed result', 'Recorded evidence': display(result.get('summary'),
                                                               'No additional finding reported')},
        {'Check': 'New candidate findings', 'Recorded evidence': str(len(result.get('findings') or []))},
        {'Check': 'Execution verification', 'Recorded evidence': verified},
        {'Check': 'Reviewer implication', 'Recorded evidence':
            'A zero-finding check does not justify deleting or changing columns. Review and close unchanged, or discuss a new explicit action.'},
    ]


def observed_check_data(finding: dict, source, *, staged: bytes | None = None, removed_rows=(), removed_columns=()) -> list[dict]:
    """Independently count values in exact named control columns; no inferred anomalies."""
    import csv
    import re
    from decimal import Decimal, InvalidOperation
    from pathlib import Path
    if source is None:
        return []
    details = finding.get('gemma_check_details') or {}
    proposal = details.get('proposal') or {}
    text = str(proposal.get('hypothesis') or '') + ' ' + str(proposal.get('method') or '')
    if not re.search(r'\b(?:blank|negative.control)\b', text, re.I) or not re.search(r'\b(?:zero|non.zero)\b', text, re.I):
        return []
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream, strict=True))
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        return []
    working = rows
    if staged is not None:
        from healthcare_data_quality.cleaning.in_memory_cleaning import _records
        working = _records(staged)
        if len(working) != len(rows) or any(len(a) != len(b) for a,b in zip(working,rows)):
            raise ValueError('Current coordinates cannot be verified')
    out = []
    for i, name in enumerate(rows[0]):
        if not name or not re.search(r'(?<![\w])'+re.escape(name)+r'(?![\w])', text, re.I) or 'blank' not in name.casefold():
            continue
        zero = nonzero = missing = invalid = 0
        for r, row in enumerate(working[1:], 2):
            if r in removed_rows:
                continue
            value = row[i].strip()
            if not value or value.upper() == 'NA':
                missing += 1; continue
            try:
                number = Decimal(value)
                if not number.is_finite():
                    raise InvalidOperation
            except InvalidOperation:
                invalid += 1; continue
            if number == 0: zero += 1
            else: nonzero += 1
        item = {'Column': i+1, 'Name': name, 'Rows': zero+nonzero+missing+invalid, 'Zero': zero, 'Nonzero': nonzero, 'Missing': missing, 'Other invalid': invalid}
        if i+1 in removed_columns:
            item['Status'] = 'Removal approved'
        out.append(item)
    return out
