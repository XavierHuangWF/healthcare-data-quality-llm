"""Resolve unambiguous reviewer directives to delete full CSV columns.

No model inference, approval, or data mutation. Only literal column numbers
stated by the reviewer are accepted; uncertain messages defer to Agent C.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path


def explicit_column_deletion(question: str, finding: dict, source: Path | None) -> dict | None:
    if source is None or not isinstance(question, str):
        return None
    text = ' '.join(question.strip().casefold().split())
    if (re.search(r"\b(?:don't|do not|never|without|instead of|not)\s+(?:delete|drop|remove)\b", text)
            or 'clear cells' in text or 'only cells' in text):
        return None
    # Reviewer shorthand is only safe inside the duplicate-column finding,
    # where the evidence identifies the intended *physical* column numbers.
    # "delete 105" means JUST column 105; never infer column 106 because it
    # has identical values or because another suggestion mentioned it.
    match = re.fullmatch(
        r'(?:please\s+)?(?:delete|deleted|drop|remove)\s+'
        r'(?:the\s+)?(?:(?:entire|whole)\s+)?columns?\s+'
        r'(\d+(?:\s*(?:,|and|&)\s*(?:columns?\s+)?\d+)*)'
        r'(?:\s*(?:entirely|completely|from\s+(?:the\s+)?(?:csv|dataset)))?[.!]?', text,
    )
    short_match = None
    if not match and finding.get('check_key') == 'compare_repeated_headers':
        short_match = re.fullmatch(r'(?:please\s+)?(?:delete|deleted|drop|remove)\s+(\d+)[.!]?', text)
    if not match and not short_match:
        return None
    values = [int(v) for v in re.findall(r'\d+', (match or short_match).group(1))]
    if short_match:
        evidenced = {pos for item in (finding.get('evidence') or []) if isinstance(item, dict)
                     for key in ('Column A', 'Column B')
                     if type(pos := item.get(key)) is int}
        if not set(values).issubset(evidenced):
            return None
    if len(set(values)) != len(values) or not 1 <= len(values) <= 16:
        return None
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        all_rows = list(csv.reader(stream, strict=True))
    if not all_rows or len(all_rows[0]) < 2 or any(len(row) != len(all_rows[0]) for row in all_rows):
        raise ValueError('Original CSV is not rectangular')
    if any(pos <= 1 or pos > len(all_rows[0]) for pos in values):
        raise ValueError('Cannot delete the identifier column or a nonexistent column')
    operations = [
        {'operation': 'drop_column', 'dataset_id': finding['dataset_id'],
         'changes': [{'position': pos, 'before': all_rows[0][pos-1], 'after': ''}]}
        for pos in values
    ]
    return {'reply': 'Delete entire CSV columns ' + ', '.join(map(str, values)) + '. Review Before → After.',
            'operation': 'none', 'dataset_id': '', 'changes': [],
            'operations': operations, 'clarification_options': [],
            'explicit_columns': frozenset(values)}
