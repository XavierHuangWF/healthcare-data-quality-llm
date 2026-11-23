"""Match a reviewer-specified physical target set to the actual draft.

A message like `delete 105` on a duplicate-column finding means one physical
column, regardless of what Agent B previously proposed. No model output or
remembered plan may silently widen those coordinates.
"""
from __future__ import annotations

import re


def explicitly_requested_columns(message: str, finding: dict) -> set[int] | None:
    if not isinstance(message, str):
        return None
    s = ' '.join(message.strip().casefold().split()).rstrip('.!')
    m = re.fullmatch(
        r'(?:please\s+)?(?:delete|deleted|drop|remove)\s+(?:(?:the\s+)?(?:whole|entire)\s+)?'
        r'columns?\s+(\d+(?:\s*(?:,|and|&)\s*(?:columns?\s+)?\d+)*)', s)
    if not m and finding.get('check_key') == 'compare_repeated_headers':
        m = re.fullmatch(r'(?:please\s+)?(?:delete|deleted|drop|remove)\s+(\d+)', s)
    if not m:
        return None
    result = set(map(int, re.findall(r'\d+', m.group(1))))
    return result if 1 <= len(result) <= 16 else None


def reject_unrequested_extra_columns(request: str, finding: dict,
                                      operations: list[dict]) -> None:
    expected = explicitly_requested_columns(request, finding)
    if expected is None:
        return
    actual = {c.get('position') for op in operations if op.get('operation') == 'drop_column'
              for c in (op.get('changes') or [])}
    if actual != expected or any(op.get('operation') != 'drop_column' for op in operations):
        raise PermissionError(
            'The proposed operations do not match the exact column deletion scope '
            f'you requested ({sorted(expected)}); the model may have carried over '
            'a previously suggested column. No changes were made.')
