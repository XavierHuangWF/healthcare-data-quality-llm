"""Separate human-visible conversation from internal review diagnostics.

All records remain in the session/audit. Only the presentation is compacted.
"""
from __future__ import annotations

def partition_review_thread(thread: list[dict]) -> tuple[list[dict], list[dict]]:
    visible, diagnostics = [], []
    seen_in_turn = set()
    for index, item in enumerate(thread):
        if not isinstance(item, dict) or item.get('role') not in {'assistant','user'}:
            continue
        text = str(item.get('content') or '').strip()
        if not text:
            continue
        if item.get('automatic') or item.get('background_diagnostic'):
            diagnostics.append(item)
            continue
        if item.get('role') == 'user':
            seen_in_turn.clear()
            visible.append(item)
            continue
        normalized = ' '.join(text.casefold().split())
        if normalized in seen_in_turn:
            diagnostics.append(item)
            continue
        seen_in_turn.add(normalized)
        # A failed current proposal is still a useful user-facing answer once.
        # Historic failure details are retained but not repeated on screen.
        if item.get('proposal_preparation_failed') and any(m.get('role') == 'user' and not m.get('automatic') for m in thread[index+1:]):
            diagnostics.append(item)
        else:
            visible.append(item)
    return visible, diagnostics
