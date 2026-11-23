"""Explicit reviewer decisions that must not be redirected to an LLM.

These are *review* decisions, never data edits. A named reviewer can choose
not to change a finding. A no-change review cannot cancel an approved edit;
completed approvals must be revised through the separate guarded workflow.
"""
from __future__ import annotations

import re


_NO_CHANGE = re.compile(
    r"(?:"
    r"(?:please\s+)?(?:keep|leave|preserve|retain)\s+(?:it\s+)?(?:as\s+is|as\s+it\s+is|unchanged|the\s+same)"
    r"|(?:please\s+)?(?:keep|leave)\s+(?:the\s+)?(?:first\s+)?(?:header|column\s+1(?:\s+header)?)\s+blank"
    r"|(?:please\s+)?(?:leave|keep)\s+(?:it\s+)?blank"
    r"|(?:no|without|don't\s+make\s+any|do\s+not\s+make\s+any)\s+changes?"
    r"|(?:no\s+change\s+(?:needed|required|necessary))"
    r"|(?:do\s+not|don't)\s+(?:change|modify|rename)\s+(?:it|this|the\s+header)"
    r")",
    re.IGNORECASE,
)


def requests_no_change(message: str) -> bool:
    """Accept an unambiguous *entire message*, not a compound instruction.

    In particular, 'keep row 2 as is and delete row 3' must NOT erase the
    delete request. A no-change request is an explicit terminal review only.
    """
    if not isinstance(message, str) or len(message) > 240:
        return False
    normalized = re.sub(r"\s+", " ", message.strip()).strip(".! ")
    return bool(_NO_CHANGE.fullmatch(normalized))
