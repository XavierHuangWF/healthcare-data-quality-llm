"""Human-controlled, no-new-edit completion for any source-backed finding/check.

This module interprets reviewer *intent*, not permission to mutate the CSV.
A close outcome is an audited decision distinct from evidence-based remediation.
"""
from __future__ import annotations

import re


_CLOSURE_COMMAND = re.compile(
    r"(?:"
    r"(?:please\s+)?(?:close|finish|complete|end|pass)\s+(?:this\s+|the\s+)?(?:review|finding|check|session)"
    r"|(?:please\s+)?(?:mark\s+)?(?:this\s+|the\s+)?(?:review|finding|check)\s+(?:as\s+)?(?:reviewed|complete|completed|closed|done)"
    r"|(?:please\s+)?(?:mark(?:ed)?\s+(?:it\s+)?(?:as\s+)?(?:reviewed\s+and\s+closed|closed|complete|completed))"
    r"|(?:reviewed\s+and\s+closed|marked\s+and\s+closed|closed|completed?|done|ignore\s+it)"
    r"|(?:no\s+(?:more|further|additional)\s+changes?\s+(?:are\s+)?(?:needed|required))"
    r")",
    re.IGNORECASE,
)


def requests_review_closure(message: str) -> bool:
    """Recognize an *entire*, unambiguous closure command only.

    Compound requests containing a new edit ("close and delete row 5")
    MUST go through ordinary scoped approval, not this zero-edit path.
    """
    if not isinstance(message, str) or not 1 <= len(message) <= 160:
        return False
    text = re.sub(r"\s+", " ", message.strip()).strip(".! ")
    return bool(_CLOSURE_COMMAND.fullmatch(text))


def closure_button_label(*, has_edits: bool, pending_proposal: bool,
                         previous_approval_proof: bool = False) -> str:
    """One final user choice, never a claim that a pending change executed."""
    if previous_approval_proof:
        return "Mark reviewed — resolved by earlier approvals"
    if has_edits:
        return ("Mark reviewed — keep approved edits, decline remaining proposal"
                if pending_proposal else "Mark reviewed — keep approved edits")
    return ("Mark reviewed — decline proposal, leave data unchanged"
            if pending_proposal else "Mark reviewed — leave unchanged")
