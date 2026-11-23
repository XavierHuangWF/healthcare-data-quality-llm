"""Track read-only proposal preparation, not editing authorization.

A reviewer must see the exact verified Before→After table and click its
corresponding choice. Background requests do not grant execution permission.
"""
from __future__ import annotations

from hashlib import sha256


def proposal_request_once(state, finding_id: str, assistant_message: str,
                          source_bytes: bytes, *, user_turn: str = "") -> str | None:
    """Deduplicate the intermediate read-only planning call for one reply.

    Returns a receipt for a new, never-before-queued reply; None otherwise.
    A receipt is NOT reviewer approval and cannot commit any cleaning.
    """
    if not isinstance(finding_id, str) or not finding_id:
        raise ValueError("Finding identity is required")
    if not isinstance(assistant_message, str) or not assistant_message.strip():
        return None
    if not isinstance(source_bytes, bytes):
        raise TypeError("Original CSV must be bytes")
    # Bind to the human turn, NOT to an assistant reply. If a read-only
    # planning request returns another prose suggestion, it must not trigger
    # an infinite chain of model calls for the same reviewer instruction.
    marker = user_turn if user_turn else assistant_message
    digest = sha256(b"HDQ_READ_ONLY_PLAN_V1\0" + finding_id.encode('utf8') + b"\0" +
                    sha256(source_bytes).digest() + b"\0" + marker.encode('utf8')).hexdigest()
    requested = state.setdefault("cleaning_readonly_plan_requests_by_finding", {})
    if requested.get(finding_id) == digest:
        return None
    requested[finding_id] = digest
    return digest
