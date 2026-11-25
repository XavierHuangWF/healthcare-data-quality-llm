"""Process-local, read-only Gemma connectivity status for background/UI workers.

Only sanitized connection metadata is retained; prompts, source cells and model
responses never enter this status. Every failed request owns its own offline
marker so one recovered worker cannot clear another worker's pending outage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock, get_ident
from time import monotonic

_LOCK = Lock()
_WAITING: dict[int, dict] = {}


def note_connection_failure(kind: str) -> None:
    """Mark the current request as waiting for local Gemma to reconnect."""
    ident = get_ident()
    with _LOCK:
        previous = _WAITING.get(ident)
        now = datetime.now(timezone.utc).isoformat()
        _WAITING[ident] = {
            "offline_since": previous["offline_since"] if previous else now,
            "last_attempt": now,
            "attempts": (previous["attempts"] + 1) if previous else 1,
            "error_kind": kind[:80],
        }


def note_connection_success() -> None:
    """A request completed; do not clear any *other* pending request's marker."""
    with _LOCK:
        _WAITING.pop(get_ident(), None)


def model_connection_snapshot() -> dict:
    """UI-safe state; no ping/network I/O on Streamlit's render thread."""
    with _LOCK:
        values = list(_WAITING.values())
        return {
            "offline": bool(values),
            "waiting_requests": len(values),
            "offline_since": min((v["offline_since"] for v in values), default=None),
            "last_attempt": max((v["last_attempt"] for v in values), default=None),
            "attempts": max((v["attempts"] for v in values), default=0),
            "error_kind": values[0]["error_kind"] if values else None,
        }


def forget_finished_request() -> None:
    """Clean up a thread-local outage marker on non-retryable failure too."""
    note_connection_success()
