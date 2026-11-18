"""Bounded process-local interactive Gemma requests with optional idle-role borrowing.

The researcher's finding remains scoped to the same validator regardless of
which idle logical agent handles the inference. All workers share one local
model server. Capacity is controlled by HDQ_LLM_MAX_ACTIVE (default 1), so
role borrowing NEVER implies extra model instances or guaranteed parallelism.
Worker threads must not access Streamlit state or modify any CSV.
"""
from __future__ import annotations

from copy import deepcopy
import threading
from uuid import uuid4
from typing import Callable

from healthcare_data_quality.llm_client import model_request_priority

_ROLE_PRIORITY = {"A": "investigation", "B": "recommendation", "C": "interactive"}


class InteractiveChatJobs:
    """One active finding conversation per logical role, queued C fallback."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        # Reservations are made BEFORE a worker thread starts, preventing two
        # simultaneous clicks from claiming the same supposedly idle role.
        self._reserved = {"A": 0, "B": 0, "C": 0}
        self._lanes = {role: threading.Lock() for role in _ROLE_PRIORITY}

    def active_count(self, role: str = "C") -> int:
        if role not in _ROLE_PRIORITY:
            raise ValueError("Unknown agent role")
        with self._lock:
            return self._reserved[role]

    def submit(self, action: Callable, *args, **kwargs) -> str:
        """Existing Agent C API. Other call sites remain compatible."""
        job_id, _ = self._enqueue(action, args, kwargs, allow_helpers=False)
        return job_id

    def submit_routed(
        self, action: Callable, *args, agent_a_idle: bool = False,
        agent_b_idle: bool = False, **kwargs,
    ) -> tuple[str, str]:
        """Borrow A/B ONLY when C is busy and that role finished its own job.

        If neither helper is free the task enters C's existing serialized lane.
        This method does not bypass the host's model admission gate or grants.
        """
        return self._enqueue(
            action, args, kwargs, allow_helpers=True,
            agent_a_idle=agent_a_idle, agent_b_idle=agent_b_idle,
        )

    def _enqueue(
        self, action: Callable, args: tuple, kwargs: dict, *,
        allow_helpers: bool, agent_a_idle: bool = False,
        agent_b_idle: bool = False,
    ) -> tuple[str, str]:
        if not callable(action):
            raise TypeError("Interactive action must be callable")
        with self._lock:
            if sum(self._reserved.values()) >= 24:
                raise RuntimeError("Too many pending interactive requests")
            role = "C"
            if allow_helpers and self._reserved["C"]:
                if agent_a_idle and not self._reserved["A"]:
                    role = "A"
                elif agent_b_idle and not self._reserved["B"]:
                    role = "B"
            self._reserved[role] += 1
            job_id = uuid4().hex
            # Prune consumed/abandoned finished jobs, never discard active work.
            if len(self._jobs) >= 256:
                for key in list(self._jobs):
                    if self._jobs[key]["status"] != "running":
                        self._jobs.pop(key)
                    if len(self._jobs) < 128:
                        break
            self._jobs[job_id] = {"status": "running", "agent": role}

        def run() -> None:
            try:
                # Only one user-level conversation per A/B role. C requests
                # queue here when helpers are unavailable; no duplicate calls.
                # Legacy submit() remains independent for compatibility;
                # routed helper conversations serialize on their borrowed lane.
                # Actual model requests always honor the shared LLM gate.
                if allow_helpers:
                    with self._lanes[role]:
                        with model_request_priority(_ROLE_PRIORITY[role]):
                            result = action(*args, **kwargs)
                else:
                    with model_request_priority(_ROLE_PRIORITY[role]):
                        result = action(*args, **kwargs)
                if not isinstance(result, (dict, str)):
                    raise ValueError("Gemma returned an unsupported response")
                outcome = {"status": "done", "result": result, "agent": role}
            except Exception as exc:
                outcome = {"status": "failed", "error":
                           f"{type(exc).__name__}: {str(exc)[:350]}", "agent": role}
            finally:
                with self._lock:
                    self._reserved[role] -= 1
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id] = outcome

        threading.Thread(
            target=run, name=f"hdq-chat-{role.lower()}-{job_id[:8]}", daemon=True,
        ).start()
        return job_id, role

    def take(self, job_id: str) -> dict:
        """Return a completed result exactly once, without stealing active jobs."""
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return {"status": "lost"}
            if state["status"] == "running":
                return {"status": "running"}
            self._jobs.pop(job_id, None)
            # Legacy callers expect the same exact shape as earlier versions.
            return {k: deepcopy(v) for k, v in state.items() if k != "agent"}


_MANAGER = InteractiveChatJobs()


def interactive_chat_manager() -> InteractiveChatJobs:
    return _MANAGER
