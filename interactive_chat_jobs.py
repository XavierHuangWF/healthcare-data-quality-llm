"""Finding-scoped interactive Gemma jobs with opportunistic idle-agent dispatch.

A/B are borrowed only when their background workflows are terminal. C has one
active conversation lane; requests waiting for C can move to a newly idle A/B
before inference begins. In-flight requests are never transferred. All workers
share the same local model and their user actions go through existing validation.
"""
from __future__ import annotations

from copy import deepcopy
import threading
from uuid import uuid4
from typing import Callable

from healthcare_data_quality.llm_client import model_request_priority

_ROLE_PRIORITY = {"A": "investigation", "B": "recommendation", "C": "interactive"}


class InteractiveChatJobs:
    """One generating finding request per logical role; bounded queued C jobs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._reserved = {"A": 0, "B": 0, "C": 0}
        self._lanes = {role: threading.Lock() for role in _ROLE_PRIORITY}

    def active_count(self, role: str = "C") -> int:
        if role not in _ROLE_PRIORITY:
            raise ValueError("Unknown agent role")
        with self._lock:
            return self._reserved[role]

    def agent_for(self, job_id: str) -> str | None:
        """Report the actual owner, including reassignment before generation."""
        with self._lock:
            state = self._jobs.get(job_id)
            return state.get("agent") if state else None

    def submit(self, action: Callable, *args, **kwargs) -> str:
        """Legacy API retained for direct callers/tests."""
        job_id, _ = self._enqueue(action, args, kwargs, allow_helpers=False)
        return job_id

    def submit_routed(
        self, action: Callable, *args, agent_a_idle: bool = False,
        agent_b_idle: bool = False,
        idle_probe: Callable[[], tuple[bool, bool]] | None = None,
        **kwargs,
    ) -> tuple[str, str]:
        """Route NOW, or borrow A/B LATER if they become free while C is busy.

        ``idle_probe`` is a background-safe callable capturing immutable run
        IDs. It must never access Streamlit state. Only a queued (not started)
        C job can change roles. The existing model admission gate still applies.
        """
        return self._enqueue(
            action, args, kwargs, allow_helpers=True,
            agent_a_idle=agent_a_idle, agent_b_idle=agent_b_idle,
            idle_probe=idle_probe,
        )

    def _enqueue(
        self, action: Callable, args: tuple, kwargs: dict, *,
        allow_helpers: bool, agent_a_idle: bool = False,
        agent_b_idle: bool = False,
        idle_probe: Callable[[], tuple[bool, bool]] | None = None,
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
            if len(self._jobs) >= 256:
                for key in list(self._jobs):
                    if self._jobs[key]["status"] != "running":
                        self._jobs.pop(key)
                    if len(self._jobs) < 128:
                        break
            self._jobs[job_id] = {"status": "running", "agent": role}

        def run() -> None:
            effective_role = role
            held_lane: threading.Lock | None = None
            try:
                if allow_helpers:
                    if role == "C":
                        # Only one C request generates at a time. While queued,
                        # recheck the background agents; a newly idle A/B can
                        # take this job without restarting or duplicating it.
                        while held_lane is None:
                            if self._lanes["C"].acquire(timeout=0.15):
                                held_lane = self._lanes["C"]
                                break
                            if idle_probe is None:
                                continue
                            try:
                                a_idle, b_idle = idle_probe()
                            except Exception:
                                # Failure must never borrow a potentially busy
                                # background agent.
                                continue
                            with self._lock:
                                for helper, is_idle in (("A", a_idle), ("B", b_idle)):
                                    if not is_idle or self._reserved[helper]:
                                        continue
                                    if not self._lanes[helper].acquire(blocking=False):
                                        continue
                                    self._reserved["C"] -= 1
                                    self._reserved[helper] += 1
                                    effective_role = helper
                                    self._jobs[job_id]["agent"] = helper
                                    held_lane = self._lanes[helper]
                                    break
                    else:
                        self._lanes[role].acquire()
                        held_lane = self._lanes[role]
                with model_request_priority(_ROLE_PRIORITY[effective_role]):
                    result = action(*args, **kwargs)
                if not isinstance(result, (dict, str)):
                    raise ValueError("Gemma returned an unsupported response")
                outcome = {"status": "done", "result": result, "agent": effective_role}
            except Exception as exc:
                outcome = {"status": "failed", "error":
                           f"{type(exc).__name__}: {str(exc)[:350]}",
                           "agent": effective_role}
            finally:
                if held_lane is not None:
                    held_lane.release()
                with self._lock:
                    self._reserved[effective_role] -= 1
                    if job_id in self._jobs:
                        self._jobs[job_id] = outcome

        threading.Thread(
            target=run, name=f"hdq-chat-{role.lower()}-{job_id[:8]}", daemon=True,
        ).start()
        return job_id, role

    def take(self, job_id: str) -> dict:
        """Consume a completed result once; never steal an active request."""
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return {"status": "lost"}
            if state["status"] == "running":
                return {"status": "running"}
            self._jobs.pop(job_id, None)
            # Preserve the pre-existing consumer API.
            return {k: deepcopy(v) for k, v in state.items() if k != "agent"}


_MANAGER = InteractiveChatJobs()


def interactive_chat_manager() -> InteractiveChatJobs:
    return _MANAGER
