"""Nonblocking, process-local inference jobs for the two cleaning discussions.

Worker threads never touch Streamlit session state or CSV files. The UI stores
submitted messages immediately and accepts model responses on a later rerun.
This module holds private research prompts in memory only, not in Git or logs.
"""

from __future__ import annotations

from copy import deepcopy
import threading
from uuid import uuid4
from typing import Callable


class InteractiveChatJobs:
    """Bounded snapshots of independently running local Gemma conversations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}

    def submit(self, action: Callable, *args, **kwargs) -> str:
        job_id = uuid4().hex
        with self._lock:
            # Finished jobs abandoned by closed browser sessions are not
            # retained indefinitely. Active jobs are never thrown away.
            if len(self._jobs) >= 256:
                for key in list(self._jobs):
                    if self._jobs[key]["status"] != "running":
                        self._jobs.pop(key)
                    if len(self._jobs) < 128:
                        break
            self._jobs[job_id] = {"status": "running"}

        def run() -> None:
            try:
                result = action(*args, **kwargs)
                if not isinstance(result, (dict, str)):
                    raise ValueError("Gemma returned an unsupported response")
                status = {"status": "done", "result": result}
            except Exception as exc:
                status = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:350]}"}
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id] = status

        threading.Thread(target=run, name=f"hdq-chat-{job_id[:8]}", daemon=True).start()
        return job_id

    def take(self, job_id: str) -> dict:
        """Return a completed result once; leave ongoing jobs untouched."""
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return {"status": "lost"}
            if state["status"] == "running":
                return {"status": "running"}
            self._jobs.pop(job_id, None)
            return deepcopy(state)


_MANAGER = InteractiveChatJobs()


def interactive_chat_manager() -> InteractiveChatJobs:
    return _MANAGER
