"""Nonblocking, process-local read-only exploration jobs for Streamlit.

Never invoke Streamlit APIs in worker threads. Each verified round is published
atomically to the UI and independently checkpointed by the caller. The worker
shares the existing vLLM request gate and never modifies uploaded data.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Callable

from healthcare_data_quality.exploratory_combined import run_exploratory_series
from healthcare_data_quality.llm_client import model_request_priority


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveExplorationJobs:
    """One run per explicit investigation ID. No implicit retries on reruns."""

    def __init__(self, runner=None):
        self._runner = runner or run_exploratory_series
        self._lock = threading.RLock()
        self._states: dict[str, dict] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start(
        self, run_id: str, datasets: dict[str, Path], audit: dict,
        *, on_round_saved: Callable[[list[dict]], None] | None = None,
        on_finished: Callable[[list[dict], dict | None], None] | None = None,
    ) -> bool:
        """Start exactly once, returning False if the job already exists."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("An investigation run ID is required")
        if not datasets:
            raise ValueError("At least one dataset is required")
        with self._lock:
            if run_id in self._states:
                return False
            self._states[run_id] = {
                "run_id": run_id, "status": "running", "started_at": _utc_now(),
                "activity": "Preparing the first Gemma exploratory test",
                "runs": [], "stop": None, "error": None, "warning": None,
            }
            thread = threading.Thread(
                target=self._run,
                args=(run_id, dict(datasets), audit, on_round_saved, on_finished),
                name="hdq-exploration-" + run_id[:8], daemon=True,
            )
            self._threads[run_id] = thread
            thread.start()
            return True

    def snapshot(self, run_id: str) -> dict | None:
        with self._lock:
            state = self._states.get(run_id)
            return deepcopy(state) if state is not None else None

    def _update(self, run_id: str, **changes) -> None:
        with self._lock:
            self._states[run_id].update(changes)

    def _warn(self, run_id: str, text: str) -> None:
        with self._lock:
            self._states[run_id]["warning"] = text[:500]

    def _run(
        self, run_id: str, datasets: dict[str, Path], audit: dict,
        on_round_saved, on_finished,
    ) -> None:
        def progress(message: str) -> None:
            self._update(run_id, activity=str(message)[:500])

        def round_complete(runs: list[dict]) -> None:
            # Publish verified results before attempting optional persistence.
            self._update(run_id, runs=deepcopy(runs),
                         activity=f"{len(runs)} verified Gemma check(s) available")
            if on_round_saved is not None:
                try:
                    on_round_saved(deepcopy(runs))
                except (OSError, ValueError, RuntimeError, TypeError) as exc:
                    self._warn(run_id, f"Round checkpoint/recommendation failed: {type(exc).__name__}: {exc}")

        try:
            # Exploration and recommendation generation are background work;
            # the existing request gate favors a waiting interactive chat.
            with model_request_priority("investigation"):
                report = self._runner(datasets, audit, on_round=round_complete,
                                      on_progress=progress)
            if not isinstance(report, dict):
                raise ValueError("Exploration did not return a result record")
            runs = report.get("runs") or []
            stop = report.get("stop")
            self._update(run_id, runs=deepcopy(runs), stop=deepcopy(stop),
                         status="complete", activity="Gemma exploration ended")
            if on_finished is not None:
                try:
                    on_finished(deepcopy(runs), deepcopy(stop))
                except (OSError, ValueError, RuntimeError, TypeError) as exc:
                    self._warn(run_id, f"Final checkpoint failed: {type(exc).__name__}: {exc}")
        except Exception as exc:
            # Completed verified rounds stay visible; never claim completeness.
            self._update(run_id, status="incomplete",
                         error=f"{type(exc).__name__}: {str(exc)[:500]}",
                         activity="Exploration stopped before completion")
        finally:
            with self._lock:
                self._threads.pop(run_id, None)


_MANAGER: LiveExplorationJobs | None = None
_MANAGER_LOCK = threading.Lock()


def exploration_manager() -> LiveExplorationJobs:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = LiveExplorationJobs()
        return _MANAGER
