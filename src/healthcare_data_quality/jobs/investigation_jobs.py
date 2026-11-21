"""Process-local nonblocking registered-tool investigation coordinator.

Only immutable snapshots cross into Streamlit. Workers never call Streamlit
or change the uploaded CSV. The caller starts recommendation and exploration
jobs from the callbacks using ordinary Python code.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Callable


def _now():
    return datetime.now(timezone.utc).isoformat()


class InvestigationJobs:
    def __init__(self):
        self._lock = Lock()
        self._states = {}

    def start(self, run_id: str, paths: list[Path], *, on_baseline: Callable,
              on_finished: Callable, runner: Callable) -> bool:
        if not run_id or not paths:
            raise ValueError('A run ID and dataset paths are required')
        with self._lock:
            if run_id in self._states:
                return False
            self._states[run_id] = {
                'run_id': run_id, 'status': 'running', 'phase': 1,
                'activity': 'Running deterministic data-quality checks',
                'started_at': _now(), 'partial_audit': None,
                'audit': None, 'error': None,
            }
        thread = Thread(target=self._run, daemon=True,
                        name='hdq-investigator-'+run_id[:8],
                        args=(run_id, list(paths), on_baseline, on_finished, runner))
        thread.start()
        return True

    def _update(self, run_id, **values):
        with self._lock:
            self._states[run_id].update(values)

    def snapshot(self, run_id: str):
        with self._lock:
            state = self._states.get(run_id)
            return deepcopy(state) if state is not None else None

    def _run(self, run_id, paths, on_baseline, on_finished, runner):
        def progress(message):
            message = str(message)[:400]
            phase = 2 if ('baseline checks completed' in message.lower()
                          or 'step ' in message.lower()
                          or 'investigation objectives:' in message.lower()) else None
            self._update(run_id, activity=message, **({'phase': phase} if phase else {}))

        def publish_baseline(partial):
            partial = deepcopy(partial)
            partial['cleaning_recommendation_run_id'] = run_id
            self._update(run_id, partial_audit=partial, phase=2,
                         activity='QC findings ready; Gemma registered-tool investigation running')
            # The queue can begin immediately, before registered tools end.
            # A recommendation-service failure must not invalidate verified QC.
            try:
                on_baseline(deepcopy(partial))
            except Exception as exc:
                self._update(run_id, callback_warning=(
                    f"Background recommendation could not start: {type(exc).__name__}: {str(exc)[:300]}"))

        try:
            audit = runner(paths[0], additional_datasets=paths[1:],
                           progress_callback=progress,
                           baseline_ready_callback=publish_baseline)
            if not isinstance(audit, dict):
                raise ValueError('Investigation returned no audit')
            audit['cleaning_recommendation_run_id'] = run_id
            # Publish the complete audit independently of cache and sandbox.
            self._update(run_id, audit=deepcopy(audit), status='completed',
                         phase=3, activity='Registered-tool investigation completed')
            try:
                on_finished(deepcopy(audit))
            except Exception as exc:
                self._update(run_id, callback_warning=(
                    f"Post-investigation continuation failed: {type(exc).__name__}: {str(exc)[:300]}"))
        except Exception as exc:
            # Preserve the independently verified baseline when later steps fail.
            self._update(run_id, status='failed',
                         error=f'{type(exc).__name__}: {str(exc)[:500]}',
                         activity='Investigation interrupted; completed QC remains available')


_MANAGER = InvestigationJobs()


def investigation_manager():
    return _MANAGER
