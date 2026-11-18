"""Persistent, finding-by-finding Gemma recommendation worker.

The worker NEVER calls Streamlit, executes cleaning, reads entire datasets,
changes user decisions, or runs generated Python. Each recommendation is saved
atomically to a private local directory and can survive Streamlit reruns.
Only one background model request is issued at a time.
"""

import hashlib
import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from cleaning_review import recommend_single_finding, review_question
from healthcare_data_quality.llm_client import model_request_priority

_JOB_SCHEMA = 1
_DEFAULT_CACHE_DIR = (Path.home() / ".local" / "share" /
                      "healthcare-data-quality-llm" / "cleaning-recommendations")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _minimal_finding(item: dict) -> dict:
    """Do not persist full source-cell evidence or files in the job queue."""
    return {
        "id": str(item["id"]),
        "dataset_id": str(item.get("dataset_id") or ""),
        "source": str(item.get("source") or "")[:60],
        "title": str(item.get("title") or "")[:500],
        "check_key": str(item.get("check_key") or "")[:100],
        "method": str(item.get("method") or "")[:900],
        "affected_record_count": (item.get("total_evidence_records")
                                  if isinstance(item.get("total_evidence_records"), int)
                                  and item["total_evidence_records"] >= 0
                                  else len(item.get("evidence") or [])),
        "actionable": bool(item.get("actionable", False)),
    }


def _prompt_finding(item: dict, runtime: dict | None = None) -> dict:
    """Read-only prompt context; source-cell samples never enter job JSON."""
    sample = list((runtime or {}).get("evidence", []))[:8]
    return {**item, "evidence": sample,
            "total_evidence_records": item.get("affected_record_count", 0),
            "evidence_is_a_sample": True}


def job_fingerprint(context_key: str, catalog: list[dict], versions: dict | None = None, *,
                    run_id: str | None = None) -> str:
    if not context_key:
        raise ValueError("An investigation key is required")
    if run_id is not None:
        # The first full investigation owns exactly one Agent B job, independent
        # of the number of findings discovered later and of cleaned CSV versions.
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("A valid investigation run ID is required")
        manifest = {"schema": _JOB_SCHEMA, "investigation": context_key,
                    "run_id": run_id, "model_revision": os.environ.get("HDQ_MODEL_REVISION", "")}
        return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    summaries = [_minimal_finding(item) for item in catalog]
    ids = [item["id"] for item in summaries]
    if len(ids) != len(set(ids)):
        raise ValueError("Finding identifiers must be unique")
    manifest = {"schema": _JOB_SCHEMA, "investigation": context_key,
                "model_revision": os.environ.get("HDQ_MODEL_REVISION", ""),
                "versions": versions or {}, "findings": summaries}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CleaningRecommendationJobs:
    def __init__(self, directory: Path | None = None, *, recommender=None):
        self.directory = Path(directory) if directory is not None else _DEFAULT_CACHE_DIR
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._runtime_context: dict[str, dict[str, dict]] = {}
        self._recommender = recommender or recommend_single_finding

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid job fingerprint")
        return self.directory / (key + ".json")

    def _load(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("key") != key or state.get("schema") != _JOB_SCHEMA:
            raise ValueError("Saved recommendation queue is incompatible")
        return state

    def _save(self, state: dict) -> None:
        """Atomic replacement; do not leak datasets into the source tree."""
        path = self._path(state["key"])
        partial = path.with_name(path.name + "." + str(threading.get_ident()) + ".tmp")
        try:
            with os.fdopen(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)

    def start(self, context_key: str, catalog: list[dict], *, versions: dict | None = None,
              retry_failed: bool = False, run_id: str | None = None) -> str:
        """Idempotently start/resume the queue; failed entries retry only on request."""
        key = job_fingerprint(context_key, catalog, versions, run_id=run_id)
        ids = [item["id"] for item in catalog]
        if len(ids) != len(set(ids)):
            raise ValueError("Finding identifiers must be unique")
        with self._lock:
            # Keep a bounded snapshot of original evidence only in this process.
            # It is never written into the persisted recommendation queue.
            context = self._runtime_context.setdefault(key, {})
            for item in catalog:
                context[str(item["id"])] = {
                    "evidence": list(item.get("evidence") or [])[:8],
                }
            state = self._load(key)
            if state is None:
                items = [_minimal_finding(item) for item in catalog]
                state = {"schema": _JOB_SCHEMA, "key": key, "created_at": _now(),
                         "findings": items,
                         "entries": {item["id"]: {"status": "pending", "updated_at": _now()}
                                     for item in items}}
            else:
                # During the SAME full investigation, verified Gemma explorations
                # may add new finding IDs. Append only; existing Agent B output is
                # immutable, including failed entries, and never regenerated.
                if run_id is not None:
                    known = set(state["entries"])
                    for finding in catalog:
                        if finding["id"] in known:
                            continue
                        summary = _minimal_finding(finding)
                        state["findings"].append(summary)
                        state["entries"][summary["id"]] = {
                            "status": "pending", "updated_at": _now()}
                        known.add(summary["id"])
                # For the once-only Agent B workflow, an interrupted request
                # cannot be safely replayed without possibly generating twice.
                # Older non-run-specific queues retain their resumable behavior.
                running = bool(self._workers.get(key) and self._workers[key].is_alive())
                if not running:
                    for entry in state["entries"].values():
                        if entry["status"] == "running":
                            if run_id is not None:
                                entry["status"] = "failed"
                                entry["error"] = "Interrupted; once-only Agent B request will not rerun"
                            else:
                                entry["status"] = "pending"
            if retry_failed and run_id is None:
                for entry in state["entries"].values():
                    if entry["status"] == "failed":
                        entry["status"] = "pending"
                        entry.pop("error", None)
            self._save(state)
            if any(entry["status"] == "pending" for entry in state["entries"].values()):
                if not (self._workers.get(key) and self._workers[key].is_alive()):
                    worker = threading.Thread(target=self._run, args=(key,),
                                              name="hdq-recommendations-" + key[:8], daemon=True)
                    self._workers[key] = worker
                    worker.start()
        return key

    def snapshot(self, key: str) -> dict | None:
        with self._lock:
            result = self._load(key)
            return deepcopy(result) if result else None

    def _run(self, key: str) -> None:
        try:
            self._run_queue(key)
        finally:
            # If an exploration appended a new item exactly as the last worker
            # exited, guarantee the pending item eventually runs.
            with self._lock:
                if self._workers.get(key) is threading.current_thread():
                    self._workers.pop(key, None)
                    state = self._load(key)
                    if not state or not any(e["status"] == "pending" for e in state["entries"].values()):
                        self._runtime_context.pop(key, None)
                    if (state and not state.get("cancel_requested") and
                            any(e["status"] == "pending" for e in state["entries"].values())):
                        worker = threading.Thread(target=self._run, args=(key,),
                                                  name="hdq-recommendations-" + key[:8], daemon=True)
                        self._workers[key] = worker
                        worker.start()

    def _run_queue(self, key: str) -> None:
        while True:
            with self._lock:
                state = self._load(key)
                if not state or state.get("cancel_requested"):
                    return
                next_item = next((x for x in state["findings"]
                                  if state["entries"][x["id"]]["status"] == "pending"), None)
                if next_item is None:
                    return
                fid = next_item["id"]
                state["entries"][fid] = {"status": "running", "updated_at": _now()}
                self._save(state)
            try:
                # The background prompt is short. An interactive request already
                # waiting gets priority before the NEXT background inference.
                with model_request_priority("recommendation"):
                    with self._lock:
                        runtime = self._runtime_context.get(key, {}).get(fid)
                    recommendation = self._recommender(_prompt_finding(next_item, runtime))
                if not isinstance(recommendation, dict) or not str(recommendation.get("suggestion", "")).strip():
                    raise ValueError("Empty Gemma recommendation")
                item = {"status": "ready", "updated_at": _now(),
                        "suggestion": str(recommendation["suggestion"])[:2200],
                        "decision_needed": str(recommendation.get("decision_needed", ""))[:1400]}
            except Exception as exc:  # Preserve remaining queue even if an individual LLM call fails.
                item = {"status": "failed", "updated_at": _now(),
                        "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            with self._lock:
                state = self._load(key)
                if not state:
                    return
                state["entries"][fid] = item
                self._save(state)
                self._runtime_context.get(key, {}).pop(fid, None)

    def cancel(self, key: str) -> None:
        """Stop after current request, never kill an active HTTP request."""
        with self._lock:
            state = self._load(key)
            if state:
                state["cancel_requested"] = True
                self._save(state)


_MANAGER: CleaningRecommendationJobs | None = None
_MANAGER_LOCK = threading.Lock()


def recommendation_manager() -> CleaningRecommendationJobs:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = CleaningRecommendationJobs()
        return _MANAGER
