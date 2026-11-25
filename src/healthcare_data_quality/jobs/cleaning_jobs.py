"""Persistent, finding-by-finding Gemma recommendation worker.

The worker NEVER calls Streamlit, executes cleaning, reads entire datasets,
changes user decisions, or runs generated Python. Each recommendation is saved
atomically to a private local directory and can survive Streamlit reruns.
One primary recommendation worker runs continuously. When Agent C is idle,
an optional second worker borrows its request lane for independent findings.
All claims are atomic; a finding is never processed by both workers.
"""

import csv
import hashlib
import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from healthcare_data_quality.ui.cleaning_review import recommend_single_finding
from healthcare_data_quality.services.llm_client import (
    model_request_priority, model_reconnect_mode, _max_active_requests, interactive_model_request_pending,
)

_JOB_SCHEMA = 5
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
        "evidence_registry": (
            {k: v for k, v in (item.get("evidence_registry") or {}).items()
             if k not in {"finding_affected_row_ids", "finding_empty_count_row_ids",
                          "source_empty_count_row_ids"}}
            if isinstance(item.get("evidence_registry"), dict) else None),
    }


def preferred_reviewed_solution(finding: dict, source: Path,
                                path: Path | str | None = None) -> dict | None:
    """Choose a reusable *approved edit*, not the latest no-change marker.

    UI-only/recommendation-only logic intentionally lives outside src/, so
    upgrading it does not invalidate the saved investigation's source-code
    fingerprint. Every returned edit still requires fresh host validation and
    human approval; same-source changed evidence requires Gemma reassessment.
    """
    from collections import deque
    from healthcare_data_quality.learning.interaction_memory import (
        _ALLOWED_OPERATIONS, _VERSION, _LOCK, _identity, _location,
        review_memory_identity,
    )
    destination = _location(path)
    if not destination.is_file():
        return None
    wanted = _identity(finding, source)
    recent = deque(maxlen=2000)
    with _LOCK:
        with destination.open("r", encoding="utf-8") as stream:
            for raw in stream:
                if len(raw) <= 1_000_000:
                    recent.append(raw)
    candidates = {}
    revoked = set()
    for raw in reversed(recent):
        try:
            record = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict) or record.get("schema_version") != _VERSION:
            continue
        identity = review_memory_identity(record)
        if record.get("decision") == "revision_reset":
            revoked.add(identity)
            continue
        if identity in revoked or record.get("finding_type") != wanted["finding_type"]:
            continue
        decision = record.get("decision")
        if decision not in {"approved_edit", "reviewed_no_change"}:
            continue
        same_source = record.get("source_sha256") == wanted["source_sha256"]
        exact = same_source and record.get("evidence_sha256") == wanted["evidence_sha256"]
        if decision == "approved_edit":
            operations = record.get("operations")
            if (not isinstance(operations, list) or not 1 <= len(operations) <= 16
                    or any(not isinstance(op, dict)
                           or op.get("operation") not in _ALLOWED_OPERATIONS
                           or not isinstance(op.get("changes"), list)
                           or not op["changes"] for op in operations)):
                continue
            tier = ("exact_edit" if exact else
                    "same_source_edit" if same_source else "similar_edit")
        else:
            # No-change is an outcome of a past review, not a transferable
            # cleaning rule, and it must not override an approved edit.
            if not exact:
                continue
            tier = "exact_no_change"
        candidates.setdefault(tier, record)
    for tier in ("exact_edit", "same_source_edit", "exact_no_change", "similar_edit"):
        record = candidates.get(tier)
        if record is not None:
            return {"match": "exact" if tier in {"exact_edit", "exact_no_change"}
                    else "similar", "record": record}
    return None


def _memory_issue_family(finding_type: str) -> str:
    """Coarse *retrieval* category, never a permission to reuse a rule.

    Categories allow Gemma to compare lessons across individual findings and
    Gemma checks while preserving the exact before/after source validator.
    """
    text = str(finding_type).casefold().replace("_", " ")
    if any(word in text for word in (
            "duplicate", "multiple rows", "repeated taxon", "same taxa")):
        return "duplicate"
    if any(word in text for word in (
            "taxonomy", "taxon", "classification", "delimiter", "separator", "rank", "level")):
        return "taxonomy"
    if any(word in text for word in (
            "missing values", "negative", "fractional", "non numeric", "non-numeric",
            "count values", "invalid counts", "count cells", "non-zero counts")):
        return "counts"
    if any(word in text for word in (
            "header", "blank control", "identical columns", "duplicate column")):
        return "columns"
    return "other"


def ranked_reviewed_solutions(finding: dict, source: Path,
                              path: Path | str | None = None, *, limit: int = 3) -> list[dict]:
    """Retrieve approved methods for *this issue and related findings*.

    A related method is context for Gemma's fresh appraisal, not executable
    authorization. Same exact issue/source ranks first. If available, reserve
    one slot for a related finding to avoid repetitive memory tunnel vision.
    """
    from collections import deque
    from healthcare_data_quality.learning.interaction_memory import (
        _ALLOWED_OPERATIONS, _VERSION, _LOCK, _identity, _location,
        review_memory_identity,
    )
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError("Select between 1 and 16 memory candidates")
    destination = _location(path)
    if not destination.is_file():
        return []
    wanted = _identity(finding, source)
    family = _memory_issue_family(wanted["finding_type"])
    recent = deque(maxlen=2000)
    with _LOCK:
        with destination.open("r", encoding="utf-8") as stream:
            for line in stream:
                if len(line) <= 1_000_000:
                    recent.append(line)
    from healthcare_data_quality.ui.human_choice_replay import action_key, _history
    from collections import Counter
    frequencies = Counter()
    for r in _history(path):
        if r.get("decision") != "approved_edit":
            continue
        try:
            frequencies[(r.get("finding_type"), r.get("source_sha256"), action_key(r.get("operations") or []))] += 1
        except (TypeError, AttributeError, ValueError):
            continue
    ranked, seen_rules, revoked = [], set(), set()
    for line in reversed(recent):
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("schema_version") != _VERSION:
            continue
        identity = review_memory_identity(record)
        if record.get("decision") == "revision_reset":
            revoked.add(identity)
            continue
        if identity in revoked or record.get("decision") != "approved_edit":
            continue
        ops = record.get("operations")
        if (not isinstance(ops, list) or not 1 <= len(ops) <= 16 or
                any(not isinstance(op, dict) or op.get("operation") not in _ALLOWED_OPERATIONS
                    or not isinstance(op.get("changes"), list) or not op["changes"]
                    for op in ops)):
            continue
        fingerprint = action_key(ops)
        # Deduplicate the *method within the same issue*. A structurally
        # similar edit in another kind of finding can still inform Gemma.
        seen_key = (record.get("finding_type"), record.get("source_sha256"), fingerprint)
        if seen_key in seen_rules:
            continue
        seen_rules.add(seen_key)
        same_source = record.get("source_sha256") == wanted["source_sha256"]
        same_issue = record.get("finding_type") == wanted["finding_type"]
        exact = (same_issue and same_source and
                 record.get("evidence_sha256") == wanted["evidence_sha256"])
        related = (not same_issue and family != "other" and
                   _memory_issue_family(record.get("finding_type", "")) == family)
        # Cross-check memories can be considered even when their diagnostic
        # phrasing differs, if they concern the same CSV. Avoid global unrelated
        # examples from other patients/datasets.
        if not (same_issue or related or same_source):
            continue
        tier = (0 if exact else 1 if same_issue and same_source else
                2 if related and same_source else 3 if same_issue else
                4 if related else 5)
        match = {"match": "exact" if exact else "similar", "record": record,
                 "source_match": same_source,
                 "historical_approval_count": frequencies[(record.get("finding_type"), record.get("source_sha256"), action_key(ops))],
                 "issue_relation": "same" if same_issue else "related" if related else "same_csv_other"}
        ranked.append((tier, match))
    ranked.sort(key=lambda entry: (entry[0], -entry[1]["historical_approval_count"]))
    main = [entry for _, entry in ranked if entry["issue_relation"] == "same"]
    other = [entry for _, entry in ranked if entry["issue_relation"] != "same"]
    if other and limit >= 2:
        # Preserve a diverse comparator while prioritizing strong same-issue
        # evidence, not just the latest nearly identical historical edits.
        result = main[:limit-1] + other[:1]
        for _, entry in ranked:
            if len(result) >= limit:
                break
            if entry not in result:
                result.append(entry)
        return result
    return [entry for _, entry in ranked[:limit]]


def reviewed_method_context(finding: dict, source: Path, *, limit: int = 16) -> list[dict]:
    """Relevant saved *methods* as bounded, untrusted context for Agent B.

    This function returns neither an executable plan nor approval. The queue
    retains candidate summaries only in process memory; its disk cache never
    contains original cell values or historical review records.
    """
    from healthcare_data_quality.learning.interaction_memory import short_description

    methods = []
    for match in ranked_reviewed_solutions(finding, source, limit=limit):
        record = match["record"]
        ops = record.get("operations") or []
        methods.append({
            "previous_issue": str(record.get("finding_title") or "")[:140],
            "method": short_description(record)[:360],
            "operation_types": [str(op.get("operation") or "") for op in ops[:16]],
            "same_original_csv": bool(match.get("source_match")),
            "issue_relation": str(match.get("issue_relation") or "")[:28],
            "historical_approval_count": match.get("historical_approval_count", 0),
            "historical_target_count": sum(len(op.get("changes") or []) for op in ops),
            **({"previous_approved_button_wording": str(record["button_label"])[:140]}
               if record.get("button_label") else {}),
        })
    return methods


def _review_memory_entry(match: dict | None) -> dict | None:
    """A review-only queue result, never a model recommendation or approval.

    Exactly matching reviews can go to host revalidation. Similar reviews must
    be assessed by Gemma on the selected current-source finding before approval.
    No historical operations or source data are persisted in the job queue.
    """
    if not isinstance(match, dict) or match.get("match") not in {"exact", "similar"}:
        return None
    record = match.get("record")
    if not isinstance(record, dict) or record.get("decision") != "approved_edit":
        # A no-change review is not a reusable cleaning proposal. Agent B
        # should still be able to recommend a new approach for that finding.
        return None
    match_type = match["match"]
    return {
        "status": "ready", "updated_at": _now(), "origin": "review_memory",
        "memory_match": match_type,
        "suggestion": ("Saved reviewed solution available for source validation"
                       if match_type == "exact" else
                       "Prior approach found; new-source Gemma assessment required"),
        "decision_needed": "New human approval is required before any edit is staged",
    }


def _prompt_finding(item: dict, runtime: dict | None = None) -> dict:
    """Read-only prompt context; source-cell samples never enter job JSON."""
    sample = list((runtime or {}).get("evidence", []))[:8]
    from healthcare_data_quality.investigation.evidence_registry import prompt_evidence
    registry = (runtime or {}).get("evidence_registry") or item.get("evidence_registry")
    return {**item, "evidence_registry": prompt_evidence(registry) if isinstance(registry, dict) else None,
            "evidence": sample,
            "reviewed_methods": list((runtime or {}).get("reviewed_methods") or []),
            "learned_skills": list((runtime or {}).get("learned_skills") or []),
            "total_evidence_records": item.get("affected_record_count", 0),
            "evidence_is_a_sample": True,
            "source_pattern_summary": (runtime or {}).get("source_pattern_summary"),
            "exact_prior_no_change": bool((runtime or {}).get("exact_prior_no_change")),
            "recent_failure_patterns": list((runtime or {}).get('recent_failure_patterns') or []),
            "related_findings": list((runtime or {}).get('related_findings') or []),
            }


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


def _learned_skills_for_finding(finding: dict) -> list[dict]:
    # Read-only, concise context. Skills are advisory text, never executable code.
    try:
        from healthcare_data_quality.learning.adaptive_skill_memory import skill_context
        return skill_context(finding, limit=3)
    except (OSError, ValueError, KeyError, TypeError):
        return []


class CleaningRecommendationJobs:
    def __init__(self, directory: Path | None = None, *, recommender=None,
                 choice_assessor=None, assist_enabled: bool | None = None):
        self.directory = Path(directory) if directory is not None else _DEFAULT_CACHE_DIR
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._assist_workers: dict[str, threading.Thread] = {}
        # Production default: use otherwise idle Agent C inference capacity.
        # Injected fake recommenders retain historical sequential test behavior
        # unless the test explicitly opts into assistance.
        self._assist_enabled = (recommender is None if assist_enabled is None
                                else bool(assist_enabled))
        self._runtime_context: dict[str, dict[str, dict]] = {}
        self._recommender = recommender or recommend_single_finding
        # Injected fake recommendation workers in tests are not allowed to
        # contact a real GPU endpoint. Production always assesses options.
        if choice_assessor is not None:
            self._choice_assessor = choice_assessor
        elif recommender is None:
            from healthcare_data_quality.services.model_plan_review import assess_review_option
            self._choice_assessor = assess_review_option
        else:
            self._choice_assessor = None

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

    def _can_assist(self) -> bool:
        return (self._assist_enabled and
                os.environ.get("HDQ_AGENT_C_ASSIST_B", "1").lower()
                not in {"0", "false", "off", "no"} and
                _max_active_requests() == 3)

    def _maybe_launch_assist(self, key: str, state: dict) -> None:
        """Borrow C's lane only when extra distinct findings are waiting.

        This is called under self._lock. The worker itself re-checks chat
        activity before each claim. Neither worker can duplicate a claim.
        """
        if not self._can_assist() or interactive_model_request_pending():
            return
        if self._assist_workers.get(key) and self._assist_workers[key].is_alive():
            return
        pending = sum(e.get("status") == "pending" for e in state["entries"].values())
        running = sum(e.get("status") == "running" for e in state["entries"].values())
        if pending < 2 and not (pending and running):
            return
        helper = threading.Thread(
            target=self._run, args=(key,), kwargs={"assist": True},
            name="hdq-agent-c-idle-assist-" + key[:8], daemon=True)
        self._assist_workers[key] = helper
        helper.start()

    def start(self, context_key: str, catalog: list[dict], *, versions: dict | None = None,
              retry_failed: bool = False, run_id: str | None = None,
              review_memory: dict[str, dict] | None = None,
              reviewed_methods: dict[str, list[dict]] | None = None,
              source_paths_by_finding: dict[str, Path] | None = None) -> str:
        """Queue suggestions only for findings without a remembered review.

        Memory-backed entries are informational markers, not model outputs or
        execution authorization. Existing completed/running entries remain
        immutable; a queued entry can be replaced before inference begins.
        """
        key = job_fingerprint(context_key, catalog, versions, run_id=run_id)
        ids = [item["id"] for item in catalog]
        if len(ids) != len(set(ids)):
            raise ValueError("Finding identifiers must be unique")
        remembered = {fid: entry for fid in ids
                      if (entry := _review_memory_entry(
                          (review_memory or {}).get(fid))) is not None}
        with self._lock:
            # Keep a bounded snapshot of original evidence only in this process.
            # It is never written into the persisted recommendation queue.
            context = self._runtime_context.setdefault(key, {})
            for item in catalog:
                context[str(item["id"])] = {
                    "evidence": list(item.get("evidence") or [])[:8],
                    # Full evidence is held only in this process so memory
                    # identity can match all original affected cells. The
                    # persistent job manifest remains _minimal_finding() and
                    # never serializes these potentially sensitive values.
                    "finding_for_memory": deepcopy(item),
                    "related_findings": [
                        {"id": f.get("id"), "title": str(f.get("title") or "")[:160],
                         "check_key": f.get("check_key"), "affected_count": f.get("total_evidence_records", len(f.get("evidence") or []))}
                        for f in catalog if f.get("dataset_id") == item.get("dataset_id") and f.get("id") != item.get("id")][:50],
                    "evidence_registry": item.get("evidence_registry"),
                    "reviewed_methods": list((reviewed_methods or {}).get(str(item["id"])) or []),
                    "learned_skills": _learned_skills_for_finding(item),
                    # Only process-local. Never persist source file paths or
                    # raw evidence to the background queue JSON.
                    "source_path": (Path(source_paths_by_finding[str(item["id"])])
                                    if source_paths_by_finding and
                                    str(item["id"]) in source_paths_by_finding else None),
                }
            state = self._load(key)
            if state is None:
                items = [_minimal_finding(item) for item in catalog]
                state = {"schema": _JOB_SCHEMA, "key": key, "created_at": _now(),
                         "findings": items,
                         "entries": {item["id"]: remembered.get(item["id"], {
                             "status": "pending", "updated_at": _now()})
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
                        state["entries"][summary["id"]] = remembered.get(
                            summary["id"], {"status": "pending", "updated_at": _now()})
                        known.add(summary["id"])
                # Reconcile old non-executable memory markers. Earlier
                # versions cached a no-change review as a ready suggestion;
                # it must not block future Agent B work or hide an approved edit.
                # Only memory-origin entries are changed, never completed model
                # outputs or active generations.
                for fid, entry in list(state["entries"].items()):
                    if entry.get("origin") == "review_memory":
                        state["entries"][fid] = remembered.get(
                            fid, {"status": "pending", "updated_at": _now()})
                # A review approved after queue creation can suppress any
                # still-pending inference, without overwriting finished work.
                for fid, memory_entry in remembered.items():
                    if state["entries"].get(fid, {}).get("status") == "pending":
                        state["entries"][fid] = memory_entry
                # For the once-only Agent B workflow, an interrupted request
                # cannot be safely replayed without possibly generating twice.
                # Older non-run-specific queues retain their resumable behavior.
                running = bool(
                    (self._workers.get(key) and self._workers[key].is_alive()) or
                    (self._assist_workers.get(key) and self._assist_workers[key].is_alive()))
                if not running:
                    for entry in state["entries"].values():
                        if entry["status"] == "running":
                            if run_id is not None:
                                entry["status"] = "failed"
                                entry["error"] = "Interrupted; once-only Agent B request will not rerun"
                            else:
                                entry["status"] = "pending"
            if retry_failed:
                for entry in state["entries"].values():
                    if entry["status"] == "failed" and int(entry.get("manual_retries", 0)) < 1:
                        entry["status"] = "pending"
                        entry["manual_retries"] = int(entry.get("manual_retries", 0)) + 1
                        entry.pop("error", None)
            self._save(state)
            if any(entry["status"] == "pending" for entry in state["entries"].values()):
                if not (self._workers.get(key) and self._workers[key].is_alive()):
                    worker = threading.Thread(target=self._run, args=(key,),
                                              name="hdq-recommendations-" + key[:8], daemon=True)
                    self._workers[key] = worker
                    worker.start()
                self._maybe_launch_assist(key, state)
        return key

    def snapshot(self, key: str) -> dict | None:
        # _save publishes an atomic replacement; readers see an entire old or
        # new JSON file, never a partially written queue. Do not hold the UI
        # behind a writer's fsync (particularly on WSL-mounted storage).
        try:
            return self._load(key)
        except FileNotFoundError:
            return None

    def _run(self, key: str, *, assist: bool = False) -> None:
        try:
            self._run_queue(key, assist=assist)
        finally:
            # If an exploration appended a new item exactly as the last worker
            # exited, guarantee the pending item eventually runs.
            with self._lock:
                if assist:
                    if self._assist_workers.get(key) is threading.current_thread():
                        self._assist_workers.pop(key, None)
                    current = self._load(key)
                    primary_active = (self._workers.get(key) and
                                      self._workers[key].is_alive())
                    if (not primary_active and current and
                            not any(e["status"] in {"pending", "running"}
                                    for e in current["entries"].values())):
                        self._runtime_context.pop(key, None)
                elif self._workers.get(key) is threading.current_thread():
                    self._workers.pop(key, None)
                    state = self._load(key)
                    if (not state or (not any(e["status"] in {"pending", "running"}
                                           for e in state["entries"].values())
                                      and not (self._assist_workers.get(key) and
                                               self._assist_workers[key].is_alive()))):
                        self._runtime_context.pop(key, None)
                    if (state and not state.get("cancel_requested") and
                            any(e["status"] == "pending" for e in state["entries"].values())):
                        worker = threading.Thread(target=self._run, args=(key,),
                                                  name="hdq-recommendations-" + key[:8], daemon=True)
                        self._workers[key] = worker
                        worker.start()

    def _run_queue(self, key: str, *, assist: bool = False) -> None:
        while True:
            # Never begin another borrowed-lane request while a human chat is
            # queued/active. Already running inference cannot be preempted.
            if assist and interactive_model_request_pending():
                return
            with self._lock:
                state = self._load(key)
                if not state or state.get("cancel_requested"):
                    return
                next_item = next((x for x in state["findings"]
                                  if state["entries"][x["id"]]["status"] == "pending"), None)
                if next_item is None:
                    return
                fid = next_item["id"]
                prior_retry_count = int(state["entries"][fid].get("manual_retries", 0))
                state["entries"][fid] = {
                    "status": "running", "updated_at": _now(),
                    "manual_retries": prior_retry_count,
                    "worker_role": "agent_c_idle_assist" if assist else "agent_b",
                }
                self._save(state)
            try:
                # The background prompt is short. An interactive request already
                # waiting gets priority before the NEXT background inference.
                with model_request_priority("idle_assist" if assist else "recommendation"), model_reconnect_mode():
                    with self._lock:
                        runtime = self._runtime_context.get(key, {}).get(fid)
                    # Re-read approved methods and skills immediately before
                    # EACH request: decisions approved during this investigation
                    # can guide later queued findings without regenerating
                    # suggestions already delivered to the user.
                    current = dict(runtime or {})
                    source_path = current.get("source_path")
                    full_finding = current.get("finding_for_memory") or next_item
                    if source_path is not None:
                        try:
                            current["reviewed_methods"] = reviewed_method_context(
                                full_finding, Path(source_path), limit=16)
                        except (OSError, ValueError, TypeError):
                            # Memory is best-effort; never invent a preference.
                            pass
                    current["learned_skills"] = _learned_skills_for_finding(next_item)
                    from healthcare_data_quality.learning.model_failure_patterns import recent_failure_hints
                    current['recent_failure_patterns'] = recent_failure_hints(full_finding)
                    if source_path is not None:
                        # Compute aggregate source-wide patterns before inference;
                        # never persist raw data or file paths in the job JSON.
                        try:
                            from healthcare_data_quality.services.source_patterns import summarize_source_patterns
                            current["source_pattern_summary"] = summarize_source_patterns(Path(source_path))
                        except (OSError, ValueError, csv.Error, UnicodeError):
                            current["source_pattern_summary"] = None
                        # A past no-change review is a preference for the exact
                        # SAME source+finding, not a transferable cleaning rule.
                        try:
                            preference = preferred_reviewed_solution(full_finding, Path(source_path))
                            record = preference.get("record", {}) if preference else {}
                            current["exact_prior_no_change"] = bool(
                                preference and preference["match"] == "exact" and
                                record.get("decision") == "reviewed_no_change")
                        except (OSError, ValueError, TypeError):
                            current["exact_prior_no_change"] = False
                    # Prepare exact remembered and source-backed candidates independently
                    # of free-form drafting. When two choices already exist including
                    # a prior human approval, Gemma reviews those directly rather than
                    # requiring another speculative recommendation first.
                    from healthcare_data_quality.ui.human_choice_replay import memory_first_candidates
                    seeds = memory_first_candidates(full_finding, Path(source_path)) if source_path else []
                    has_preferred = any(c.get('historical_approval_count', 0) > 0 for c in seeds)
                    recommendation = None
                    if len(seeds) >= 2 and has_preferred and self._choice_assessor is not None:
                        recommendation = {'suggestion': 'Review previously approved and source-backed alternatives.',
                                          'choices': []}
                    if recommendation is None:
                        for attempt in range(2):
                            try:
                                recommendation = self._recommender(_prompt_finding(next_item, current))
                                if not isinstance(recommendation, dict) or not str(recommendation.get("suggestion", "")).strip():
                                    raise ValueError("Empty Gemma recommendation")
                                break
                            except (RuntimeError, ValueError, TimeoutError, OSError):
                                if attempt:
                                    if not seeds or self._choice_assessor is None:
                                        raise
                                    recommendation = {'suggestion': 'Review independently prepared alternatives.',
                                                      'choices': []}
                    reviewed_options = []
                    source_path = current.get('source_path')
                    if source_path is not None:
                        full = current.get('finding_for_memory') or next_item
                        candidates = memory_first_candidates(full, Path(source_path), {
                            **recommendation,
                            'memory_context': current.get('reviewed_methods') or [],
                            'approved_row_deletion_method_seen': any(
                                isinstance(m, dict) and 'drop_rows' in (m.get('operation_types') or [])
                                for m in current.get('reviewed_methods') or []),
                        })
                        if self._choice_assessor is not None:
                            for candidate in candidates[:5]:
                                # A transient inference failure should not
                                # strand this one finding. The second attempt
                                # reviews the identical source-bound candidate;
                                # never auto-expand its target scope.
                                for assessment_attempt in range(2):
                                    try:
                                        assessment = self._choice_assessor(
                                            full, Path(source_path), candidate,
                                            methods=current.get('reviewed_methods'),
                                            skills=current.get('learned_skills'),
                                            review_context={'findings': current.get('related_findings') or []})
                                        if not isinstance(assessment, dict) or 'can_offer' not in assessment:
                                            raise ValueError('Gemma option appraisal omitted its decision')
                                        reviewed_options.append(assessment)
                                        break
                                    except (RuntimeError, ValueError, TimeoutError, OSError, ConnectionError) as exc:
                                        if assessment_attempt:
                                            # A timed-out/invalid *advisory* review must
                                            # not discard an independently source-verified
                                            # recommendation and strand its reviewer UI.
                                            # Record failure as not assessed, not endorsed.
                                            from healthcare_data_quality.services.model_plan_review import option_fingerprint
                                            reviewed_options.append({
                                                'fingerprint': option_fingerprint(
                                                    full, Path(source_path), candidate),
                                                'can_offer': False,
                                                'decision': 'unavailable',
                                                'scope_matches': None,
                                                'has_extra_changes': None,
                                                'reason': ('Gemma assessment unavailable ('
                                                           + type(exc).__name__ + ').'),
                                            })
                item = {"status": "ready", "updated_at": _now(),
                        "assisted_by_agent_c": bool(assist),
                        "suggestion": str(recommendation["suggestion"])[:2200],
                        "decision_needed": str(recommendation.get("decision_needed", ""))[:1400],
                        "memory_assessed": bool(current.get("reviewed_methods") or current.get("learned_skills")),
                        "method_candidates_considered": len(current.get("reviewed_methods") or []),
                        "skill_candidates_considered": len(current.get("learned_skills") or []),
                        "memory_assessment": recommendation.get("memory_assessment") or {},
                        "approved_row_deletion_method_seen": any(
                            isinstance(m, dict) and 'drop_rows' in (m.get('operation_types') or [])
                            and m.get('issue_relation') in {'same', 'related'}
                            for m in (current.get('reviewed_methods') or [])
                        ),
                        "memory_context": [
                            {"operation_types": [str(t)[:30] for t in (m.get("operation_types") or [])[:5]],
                             "issue_relation": str(m.get("issue_relation") or "")[:24],
                             "same_original_csv": bool(m.get("same_original_csv"))}
                            for m in (current.get("reviewed_methods") or [])[:3]
                            if isinstance(m, dict)],
                        "choices": list(recommendation.get("choices") or [])[:5],
                        "gemma_option_reviews": reviewed_options,
                        # Bounded, aggregate-only trace of actual source
                        # patterns and prior exact-source no-change preference.
                        "source_pattern_summary": current.get("source_pattern_summary"),
                        "exact_prior_no_change": bool(current.get("exact_prior_no_change"))}
            except Exception as exc:  # Preserve remaining queue even if an individual LLM call fails.
                try:
                    from healthcare_data_quality.learning.model_failure_patterns import record_failure
                    record_failure((runtime or {}).get('finding_for_memory') or next_item,
                                   'recommendation', exc)
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # Diagnostics must never break another worker.
                item = {"status": "failed", "updated_at": _now(),
                        "assisted_by_agent_c": bool(assist),
                        "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            # Durable producer provenance on BOTH success and failure.
            # Rendering must not guess the worker from the global Agent B
            # queue title after C borrows the idle recommendation lane.
            item["worker_role"] = "agent_c_idle_assist" if assist else "agent_b"
            item["manual_retries"] = prior_retry_count
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
