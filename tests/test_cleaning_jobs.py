"""Background Gemma worker does not mutate datasets or approval state."""
import json
import threading
from pathlib import Path

import pytest

from cleaning_jobs import CleaningRecommendationJobs, job_fingerprint
from healthcare_data_quality.llm_client import _RequestGate, model_request_priority


def _findings():
    return [
        {"id": "dataset_1:qc:1", "dataset_id": "dataset_1", "source": "QC",
         "title": "Fractional counts", "method": "Check integer counts",
         "evidence": [{"Original value": "SENSITIVE ORIGINAL CELL"}], "actionable": True},
        {"id": "gemma:1:check", "source": "Gemma check", "title": "Control check",
         "actionable": False, "evidence": []},
    ]


def _wait(jobs, key):
    worker = jobs._workers[key]
    worker.join(3)
    assert not worker.is_alive(), "Fake model never returned"
    return jobs.snapshot(key)


def test_one_finding_at_a_time_and_persisted_partial_results(tmp_path):
    started = threading.Event()
    release = threading.Event()
    seen = []

    def fake(item):
        seen.append(item["id"])
        if len(seen) == 2:
            started.set()
            assert release.wait(2)
        return {"suggestion": "Ask researcher first", "decision_needed": "Policy?"}

    jobs = CleaningRecommendationJobs(tmp_path, recommender=fake)
    key = jobs.start("test-audit", _findings())
    assert started.wait(2)
    interim = jobs.snapshot(key)
    assert interim["entries"]["dataset_1:qc:1"]["status"] == "ready"
    assert interim["entries"]["gemma:1:check"]["status"] == "running"
    # Source-cell contents are not persisted into the recommendation queue.
    assert "SENSITIVE ORIGINAL CELL" not in (tmp_path / (key + ".json")).read_text()
    release.set()
    done = _wait(jobs, key)
    assert [x["status"] for x in done["entries"].values()] == ["ready", "ready"]
    assert seen == ["dataset_1:qc:1", "gemma:1:check"]
    # Re-render/start cannot re-run already completed model calls.
    assert jobs.start("test-audit", _findings()) == key
    assert seen == ["dataset_1:qc:1", "gemma:1:check"]


def test_failed_individual_request_does_not_block_other_findings(tmp_path):
    calls = []

    def fake(item):
        calls.append(item["id"])
        if len(calls) == 1:
            raise RuntimeError("local test failure")
        return {"suggestion": "No data change needed", "decision_needed": "Confirm"}

    jobs = CleaningRecommendationJobs(tmp_path, recommender=fake)
    key = jobs.start("audit-b", _findings())
    record = _wait(jobs, key)
    assert [x["status"] for x in record["entries"].values()] == ["failed", "ready"]
    jobs.start("audit-b", _findings())
    assert len(calls) == 2  # Failures do not silently retry every UI refresh.
    jobs.start("audit-b", _findings(), retry_failed=True)
    record = _wait(jobs, key)
    assert all(x["status"] == "ready" for x in record["entries"].values())
    assert len(calls) == 3


def test_worker_resumes_interrupted_job_from_disk(tmp_path):
    jobs = CleaningRecommendationJobs(tmp_path, recommender=lambda item: {
        "suggestion": "Review value", "decision_needed": "Policy needed"})
    key = job_fingerprint("audit-c", _findings())
    jobs.start("audit-c", _findings())
    _wait(jobs, key)
    # Simulate a process crash during the second item of a previous job.
    state = jobs.snapshot(key)
    state["entries"]["gemma:1:check"] = {"status": "running"}
    jobs._save(state)
    fresh_process = CleaningRecommendationJobs(tmp_path, recommender=lambda item: {
        "suggestion": "Recovered", "decision_needed": "Confirm"})
    fresh_process.start("audit-c", _findings())
    result = _wait(fresh_process, key)
    assert result["entries"]["gemma:1:check"]["suggestion"] == "Recovered"


def test_version_change_isolated_and_identifiers_validated(tmp_path):
    first = job_fingerprint("audit-d", _findings())
    second = job_fingerprint("audit-d", _findings(), {"dataset_1": "new-hash"})
    assert first != second
    with pytest.raises(ValueError, match="unique"):
        job_fingerprint("audit-d", _findings() + _findings())


def test_cancel_prevents_new_requests_after_current_finishes(tmp_path):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake(item):
        calls.append(item["id"])
        started.set()
        assert release.wait(2)
        return {"suggestion": "Review", "decision_needed": "Policy"}

    jobs = CleaningRecommendationJobs(tmp_path, recommender=fake)
    key = jobs.start("audit-e", _findings())
    assert started.wait(2)
    jobs.cancel(key)
    release.set()
    _wait(jobs, key)
    assert calls == ["dataset_1:qc:1"]
    assert jobs.snapshot(key)["cancel_requested"] is True


def test_gate_prefers_waiting_interactive_request():
    gate = _RequestGate()
    acquired = []
    started = threading.Event()
    release = threading.Event()
    chat_waiting = threading.Event()

    def busy():
        with gate.admit("background"):
            started.set()
            assert release.wait(2)

    def chat():
        chat_waiting.set()
        with gate.admit("interactive"):
            acquired.append("chat")

    def background():
        with gate.admit("background"):
            acquired.append("background")

    t1 = threading.Thread(target=busy)
    t1.start()
    assert started.wait(2)
    t2 = threading.Thread(target=chat)
    t2.start()
    assert chat_waiting.wait(2)
    # Ensure chat actually registered at the gate before launching background.
    with gate._condition:
        assert gate._waiting_interactive == 1
        t3 = threading.Thread(target=background)
        t3.start()
    release.set()
    for t in (t1, t2, t3):
        t.join(2)
        assert not t.is_alive()
    assert acquired == ["chat", "background"]


def test_once_per_investigation_and_gemma_checks_appended_without_regeneration(tmp_path):
    calls = []
    def fake(item):
        calls.append(item["id"])
        return {"suggestion": "Review once", "decision_needed": "Scientific decision?"}

    jobs = CleaningRecommendationJobs(tmp_path, recommender=fake)
    original = _findings()[:1]
    run_id = "full-investigation-123"
    key = jobs.start("same-data", original, run_id=run_id)
    _wait(jobs, key)
    assert calls == ["dataset_1:qc:1"]

    # New Gemma check arrives after exploration. A previous finding is present
    # again in the catalog, but must NOT be generated twice.
    assert jobs.start("same-data", _findings(), run_id=run_id) == key
    _wait(jobs, key)
    assert calls == ["dataset_1:qc:1", "gemma:1:check"]
    assert jobs.start("same-data", _findings(), run_id=run_id,
                      versions={"dataset_1": "cleaned-version"}, retry_failed=True) == key
    assert calls == ["dataset_1:qc:1", "gemma:1:check"]
    assert all(x["status"] == "ready" for x in jobs.snapshot(key)["entries"].values())

    # A deliberately NEW full investigation is permitted its own unique advice.
    new_key = jobs.start("same-data", _findings(), run_id="new-investigation")
    _wait(jobs, new_key)
    assert new_key != key
    assert len(calls) == 4


def test_once_only_failed_advice_is_not_regenerated_on_retry(tmp_path):
    calls = []
    def failure(item):
        calls.append(item["id"])
        raise ValueError("Synthetic model failure")

    jobs = CleaningRecommendationJobs(tmp_path, recommender=failure)
    key = jobs.start("dataset", _findings()[:1], run_id="run-with-failure")
    _wait(jobs, key)
    assert jobs.snapshot(key)["entries"]["dataset_1:qc:1"]["status"] == "failed"
    jobs.start("dataset", _findings()[:1], run_id="run-with-failure", retry_failed=True)
    assert len(calls) == 1  # No generation more than once during this run.


def test_once_only_interrupted_call_stays_failed_after_restart(tmp_path):
    calls = []
    jobs = CleaningRecommendationJobs(tmp_path, recommender=lambda item: {
        "suggestion": "Saved", "decision_needed": "Check policy"})
    key = jobs.start("audit", _findings()[:1], run_id="interrupt-run")
    _wait(jobs, key)
    state = jobs.snapshot(key)
    state["entries"]["dataset_1:qc:1"] = {"status": "running"}
    jobs._save(state)
    new_manager = CleaningRecommendationJobs(tmp_path, recommender=lambda item: calls.append(item))
    new_manager.start("audit", _findings()[:1], run_id="interrupt-run")
    result = new_manager.snapshot(key)
    assert result["entries"]["dataset_1:qc:1"]["status"] == "failed"
    assert not calls


def test_recommendation_prompt_preserves_full_count_when_evidence_is_sampled():
    from cleaning_jobs import _minimal_finding, _prompt_finding
    finding = {
        "id": "dataset_1:qc:2", "dataset_id": "dataset_1", "source": "QC",
        "title": "618 missing count values", "method": "Inspect count matrix",
        "actionable": True, "total_evidence_records": 618,
        "evidence": [{"CSV row": 12, "Column": 9, "Original value": ""}],
    }
    summary = _minimal_finding(finding)
    assert summary["affected_record_count"] == 618
    assert "Original value" not in summary
    prompt = _prompt_finding(summary, {"evidence": finding["evidence"]})
    assert prompt["total_evidence_records"] == 618
    assert len(prompt["evidence"]) == 1
