"""Nonblocking Gemma exploration, verified checkpoints and model-slot fairness."""
import threading
import time

import pytest

from healthcare_data_quality.jobs.exploration_jobs import LiveExplorationJobs
from healthcare_data_quality.services.llm_client import _RequestGate


def _wait(jobs, key, *, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = jobs.snapshot(key)
        if state and state["status"] != "running":
            return state
        time.sleep(0.01)
    raise AssertionError("The test worker did not complete")


def test_worker_yields_ui_control_and_publishes_verified_rounds(tmp_path):
    begin = threading.Event()
    release = threading.Event()
    report_round = [{"result": {"summary": "Verified test", "findings": []}}]
    persisted = []
    stopped = []

    def run(datasets, audit, *, on_round, on_progress):
        assert audit == {"baseline": {}}
        on_progress("Planning test 1")
        begin.set()
        assert release.wait(2)
        on_round(report_round)
        return {"runs": report_round, "stop": {"decision": "stop"}}

    jobs = LiveExplorationJobs(runner=run)
    csv = tmp_path / "data.csv"
    csv.write_text("A\n1\n")
    assert jobs.start("run-1", {"dataset_1": csv}, {"baseline": {}},
                      on_round_saved=lambda rounds: persisted.append(rounds),
                      on_finished=lambda runs, stop: stopped.append((runs, stop)))
    assert begin.wait(2)
    # Immediately available to the Streamlit rendering thread.
    snapshot = jobs.snapshot("run-1")
    assert snapshot["status"] == "running"
    assert snapshot["activity"] == "Planning test 1"
    assert snapshot["runs"] == []
    assert not jobs.start("run-1", {"dataset_1": csv}, {})  # never duplicate
    release.set()
    result = _wait(jobs, "run-1")
    assert result["status"] == "complete"
    assert result["runs"] == report_round
    assert persisted == [report_round]
    assert stopped == [(report_round, {"decision": "stop"})]
    # Caller cannot change published results by mutating a snapshot.
    result["runs"].clear()
    assert jobs.snapshot("run-1")["runs"] == report_round


def test_worker_keeps_verified_rounds_on_exploration_failure(tmp_path):
    seen = [{"result": {"summary": "Round 1", "findings": []}}]

    def crash(datasets, audit, *, on_round, on_progress):
        on_round(seen)
        raise RuntimeError("No next distinct test")

    jobs = LiveExplorationJobs(runner=crash)
    source = tmp_path / "x.csv"
    source.write_text("A\n1\n")
    jobs.start("run-2", {"dataset_1": source}, {})
    result = _wait(jobs, "run-2")
    assert result["status"] == "incomplete"
    assert "RuntimeError" in result["error"]
    assert result["runs"] == seen
    assert result["stop"] is None


def test_persistence_warning_does_not_discard_verified_result(tmp_path):
    test = [{"result": {"summary": "ok", "findings": []}}]

    def complete(datasets, audit, *, on_round, on_progress):
        on_round(test)
        return {"runs": test, "stop": None}

    jobs = LiveExplorationJobs(runner=complete)
    csv = tmp_path / "x.csv"
    csv.write_text("A\n1\n")

    def fail_saving(rounds):
        raise OSError("Disk full")

    jobs.start("run-3", {"dataset_1": csv}, {}, on_round_saved=fail_saving)
    snapshot = _wait(jobs, "run-3")
    assert snapshot["status"] == "complete"
    assert snapshot["runs"] == test
    assert "Disk full" in snapshot["warning"]


def test_reject_empty_job_identity():
    jobs = LiveExplorationJobs()
    with pytest.raises(ValueError):
        jobs.start("", {"a": "b"}, {})
    with pytest.raises(ValueError):
        jobs.start("id", {}, {})


def test_two_slots_reserve_one_for_interactive_requests(monkeypatch):
    monkeypatch.setenv("HDQ_LLM_MAX_ACTIVE", "2")
    gate = _RequestGate()
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    interactive_admitted = threading.Event()

    def background_first():
        with gate.admit("background"):
            first_started.set()
            assert release_first.wait(2)

    def background_second():
        with gate.admit("background"):
            second_started.set()

    def interactive():
        with gate.admit("interactive"):
            interactive_admitted.set()

    one = threading.Thread(target=background_first)
    two = threading.Thread(target=background_second)
    chat = threading.Thread(target=interactive)
    one.start()
    assert first_started.wait(2)
    two.start()
    chat.start()
    assert interactive_admitted.wait(2)
    assert not second_started.is_set()  # background can't fill reserved slot
    release_first.set()
    for t in (one, two, chat):
        t.join(2)
        assert not t.is_alive()
    assert second_started.is_set()
