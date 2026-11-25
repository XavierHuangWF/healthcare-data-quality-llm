"""Gemma outage: retry same request, preserve job state, lock UI, restore it."""
from __future__ import annotations

import io
import json
from threading import Event, Thread
from time import sleep
from urllib.error import HTTPError, URLError

import pytest

from healthcare_data_quality.jobs.investigation_jobs import InvestigationJobs
from healthcare_data_quality.services import llm_client
from healthcare_data_quality.services.model_connectivity import model_connection_snapshot


class Response:
    def __init__(self):
        self.data = io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}]
        }).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self, n=-1):
        return self.data.read(n)


def test_connection_refused_then_recovers_same_request(monkeypatch):
    seen = []
    offline_snapshot = []

    def request(req, timeout):
        seen.append((req.full_url, req.data))
        if len(seen) == 1:
            raise URLError(ConnectionRefusedError("server shut down"))
        return Response()

    monkeypatch.setattr(llm_client, "urlopen", request)
    monkeypatch.setattr(llm_client, "sleep", lambda secs: offline_snapshot.append(model_connection_snapshot()))
    monkeypatch.setenv("HDQ_GEMMA_AUTO_RECONNECT", "1")
    assert llm_client.query_local_model("sensitive source prompt") == "OK"
    assert len(seen) == 2 and seen[0] == seen[1]
    assert offline_snapshot[0]["offline"] is True
    assert offline_snapshot[0]["waiting_requests"] == 1
    assert "sensitive source prompt" not in str(offline_snapshot)
    assert model_connection_snapshot()["offline"] is False


def test_http_503_is_retried_but_invalid_request_is_not(monkeypatch):
    monkeypatch.setenv("HDQ_GEMMA_AUTO_RECONNECT", "1")
    count = 0

    def unavailable(*_args, **_kwargs):
        nonlocal count
        count += 1
        if count < 3:
            raise HTTPError("http://127.0.0.1:8000", 503, "unavailable", {}, io.BytesIO(b"down"))
        return Response()

    monkeypatch.setattr(llm_client, "urlopen", unavailable)
    monkeypatch.setattr(llm_client, "sleep", lambda _secs: None)
    assert llm_client.query_local_model("check") == "OK"
    assert count == 3

    def invalid(*_args, **_kwargs):
        raise HTTPError("http://127.0.0.1:8000", 400, "invalid schema", {}, io.BytesIO(b"invalid schema"))

    monkeypatch.setattr(llm_client, "urlopen", invalid)
    with pytest.raises(RuntimeError, match="vLLM HTTP 400"):
        llm_client.query_local_model("invalid")
    assert model_connection_snapshot()["offline"] is False


def test_auto_reconnect_can_be_disabled_for_diagnostics(monkeypatch):
    monkeypatch.setenv("HDQ_GEMMA_AUTO_RECONNECT", "0")
    monkeypatch.setattr(llm_client, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(URLError("down")))
    monkeypatch.setattr(llm_client, "sleep", lambda _secs: pytest.fail("should not retry"))
    with pytest.raises(URLError):
        llm_client.query_local_model("test")
    assert not model_connection_snapshot()["offline"]


def test_waiting_for_server_releases_agent_gpu_slot(monkeypatch):
    monkeypatch.setenv("HDQ_GEMMA_AUTO_RECONNECT", "1")
    was_offline = Event()
    continue_retry = Event()
    result = []
    monkeypatch.setenv("HDQ_LLM_MAX_ACTIVE", "1")
    monkeypatch.setattr(llm_client, "_GATE", llm_client._RequestGate())
    attempts = 0

    def request(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("refused")
        return Response()

    def on_wait(_seconds):
        was_offline.set()
        assert continue_retry.wait(3)

    monkeypatch.setattr(llm_client, "urlopen", request)
    monkeypatch.setattr(llm_client, "sleep", on_wait)
    t = Thread(target=lambda: result.append(llm_client.query_local_model("question")), daemon=True)
    t.start()
    assert was_offline.wait(2)
    assert model_connection_snapshot()["offline"]
    # The background worker is sleeping OUTSIDE _GATE.admit.
    with llm_client._GATE.admit("interactive"):
        pass
    continue_retry.set()
    t.join(3)
    assert result == ["OK"]
    assert not model_connection_snapshot()["offline"]


def test_worker_keeps_same_audit_and_run_id_during_outage(monkeypatch, tmp_path):
    monkeypatch.setenv("HDQ_GEMMA_AUTO_RECONNECT", "1")
    started_wait = Event()
    restore_model = Event()
    seen = []
    count = 0
    original = llm_client._GATE
    monkeypatch.setattr(llm_client, "_GATE", llm_client._RequestGate())

    def request(*_args, **_kwargs):
        nonlocal count
        count += 1
        if not restore_model.is_set():
            raise URLError("connection refused")
        return Response()

    monkeypatch.setattr(llm_client, "urlopen", request)
    monkeypatch.setattr(llm_client, "sleep", lambda _secs: (started_wait.set(), restore_model.wait(3)))
    source = tmp_path / "source.csv"
    source.write_text("taxon,count\na,3\n")
    manager = InvestigationJobs()

    def runner(_path, *, additional_datasets, progress_callback, baseline_ready_callback):
        baseline_ready_callback({"dataset_baselines": {"dataset_1": {"checks": {}}}})
        result = llm_client.query_local_model("continue original investigation")
        return {"result": result, "status": "complete"}

    assert manager.start("same-run", [source], on_baseline=lambda partial: seen.append("QC"),
                         on_finished=lambda complete: seen.append("complete"), runner=runner)
    try:
        assert started_wait.wait(3)
        during = manager.snapshot("same-run")
        assert during["status"] == "running", during
        assert during["partial_audit"] is not None
        assert during["audit"] is None
        assert seen == ["QC"]
        assert model_connection_snapshot()["offline"]
    finally:
        restore_model.set()
    for _ in range(150):
        if manager.snapshot("same-run")["status"] == "completed":
            break
        sleep(.01)
    state = manager.snapshot("same-run")
    assert state["status"] == "completed" and state["run_id"] == "same-run"
    assert state["audit"]["result"] == "OK"
    assert seen == ["QC", "complete"]
    assert count >= 2
    assert not model_connection_snapshot()["offline"]


def test_offline_ui_blocks_editing_then_triggers_rerun(monkeypatch):
    from test_cleaning_review_cards import StubUI, _app
    stub = StubUI()
    app = _app(monkeypatch, stub)
    records = []
    stub.warning = lambda msg, **kw: records.append(("warning", msg))
    stub.caption = lambda msg, **kw: records.append(("caption", msg))
    stub.rerun = lambda: records.append(("rerun", ""))
    monkeypatch.setattr(app, "model_connection_snapshot", lambda: {
        "offline": True, "waiting_requests": 1, "attempts": 2,
    })
    assert app.render_model_reconnect_gate({"status": "running"}) is True
    assert any("paused, not failed" in msg for _, msg in records)
    assert not any(key == "rerun" for key, _ in records)
    assert app.render_model_reconnect_gate({"status": "failed"}) is False
    monkeypatch.setattr(app, "model_connection_snapshot", lambda: {
        "offline": False, "waiting_requests": 0, "attempts": 0,
    })
    assert app.render_model_reconnect_gate({"status": "running"}) is False


def test_permanent_model_500_and_bare_library_calls_fail_fast(monkeypatch):
    monkeypatch.delenv("HDQ_GEMMA_AUTO_RECONNECT", raising=False)
    monkeypatch.setattr(llm_client, "urlopen", lambda *_a, **_kw: (_ for _ in ()).throw(
        URLError("not running")))
    monkeypatch.setattr(llm_client, "sleep", lambda _s: pytest.fail("unscoped call must not wait"))
    with pytest.raises(URLError):
        llm_client.query_local_model("bare diagnostic")

    monkeypatch.setenv("HDQ_GEMMA_AUTO_RECONNECT", "1")
    monkeypatch.setattr(llm_client, "urlopen", lambda *_a, **_kw: (_ for _ in ()).throw(
        HTTPError("http://127.0.0.1:8000", 500, "CUDA OOM", {}, io.BytesIO(b"CUDA out of memory"))))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        llm_client.query_local_model("context overflow")
    assert not model_connection_snapshot()["offline"]
