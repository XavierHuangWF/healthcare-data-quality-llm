"""Engine-death recovery and one owner for the review render lifecycle."""
import io
from pathlib import Path
from urllib.error import HTTPError, URLError
import pytest
from healthcare_data_quality.services import llm_client
from healthcare_data_quality.services.model_connectivity import model_connection_snapshot
from test_gemma_server_recovery_20261123 import Response
from test_cleaning_review_cards import StubUI, _app

ROOT = Path(__file__).resolve().parents[1]


def http_error(code, message):
    return HTTPError('http://127.0.0.1:8000', code, message, {}, io.BytesIO(message.encode()))


@pytest.mark.parametrize('message', ['EngineCore encountered an issue. See stack trace (above) for the root cause.', 'Internal Server Error'])
def test_engine_death_waits_for_health_then_retries_identical_read_only_request(monkeypatch, message):
    monkeypatch.setenv('HDQ_GEMMA_AUTO_RECONNECT', '1')
    requests, health, waits = [], [], []
    class Healthy:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
    def send(req, timeout):
        if req.get_method() == 'GET':
            assert req.full_url == llm_client.HEALTH_URL
            health.append(timeout)
            if len(health) <= 2:
                raise http_error(503, 'engine offline')
            return Healthy()
        requests.append(req.data)
        if len(requests) == 1:
            raise http_error(500, message)
        return Response()
    monkeypatch.setattr(llm_client, 'urlopen', send)
    monkeypatch.setattr(llm_client, 'sleep', lambda _: waits.append(model_connection_snapshot()))
    assert llm_client.query_local_model('read-only model task') == 'OK'
    assert len(requests) == 2 and requests[0] == requests[1]
    assert len(health) == 3 and all(w['offline'] for w in waits)
    assert not model_connection_snapshot()['offline']


def test_healthy_generic_500_still_fails_fast(monkeypatch):
    monkeypatch.setenv('HDQ_GEMMA_AUTO_RECONNECT', '1')
    class Healthy:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
    def send(req, timeout):
        if req.get_method() == 'GET': return Healthy()
        raise http_error(500, 'Internal Server Error')
    monkeypatch.setattr(llm_client, 'urlopen', send)
    monkeypatch.setattr(llm_client, 'sleep', lambda _: pytest.fail('Healthy engine 500 is not an outage'))
    with pytest.raises(RuntimeError, match='vLLM HTTP 500'):
        llm_client.query_local_model('bad inference')
    assert not model_connection_snapshot()['offline']


def test_decline_uses_local_fragment_rerun_and_keeps_other_approvals(monkeypatch):
    st = StubUI(); app = _app(monkeypatch, st)
    st.session_state.cleaning_plans_by_finding = {'f': {'operation': 'drop_rows'}, 'other': {'operation': 'replace_cells'}}
    st.session_state.cleaning_discussion_mode_by_finding = {}
    st.session_state.cleaning_review_statuses = {}
    st.session_state.cleaning_history = [{'finding_id': 'other'}]
    calls = []
    def rerun(*, scope='app'):
        calls.append(scope)
        raise StopIteration
    st.rerun = rerun
    app._decline_and_discuss('f')
    with pytest.raises(StopIteration): app._rerun_cleaning_fragment('f')
    assert calls == ['fragment']
    assert app._waiting_for_new_discussion(st.session_state, 'f')
    assert st.session_state.cleaning_plans_by_finding == {'other': {'operation': 'replace_cells'}}
    assert st.session_state.cleaning_history == [{'finding_id': 'other'}]


def test_supported_iframe_and_single_cleaning_render_owner():
    source = (ROOT/'app.py').read_text()
    wait = (ROOT/'src/healthcare_data_quality/ui/recommendation_wait.py').read_text()
    assert 'streamlit.components.v1' not in source + wait
    assert 'st.iframe(' in source and 'st.iframe(' in wait
    assert 'on_click=_decline_and_discuss' in source
    callback = source.split('def _decline_and_discuss(',1)[1].split('\ndef ',1)[0]
    assert 'st.rerun(' not in callback and 'st.markdown(' not in callback
    assert source.count('st.fragment(run_every="5s", key="hdq_cleaning_review")(refresh_cleaning)()') == 1
    assert 'st.fragment(run_every="3s")(_poll_agent_c_completion)' not in source
    assert 'st.fragment(run_every="3s")(_refresh_parent_for_new_agent_b_choices)' not in source
    assert '_poll_agent_c_completion()\n            current' in source


def test_diagnostic_launcher_options_preserve_context_and_concurrency(tmp_path, monkeypatch):
    import os, subprocess
    fake = tmp_path/'venvs/healthcare-vllm/bin/vllm'
    fake.parent.mkdir(parents=True)
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    env = dict(os.environ, HOME=str(tmp_path), HDQ_MAX_MODEL_LEN='12288', HDQ_MAX_NUM_SEQS='3', HDQ_ENFORCE_EAGER='1', HDQ_ASYNC_SCHEDULING='0', HDQ_GPU_MEMORY_UTILIZATION='0.85')
    result = subprocess.run(['bash', str(ROOT/'scripts/serve_gemma4.sh')], env=env, capture_output=True, text=True, check=True)
    assert '--enforce-eager' in result.stdout and '--no-async-scheduling' in result.stdout
    assert '--gpu-memory-utilization\n0.85' in result.stdout
    assert '--max-num-seqs\n3' in result.stdout and '--max-model-len\n12288' in result.stdout
