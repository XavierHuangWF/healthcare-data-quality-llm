import pytest
from urllib.error import URLError
from healthcare_data_quality.services import llm_client as llm

@pytest.mark.parametrize('error', [TimeoutError('read timed out'), URLError(TimeoutError('timed out'))])
def test_ambiguous_completion_timeout_is_not_resubmitted(monkeypatch, error):
    calls, stages = [], []
    monkeypatch.setenv('HDQ_GEMMA_AUTO_RECONNECT', '1')
    monkeypatch.setenv('HDQ_GEMMA_REQUEST_TIMEOUT_SECONDS', '30')
    monkeypatch.setattr(llm, '_GATE', llm._RequestGate())
    def transport(req, timeout):
        calls.append((req.full_url, timeout))
        raise error
    monkeypatch.setattr(llm,'urlopen',transport)
    monkeypatch.setattr(llm,'sleep',lambda _:pytest.fail('ambiguous timeout must not retry'))
    with llm.observe_model_requests(lambda stage, metrics:stages.append(stage)):
        with pytest.raises(llm.ModelCompletionTimeout, match='may still be active'):
            llm.query_local_model('short synthetic request')
    assert calls == [(llm.MODEL_URL,30)]
    assert stages == ['waiting_for_inference_slot','completion_sent','inference_timeout']
    assert llm._GATE._active == 0
    assert not llm.model_connection_snapshot()['offline']


def test_completed_response_at_expired_deadline_is_not_resent(monkeypatch):
    import io, json
    clock=[0.]; calls=[]
    monkeypatch.setattr(llm,'monotonic',lambda:clock[0])
    monkeypatch.setenv('HDQ_RECOMMENDATION_BUDGET_SECONDS','15')
    monkeypatch.setenv('HDQ_GEMMA_AUTO_RECONNECT','1')
    monkeypatch.setattr(llm,'_GATE',llm._RequestGate())
    def transport(req,timeout):
        calls.append(timeout); clock[0]=16
        return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'OK'}}]}).encode())
    monkeypatch.setattr(llm,'urlopen',transport)
    monkeypatch.setattr(llm,'sleep',lambda _:pytest.fail('expired budget must not retry'))
    with llm.bounded_recommendation_review():
        with pytest.raises(TimeoutError,match='recommendation assessment timed out'):
            llm.query_local_model('short')
    assert calls==[15] and llm._GATE._active==0


def test_diagnostic_validates_exact_batch_ids():
    from diagnose_gemma_inference import valid
    import json
    reviews=[{'option_id':f'option_{i}','assessment':'supported','finding_scope_matches':True,
              'contains_unrequested_change':False,'reason':'synthetic'} for i in range(1,4)]
    assert valid('batch',json.dumps({'reviews':reviews}))
    reviews[2]['option_id']='option_1'
    assert not valid('batch',json.dumps({'reviews':reviews}))


def test_worker_budget_expires_on_ambiguous_completion_timeout(monkeypatch):
    monkeypatch.setenv('HDQ_RECOMMENDATION_BUDGET_SECONDS','120')
    monkeypatch.setattr(llm,'_GATE',llm._RequestGate())
    calls=[]
    def transport(req,timeout):
        calls.append(timeout)
        raise TimeoutError('stalled')
    monkeypatch.setattr(llm,'urlopen',transport)
    with llm.bounded_recommendation_review():
        with pytest.raises(llm.ModelCompletionTimeout):
            llm.query_local_model('short')
        with pytest.raises(TimeoutError,match='recommendation assessment timed out'):
            llm.query_local_model('alternate draft')
    assert len(calls)==1 and llm._REVIEW_DEADLINE.get() is None


def test_diagnostic_stops_after_first_failed_completion(monkeypatch, capsys):
    import diagnose_gemma_inference as probe
    import sys
    calls=[]
    def fetch(path,payload=None,timeout=5):
        calls.append(path)
        if path=='/health': return 200,''
        if path=='/metrics': return 200,'vllm:generation_tokens_total{model_name="gemma4-12b"} 100\n'
        assert payload is not None and 'response_format' not in payload
        raise TimeoutError('test stall')
    monkeypatch.setattr(probe,'fetch',fetch)
    monkeypatch.setattr(sys,'argv',['diagnose_gemma_inference.py'])
    assert probe.main()==1
    assert calls.count('/v1/chat/completions')==1
    assert 'Global generated-token delta: 0' in capsys.readouterr().out
