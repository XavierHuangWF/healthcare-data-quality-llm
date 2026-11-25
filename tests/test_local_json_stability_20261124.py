import io
import json
import pytest
from healthcare_data_quality.services import llm_client as llm

SCHEMA = {'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}

@pytest.fixture
def transport(monkeypatch):
    calls=[]
    response={'content':'{"ok":true}'}
    def fake(req, timeout):
        calls.append(json.loads(req.data))
        return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':response}]}).encode())
    monkeypatch.setattr(llm,'urlopen',fake)
    monkeypatch.setattr(llm,'_GATE',llm._RequestGate())
    monkeypatch.delenv('HDQ_GEMMA_JSON_MODE',raising=False)
    return calls,response


def test_default_does_not_request_guided_decoding(transport):
    calls,_=transport
    assert json.loads(llm.query_local_model('TASK DATA',json_schema=SCHEMA))=={'ok':True}
    assert len(calls)==1 and 'response_format' not in calls[0]
    assert 'TASK DATA' in calls[0]['messages'][0]['content']
    assert '"additionalProperties":false' in calls[0]['messages'][0]['content']


@pytest.mark.parametrize('bad',['{"ok":"true"}','{"ok":true,"extra":1}','{"ok":true,"ok":false}','{"ok":','{"ok":NaN}','prefix {"ok":true}'])
def test_invalid_output_fails_closed(transport,bad):
    calls,response=transport; response['content']=bad
    with pytest.raises(ValueError,match='schema validation'):
        llm.query_local_model('TASK',json_schema=SCHEMA)
    assert len(calls)==1


def test_guided_mode_is_explicit_and_validated(monkeypatch,transport):
    calls,response=transport; monkeypatch.setenv('HDQ_GEMMA_JSON_MODE','guided')
    assert json.loads(llm.query_local_model('TASK',json_schema=SCHEMA))=={'ok':True}
    assert calls[0]['response_format']['json_schema']['schema']==SCHEMA
    response['content']='{"ok":42}'
    with pytest.raises(ValueError): llm.query_local_model('TASK',json_schema=SCHEMA)


def test_plain_text_unchanged(transport):
    calls,response=transport; response['content']='OK'
    assert llm.query_local_model('TASK')=='OK'
    assert calls[0]['messages'][0]['content']=='TASK'


def test_whole_json_fence_supported(transport):
    _,response=transport; response['content']='```json\n{"ok":true}\n```'
    assert json.loads(llm.query_local_model('TASK',json_schema=SCHEMA))=={'ok':True}


def test_stability_launcher_defaults_keep_three_sequences(tmp_path, monkeypatch):
    import os, subprocess
    from pathlib import Path
    executable = tmp_path/'venvs/healthcare-vllm/bin/vllm'
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    env = {k:v for k,v in os.environ.items() if not k.startswith('HDQ_')}
    env['HOME'] = str(tmp_path)
    launcher = Path(__file__).resolve().parents[1]/'scripts/serve_gemma4.sh'
    output = subprocess.check_output(['bash',str(launcher)],env=env,text=True)
    assert '--no-async-scheduling' in output
    assert '--max-num-seqs\n3' in output and '--max-model-len\n12288' in output
    assert '--enforce-eager' not in output
    env['HDQ_ASYNC_SCHEDULING']='1'
    assert '\n--async-scheduling\n' in subprocess.check_output(['bash',str(launcher)],env=env,text=True)


def test_diagnostic_default_avoids_guided_schema(monkeypatch):
    import sys
    import diagnose_gemma_inference as probe
    calls=[]
    def fetch(path,payload=None,timeout=5):
        if path=='/health': return 200,''
        if path=='/metrics': return 200,''
        calls.append(payload)
        assert 'response_format' not in payload
        content = 'OK' if len(calls)==1 else json.dumps({'reviews':[
            {'option_id':f'option_{i}','assessment':'supported','finding_scope_matches':True,
             'contains_unrequested_change':False,'reason':'synthetic'} for i in range(1,4)]})
        return 200,json.dumps({'choices':[{'finish_reason':'stop','message':{'content':content}}]})
    monkeypatch.setattr(probe,'fetch',fetch)
    monkeypatch.setattr(sys,'argv',['diagnose_gemma_inference.py'])
    assert probe.main()==0 and len(calls)==2


def test_actual_batch_transport_preserves_option_binding(monkeypatch):
    from healthcare_data_quality.services import model_plan_review as mr
    from healthcare_data_quality.ui.human_choice_replay import memory_first_candidates
    from test_review_lookahead_20261124 import setup
    from test_automatic_final_actions_20261124 import SRC
    _,target,_=setup(monkeypatch)
    options=memory_first_candidates(target,SRC)
    assert len(options)>=2
    calls=[]
    monkeypatch.delenv('HDQ_GEMMA_JSON_MODE',raising=False)
    monkeypatch.setattr(llm,'_GATE',llm._RequestGate())
    def fake(req, timeout):
        payload=json.loads(req.data)
        if req.full_url.endswith('/tokenize'):
            assert 'output schema' in payload['messages'][0]['content']
            return io.BytesIO(json.dumps({'count':3500,'max_model_len':12288}).encode())
        calls.append(payload)
        assert 'response_format' not in payload
        candidates=json.loads(payload['messages'][0]['content'].split('DATA: ',1)[1])['candidates']
        reviews=[{'option_id':c['option_id'],'assessment':'supported','finding_scope_matches':True,
                  'contains_unrequested_change':False,'reason':'synthetic assessment'} for c in reversed(candidates)]
        return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'reviews':reviews})}}]}).encode())
    monkeypatch.setattr(llm,'urlopen',fake)
    reviews=mr.assess_review_batch(target,SRC,options)
    assert len(calls)==1 and len(reviews)==len(options)
    assert [r['fingerprint'] for r in reviews]==[mr.option_fingerprint(target,SRC,o) for o in options]
    assert all(r['can_offer'] for r in reviews)
