import json
import time
from threading import Event
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from healthcare_data_quality.services import llm_client as llm, model_plan_review as mr
from healthcare_data_quality.jobs.interactive_chat_jobs import InteractiveChatJobs
from healthcare_data_quality.ui.human_choice_replay import memory_first_candidates
from test_review_lookahead_20261124 import setup
from test_automatic_final_actions_20261124 import SRC


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('stage did not arrive')


def test_actual_http_dispatch_and_preflight_are_distinct(monkeypatch, capsys):
    entered, release = Event(), Event()
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            if self.path == '/tokenize':
                result = {'count':1800, 'max_model_len':12288}
            else:
                entered.set()
                release.wait(3)
                result = {'choices':[{'finish_reason':'stop','message':{'content':'ok'}}],
                          'usage':{'prompt_tokens':1800,'completion_tokens':1}}
            payload = json.dumps(result).encode()
            self.send_response(200); self.send_header('Content-Length',str(len(payload))); self.end_headers();self.wfile.write(payload)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    Thread(target=server.serve_forever,daemon=True).start()
    monkeypatch.setattr(llm,'MODEL_URL',f'http://127.0.0.1:{server.server_port}/v1/chat/completions')
    monkeypatch.setattr(llm,'_GATE',llm._RequestGate())
    manager=InteractiveChatJobs()
    try:
        job=manager.submit_role('B',llm.query_local_model,'PRIVATE SOURCE TEXT '*400)
        assert entered.wait(3)
        assert manager.model_stage_for(job)['stage']=='completion_sent'
        assert calls==['/tokenize','/v1/chat/completions']
        release.set()
        wait_until(lambda:manager.phase_for(job)=='done')
        assert manager.take(job)['result']=='ok'
        trace=capsys.readouterr().out
        assert 'tokenized' in trace and 'completion_received' in trace
        assert 'PRIVATE SOURCE TEXT' not in trace
    finally:
        release.set();server.shutdown();server.server_close()


def test_waiting_for_gate_is_not_reported_as_http_dispatch(monkeypatch):
    monkeypatch.setattr(llm,'_GATE',llm._RequestGate())
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE','3')
    manager=InteractiveChatJobs(); calls=[]
    def fake_submit(request):
        calls.append(request)
        return {'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]}
    # Use real gate and fake transport, rather than faking admission.
    import io
    monkeypatch.setattr(llm,'urlopen',lambda req,**kw:io.BytesIO(json.dumps(fake_submit(req)).encode()))
    with llm._GATE.admit('recommendation'):
        job=manager.submit_role('B',llm.query_local_model,'short prompt')
        wait_until(lambda:manager.model_stage_for(job).get('stage')=='waiting_for_inference_slot')
        assert not calls
    wait_until(lambda:manager.phase_for(job)=='done')
    assert len(calls)==1


def test_batch_overflow_retries_compact_scope(monkeypatch):
    state,target,_=setup(monkeypatch);opts=memory_first_candidates(target,SRC);seen=[]
    def model(prompt,**kwargs):
        payload=json.loads(prompt.split('DATA: ',1)[1]);seen.append(payload)
        if len(seen)==1:
            raise RuntimeError('Context budget exceeded')
        return json.dumps({'reviews':[{'option_id':c['option_id'],'assessment':'supported',
            'finding_scope_matches':True,'contains_unrequested_change':False,'reason':'Reasonable optional action'} for c in payload['candidates']]})
    monkeypatch.setattr(mr,'query_local_model',model)
    result=mr.assess_review_batch(target,SRC,opts,review_context={'previous_reviewer_turns':['x'*500]*5})
    assert len(seen)==2 and seen[0]['candidates']==seen[1]['candidates']
    assert len(json.dumps(seen[1]))<len(json.dumps(seen[0]))
    assert len(result)==len(opts) and all(r['can_offer'] for r in result)
