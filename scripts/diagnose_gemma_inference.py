"""Local synthetic Gemma probe. No dataset, prompts from users, or memory are read."""
import argparse
import json
import re
import sys
from pathlib import Path
from time import monotonic
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE = 'http://127.0.0.1:8000'
MODEL = 'gemma4-12b'
METRICS = {'num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc', 'generation_tokens_total'}


def fetch(path, payload=None, timeout=5):
    req = Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as response:
        return response.status, response.read().decode('utf-8')


def metrics():
    try:
        _, raw = fetch('/metrics')
        result = {}
        for line in raw.splitlines():
            m = re.match(r'^vllm:([a-z_]+)(?:\{([^}]*)\})?\s+([\d.eE+-]+)(?:\s|$)', line)
            if m and m[1] in METRICS and (not m[2] or f'model_name="{MODEL}"' in m[2] or 'model_name=' not in m[2]):
                result[m[1]] = result.get(m[1], 0.0) + float(m[3])
        return result
    except (OSError, HTTPError, ValueError) as exc:
        print(f'Metrics unavailable: {type(exc).__name__}')
        return {}


def payload(mode):
    p = {'model': MODEL, 'temperature': 0, 'max_tokens': 64,
         'messages': [{'role':'user','content':'Reply with OK only.'}]}
    if mode == 'plain':
        return p
    props = {'option_id': {'type':'string'}, 'assessment': {'type':'string','enum':['supported','uncertain','reject']},
             'finding_scope_matches': {'type':'boolean'}, 'contains_unrequested_change': {'type':'boolean'},
             'reason': {'type':'string'}}
    item = {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
    if mode == 'schema':
        schema = {'type':'object','properties':{'answer':{'type':'string','enum':['OK']}},'required':['answer'],'additionalProperties':False}
        p['messages'][0]['content'] = 'Return the JSON object {"answer":"OK"} only.'
    else:
        schema = {'type':'object','properties':{'reviews':{'type':'array','items':item,'minItems':1,'maxItems':5}},'required':['reviews'],'additionalProperties':False}
        p['messages'][0]['content'] = ('Return JSON only. Independently assess three synthetic formatting options option_1, option_2, option_3. '
            'Each is a safe formatting-only operation in its own scope. For each, return assessment supported, '
            'finding_scope_matches true, contains_unrequested_change false, and a short nonempty reason. '
            'Use a reviews array and each exact option_id once.')
        p['max_tokens'] = 400
    p['response_format'] = {'type':'json_schema','json_schema':{'name':'structured_response','schema':schema}}
    return p


def valid(mode, text):
    if mode == 'plain':
        return text.strip() == 'OK'
    import jsonschema
    obj = json.loads(text)
    jsonschema.validate(obj, payload(mode)['response_format']['json_schema']['schema'])
    if mode == 'schema':
        return obj == {'answer':'OK'}
    reviews = obj['reviews']
    return (len(reviews) == 3 and {r['option_id'] for r in reviews} == {'option_1','option_2','option_3'}
            and all(r['assessment']=='supported' and r['finding_scope_matches'] is True
                    and r['contains_unrequested_change'] is False and r['reason'].strip() for r in reviews))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all','plain','schema','batch','local-batch','guided-all'], default='all',
                        help='all tests plain and local JSON; guided modes are explicit opt-in')
    parser.add_argument('--timeout', type=float, default=45)
    args=parser.parse_args()
    if not 5 <= args.timeout <= 600:
        parser.error('timeout must be between 5 and 600 seconds')
    try:
        status, _ = fetch('/health')
        print(f'Health: HTTP {status}')
    except (OSError, HTTPError) as exc:
        print(f'Health check failed: {exc}')
        return 2
    modes = (['plain','local-batch'] if args.mode=='all' else
             ['plain','schema','batch'] if args.mode=='guided-all' else [args.mode])
    for mode in modes:
        before = metrics()
        print(f'Before {mode}: {before}', flush=True)
        started = monotonic()
        try:
            request_payload = payload('batch' if mode == 'local-batch' else mode)
            if mode == 'local-batch':
                schema = request_payload.pop('response_format')['json_schema']['schema']
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
                from healthcare_data_quality.services.llm_client import _json_schema_prompt
                request_payload['messages'][0]['content'] = _json_schema_prompt(request_payload['messages'][0]['content'], schema)
            status, raw = fetch('/v1/chat/completions', request_payload, args.timeout)
            response=json.loads(raw); choice=response['choices'][0]
            text = choice['message']['content']
            if mode == 'local-batch' and choice['finish_reason'] == 'stop':
                from healthcare_data_quality.services.llm_client import _validate_json_response
                text = _validate_json_response(text, schema)
            ok = choice['finish_reason']=='stop' and valid('batch' if mode == 'local-batch' else mode, text)
            print(f'{mode}: HTTP {status}, {monotonic()-started:.1f}s, finish={choice["finish_reason"]}, valid={ok}')
            print(f'Usage: {response.get("usage", {})}')
        except Exception as exc:
            print(f'{mode}: FAILED after {monotonic()-started:.1f}s ({type(exc).__name__}: {exc})')
            ok=False
        after = metrics()
        print(f'After {mode}: {after}', flush=True)
        if 'generation_tokens_total' in before and 'generation_tokens_total' in after:
            print(f'Global generated-token delta: {after["generation_tokens_total"]-before["generation_tokens_total"]:.0f}')
        if not ok:
            print('Stopping probes. A timed-out request may still be active; do not blindly resubmit.')
            return 1
    print('Selected probes passed. This does not prove all application requests are healthy.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
