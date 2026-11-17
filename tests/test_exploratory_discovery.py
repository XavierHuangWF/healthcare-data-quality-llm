"""Offline regression coverage for the exploratory agent; no live Gemma needed."""

import json
from pathlib import Path

import pytest

from healthcare_data_quality import exploratory_agent as agent
from healthcare_data_quality import exploratory_sandbox as sandbox


GOOD_CODE = '''from statistics import mean

def investigate(datasets):
    rows = datasets["dataset_1"]["rows"]
    values = [float(row[1]) for row in rows if len(row) > 1 and row[1]]
    average = mean(values) if values else 0
    return {"summary": "Average measured: " + str(round(average, 2)), "findings": [
        {"title": "Unexpected value relationship", "description": "Follow up with a domain scientist.",
         "evidence": [{"dataset_id": "dataset_1", "csv_row": 2,
                       "column": 2, "value": rows[0][1]}]}
    ]}
'''


def _files(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("sample,count,label\nA,3,NA\nB,4,unclassified\n", encoding="utf-8")
    b.write_text("id,batch\nA,1\nB,2\n", encoding="utf-8")
    return {"dataset_1": a, "dataset_2": b}


def _result(value="3"):
    return {"summary": "A candidate pattern was detected", "findings": [{
        "title": "Unexpected pattern", "description": "An association deserves review.",
        "evidence": [{"dataset_id": "dataset_1", "csv_row": 2, "column": 2, "value": value}],
    }]}


def _proposal(code=GOOD_CODE):
    return {"hypothesis": "Hidden batch pattern", "rationale": "May reveal a mismatch",
            "method": "Read all rows and compare batches", "assumptions": "Batch ID is meaningful",
            "code": code}


def test_profile_uses_all_rows_and_preserves_column_positions(tmp_path):
    datasets = _files(tmp_path)
    profile = agent.profile_datasets(datasets)
    assert profile["dataset_1"]["rows"] == 2
    assert profile["dataset_1"]["columns"] == 3
    assert profile["dataset_1"]["column_profile"][2]["examples"] == ["NA", "unclassified"]
    assert profile["dataset_2"]["column_profile"][1]["numeric"] == 2


@pytest.mark.parametrize("code", [
    "import os\n" + GOOD_CODE,
    "def investigate(datasets):\n    open('/etc/passwd')\n",
    "def investigate(datasets):\n    return datasets.__class__\n",
    "def investigate(datasets):\n    import subprocess\n    return {}\n",
    "def investigate(datasets):\n    eval('1+1')\n",
    "print('top-level')\n" + GOOD_CODE,
    "def wrong(datasets): return {}",
])
def test_rejects_dangerous_or_ill_formed_code(code):
    with pytest.raises(ValueError):
        agent.validate_generated_code(code)


def test_allows_dynamic_stdlib_code():
    agent.validate_generated_code(GOOD_CODE)


@pytest.mark.parametrize("code", [
    "def investigate(datasets):\n    return {'zero_frequencies': []}\n",
    "def investigate(datasets):\n    print('complete')\n",
    "def investigate(datasets):\n    result = {'summary': 'x', 'findings': []}\n    return result\n",
    "def investigate(datasets):\n    return {'summary': 'x', 'issues': []}\n",
    "def investigate(datasets):\n    if datasets:\n        return []\n    return {'summary': 'x', 'findings': []}\n",
])
def test_bad_output_contract_is_rejected_before_approval(code):
    with pytest.raises(ValueError, match="investigate must"):
        agent.validate_generated_code(code)


def test_runtime_contract_errors_are_specific(tmp_path):
    datasets = _files(tmp_path)
    with pytest.raises(ValueError, match="text 'summary'"):
        agent.verify_result({"summary": 12, "findings": []}, datasets)
    with pytest.raises(ValueError, match="list named 'findings'"):
        agent.verify_result({"summary": "Result", "findings": None}, datasets)


def test_repair_produces_new_valid_code_without_execution(monkeypatch):
    broken = _proposal("def investigate(datasets):\n    return {'count': 0}\n")
    prompts = []

    def model(prompt, **kwargs):
        prompts.append(prompt)
        assert kwargs["json_schema"] == agent.PROPOSAL_SCHEMA
        return json.dumps(_proposal())

    monkeypatch.setattr(agent, "query_local_model", model)
    monkeypatch.setattr(agent, "run_in_sandbox",
                        lambda *_args: pytest.fail("Repair must never run generated code"))
    repaired = agent.repair_exploration(broken, "Missing summary and findings")
    assert repaired["code"] == GOOD_CODE
    assert "Missing summary and findings" in prompts[0]


def test_initial_generation_repairs_bad_static_contract_once(tmp_path, monkeypatch):
    calls = []

    def model(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(_proposal("def investigate(datasets):\n    return {'count': 0}\n")
                          if len(calls) == 1 else _proposal())

    monkeypatch.setattr(agent, "query_local_model", model)
    monkeypatch.setattr(agent, "run_in_sandbox",
                        lambda *_args: pytest.fail("No code should execute during generation"))
    assert agent.propose_exploration(_files(tmp_path), {}, [])["code"] == GOOD_CODE
    assert len(calls) == 2


def test_repair_must_not_accept_unchanged_script(monkeypatch):
    original = _proposal()
    monkeypatch.setattr(agent, "query_local_model",
                        lambda *_args, **_kwargs: json.dumps(original))
    with pytest.raises(ValueError, match="unchanged"):
        agent.repair_exploration(original, "runtime failure")


def test_rejects_fabricated_source_cells(tmp_path):
    datasets = _files(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        agent.verify_result(_result(value="99"), datasets)
    with pytest.raises(ValueError, match="outside"):
        bad = _result()
        bad["findings"][0]["evidence"][0]["csv_row"] = 999
        agent.verify_result(bad, datasets)


def test_rejects_findings_without_exact_evidence(tmp_path):
    datasets = _files(tmp_path)
    bad = _result()
    bad["findings"][0]["evidence"] = []
    with pytest.raises(ValueError, match="citations"):
        agent.verify_result(bad, datasets)


def test_no_findings_is_allowed_but_not_clean_certification(tmp_path):
    assert agent.verify_result({"summary": "No hypothesis supported", "findings": []},
                               _files(tmp_path)) == {"summary": "No hypothesis supported", "findings": []}


def test_code_executes_twice_with_independent_source_verification(tmp_path, monkeypatch):
    datasets = _files(tmp_path)
    calls = []
    def isolated(code, data):
        calls.append((code, list(data)))
        return _result()
    monkeypatch.setattr(agent, "run_in_sandbox", isolated)
    completed = agent.execute_exploration(_proposal(), datasets)
    assert len(calls) == 2
    assert completed["verification"]["sandbox_runs"] == 2
    assert completed["verification"]["scientific_method_validated"] is False
    assert set(completed["verification"]["dataset_sha256"]) == {"dataset_1", "dataset_2"}


def test_nonreproducible_analysis_rejected(tmp_path, monkeypatch):
    calls = 0
    def isolated(*_):
        nonlocal calls
        calls += 1
        x = _result()
        x["summary"] = str(calls)
        return x
    monkeypatch.setattr(agent, "run_in_sandbox", isolated)
    with pytest.raises(ValueError, match="not reproducible"):
        agent.execute_exploration(_proposal(), _files(tmp_path))


def test_gemma_proposes_new_code_and_receives_prior_results(tmp_path, monkeypatch):
    proposed = _proposal()
    prompt_seen = []
    def local_model(prompt, **kwargs):
        prompt_seen.append(prompt)
        if kwargs["json_schema"] == agent.CONTINUATION_SCHEMA:
            return json.dumps({"decision": "continue", "reason": "New cross-dataset question",
                               "remaining_questions": []})
        assert kwargs["json_schema"] == agent.PROPOSAL_SCHEMA
        return json.dumps(proposed)
    monkeypatch.setattr(agent, "query_local_model", local_model)
    prior = [{"proposal": {"hypothesis": "Prior anomaly"},
              "result": {"summary": "Earlier result", "findings": []}}]
    result = agent.propose_exploration(_files(tmp_path), {"dataset_baselines": {}}, prior)
    assert result == proposed
    assert len(prompt_seen) == 2  # Gemma continuation decision, then hypothesis
    assert "Prior anomaly" in prompt_seen[0]
    assert "dataset_2" in prompt_seen[0]
    assert "code" in prompt_seen[1]


def test_gemma_ill_formed_proposal_never_runs_code(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "query_local_model", lambda *_args, **_kwargs: "{}")
    with pytest.raises(ValueError, match="omitted"):
        agent.propose_exploration(_files(tmp_path), {}, [])


def test_round_limit_prevents_further_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "query_local_model", lambda *_args, **_kwargs: pytest.fail("Must not call model"))
    with pytest.raises(ValueError, match="safety ceiling reached"):
        agent.propose_exploration(_files(tmp_path), {}, [{}] * agent.MAX_ROUNDS)


def test_sandbox_fails_closed_when_bwrap_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "sandbox_available", lambda: False)
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *_a, **_kw: pytest.fail("No host execution"))
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.run_in_sandbox(GOOD_CODE, _files(tmp_path))


def test_bwrap_never_binds_host_home_or_root(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda *_: "/usr/bin/bwrap")
    cmd = sandbox._bwrap_command(tmp_path / "datasets", tmp_path / "analysis")
    assert "--unshare-all" in cmd and "--clearenv" in cmd
    assert "--ro-bind" in cmd
    assert not any(cmd[i:i + 3] == ["--ro-bind", "/", "/"]
                   for i in range(len(cmd)))
    assert "/home/xavier" not in cmd
    assert "--share-net" not in cmd
    assert ["/usr/bin/python3", "-I", "-B", "/analysis/runner.py"] == cmd[-4:]

# Regression: local Gemma generated `collections.defaultdict` without importing
# `collections`, causing a NameError only after human approval and sandbox run.
MISSING_IMPORT_CODE = '''def investigate(datasets):
    name_to_indices = collections.defaultdict(list)
    return {'summary': str(name_to_indices), 'findings': []}
'''


def test_missing_global_module_is_rejected_before_approval():
    with pytest.raises(ValueError, match="undefined global name.*collections"):
        agent.validate_generated_code(MISSING_IMPORT_CODE)


def test_explicit_collections_import_passes_preflight():
    agent.validate_generated_code("import collections\n" + MISSING_IMPORT_CODE)
    agent.validate_generated_code(
        "from collections import defaultdict\n"
        + MISSING_IMPORT_CODE.replace("collections.defaultdict", "defaultdict")
    )


def test_initial_generation_repairs_missing_import_before_approval(tmp_path, monkeypatch):
    scripts = [MISSING_IMPORT_CODE, "import collections\n" + MISSING_IMPORT_CODE]
    calls = []

    def model(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(_proposal(scripts.pop(0)))

    monkeypatch.setattr(agent, "query_local_model", model)
    monkeypatch.setattr(agent, "run_in_sandbox",
                        lambda *_args: pytest.fail("No execution permitted during repair"))
    result = agent.propose_exploration(_files(tmp_path), {}, [])
    assert result["code"].startswith("import collections")
    assert len(calls) == 2


def test_hypothesis_prompt_includes_existing_duplicate_findings(tmp_path, monkeypatch):
    prompts = []

    def model(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(_proposal())

    monkeypatch.setattr(agent, "query_local_model", model)
    agent.propose_exploration(_files(tmp_path), {
        "ingestion": {"duplicate_headers": {"ABC99": [105, 106]},
                      "blank_header_positions": [1], "ragged_records": []},
        "dataset_baselines": {"dataset_1": {"checks": {
            "find_invalid_counts": {"status": "completed", "observations": [1, 2]}
        }}},
    }, [])
    assert "ABC99" in prompts[0]
    assert "105" in prompts[0]
    assert "identical columns" in prompts[0]
    assert '"observations": 2' in prompts[0]


def test_auto_repairs_missing_import_then_returns_only_verified_result(tmp_path, monkeypatch):
    """The exact undefined `collections` failure seen in the user's UI."""
    original = _proposal(MISSING_IMPORT_CODE)
    repaired_code = 'import collections\n' + MISSING_IMPORT_CODE
    model_prompts = []
    sandbox_calls = []

    def model(prompt, **kwargs):
        model_prompts.append(prompt)
        return json.dumps(_proposal(repaired_code))

    def sandbox_run(code, inputs):
        sandbox_calls.append(code)
        assert code == repaired_code
        return _result()

    monkeypatch.setattr(agent, 'query_local_model', model)
    monkeypatch.setattr(agent, 'run_in_sandbox', sandbox_run)
    report = agent.execute_with_auto_repair(original, _files(tmp_path))
    assert len(model_prompts) == 1
    assert len(sandbox_calls) == 2  # both verification runs are sandboxed
    assert report['verification']['approved_auto_repairs'] == 1
    assert report['verification']['analysis_attempts'] == 2
    assert report['original_proposal']['code'] == MISSING_IMPORT_CODE
    assert report['result']['findings'][0]['evidence'][0]['value'] == '3'


def test_auto_repairs_runtime_output_contract_without_leaking_failed_result(tmp_path, monkeypatch):
    original = _proposal()
    corrected = _proposal(GOOD_CODE + '\n# repaired code\n')
    sandbox_calls = []

    def sandbox_run(code, inputs):
        sandbox_calls.append(code)
        if code == original['code']:
            return {'summary': 13, 'findings': 'wrong type'}
        return _result()

    monkeypatch.setattr(agent, 'run_in_sandbox', sandbox_run)
    monkeypatch.setattr(agent, 'query_local_model',
                        lambda *_args, **_kwargs: json.dumps(corrected))
    report = agent.execute_with_auto_repair(original, _files(tmp_path))
    assert sandbox_calls.count(original['code']) == 1
    assert sandbox_calls.count(corrected['code']) == 2
    assert isinstance(report['result']['summary'], str)
    assert report['verification']['approved_auto_repairs'] == 1
    assert len(report['verification']['rejected_attempts']) == 1
    assert 'summary' not in str(report['verification']['rejected_attempts'])


def test_auto_repair_exhausts_after_three_failed_executions(tmp_path, monkeypatch):
    original = _proposal()
    serial = iter([1, 2])
    monkeypatch.setattr(agent, 'query_local_model', lambda *_args, **_kw: json.dumps(
        _proposal(GOOD_CODE + f'\n# attempt {next(serial)}\n')
    ))
    sandbox_calls = []

    def fail(code, _datasets):
        sandbox_calls.append(code)
        raise sandbox.SandboxExecutionError(
            'Isolated analysis failed (exit code 1). Sandbox runtime error: Traceback: NameError'
        )

    monkeypatch.setattr(agent, 'run_in_sandbox', fail)
    with pytest.raises(agent.AutoRepairExhausted) as info:
        agent.execute_with_auto_repair(original, _files(tmp_path))
    assert len(sandbox_calls) == 3
    assert len(info.value.diagnostics) == 3
    assert 'verified result' in str(info.value)


def test_auto_repair_never_bypasses_sandbox_or_nonreproducible_output(tmp_path, monkeypatch):
    original = _proposal()
    monkeypatch.setattr(agent, 'query_local_model', lambda *_args, **_kw: pytest.fail('No repair'))
    monkeypatch.setattr(agent, 'run_in_sandbox', lambda *_args, **_kw: (
        (_ for _ in ()).throw(sandbox.SandboxUnavailable('bwrap missing'))
    ))
    with pytest.raises(sandbox.SandboxUnavailable):
        agent.execute_with_auto_repair(original, _files(tmp_path))

    calls = [0]

    def vary(*_):
        calls[0] += 1
        x = _result()
        x['summary'] = str(calls[0])
        return x

    monkeypatch.setattr(agent, 'run_in_sandbox', vary)
    with pytest.raises(agent.AutoRepairExhausted, match='repair budget'):
        agent.execute_with_auto_repair(original, _files(tmp_path))
    assert calls[0] == 2


def test_auto_repair_of_existing_failed_proposal_skips_old_code(tmp_path, monkeypatch):
    original = _proposal(MISSING_IMPORT_CODE)
    corrected = 'import collections\n' + MISSING_IMPORT_CODE
    monkeypatch.setattr(agent, 'query_local_model',
                        lambda *_args, **_kwargs: json.dumps(_proposal(corrected)))
    executions = []

    def success(code, datasets):
        executions.append(code)
        return _result()

    monkeypatch.setattr(agent, 'run_in_sandbox', success)
    result = agent.execute_with_auto_repair(
        original, _files(tmp_path), initial_failure="NameError: name 'collections' is not defined"
    )
    assert executions == [corrected, corrected]
    assert result['verification']['approved_auto_repairs'] == 1
    assert result['verification']['analysis_attempts'] == 1


def test_auto_repair_cannot_change_approved_hypothesis(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, 'query_local_model', lambda *_args, **kwargs: json.dumps({
        **_proposal('import collections\n' + MISSING_IMPORT_CODE),
        'hypothesis': 'A different scientific question',
    }))
    monkeypatch.setattr(agent, 'run_in_sandbox', lambda *_args, **_kw: pytest.fail('Not approved'))
    with pytest.raises(agent.AutoRepairExhausted, match='repair budget'):
        agent.execute_with_auto_repair(_proposal(MISSING_IMPORT_CODE), _files(tmp_path), max_repairs=1)


def test_sandbox_preserves_last_python_exception_for_local_repair(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(sandbox, 'sandbox_available', lambda: True)
    monkeypatch.setattr(sandbox, '_bwrap_command', lambda *_args: ['bwrap', 'fake'])

    def fake_run(*_args, stdout, stderr, **_kwargs):
        stderr.write(b'Traceback (most recent call last):\n')
        stderr.write(b'  File "/analysis/proposal.py", line 25, in investigate\n')
        stderr.write(b'    name_to_indices = collections.defaultdict(list)\n')
        stderr.write(b"NameError: name 'collections' is not defined\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(sandbox.subprocess, 'run', fake_run)
    with pytest.raises(sandbox.SandboxGeneratedCodeError, match='collections') as result:
        sandbox.run_in_sandbox(GOOD_CODE, _files(tmp_path))
    assert agent._repairable_error(result.value)


def test_sandbox_infrastructure_failure_does_not_trigger_gemma_repair(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(sandbox, 'sandbox_available', lambda: True)
    monkeypatch.setattr(sandbox, '_bwrap_command', lambda *_args: ['bwrap', 'fake'])

    def fail_bwrap(*_args, stdout, stderr, **_kwargs):
        stderr.write(b'bwrap: cannot create new user namespace: Operation not permitted')
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(sandbox.subprocess, 'run', fail_bwrap)
    with pytest.raises(sandbox.SandboxExecutionError) as result:
        sandbox.run_in_sandbox(GOOD_CODE, _files(tmp_path))
    assert not isinstance(result.value, sandbox.SandboxGeneratedCodeError)
    assert not agent._repairable_error(result.value)
