"""Regression tests for sparse microbiome count-matrix hypothesis selection.

No local LLM or Bubblewrap installation is needed for these tests.
"""
import csv
import json
from pathlib import Path

import pytest

from healthcare_data_quality import exploratory_agent as agent


VALID_CODE = '''def investigate(datasets):
    summary = 'Checked cross-dataset consistency'
    findings = []
    return {'summary': str(summary), 'findings': findings}
'''


def microbiome_inputs(tmp_path: Path) -> dict[str, Path]:
    counts = tmp_path / "taxa_counts.csv"
    with counts.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["", "ABC1", "ABC2", "ABC3", "ABC4", "P7_Blank"])
        for rank in range(12):
            writer.writerow([
                f"Bacteria; Firmicutes; Clostridia; Order_{rank}; Family; Genus; NA",
                0 if rank % 3 else 17, 0, 0 if rank % 2 else 3, rank, 0,
            ])
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "SampleID,Group,Age\nABC1,Case,31\nABC2,Control,29\nABC3,Case,35\n",
        encoding="utf-8",
    )
    return {"dataset_1": counts, "dataset_2": metadata}


def candidate(hypothesis: str, method: str, assumptions="Uses only supplied data") -> dict:
    return {
        "hypothesis": hypothesis,
        "rationale": "Evaluate a reproducible relationship rather than assuming an error.",
        "method": method,
        "assumptions": assumptions,
        "code": VALID_CODE,
    }


def bad_zero_candidate() -> dict:
    return candidate(
        "ABC1 through ABC4 are sequential experimental replicates with high zero-inflation and a floor effect",
        "Calculate the percentage of '0' values in every ABC column and flag >80% zeros",
        "ABC-prefixed columns represent experimental replicates",
    )


def good_cross_dataset_candidate() -> dict:
    return candidate(
        "Some abundance-matrix sample IDs may be absent from the supplied clinical metadata",
        "Compare the complete sample-column names with the SampleID values in the second dataset, preserving originals",
        "Column names identify samples, and any proposed matches require confirmation",
    )


def test_microbiome_role_uses_taxonomy_content_not_abc_prefix(tmp_path):
    datasets = microbiome_inputs(tmp_path)
    profiles = agent.profile_datasets(datasets)
    role = profiles["dataset_1"]["inferred_role"]
    assert role["type"] == "microbiome_count_matrix"
    assert "rows_are_taxa" in role["orientation"]
    assert role["evidence"]["taxonomy_like_first_column_rows"] == 12
    assert profiles["dataset_2"]["inferred_role"]["type"] == "unknown_tabular_data"


def test_actual_uploaded_bug_dataset_is_recognized_if_available():
    path = Path(__file__).resolve().parents[1] / "data" / "dirtyL7PVDforXav.csv"
    if not path.exists():
        pytest.skip("Only needed when the reference test dataset is present")
    profile = agent.profile_datasets({"dataset_1": path})["dataset_1"]
    assert profile["rows"] == 158
    assert profile["inferred_role"]["type"] == "microbiome_count_matrix"


def test_gemma_rejects_zero_frequency_only_and_generates_different_question(tmp_path, monkeypatch):
    inputs = microbiome_inputs(tmp_path)
    alternatives = [bad_zero_candidate(), good_cross_dataset_candidate()]
    prompts = []

    def fake_gemma(prompt, **_):
        prompts.append(prompt)
        return json.dumps(alternatives.pop(0))

    monkeypatch.setattr(agent, "query_local_model", fake_gemma)
    monkeypatch.setattr(agent, "run_in_sandbox", lambda *_: pytest.fail("Must not execute a proposal"))
    result = agent.propose_exploration(inputs, {}, [])
    assert result["hypothesis"] == good_cross_dataset_candidate()["hypothesis"]
    assert len(prompts) == 2
    assert "DOMAIN-CRITICAL" in prompts[0]
    assert "ABC-prefixed COLUMN names are sample IDs" in prompts[0]
    assert "ABC-prefixed sample IDs" in prompts[1] or "sparse microbiome count matrix" in prompts[1]
    assert "REJECTED on scientific grounds" in prompts[1]


def test_repeated_zero_only_proposals_fail_closed(tmp_path, monkeypatch):
    datasets = microbiome_inputs(tmp_path)
    calls = []

    def model(prompt, **_):
        calls.append(prompt)
        return json.dumps(bad_zero_candidate())

    monkeypatch.setattr(agent, "query_local_model", model)
    monkeypatch.setattr(agent, "run_in_sandbox", lambda *_: pytest.fail("Must not execute"))
    with pytest.raises(ValueError, match="scientifically unsupported"):
        agent.propose_exploration(datasets, {}, [])
    assert len(calls) == 4  # one attempt and three bounded retries


def test_zero_mentioned_in_metadata_conditional_hypothesis_is_not_blanket_banned(tmp_path):
    profiles = agent.profile_datasets(microbiome_inputs(tmp_path))
    conditional = candidate(
        "Unexpected zero count distribution conditional on sequenced sample batch in supplied metadata",
        "Compare zero proportions across independently labelled sequencing runs or batches, while adjusting for sequencing depth",
    )
    assert agent._semantic_proposal_issue(conditional, profiles) is None


def test_generic_tabular_data_not_subject_to_microbiome_only_rule(tmp_path):
    generic = tmp_path / "measurements.csv"
    generic.write_text("time,measurement\n0,0\n1,0\n2,1\n", encoding="utf-8")
    profile = agent.profile_datasets({"dataset_1": generic})
    assert profile["dataset_1"]["inferred_role"]["type"] == "unknown_tabular_data"
    assert agent._semantic_proposal_issue(bad_zero_candidate(), profile) is None


def test_bad_json_recovers_without_execution(tmp_path, monkeypatch):
    inputs = microbiome_inputs(tmp_path)
    responses = ["{not json", json.dumps(good_cross_dataset_candidate())]
    monkeypatch.setattr(agent, "query_local_model", lambda *_a, **_kw: responses.pop(0))
    monkeypatch.setattr(agent, "run_in_sandbox", lambda *_a: pytest.fail("Must not run"))
    result = agent.propose_exploration(inputs, {}, [])
    assert result["hypothesis"] == good_cross_dataset_candidate()["hypothesis"]
    assert not responses


def test_missing_fields_recovers_without_execution(tmp_path, monkeypatch):
    inputs = microbiome_inputs(tmp_path)
    responses = ["{}", json.dumps(good_cross_dataset_candidate())]
    monkeypatch.setattr(agent, "query_local_model", lambda *_a, **_kw: responses.pop(0))
    monkeypatch.setattr(agent, "run_in_sandbox", lambda *_a: pytest.fail("Must not run"))
    assert agent.propose_exploration(inputs, {}, [])["hypothesis"] == good_cross_dataset_candidate()["hypothesis"]


def test_unrepairable_generated_code_recovers_with_new_proposal(tmp_path, monkeypatch):
    inputs = microbiome_inputs(tmp_path)
    bad = good_cross_dataset_candidate().copy()
    bad["code"] = "import os\ndef investigate(datasets):\n    return {'summary': 'x', 'findings': []}"
    responses = [json.dumps(bad), json.dumps(good_cross_dataset_candidate())]
    monkeypatch.setattr(agent, "query_local_model", lambda *_a, **_kw: responses.pop(0))
    monkeypatch.setattr(agent, "repair_exploration", lambda *_a: (_ for _ in ()).throw(ValueError("repair failed")))
    monkeypatch.setattr(agent, "run_in_sandbox", lambda *_a: pytest.fail("Must not run"))
    assert agent.propose_exploration(inputs, {}, [])["hypothesis"] == good_cross_dataset_candidate()["hypothesis"]


def test_truncated_model_completion_recovers(tmp_path, monkeypatch):
    inputs = microbiome_inputs(tmp_path)
    responses = [RuntimeError("Incomplete generation: length"), json.dumps(good_cross_dataset_candidate())]
    def model(*_args, **_kwargs):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(agent, "query_local_model", model)
    assert agent.propose_exploration(inputs, {}, [])["hypothesis"] == good_cross_dataset_candidate()["hypothesis"]


def test_model_unavailable_is_not_hidden_by_retries(tmp_path, monkeypatch):
    inputs = microbiome_inputs(tmp_path)
    calls = []
    def fail(*_a, **_k):
        calls.append(1)
        raise RuntimeError("vLLM HTTP 503")
    monkeypatch.setattr(agent, "query_local_model", fail)
    with pytest.raises(RuntimeError, match="503"):
        agent.propose_exploration(inputs, {}, [])
    assert len(calls) == 1


def test_repeated_bad_format_fails_closed_with_actionable_reason(tmp_path, monkeypatch):
    inputs = microbiome_inputs(tmp_path)
    calls = []
    def model(*_a, **_k):
        calls.append(1)
        return "{not json"
    monkeypatch.setattr(agent, "query_local_model", model)
    monkeypatch.setattr(agent, "run_in_sandbox", lambda *_a: pytest.fail("Must not run"))
    with pytest.raises(ValueError, match="Last rejection: Invalid proposal JSON"):
        agent.propose_exploration(inputs, {}, [])
    assert len(calls) == agent.MAX_HYPOTHESIS_RETRIES + 1
