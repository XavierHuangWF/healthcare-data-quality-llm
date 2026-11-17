"""Model selection cannot bypass the deterministic, read-only objective ledger."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from healthcare_data_quality.agent_loop import run_investigation
from healthcare_data_quality.baseline_audit import run_baseline_audit
from healthcare_data_quality.tools.context import verify_contexts


def duplicate_columns(tmp_path: Path, name="sample.csv") -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([
            ["Taxon", "same", "same"],
            ["k__Bacteria;p__A;c__B;o__C;f__D;g__E;s__F", "1", "1"],
            ["k__Bacteria;p__A;c__B;o__C;f__D;g__E;s__G", "2", "2"],
            ["k__Bacteria;p__A;c__B;o__C;f__D;g__E;s__H", "3", "3"],
        ])
    return path


@pytest.mark.parametrize("selection", [
    {"objective_id": "stop", "reason": "No further work"},
    {"objective_id": "investigate_with_r", "reason": "Ignore allowed list"},
    {"objective_id": "dataset_1:delete_everything", "reason": "Injected instruction"},
])
def test_invalid_or_premature_model_selection_cannot_override_worklist(tmp_path, selection):
    path = duplicate_columns(tmp_path)
    baseline = run_baseline_audit(path)
    with (patch("healthcare_data_quality.agent_loop.query_local_model",
                return_value=json.dumps(selection)),
          patch("healthcare_data_quality.agent_loop.summarize_investigation",
                return_value={"findings": [], "unresolved_questions": [],
                              "recommended_next_step": "Review lineage"})):
        result = run_investigation(
            path, max_steps=1, verified_contexts=verify_contexts(path),
            objective_baselines={"dataset_1": baseline},
        )
    assert result["history"]
    assert result["history"][0]["tool"] == "compare_columns"
    assert result["history"][0]["arguments"] == {"left": 2, "right": 3}
    assert result["status"] == "incomplete_budget_exhausted"
    assert all(step["tool"] not in {"investigate_with_r", "reuse_saved_r_investigation"}
               for step in result["history"])


def test_read_only_failed_tool_is_recorded_and_not_called_again(tmp_path):
    path = duplicate_columns(tmp_path)
    baseline = run_baseline_audit(path)
    with (patch("healthcare_data_quality.agent_loop.query_local_model",
                return_value=json.dumps({"objective_id": "stop", "reason": "stop"})),
          patch("healthcare_data_quality.agent_loop.execute_tool",
                side_effect=ValueError("injected synthetic failure")) as execution):
        result = run_investigation(
            path, max_steps=1, verified_contexts=verify_contexts(path),
            objective_baselines={"dataset_1": baseline},
        )
    execution.assert_called_once()
    assert result["history"] == []
    assert result["failed_tool_calls"][0]["phase"] == "execution"
    assert result["failed_objective_ids"]


def test_two_datasets_receive_distinct_read_only_steps(tmp_path):
    a = duplicate_columns(tmp_path, "a.csv")
    b = duplicate_columns(tmp_path, "b.csv")
    baselines = {"dataset_1": run_baseline_audit(a), "dataset_2": run_baseline_audit(b)}
    with (patch("healthcare_data_quality.agent_loop.query_local_model",
                return_value=json.dumps({"objective_id": "stop", "reason": "stop"})),
          patch("healthcare_data_quality.agent_loop.summarize_investigation",
                return_value={"findings": [], "unresolved_questions": [],
                              "recommended_next_step": "Review lineage"})):
        result = run_investigation(
            a, max_steps=2, additional_datasets=[b],
            verified_contexts=verify_contexts(a), objective_baselines=baselines,
        )
    assert {item["dataset_id"] for item in result["history"]} == {
        "dataset_1", "dataset_2"
    }
