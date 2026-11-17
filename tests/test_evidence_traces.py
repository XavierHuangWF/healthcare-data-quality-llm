"""Evidence is a causal test trace, not a second copy of issue cards."""
import csv
from pathlib import Path

from evidence_view import baseline_evidence_trace, registered_investigation_evidence
from healthcare_data_quality.finding_evidence import dataset_finding_evidence
from healthcare_data_quality.finding_presentation import header_findings
from healthcare_data_quality.ingestion import inspect_headers
from healthcare_data_quality.tools.column_inspection import compare_columns


def _make_csv(tmp_path, headers, rows):
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def test_count_evidence_explains_thresholds_and_actual_aggregation(tmp_path):
    path = _make_csv(tmp_path, ["Taxon", "A", "B"], [
        ["k__A", "0", ""], ["k__B", "-2", "1.5"], ["k__C", "", "NaN"],
    ])
    observations = [
        {"row": 2, "column": 3, "sample_id": "B", "taxon": "k__A", "original_value": "", "issue_type": "missing"},
        {"row": 3, "column": 2, "sample_id": "A", "taxon": "k__B", "original_value": "-2", "issue_type": "negative"},
        {"row": 3, "column": 3, "sample_id": "B", "taxon": "k__B", "original_value": "1.5", "issue_type": "fractional"},
        {"row": 4, "column": 2, "sample_id": "A", "taxon": "k__C", "original_value": "", "issue_type": "missing"},
        {"row": 4, "column": 3, "sample_id": "B", "taxon": "k__C", "original_value": "NaN", "issue_type": "non_numeric"},
    ]
    baseline = {"checks": {"find_invalid_counts": {"status": "completed", "observations": observations}}}
    traces = [baseline_evidence_trace(e, baseline, path)
              for e in dataset_finding_evidence(baseline, path, inspect_headers(path))]
    counts = {t["measurements"][0]["Observed"]: t for t in traces}
    assert 2 in counts
    assert counts[2]["measurements"][1]["Observed"] == 2
    assert "zero is not missing" in counts[2]["rule"].lower()
    assert not any(row.get("Original value") == "0" for trace in traces for row in trace["examples"])
    assert any("below zero" in t["rule"] for t in traces)
    assert any("integral value" in t["rule"] for t in traces)


def test_repeated_header_evidence_includes_actual_comparison_and_csv_examples(tmp_path):
    path = _make_csv(tmp_path, ["Taxon", "ABC99", "ABC99"], [
        ["T1", "2", "2"], ["T2", "", ""], ["T3", "7", "7"],
    ])
    header = inspect_headers(path)
    baseline = {"checks": {}}
    entries = dataset_finding_evidence(baseline, path, header)
    assert entries[0]["title"] == header_findings(path, header)[0]["title"]
    trace = baseline_evidence_trace(entries[0], baseline, path)
    assert trace["measurements"][0]["Observed"] == 3
    assert trace["measurements"][1]["Observed"] == 3
    assert trace["examples"][1]["Comparison"] == "both blank"
    assert trace["examples"][0]["CSV row"] == 2


def test_registered_gemma_result_shows_tool_arguments_counts_and_origin(tmp_path):
    path = _make_csv(tmp_path, ["Taxon", "ABC99", "ABC99"], [
        ["T1", "8", "8"], ["T2", "", ""], ["T3", "1", "2"],
        ["T4", "", "9"],
    ])
    measured = compare_columns(path, 2, 3)[0]
    audit = {"history": [{
        "step": 1, "tool": "compare_columns", "dataset_id": "dataset_1",
        "arguments": {"left": 2, "right": 3}, "observations": [measured],
    }], "report": {"findings": [{
        "claim": "These are biological replicates", "evidence_ids": ["T1.O1"],
        "claim_verification": {"status": "requires_scientific_review"},
    }]}}
    traces = registered_investigation_evidence(audit, [path])
    assert len(traces) == 1
    trace = traces[0]
    assert trace["verification"] == "requires_scientific_review"
    assert not trace["unresolved"]
    observation = trace["linked"][0]
    assert observation["arguments"] == {"left": 2, "right": 3}
    assert observation["citation"] == "T1.O1"
    assert observation["observed"] == measured
    assert observation["measurements"][0]["Observed"] == 4
    assert observation["source_examples"][2]["CSV row"] == 4
    assert observation["source_examples"][2]["Comparison"] == "conflicting nonblank"


def test_missing_gemma_citations_are_not_invented():
    traces = registered_investigation_evidence({
        "history": [], "report": {"findings": [{"claim": "Unsupported", "evidence_ids": ["T9.O1"]}]}
    })
    assert traces[0]["unresolved"] == ["T9.O1"]
    assert traces[0]["linked"] == []


def test_evidence_ui_does_not_repeat_entire_ai_history_or_empty_exploratory_banner():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    view = source.split("def render_finding_evidence(", 1)[1].split("def render_cleaning_workspace(", 1)[0]
    assert "render_technical_history(audit)" not in view
    assert 'if exploratory_runs:' in view
    assert 'st.markdown("#### Gemma exploratory findings")' not in view
    assert "registered_investigation_evidence(audit, dataset_paths)" in view
    assert "st.code(str(proposal.get(\"code\")" in view
