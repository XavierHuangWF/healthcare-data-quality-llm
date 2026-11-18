"""Presentation-only, read-only evidence traces for the Streamlit evidence tab.

This module deliberately lives outside src/healthcare_data_quality so changes
in how results are explained do not invalidate saved investigation caches.
It never makes an LLM conclusion more verified than the underlying audit does.
"""
from __future__ import annotations

import csv
import inspect
from collections import Counter
from pathlib import Path

from healthcare_data_quality.count_validation import find_invalid_counts
from healthcare_data_quality.duplicate_validation import find_duplicate_ids, find_duplicate_conflicts
from healthcare_data_quality.finding_presentation import header_findings
from healthcare_data_quality.ingestion import inspect_headers
from healthcare_data_quality.taxonomy_validation import (
    find_duplicate_taxa, inspect_taxonomy_structure, inspect_taxonomy_naming,
)

from healthcare_data_quality.investigation_report import build_evidence_index


COUNT_RULES = {
    "missing": "For each data cell after the taxonomy column, trim whitespace. An empty cell or absent trailing CSV field is classified as missing; zero is not missing.",
    "negative": "Parse a nonblank count using decimal arithmetic. Flag a finite numeric value strictly below zero.",
    "fractional": "Parse a nonblank count using decimal arithmetic. Flag a finite nonnegative value that differs from its integral value.",
    "non_numeric": "Flag nonblank values that cannot be parsed as finite decimal counts (including NaN and Infinity).",
    "unexpected_column": "Flag a field that occurs after the last header column, even if it looks numeric.",
}

TOOL_METHODS = {
    "compare_columns": (
        "Read both physical CSV columns by 1-based position for every data record. "
        "Classify each pair as matching nonblank values, both blank, exactly one blank, "
        "or conflicting nonblank values. The counts are mutually exclusive."
    ),
    "inspect_column": "Read one physical CSV column and calculate its observed missing/nonmissing and value-distribution statistics.",
    "inspect_columns": "Read the specified physical CSV columns and calculate each column's observed statistics.",
    "observe_matrix_structure": "Count CSV records and fields and inspect value types without presuming scientific measurement semantics.",
    "inspect_provenance": "Check available dataset-bound source documentation and the recorded dataset fingerprint; missing provenance does not establish the source's identity.",
    "summarize_invalid_counts": "Summarize invalid counts using the registered read-only count-validation tool.",
    "find_duplicate_ids": "Group identical values of the verified identifier fields and record their CSV row positions.",
    "find_duplicate_conflicts": "Compare metadata values in rows sharing the same sample identifier.",
    "investigate_with_r": "Run approved R code in an isolated sandbox and record the returned output; method correctness is not independently established.",
    "reuse_saved_r_investigation": "Re-run a previously approved R investigation in the isolated sandbox; method correctness is not independently established.",
}


# Human-readable provenance to the read-only implementation used by the agent.
TOOL_IMPLEMENTATIONS = {
    "compare_columns": "src/healthcare_data_quality/tools/column_inspection.py :: compare_columns",
    "inspect_column": "src/healthcare_data_quality/tools/column_inspection.py :: inspect_column",
    "inspect_columns": "src/healthcare_data_quality/tools/column_inspection.py :: inspect_columns",
    "inspect_provenance": "src/healthcare_data_quality/tools/provenance.py :: inspect_provenance",
    "summarize_invalid_counts": "src/healthcare_data_quality/tools/microbiome.py :: summarize_invalid_counts",
    "observe_matrix_structure": "src/healthcare_data_quality/tools/microbiome.py :: observe_matrix_structure",
    "find_duplicate_ids": "src/healthcare_data_quality/duplicate_validation.py :: find_duplicate_ids",
    "find_duplicate_conflicts": "src/healthcare_data_quality/duplicate_validation.py :: find_duplicate_conflicts",
}



# These are references to the *actual installed check functions*, not
# hand-written illustrative snippets or LLM explanations. Keep this an
# allowlist: a dataset or a cached audit must never select a source file.
DETECTION_IMPLEMENTATIONS = {
    "find_invalid_counts": (find_invalid_counts,),
    "find_duplicate_taxa": (find_duplicate_taxa,),
    "find_duplicate_ids": (find_duplicate_ids,),
    "find_duplicate_conflicts": (find_duplicate_ids, find_duplicate_conflicts),
    "inspect_taxonomy_structure": (inspect_taxonomy_structure,),
    "inspect_taxonomy_naming": (inspect_taxonomy_naming,),
    "compare_repeated_headers": (inspect_headers, header_findings),
    "inspect_headers": (inspect_headers,),
}


def deterministic_detection_code(check_key: str) -> list[dict[str, str]]:
    """Retrieve the real read-only Python implementations behind this QC check.

    Source is read from local installed functions, never from user-uploaded
    files, model output, or arbitrary paths in an audit. It is not executed.
    A historic cached result may have been generated with an older code
    version, so the UI labels this as the *currently installed* implementation.
    """
    result = []
    project_root = Path(__file__).resolve().parent
    for func in DETECTION_IMPLEMENTATIONS.get(check_key, ()):
        try:
            source_file = inspect.getsourcefile(func)
            code = inspect.getsource(func)
            if not source_file:
                continue
            file_path = Path(source_file).resolve()
            if not file_path.is_relative_to(project_root):
                continue
            result.append({
                "function": func.__name__,
                "path": file_path.relative_to(project_root).as_posix(),
                "code": code,
            })
        except (OSError, TypeError, ValueError):
            # Do not synthesize plausible-looking code when source is unavailable.
            continue
    return result


def _preview(rows: list[dict], limit: int = 5) -> tuple[list[dict], int]:
    valid = [r for r in rows if isinstance(r, dict)]
    return valid[:limit], max(0, len(valid) - limit)


def _metric(name: str, observed: object, calculation: str) -> dict:
    return {"Measurement": name, "Observed": observed, "How obtained": calculation}


def measurement_rows_for_display(measurements: list[dict]) -> list[dict]:
    """Return Arrow-safe evidence metrics without changing the recorded data.

    A single check may report numeric counts AND a categorical string such as
    "unprefixed" in its Observed column. PyArrow requires a consistent column
    type and otherwise logs an ArrowTypeError on each Streamlit rerender.
    All observed values are shown as text *only in this display copy*; the
    underlying audit/trace retains its original numeric values.
    """
    return [
        {**row, "Observed": "—" if row.get("Observed") is None else str(row["Observed"])}
        for row in measurements
    ]


def _cell_pair_examples(path: Path | None, left: int, right: int, limit: int = 4) -> list[dict]:
    """Read a few source examples, without re-running or supplanting the saved test."""
    if path is None or not Path(path).is_file() or left < 1 or right < 1:
        return []
    samples = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = next(reader, [])
        if max(left, right) > len(headers):
            return []
        for csv_row, record in enumerate(reader, start=2):
            if max(left, right) > len(record):
                continue
            a, b = record[left - 1], record[right - 1]
            label = ("both blank" if not a and not b else
                     "one blank" if not a or not b else
                     "matching nonblank" if a == b else "conflicting nonblank")
            samples.append({"CSV row": csv_row, f"Column {left}": a,
                            f"Column {right}": b, "Comparison": label})
            if len(samples) >= limit:
                break
    return samples


def baseline_evidence_trace(entry: dict, baseline: dict, source_path: Path | None = None) -> dict:
    """Explain the exact calculation behind an existing displayed QC issue.

    `entry` comes from dataset_finding_evidence; its sample is already tied to
    the audit's original check. We do not refit or modify any test results.
    """
    check = str(entry.get("check_key") or "")
    evidence = entry.get("evidence") or []
    raw_check = (baseline.get("checks") or {}).get(check) or {}
    rule = str(entry.get("method") or "Method not recorded")
    measurements = []
    preview, omitted = _preview(evidence)

    if check == "find_invalid_counts":
        positions = {(r.get("CSV row"), r.get("Column")) for r in evidence if isinstance(r, dict)}
        matched = [r for r in raw_check.get("observations") or []
                   if (r.get("row"), r.get("column")) in positions]
        kinds = Counter(str(r.get("issue_type")) for r in matched)
        if len(kinds) == 1:
            kind = next(iter(kinds))
            rule = COUNT_RULES.get(kind, rule)
        measurements = [
            _metric("Flagged count cells", len(evidence), "Count every original cell satisfying this rule exactly once"),
            _metric("Distinct CSV rows", len({r.get("CSV row") for r in evidence}), "Unique 1-based row numbers in those cells"),
            _metric("Distinct columns", len({r.get("Column") for r in evidence}), "Unique physical column positions in those cells"),
        ]
    elif check == "find_duplicate_taxa":
        rule = ("Group records by the original taxonomy text in Column 1. Count a group only "
                "when that exact label occurs in at least two CSV data rows; compare each full count vector independently.")
        measurements = [
            _metric("Repeated-label groups", len(evidence), "Number of exact-label groups with at least two records"),
            _metric("Groups with identical count vectors", sum(r.get("Values identical") == "Yes" for r in evidence),
                    "Compare all count cells within each group; duplicates need not have equal counts"),
        ]
    elif check == "find_duplicate_ids":
        rule = "Group the original metadata by the verified identifier field and literal identifier value; count groups with at least two records."
        measurements = [_metric("Repeated identifier groups", len(evidence), "Number of repeated field/value groups")]
    elif check == "find_duplicate_conflicts":
        rule = "Within each repeated sample ID, compare each metadata field; record fields containing more than one distinct original value."
        raw = raw_check.get("observations") or []
        measurements = [
            _metric("Conflicting metadata fields", len(raw), "Each conflicting sample-identifier/field comparison counts once"),
            _metric("Source-value records", len(evidence), "Number of original conflicting values shown in the Findings tab"),
        ]
    elif check == "inspect_taxonomy_structure" and evidence:
        if "Levels found" in evidence[0]:
            rule = ("Split the original classification at ';' or '|' and count rank positions. "
                    "Flag a row only when the count differs from 7. 'NA' is a valid position, anywhere, including multiple times.")
            dist = Counter(row.get("Levels found") for row in evidence)
            measurements = [
                _metric("Rows with rank count != 7", len(evidence), "Number of classifications whose split produces other than seven tokens"),
                *[_metric(f"Rows with {count} positions", amount, "Group flagged classifications by token count")
                  for count, amount in sorted(dist.items())],
            ]
        elif "Expected separator" in evidence[0]:
            rule = "Inspect the original classification for a literal '|' delimiter instead of the standard ';' delimiter; do not change taxonomy values."
            measurements = [_metric("Rows containing alternate delimiter", len(evidence), "Count flagged CSV rows with '|' in the original classification")]
    elif check == "inspect_taxonomy_naming":
        counts = raw_check.get("counts_by_style") or {}
        used = {k: v for k, v in counts.items() if k != "not_assessable" and v}
        majority = max(used, key=used.get) if used else "not established"
        rule = ("For seven-position classifications, check each non-NA rank against the expected positional prefix "
                "(d__ or k__ both accepted in the domain position). Classify rows as fully prefixed, "
                "unprefixed, or mixed; report rows differing from the predominant style. NA is neutral.")
        measurements = [
            _metric("Predominant naming style", majority, "Most frequently observed non-neutral style"),
            *[_metric(f"Rows classified {style}", amount, "Original naming-style classification")
              for style, amount in sorted(counts.items())],
            _metric("Rows with other styles", len(evidence), "Rows with an assessable style other than the predominant one"),
        ]
    elif check == "compare_repeated_headers" and evidence and "Column A" in evidence[0]:
        row = evidence[0]
        a, b = row["Column A"], row["Column B"]
        matching, different, incomparable = (row.get("Matching rows", 0),
                                             row.get("Differing rows", 0), row.get("Incomparable rows", 0))
        rule = ("First find identical nonblank CSV header names by physical position. "
                "For this pair, compare each complete data row cell-by-cell, including equal blanks as matches. "
                "Rows of inconsistent width cannot be compared.")
        measurements = [
            _metric("Rows compared or blocked", matching + different + incomparable, "Matching + differing + incomparable rows"),
            _metric("Matching cells", matching, "Count equal strings, including two blank strings"),
            _metric("Differing cells", different, "Count unequal strings in complete rows"),
            _metric("Rows not comparable", incomparable, "Count rows whose width differs from the header width"),
        ]
        # These are examples, not a substitute for full-column equality counts.
        preview = _cell_pair_examples(source_path, a, b)
        omitted = max(0, matching + different + incomparable - len(preview))
    elif check == "inspect_headers":
        if evidence and "Header name" in evidence[0]:
            rule = "Read every original header field and identify positions whose name is empty or whitespace-only."
            measurements = [_metric("Unnamed header positions", len(evidence), "Count empty/whitespace-only CSV header names")]
        elif evidence and "Fields found" in evidence[0]:
            rule = "Compare the physical number of CSV fields in each row with the header's field count."
            measurements = [_metric("Rows with unexpected width", len(evidence), "Count records with a field count different from the header")]
    return {"rule": rule, "measurements": measurements, "examples": preview,
            "examples_omitted": omitted, "limitation": entry.get("limitation", ""),
            "check_name": entry.get("check_name", check), "check_key": check}


def registered_investigation_evidence(audit: dict, dataset_paths: list[Path] | None = None) -> list[dict]:
    """Resolve each Gemma narrative citation to the *actual* recorded tool call.

    No free-floating history is emitted. A claim's citation IDs define the
    only observations presented for that claim; unresolvable IDs are labeled.
    """
    history = audit.get("history") or []
    report = audit.get("report") or {}
    index = build_evidence_index(history)
    steps = {f"T{step.get('step')}.O{n}": step
             for step in history for n, _ in enumerate(step.get("observations") or [], start=1)}
    paths = dataset_paths or []
    traces = []
    for claim_num, finding in enumerate(report.get("findings") or [], start=1):
        if not isinstance(finding, dict):
            continue
        linked = []
        unresolved = []
        for citation in finding.get("evidence_ids") or []:
            if citation not in index or citation not in steps:
                unresolved.append(str(citation))
                continue
            item = index[citation]
            step = steps[citation]
            tool = str(item.get("tool") or "")
            obs = item.get("observation") or {}
            dataset_id = str(step.get("dataset_id") or "dataset_1")
            try:
                dataset_index = int(dataset_id.rsplit("_", 1)[-1]) - 1
            except ValueError:
                dataset_index = -1
            dataset_path = paths[dataset_index] if 0 <= dataset_index < len(paths) else None
            measurements = []
            if tool == "compare_columns":
                a = obs.get("left_position")
                b = obs.get("right_position")
                fields = [("Matching filled values", "matching_nonmissing"),
                          ("Both cells blank", "both_missing"),
                          ("One cell blank", "one_missing"),
                          ("Conflicting filled values", "conflicting_nonmissing")]
                measurements = [_metric(label, obs.get(key, "unavailable"), "CSV row-by-row comparison")
                                for label, key in fields]
                if all(type(obs.get(key)) is int for _, key in fields):
                    total = sum(obs[key] for _, key in fields)
                    measurements.insert(0, _metric("Data rows classified", total, "Sum of the four mutually exclusive categories"))
                examples = _cell_pair_examples(dataset_path, a, b) if type(a) is int and type(b) is int else []
            else:
                examples = []
            linked.append({
                "citation": citation, "tool": tool, "dataset_id": dataset_id,
                "arguments": step.get("arguments") or {},
                "method": TOOL_METHODS.get(tool, "Recorded read-only tool; see its actual input and returned observations below."),
                "implementation": TOOL_IMPLEMENTATIONS.get(tool, "See registered tool configuration"),
                "reason": step.get("reason") or "Read-only evidence collection",
                "observed": obs, "measurements": measurements,
                "source_examples": examples,
            })
        traces.append({
            "claim_number": claim_num,
            "claim": str(finding.get("claim") or ""),
            "verification": (finding.get("claim_verification") or {}).get("status", "requires_scientific_review"),
            "linked": linked,
            "unresolved": unresolved,
        })
    return traces
