"""Read-only, human-readable descriptions of deterministic data-quality evidence.

This is a presentation layer. It never changes a scientific classification,
modifies a source dataset, or grants approval to remediate any finding.
"""

import csv
from itertools import combinations
from pathlib import Path


MAX_HEADER_PAIRS = 200


def natural_join(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def column_names(positions: list[int]) -> str:
    return natural_join([f"Column {p}" for p in positions])


def row_names(numbers: list[int]) -> str:
    return natural_join([f"row {p}" for p in numbers])


def header_findings(path: Path, header_report: dict) -> list[dict]:
    """Compare repeated-header columns *in this file*, never infer sample identity.

    Position numbers are 1-based. Column comparison is exact and includes blank
    values. Ragged rows make the whole-column result inconclusive.
    """
    headers = header_report.get("duplicate_headers") or {}
    pairs = [
        (name, left, right)
        for name, positions in headers.items()
        if name.strip()
        for left, right in combinations(positions, 2)
    ]
    inspected = pairs[:MAX_HEADER_PAIRS]
    results = {
        (name, left, right): {"matching": 0, "different": 0, "uncomparable": 0}
        for name, left, right in inspected
    }
    if inspected:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            expected = len(next(reader, []))
            for record in reader:
                complete = len(record) == expected
                for name, left, right in inspected:
                    counts = results[(name, left, right)]
                    if not complete:
                        counts["uncomparable"] += 1
                    elif record[left - 1] == record[right - 1]:
                        counts["matching"] += 1
                    else:
                        counts["different"] += 1
    findings = []
    for name, left, right in inspected:
        counts = results[(name, left, right)]
        total = sum(counts.values())
        label = f"Columns {left} and {right}"
        if counts["uncomparable"]:
            title = f"{label} share the name ‘{name}’; comparison is incomplete"
            explanation = (
                f"{counts['matching']:,} matching rows and {counts['different']:,} rows with different values "
                f"were compared; {counts['uncomparable']:,} rows have an inconsistent number of fields."
            )
        elif counts["different"]:
            title = f"{label} share the name ‘{name}’ but contain different values"
            explanation = (
                f"Across {total:,} records, {counts['matching']:,} values match and "
                f"{counts['different']:,} differ. These are repeated header names, not identical columns."
            )
        else:
            title = f"{label} (‘{name}’) contain identical values"
            explanation = (
                f"All {total:,} data rows match, including any blank cells. "
                "This confirms identical column contents, not that the biological specimens are the same."
            )
        findings.append({
            "title": title,
            "explanation": explanation,
            "next_step": "Check sample identity and collection/provenance records before removing either column.",
            "header": name,
            "columns": [left, right],
            "matching_rows": counts["matching"],
            "different_rows": counts["different"],
            "uncomparable_rows": counts["uncomparable"],
        })
    remaining = len(pairs) - len(inspected)
    if remaining:
        findings.append({
            "title": f"{remaining:,} additional repeated-header column pairs were not compared",
            "explanation": "The interactive display limits column-pair comparisons to avoid excessive processing.",
            "next_step": "Review the repeated-header groups or run a targeted comparison.",
            "header": "",
            "columns": [],
        })
    return findings


def header_issues(header_report: dict) -> list[dict]:
    """Missing names and inconsistent CSV row widths in plain English."""
    issues = []
    blanks = header_report.get("blank_header_positions") or []
    if blanks:
        title = f"{column_names(blanks)} {'has no header name' if len(blanks) == 1 else 'have no header names'}"
        issues.append({
            "title": title,
            "explanation": "The header cells are empty. The original column positions are preserved in the dataset view.",
            "next_step": "Confirm the intended variable name(s) from the data dictionary before renaming.",
            "type": "blank_headers",
            "rows": [{"Column": p, "Header name": "(blank)"} for p in blanks],
        })
    ragged = header_report.get("ragged_records") or []
    if ragged:
        expected = header_report.get("total_columns", 0)
        issues.append({
            "title": f"{len(ragged):,} data rows have an unexpected number of columns",
            "explanation": f"Each CSV row should contain {expected:,} fields. These rows contain more or fewer fields.",
            "next_step": "Check the original CSV export before interpreting or cleaning these rows.",
            "type": "ragged_records",
            "rows": [{"CSV row": csv_row, "Fields found": actual, "Expected": expected}
                     for csv_row, actual in ragged],
        })
    return issues


CHECK_NAMES = {
    "observe_matrix_structure": "Dataset structure",
    "find_duplicate_ids": "Repeated identifiers",
    "find_duplicate_conflicts": "Conflicting metadata",
    "find_invalid_counts": "Missing or invalid counts",
    "find_duplicate_taxa": "Repeated taxon labels",
    "inspect_taxonomy_structure": "Taxonomic hierarchy",
    "find_unclassified_species": "Unclassified species",
    "inspect_taxonomy_naming": "Taxonomy naming consistency",
}

TOOL_NAMES = {
    **CHECK_NAMES,
    "inspect_column": "Detailed column inspection",
    "inspect_columns": "Detailed column inspection",
    "compare_columns": "Column value comparison",
    "inspect_provenance": "Dataset provenance review",
    "summarize_invalid_counts": "Count validation",
    "investigate_with_r": "Additional statistical investigation",
    "reuse_saved_r_investigation": "Additional statistical investigation",
}

STATUS_LABELS = {
    "completed": "Checked",
    "resolved": "Evidence collected",
    "requires_client_input": "Needs scientific decision",
    "in_progress": "Partially checked",
    "uninvestigated": "Not checked yet",
    "blocked": "Not yet available",
    "skipped_unverified_context": "Not applicable / unconfirmed",
    "error": "Check unsuccessful",
}

COUNT_TYPE_LABELS = {
    "missing": "Missing value",
    "negative": "Negative count",
    "fractional": "Fractional count",
    "non_numeric": "Non-numeric value",
}


def readable_objective_title(objective: dict) -> str:
    raw = str(objective.get("title") or "Investigation question")
    objective_id = str(objective.get("objective_id") or "")
    if ":compare_columns:" in objective_id:
        try:
            left, right = [int(p) for p in objective_id.rsplit(":", 2)[-2:]]
            return f"Compare the values in Columns {left} and {right}"
        except ValueError:
            pass
    if raw.startswith("Compare duplicate-header columns "):
        return raw.replace("Compare duplicate-header columns ", "Compare values in Columns ", 1)
    if raw.startswith("Measure "):
        check = raw[len("Measure "):].strip().replace(" ", "_")
        if check in CHECK_NAMES:
            return CHECK_NAMES[check]
    return raw


def readable_objective_summary(objective: dict) -> tuple[str, str]:
    """Distinguish completed checks from scientific decisions in client language."""
    source = objective.get("source")
    objective_id = str(objective.get("objective_id") or "")
    status = objective.get("status")
    description = str(objective.get("finding") or "")
    next_action = str(objective.get("next_action") or "")
    if source == "deterministic_baseline":
        return (
            "The check has finished and supporting observations were recorded.",
            "See Data quality findings for the affected cells and records.",
        )
    if ":compare_columns:" in objective_id:
        if status == "resolved":
            return (
                "These two columns were compared row by row. This does not establish specimen identity.",
                "Confirm the original sample records before deciding whether either column is redundant.",
            )
        return (
            "These columns have the same name; their contents have not yet been compared by the agent.",
            "Compare the two columns and verify specimen provenance before any removal.",
        )
    if "cross_dataset_id_reconciliation" in objective_id:
        return (
            "Identifiers across separate uploaded datasets have not yet been reconciled.",
            "Review matching rules and source identifiers before combining datasets.",
        )
    if "capacity:duplicate_header_pairs" in objective_id:
        return (
            "Some pairs of repeated column names still need comparison.",
            "Review the remaining pairs before making a cleaning decision.",
        )
    return description, next_action


def _issue(title: str, explanation: str, evidence: list[dict], next_step: str = "") -> dict:
    return {"title": title, "explanation": explanation, "evidence": evidence, "next_step": next_step}


def baseline_issues(baseline: dict) -> list[dict]:
    """Provide complete, tabular source evidence without exposing internal JSON keys."""
    checks = baseline.get("checks") or {}
    issues: list[dict] = []
    check = checks.get("find_invalid_counts") or {}
    if check.get("status") == "completed":
        rows = check.get("observations") or []
        for category in sorted({str(r.get("issue_type", "unknown")) for r in rows if isinstance(r, dict)}):
            subset = [r for r in rows if isinstance(r, dict) and r.get("issue_type") == category]
            label = COUNT_TYPE_LABELS.get(category, category.replace("_", " ").capitalize())
            table = [{
                "CSV row": r.get("row"),
                "Column": r.get("column"),
                "Column name": r.get("sample_id"),
                "Taxon / feature": r.get("taxon"),
                "Original value": r.get("original_value"),
            } for r in subset]
            issues.append(_issue(
                f"{len(subset):,} cells contain {label.lower()}s" if category == "missing" else
                f"{len(subset):,} cells contain {label.lower()} values",
                f"The affected measurements were identified in {len(subset):,} cells. No values were changed.",
                table,
                "Confirm the scientific missing-data and count-handling policy before preprocessing.",
            ))
    check = checks.get("find_duplicate_taxa") or {}
    if check.get("status") == "completed":
        observations = check.get("observations") or []
        if observations:
            issues.append(_issue(
                f"{len(observations):,} taxon labels appear in multiple rows",
                "Rows share the same taxon name. Their measurements may or may not be identical.",
                [{"Taxon / feature": r.get("taxon"), "CSV rows": ", ".join(map(str, r.get("records") or [])),
                  "Values identical": "Yes" if r.get("counts_identical") else "No"} for r in observations],
                "Confirm whether repeated labels represent the same taxon before merging or removing rows.",
            ))
    check = checks.get("find_duplicate_ids") or {}
    if check.get("status") == "completed":
        observations = check.get("observations") or []
        if observations:
            issues.append(_issue(
                f"{len(observations):,} identifier values occur in multiple metadata records",
                "These identifier values repeat. Repeated values do not establish that the underlying specimens are duplicates.",
                [{"Identifier field": r.get("field"), "Identifier value": r.get("value"),
                  "CSV rows": ", ".join(map(str, r.get("records") or [])),
                  "Records": len(r.get("records") or [])} for r in observations],
                "Check specimen identifiers and source records before deciding whether to merge records.",
            ))
    check = checks.get("find_duplicate_conflicts") or {}
    if check.get("status") == "completed":
        observations = check.get("observations") or []
        if observations:
            evidence = []
            for r in observations:
                for csv_row, value in (r.get("values_by_record") or {}).items():
                    evidence.append({"Sample identifier": r.get("sample_id"),
                                     "Conflicting field": r.get("field"), "CSV row": csv_row, "Recorded value": value})
            issues.append(_issue(
                f"{len(observations):,} metadata fields have conflicting values",
                "Records sharing a sample identifier contain different values in the listed fields.",
                evidence,
                "Check the authoritative metadata source. Do not automatically select one record.",
            ))
    check = checks.get("inspect_taxonomy_structure") or {}
    if check.get("status") == "completed":
        structure = (check.get("observations") or [{}])[0]
        incomplete = structure.get("incomplete_taxonomy") or []
        alternate = structure.get("alternate_delimiters") or []
        if incomplete:
            issues.append(_issue(
                f"{len(incomplete):,} {'row has' if len(incomplete) == 1 else 'rows have'} incomplete taxonomic classifications",
                "These rows do not contain the expected seven taxonomic levels.",
                [{"CSV row": r.get("record"), "Classification levels found": r.get("level_count"),
                  "Expected levels": 7} for r in incomplete],
                "Review the taxonomy source and naming convention before attempting a mapping.",
            ))
        if alternate:
            issues.append(_issue(
                f"{len(alternate):,} {'row uses' if len(alternate) == 1 else 'rows use'} a different taxonomy separator",
                "These rows contain a vertical bar (|) instead of the usual semicolon-separated format.",
                [{"CSV row": row, "Issue": "Different separator (|)"} for row in alternate],
                "Confirm the expected taxonomy format before standardizing labels.",
            ))
    check = checks.get("find_unclassified_species") or {}
    if check.get("status") == "completed":
        observations = check.get("observations") or []
        if observations:
            issues.append(_issue(
                f"{len(observations):,} {'row has' if len(observations) == 1 else 'rows have'} no specific species assignment",
                "The species-level classification is missing, unknown, or explicitly unclassified.",
                [{"CSV row": r.get("record"), "Taxon / feature": r.get("original_taxonomy"),
                  "Species entry": r.get("species_value")} for r in observations],
                "Agree on how unknown species should be retained and reported.",
            ))
    check = checks.get("inspect_taxonomy_naming") or {}
    if check.get("status") == "completed":
        counts = check.get("counts_by_style") or {}
        used = {k: v for k, v in counts.items() if v}
        if len(used) > 1:
            names = {"fully_prefixed": "Named with taxonomy prefixes",
                     "unprefixed": "Named without prefixes", "mixed": "Mix of prefixed and unprefixed levels"}
            source = (check.get("observations") or [{}])[0]
            issues.append(_issue(
                "Taxonomic classifications use different naming formats",
                "; ".join(f"{names.get(k, k)}: {v:,} rows" for k, v in used.items()) + ".",
                [{"CSV row": row, "Naming format": names.get(k, k)}
                 for k, rows in source.items() for row in rows],
                "Confirm the accepted taxonomy naming convention before any change.",
            ))
    return issues


def history_observations(tool: str, observations: list[dict]) -> list[dict]:
    """Turn known agent-tool outputs into reader-friendly evidence tables."""
    rows: list[dict] = []
    if tool == "compare_columns":
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            rows.append({
                "Column A": obs.get("left_position"), "Column B": obs.get("right_position"),
                "Name A": obs.get("left_name"), "Name B": obs.get("right_name"),
                "Matching filled rows": obs.get("matching_nonmissing"),
                "Both blank": obs.get("both_missing"),
                "Only one blank": obs.get("one_missing"),
                "Conflicting filled rows": obs.get("conflicting_nonmissing"),
                "All values identical": "Yes" if obs.get("identical") else "No",
            })
    elif tool in {"inspect_column", "inspect_columns"}:
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            preferred = ("position", "name", "header", "nonmissing", "missing",
                         "distinct_count", "numeric", "nonempty_count", "missing_count",
                         "unique_count", "numeric_count", "numeric_fraction", "total_rows")
            readable = {"position": "Column", "name": "Column name", "header": "Header name",
                        "nonmissing": "Filled values", "missing": "Missing values",
                        "distinct_count": "Distinct values", "numeric": "Numeric values",
                        "nonempty_count": "Filled values", "missing_count": "Missing values",
                        "unique_count": "Distinct values", "numeric_count": "Numeric values",
                        "numeric_fraction": "Numeric fraction", "total_rows": "Total rows"}
            rows.append({readable[k]: obs[k] for k in preferred if k in obs})
    return rows
