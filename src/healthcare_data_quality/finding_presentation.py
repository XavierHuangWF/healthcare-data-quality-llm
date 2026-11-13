"""Read-only, human-readable descriptions of deterministic data-quality evidence.

This is a presentation layer. It never changes a scientific classification,
modifies a source dataset, or grants approval to remediate any finding.
"""

import csv
import re
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


# This project currently audits the classical seven-rank taxonomy format.
# A missing *rank entry* is not evidence that the underlying taxon is unknown.
EXPECTED_TAXONOMY_RANKS = (
    ("Kingdom / Domain", "k__"),
    ("Phylum", "p__"),
    ("Class", "c__"),
    ("Order", "o__"),
    ("Family", "f__"),
    ("Genus", "g__"),
    ("Species", "s__"),
)


def _original_taxonomy_by_csv_row(path: Path | None, row_numbers: set[int]) -> dict[int, str]:
    """Recover original text using CSV physical record numbering, preserving values verbatim."""
    if not path or not row_numbers:
        return {}
    found: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        for csv_row, record in enumerate(csv.reader(source), start=1):
            if csv_row in row_numbers:
                found[csv_row] = record[0] if record else ""
                if len(found) == len(row_numbers):
                    break
    return found


def _taxonomy_structure_evidence(
    incomplete: list[dict], source_path: Path | None
) -> tuple[list[dict], list[dict]]:
    """Show source taxonomy and the seven expected rank slots, without filling unknown taxa.

    If all tokens are rank-prefixed and strictly ordered, the prefixes identify
    which ranks are absent. Otherwise the assignment is explicitly positional,
    and must not be mistaken for an authoritative taxonomic correction.
    """
    original_by_row = _original_taxonomy_by_csv_row(
        source_path, {int(r["record"]) for r in incomplete if r.get("record") is not None}
    )
    prefix_to_index = {prefix[0]: i for i, (_, prefix) in enumerate(EXPECTED_TAXONOMY_RANKS)}
    prefix_to_index["d"] = 0  # Some datasets mark Domain as d__ instead of k__.
    rows: list[dict] = []
    details: list[dict] = []

    for record in incomplete:
        csv_row = record.get("record")
        original = original_by_row.get(csv_row, record.get("original_taxonomy"))
        # The existing audit counts both semicolons and vertical bars as separators.
        levels = [part.strip() for part in re.split(r"[;|]", original)] if original is not None else []
        declared_count = record.get("level_count")
        level_count = int(declared_count) if declared_count is not None else len(levels)

        prefix_indices = []
        for value in levels:
            match = re.match(r"^([dkpcofgs])__", value, flags=re.IGNORECASE)
            prefix_indices.append(prefix_to_index.get(match.group(1).lower()) if match else None)
        unambiguous_prefixes = (
            bool(levels)
            and all(index is not None for index in prefix_indices)
            and all(left < right for left, right in zip(prefix_indices, prefix_indices[1:]))
        )
        placement_note = (
            "Rank prefixes identify the positions shown. Taxon names have not been validated."
            if unambiguous_prefixes else
            "Provisional left-to-right alignment: rank positions have not been confirmed against the source taxonomy."
        )
        assigned: dict[int, str] = {}
        if unambiguous_prefixes:
            assigned = dict(zip(prefix_indices, levels))
        else:
            assigned = {index: value for index, value in enumerate(levels[:7])}

        missing_indices = [index for index in range(7) if index not in assigned]
        missing = ", ".join(EXPECTED_TAXONOMY_RANKS[index][0] for index in missing_indices)
        if original is None:
            missing = (
                f"{max(0, 7 - level_count)} rank(s) missing; exact positions unavailable "
                "without the original classification"
            )
            placement_note = "Original classification unavailable; the missing rank positions cannot be established."
        elif not missing:
            missing = "Not identifiable by position; inspect the original classification"
        rank_table = [
            {
                "Expected rank": rank_name,
                "Expected prefix": prefix,
                "Original entry (provisional rank)": (
                    assigned.get(index, "Not provided") if original is not None else "Source unavailable"
                ),
                "Status": (
                    "Present in source" if index in assigned else "Not provided in source"
                ) if original is not None else "Cannot assess",
            }
            for index, (rank_name, prefix) in enumerate(EXPECTED_TAXONOMY_RANKS)
        ]
        rows.append({
            "CSV row": csv_row,
            "Original classification (verbatim)": original if original is not None else "Source text unavailable",
            "Levels found": level_count,
            "Expected levels": 7,
            "Ranks not provided (provisional)": missing,
        })
        details.append({
            "csv_row": csv_row,
            "original": original,
            "missing": missing,
            "rank_table": rank_table,
            "placement_note": placement_note,
        })
    return rows, details



def baseline_issues(baseline: dict, source_path: Path | None = None) -> list[dict]:
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
            evidence, details = _taxonomy_structure_evidence(incomplete, source_path)
            issue = _issue(
                f"{len(incomplete):,} {'row has' if len(incomplete) == 1 else 'rows have'} incomplete taxonomic classifications",
                "The current file format expects seven ranks: Kingdom/Domain, Phylum, Class, "
                "Order, Family, Genus and Species. The table shows the original classification "
                "and which rank entries appear absent. Missing ranks are provisional when "
                "the source does not explicitly identify rank positions.",
                evidence,
                "Check the original taxonomy database or annotation pipeline to identify missing "
                "names. Do not assign species names or fill missing ranks automatically.",
            )
            issue["type"] = "taxonomy_structure"
            issue["details"] = details
            issues.append(issue)

        if alternate:
            original_by_row = _original_taxonomy_by_csv_row(
                source_path, {int(row) for row in alternate}
            )

            evidence = []

            for row in alternate:
                original = original_by_row.get(int(row))

                suggested = (
                    "; ".join(
                        part.strip()
                        for part in re.split(r"[;|]", original)
                    )
                    if original is not None
                    else "Source unavailable"
                )

                evidence.append({
                    "CSV row": row,
                    "Original classification": (
                        original if original is not None
                        else "Source unavailable"
                    ),
                    "Problem": "Uses | instead of ;",
                    "Expected separator": ";",
                })

            issues.append(_issue(

                f"{len(alternate):,} "
                f"{'row uses' if len(alternate) == 1 else 'rows use'} "
                "inconsistent taxonomy separators",

                "The expected format uses semicolons (;) "
                "between taxonomic ranks. The affected rows "
                "use vertical bars (|). The proposed version "
                "only standardizes separators and preserves "
                "the original taxonomic names.",
                evidence,
                "Review the suggested format before making "
                "any changes to the source data.",
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
