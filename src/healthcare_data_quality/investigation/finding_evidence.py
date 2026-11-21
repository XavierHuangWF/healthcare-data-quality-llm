"""Evidence catalog aligned one-to-one with the user-facing QC findings.

Nothing here interprets an LLM claim as a validated scientific conclusion.
The evidence is derived from the same baseline/header finding objects as the UI.
"""

from pathlib import Path

from healthcare_data_quality.ui.finding_presentation import baseline_issues, header_findings, header_issues, CHECK_NAMES

CHECK_METHODS = {
    "find_invalid_counts": (
        "Inspect the original count cells and classify missing, negative, fractional, "
        "non-numeric or unexpected values. Zero counts are valid and are not classified as missing.",
        "A flagged value may reflect the dataset's documented missing-value policy; "
        "the check alone cannot determine an appropriate replacement.",
    ),
    "find_duplicate_taxa": (
        "Group identical taxonomy labels, record all CSV row positions, "
        "and compare the associated count vectors.",
        "A repeated label does not establish that rows should be merged.",
    ),
    "find_duplicate_ids": (
        "Group repeated values in the recognized metadata identifier fields "
        "and report their CSV row positions.",
        "A repeated identifier is not proof of duplicated biological specimens.",
    ),
    "find_duplicate_conflicts": (
        "Compare metadata fields for records that share the same sample identifier; "
        "retain each distinct observed value and its source CSV row.",
        "Deciding which conflicting value is correct requires an authoritative source.",
    ),
    "inspect_taxonomy_structure": (
        "Split each original classification into rank positions using recognized delimiters; "
        "flag classifications with a level count other than seven and report alternate delimiters. "
        "NA counts as a valid rank position anywhere in the classification.",
        "This checks rank count and delimiter consistency, not biological correctness or the missing taxon's identity.",
    ),
    "inspect_taxonomy_naming": (
        "Compare taxonomy naming/prefix patterns across observed classifications "
        "and show rows that differ from the predominant format.",
        "A less common naming format is not necessarily a scientific error.",
    ),
    "compare_repeated_headers": (
        "Find repeated CSV header names by physical column position; "
        "compare corresponding cell values row-by-row without conflating columns of the same name.",
        "Matching columns do not establish that their samples are identical or that either can be removed.",
    ),
    "inspect_headers": (
        "Read the original CSV header and record widths; identify blank header positions "
        "and data records with more or fewer fields than the header.",
        "A missing header requires source documentation before it can be renamed.",
    ),
}


def dataset_finding_evidence(baseline: dict, source_path: Path, header_report: dict) -> list[dict]:
    """Return the exact finding order used in the Streamlit results tab."""
    findings = baseline_issues(baseline, source_path=source_path)
    findings.extend(header_findings(source_path, header_report))
    findings.extend(header_issues(header_report))
    result = []
    for finding in findings:
        check_key = finding.get("source_check") or "unknown"
        method, limitation = CHECK_METHODS.get(check_key, (
            "The saved result did not identify a recognized check method.",
            "Review the original audit before acting on this finding.",
        ))
        rows = finding.get("evidence")
        if rows is None:
            rows = finding.get("rows") or []
        result.append({
            "title": finding["title"],
            "evidence": rows,
            "check_key": check_key,
            "check_name": CHECK_NAMES.get(check_key, "CSV header comparison" if check_key == "compare_repeated_headers" else "CSV structure"),
            "method": method,
            "limitation": limitation,
            "verification": "Original data observation (deterministic QC)",
        })
    return result


def exploratory_finding_evidence(runs: list[dict]) -> list[dict]:
    """Preserve source verification metadata for each novel exploratory finding."""
    entries = []
    for round_number, run in enumerate(runs, start=1):
        result = run.get("result") or {}
        proposal = run.get("proposal") or {}
        verification = run.get("verification") or {}
        for finding in result.get("findings") or []:
            entries.append({
                "title": str(finding.get("title") or "Exploratory candidate finding"),
                "description": str(finding.get("description") or ""),
                "evidence": finding.get("evidence") or [],
                "check_key": "gemma_exploration",
                "check_name": f"Gemma investigation {round_number}",
                "method": str(proposal.get("method") or "Method not recorded"),
                "hypothesis": str(proposal.get("hypothesis") or ""),
                "assumptions": str(proposal.get("assumptions") or ""),
                "verification_details": verification,
                "code": str(proposal.get("code") or ""),
                "verification": "Source references reproduced; scientific conclusion unvalidated",
                "limitation": "Source-cell verification and reproducibility do not prove the analytical claim or its clinical meaning.",
            })
    return entries
