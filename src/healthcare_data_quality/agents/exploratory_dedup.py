"""Conservative, deterministic deduplication of exploratory QC proposals/results.

This is a duplication filter, NOT a scientific anomaly classifier. Reject only
investigations already answered by completed baseline checks; do not forbid
novel associations, cross-dataset relationships or causal follow-up questions.
Dataset contents are read-only and remain local.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from healthcare_data_quality.investigation.ingestion import inspect_headers


# These check names are the authoritative registry for baseline-supported types.
_CHECK_ISSUES = {
    "find_duplicate_taxa": "repeated_taxa",
    "find_duplicate_ids": "repeated_metadata_ids",
    "find_duplicate_conflicts": "conflicting_metadata",
    "inspect_taxonomy_structure": "taxonomy_structure",
    "inspect_taxonomy_naming": "taxonomy_naming",
}


def known_issue_catalog(datasets: dict[str, Path], audit: dict) -> dict:
    """Report actual observed baseline issues, including headers in ALL CSVs.

    A completed check with zero findings does not create an exclusion. If a
    historical audit is partial, the filter errs toward allowing investigation.
    """
    catalog = {}
    baselines = audit.get("dataset_baselines") or {}
    if not baselines and audit.get("baseline"):
        baselines = {"dataset_1": audit["baseline"]}
    for dataset_id, path in datasets.items():
        # The UI already scans headers during ingest; do it here as well so the
        # guard works when called independently and for non-primary datasets.
        structure = inspect_headers(Path(path))
        # Preserve exact known structural findings from the current audit.
        # Its ingestion report covers the primary dataset only; the remaining
        # datasets are independently scanned above.
        if dataset_id == "dataset_1" and isinstance(audit.get("ingestion"), dict):
            ingested = audit["ingestion"]
            combined = dict(structure.get("duplicate_headers") or {})
            combined.update(ingested.get("duplicate_headers") or {})
            structure["duplicate_headers"] = combined
            structure["blank_header_positions"] = sorted(set(
                structure.get("blank_header_positions") or []
            ).union(ingested.get("blank_header_positions") or []))
            structure["ragged_records"] = (
                structure.get("ragged_records") or ingested.get("ragged_records") or []
            )
        checks = (baselines.get(dataset_id) or {}).get("checks") or {}
        types = set()
        if structure.get("duplicate_headers"):
            types.add("duplicate_headers")
        if structure.get("blank_header_positions"):
            types.add("blank_headers")
        if structure.get("ragged_records"):
            types.add("ragged_rows")
        for check_name, kind in _CHECK_ISSUES.items():
            check = checks.get(check_name) or {}
            if check.get("status") != "completed":
                continue
            observations = check.get("observations") or []
            if check_name == "inspect_taxonomy_structure":
                details = observations[0] if observations and isinstance(observations[0], dict) else {}
                if details.get("incomplete_taxonomy") or details.get("alternate_delimiters"):
                    types.add(kind)
            elif check_name == "inspect_taxonomy_naming":
                styles = check.get("counts_by_style") or {}
                if len([style for style, count in styles.items() if count and style != "not_assessable"]) > 1:
                    types.add(kind)
            elif observations:
                types.add(kind)
        counts = checks.get("find_invalid_counts") or {}
        known_cells = {}
        if counts.get("status") == "completed":
            # Preserve the QC result's source-cell identities as a private,
            # local-only backstop for filtering duplicate generated findings.
            # Never put this potentially large index in the Gemma prompt.
            for observation in counts.get("observations") or []:
                if not isinstance(observation, dict):
                    continue
                row, column = observation.get("row"), observation.get("column")
                kind = observation.get("issue_type")
                raw = observation.get("original_value")
                if type(row) is int and type(column) is int and isinstance(kind, str) and isinstance(raw, str):
                    known_cells[f"{row}:{column}"] = {"kind": kind, "value": raw}
                    types.add("count_" + kind)
            # Support older/compact audits without full cell observations.
            for kind, quantity in (counts.get("counts_by_type") or {}).items():
                if quantity:
                    types.add("count_" + str(kind))
        catalog[dataset_id] = {
            "_known_count_cells": known_cells,
            "issue_types": sorted(types),
            "repeated_column_headers": {
                str(name)[:90]: positions[:12]
                for name, positions in (structure.get("duplicate_headers") or {}).items()
                if name.strip()
            },
            "blank_header_positions": structure.get("blank_header_positions", [])[:30],
            "ragged_row_count": len(structure.get("ragged_records") or []),
        }
    return catalog


def prompt_issue_catalog(catalog: dict) -> dict:
    """Expose concise QC categories to Gemma, not the local evidence index."""
    return {
        dataset_id: {key: value for key, value in record.items() if not key.startswith("_")}
        for dataset_id, record in catalog.items()
    }


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def _is_cross_dataset(text: str) -> bool:
    return _matches(text, r"\b(?:cross[ -]?dataset|cross[ -]?file|between (?:datasets?|files?|studies)|"
                          r"across (?:datasets?|files?|studies)|metadata (?:vs\.?|versus|against) (?:count|abundance|matrix))\b")


def _new_relationship(text: str) -> bool:
    """Do not blanket-ban baseline fields used in a new cross-variable analysis."""
    return _is_cross_dataset(text) or _matches(
        text, r"\b(?:association|correlation|correlat\w*|predict\w*|batch effect|"
              r"co[ -]?occurr\w*|outcome|between (?:two|different) (?:variables?|measurements?)|"
              r"clinically|conditional on|associated with|link(?:ed)? to (?:clinical|batch))\b"
    )


def _known_baseline_reason(text: str, catalog: dict) -> str | None:
    issues = set().union(*(set(record.get("issue_types", [])) for record in catalog.values()))
    if not issues:
        return None
    low = text.lower()

    # Header/column duplication is ALREADY checked row-by-row by header_findings.
    # A proposal to compare two ABC99 columns is therefore still redundant.
    # Comparing identifiers BETWEEN data files is a different, useful question.
    if "duplicate_headers" in issues and not _is_cross_dataset(low) and (
        _matches(low, r"\b(?:duplicate\w*|duplicated|repeated|identical|same|non[ -]?unique|shared)\b"
                      r".{0,100}\b(?:header|column names?|column labels?|column ids?|columns?)\b")
        or _matches(low, r"\b(?:columns?|headers?)\b.{0,100}\b(?:same name|same (?:id|identifier)|"
                         r"duplicate\w*|repeated|identical|shared (?:name|id|header))\b")
        or _matches(low, r"\b(?:two|multiple|different) (?:samples?|columns?)\b.{0,80}"
                         r"\b(?:same (?:id|identifier|header)|identical header)\b")
    ):
        return "Duplicate column headers and their row-by-row value comparisons are already shown under Data quality findings."
    if "blank_headers" in issues and _matches(low, r"\b(?:blank|empty|missing|unnamed) (?:column )?headers?\b"):
        return "Blank header positions are already reported by the deterministic checks."
    if "ragged_rows" in issues and _matches(low, r"\b(?:ragged|uneven|inconsistent|varying) (?:csv |record |row )?(?:widths?|lengths?|fields?)\b"):
        return "CSV rows with inconsistent field counts are already reported."

    if _new_relationship(low):
        return None

    # Count abnormalities may be described by their category alone
    # ("negative counts"), by a numeric condition ("less than zero"), or
    # by an individual cell. The old patterns incorrectly required a second
    # noun after "negative counts", allowing the screenshot's exact duplicate.
    # The baseline category must genuinely have been observed; skipped checks
    # or clean datasets must never blacklist an exploration.
    count_context = _matches(
        low, r"\b(?:counts?|abundances?|measurements?|values?|cells?|matri(?:x|ces)|"
             r"numeric|microbial|microbiome|sequencing|reads?|columns?|samples?)\b"
    )
    if count_context:
        count_patterns = (
            ("count_negative",
             r"\b(?:negative|below[ -]?zero|sub[ -]?zero|less than zero|"
             r"strictly less than zero|smaller than zero|non[ -]?physical)\b|<\s*0\b",
             "Negative measurement cells are already reported."),
            ("count_fractional",
             r"\b(?:fractional|non[ -]?integers?|non[ -]?whole|decimal|not (?:an? )?integers?)\b",
             "Non-integer count cells are already reported."),
            ("count_missing",
             r"\b(?:missing|blank|empty|null|absent)\s+(?:count|measurement|abundance|value|cell)s?\b",
             "Missing measurement cells are already reported."),
            ("count_non_numeric",
             r"\b(?:non[ -]?numeric|non[ -]?number|not numeric|unparseable numeric)\b",
             "Non-numeric measurement cells are already reported."),
        )
        for kind, pattern, reason in count_patterns:
            if kind in issues and _matches(low, pattern):
                # Negative controls are legitimate samples, not negative
                # measurements. Do not blacklist an analysis of controls.
                if kind == "count_negative" and _matches(low, r"\bnegative (?:controls?|blanks?)\b") and not _matches(
                    low, r"\bnegative\s+(?:counts?|values?|measurements?|abundances?|cells?)\b|"
                         r"\b(?:below|less than|strictly less than) zero\b|<\s*0\b"
                ):
                    continue
                return reason

    simple_checks = [
        ("repeated_taxa", r"\b(?:duplicate|duplicated|repeated|identical) (?:taxon|taxa|taxonomy labels?|species names?)\b", "Repeated taxonomy labels are already reported."),
        ("repeated_metadata_ids", r"\b(?:duplicate|duplicated|repeated) (?:sample |metadata )?(?:ids?|identifiers?)\b", "Repeated metadata identifiers are already reported."),
        ("conflicting_metadata", r"\b(?:conflicting|inconsistent) (?:metadata )?(?:values?|fields?|records?)\b", "Conflicting duplicate metadata values are already reported."),
        ("taxonomy_structure", r"\b(?:incorrect|invalid|malformed|missing|extra) (?:number of )?(?:taxonomy |taxonomic )?(?:levels?|ranks?|delimiters?)\b", "Malformed taxonomic rank counts/delimiters are already reported."),
        ("taxonomy_naming", r"\b(?:inconsistent|mixed|nonstandard) (?:taxonomy |taxonomic )?(?:prefixes?|naming|names?)\b", "Taxonomy naming inconsistencies are already reported."),
    ]
    for kind, pattern, reason in simple_checks:
        if kind in issues and _matches(low, pattern):
            return reason
    return None


def redundant_proposal_reason(proposal: dict, catalog: dict, previous: list[dict]) -> str | None:
    """Reject only clear matches, never use model self-assessed novelty."""
    hypothesis = str(proposal.get("hypothesis", ""))
    method = str(proposal.get("method", ""))
    reason = _known_baseline_reason(hypothesis + ". " + method, catalog)
    if reason:
        return reason
    proposed = _normalized(hypothesis)
    for previous_run in previous:
        old = _normalized(str((previous_run.get("proposal") or {}).get("hypothesis", "")))
        if old and proposed and (old == proposed or (
            min(len(old), len(proposed)) >= 35
            and SequenceMatcher(None, old, proposed).ratio() >= 0.91
        )):
            return "This hypothesis substantially repeats a previously completed exploratory investigation."
    return None


def _already_reported_count_evidence(finding: dict, catalog: dict) -> bool:
    """True only if every source cell is a baseline-reported count anomaly.

    A genuinely new relationship with additional independently cited evidence
    remains eligible. This is a supplement to language-based classification,
    not proof that the exploratory analysis is scientifically redundant.
    """
    text = str(finding.get("title", "")) + ". " + str(finding.get("description", ""))
    if _new_relationship(text):
        return False
    evidence = finding.get("evidence") or []
    if not evidence:
        return False
    for cell in evidence:
        if not isinstance(cell, dict):
            return False
        known = (catalog.get(cell.get("dataset_id")) or {}).get("_known_count_cells") or {}
        expected = known.get(f"{cell.get('csv_row')}:{cell.get('column')}")
        if expected is None or expected.get("value") != cell.get("value"):
            return False
    return True


def filter_redundant_findings(result: dict, catalog: dict, previous: list[dict]) -> tuple[dict, int]:
    """Filter repeated individual findings; preserve independently novel ones.

    Do not use vague semantic matching: where a match is uncertain, retain the
    finding for human review. A previous exploration is not ground truth.
    """
    accepted = []
    removed = 0
    prior_titles = {
        _normalized(str(f.get("title", "")))
        for run in previous
        for f in ((run.get("result") or {}).get("findings") or [])
        if isinstance(f, dict)
    }
    for finding in result.get("findings") or []:
        title = str(finding.get("title", ""))
        description = str(finding.get("description", ""))
        normalized = _normalized(title)
        if (normalized in prior_titles
                or _known_baseline_reason(title + ". " + description, catalog)
                or _already_reported_count_evidence(finding, catalog)):
            removed += 1
        else:
            accepted.append(finding)
    if not removed:
        return result, 0
    updated = dict(result, findings=accepted)
    # The original generated summary may describe suppressed findings as novel.
    # Avoid presenting an unsupported summary after the evidence was filtered.
    updated["summary"] = (
        f"{len(accepted)} additional candidate finding(s) remained after filtering "
        f"{removed} already-reported finding(s)." if accepted else
        "No new findings beyond existing QC checks or previous exploratory results were identified."
    )
    return updated, removed
