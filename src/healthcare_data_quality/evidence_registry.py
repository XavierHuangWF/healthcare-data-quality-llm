"""Shared, immutable, source-verified evidence contract for review agents.

The registry is a deterministic projection of the ORIGINAL CSV and its selected
QC/Gemma finding. Agent outputs are untrusted interpretations of this contract.
Never use a model-generated count, coordinate, or permission as authority.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .cleaning_authorization import file_sha256
from .source_inventory import source_inventory

REGISTRY_SCHEMA = 1


def build_finding_registry(finding: dict, source: Path) -> dict:
    """Bind one finding and full-dataset statistics to one immutable CSV hash.

    The registry does not grant edit permission; valid target sets are still
    enforced by finding_cleaning.validate_scoped_plan() at proposal and approval.
    """
    from .finding_cleaning import allowed_positions

    source = Path(source)
    inventory = source_inventory(source)
    cells, headers = allowed_positions(finding, source)
    affected_rows = sorted({r for r, _ in cells})
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        original_headers = next(reader)
        original_rows = list(reader)
    if len(original_rows) != inventory["data_rows"]:
        raise ValueError("Evidence inventory and original CSV disagree")
    empty_finding = sorted((r, c) for r, c in cells
                           if 2 <= r <= len(original_rows) + 1
                           and 1 <= c <= len(original_headers)
                           and c > 1 and not original_rows[r-2][c-1].strip())
    result = {
        "schema": REGISTRY_SCHEMA,
        "finding_id": str(finding["id"]),
        "dataset_id": str(finding.get("dataset_id") or ""),
        "check_key": str(finding.get("check_key") or ""),
        "source_sha256": file_sha256(source),
        "source_data_rows": inventory["data_rows"],
        "source_column_count": inventory["column_count"],
        "finding_affected_cells": len(cells),
        "finding_affected_rows": len(affected_rows),
        "finding_affected_row_ids": affected_rows,
        "finding_empty_count_cells": len(empty_finding),
        "finding_empty_count_rows": len({r for r, _ in empty_finding}),
        "finding_empty_count_row_ids": sorted({r for r, _ in empty_finding}),
        "source_empty_count_cells": inventory["empty_count_cells_total"],
        "source_empty_count_rows": inventory["empty_count_row_count"],
        "source_empty_count_row_ids": inventory["empty_count_rows"],
        "source_taxonomy_na_rows": inventory["taxonomy_na_row_count"],
        "source_explicit_na_cells": inventory["exact_na_cells_total"],
        "scope": "selected_finding",
        "user_may_request_source_wide_action": True,
    }
    return result


def verify_finding_registry(finding: dict, source: Path, *,
                            enforce_source_hash: bool = False) -> dict | None:
    """Reject stale or forged registries when they exist, never silently trust them."""
    record = finding.get("evidence_registry")
    if record is None:
        return None  # Backward-compatible with older saved investigations.
    if not isinstance(record, dict) or record.get("schema") != REGISTRY_SCHEMA:
        raise PermissionError("Unsupported evidence registry")
    if (record.get("finding_id") != finding.get("id") or
            record.get("dataset_id") != str(finding.get("dataset_id") or "") or
            (enforce_source_hash and record.get("source_sha256") != file_sha256(source))):
        raise PermissionError("Evidence registry is stale or belongs to a different finding")
    # Validators for deliberately chained, derived source CSVs still verify
    # every current before-value and issue a fresh source_sha256 for the plan.
    # The interactive Agent C host MUST enforce the ORIGINAL registry hash.
    return record


def prompt_evidence(registry: dict) -> dict:
    """Only machine-verified facts, compact enough for A/B/C prompt context."""
    result = {key: registry[key] for key in (
        "schema", "finding_id", "dataset_id", "source_sha256", "scope",
        "source_data_rows", "source_column_count", "finding_affected_cells",
        "finding_affected_rows", "finding_affected_row_ids",
        "finding_empty_count_cells", "finding_empty_count_rows",
        "finding_empty_count_row_ids", "source_empty_count_cells",
        "source_empty_count_rows", "source_empty_count_row_ids",
        "source_taxonomy_na_rows", "source_explicit_na_cells",
        "user_may_request_source_wide_action",
    ) if key in registry}
    for key in ("finding_affected_row_ids", "finding_empty_count_row_ids", "source_empty_count_row_ids"):
        if isinstance(result.get(key), list) and len(result[key]) > 64:
            result[key] = {"first_64": result[key][:64], "truncated": True,
                           "total": len(registry[key])}
    return result


def evidence_bound_response(message: str, registry: dict | None) -> str:
    """Do not publish numeric claims about empty-cell extent from model prose.

    A source statistic cannot be inferred from the LLM's text. Replace only
    sentences making unverifiable empty-cell/row extent claims with measured
    counts. Never modify the structured operations, which have separate gates.
    """
    if not registry or not message:
        return message
    import re
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(message))
    retained = []
    substituted = False
    for sentence in sentences:
        # Narrowly match a model asserting N empty cells 'across/in/spread'
        # N rows, or statements that N distinct rows contain blanks.
        count_claim = (re.search(r"\b\d[\d,]*\s+(?:empty|blank|missing)\s+cells?\b", sentence, re.I)
                       and re.search(r"\b(?:rows?|across|spread|distributed)\b", sentence, re.I))
        if count_claim:
            substituted = True
        else:
            retained.append(sentence)
    if not substituted:
        return message
    verified = (f"Verified: {registry['source_empty_count_cells']} empty count cells "
                f"across {registry['source_empty_count_rows']} CSV rows. "
                f"Current finding: {registry['finding_empty_count_cells']} empty cells "
                f"across {registry['finding_empty_count_rows']} rows.")
    return " ".join([verified, *retained]).strip()
