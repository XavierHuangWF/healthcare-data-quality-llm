"""Downloadable source-linked reviewer bundle for a SINGLE CSV investigation.

This runs only when the existing human-review + investigation completion gate
is open. It cannot invent decisions or agent conversations; only the current
session ledger is exported. The CSV is never decorated with Excel styling.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from healthcare_data_quality.cleaning.reviewer_audit import (
    build_reviewer_audit, reviewer_audit_json,
)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8")


def _csv_bytes(columns: list[str], records: list[dict], keys: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    for record in records:
        writer.writerow([_csv_cell(record.get(key)) for key in keys])
    return output.getvalue().encode("utf-8-sig")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    return str(value)


def _name(value: str) -> str:
    # Use one safe filename even for uploads with path separators or odd Unicode.
    leaf = str(value or "source.csv").replace("\\", "/").rsplit("/", 1)[-1]
    base = re.sub(r"[^\w. -]", "_", leaf, flags=re.UNICODE).strip(". ")[:100]
    return base if base.lower().endswith(".csv") else (base or "source") + ".csv"


def build_review_zip_bundle(
    *, original_csv: bytes, cleaned_csv: bytes, original_filename: str,
    reviewer: str, history: list[dict], decisions: list[dict],
    review_statuses: dict[str, str], conversations: dict[str, list[dict]],
    approved_row_operations: list[dict], approved_column_deletions: list[dict],
    completed_audits: dict[str, dict], finding_titles: dict[str, str],
    agent_b_suggestions: dict[str, Any] | None = None,
    excel_attachment: tuple[str, bytes] | None = None,
    dataset_id: str = "dataset_1",
) -> tuple[str, bytes]:
    """Produce one independently verifiable ZIP; fail closed on contradictions.

    :returns: archive filename and bytes. Nothing is persisted to the server.
    """
    if not str(reviewer or "").strip():
        raise PermissionError("A named reviewer is required before the final audit can be exported")
    if not original_csv or not cleaned_csv:
        raise ValueError("Original and cleaned CSV are both required")
    if not finding_titles or not review_statuses:
        raise PermissionError("No reviewed findings/checks are available for the final audit")
    final_statuses = {
        "Applied and verified", "Reviewed — no change",
        "Resolved by earlier approved edit", "Approved column deletion for final export",
        "Approved row changes for final export",
    }
    incomplete = [fid for fid in finding_titles if review_statuses.get(fid) not in final_statuses]
    if incomplete:
        raise PermissionError(f"Review is incomplete for {len(incomplete)} finding(s)/check(s)")
    audit = build_reviewer_audit(
        original_csv=original_csv, cleaned_csv=cleaned_csv,
        original_filename=original_filename, reviewer=reviewer,
        history=history, decisions=decisions, review_statuses=review_statuses,
        conversations=conversations, approved_row_operations=approved_row_operations,
        approved_column_deletions=approved_column_deletions,
        completed_audits=completed_audits, finding_titles=finding_titles,
        dataset_id=dataset_id,
    )
    audit["source_finding_titles"] = dict(finding_titles)
    audit["agent_b_suggestions_at_export"] = dict(agent_b_suggestions or {})
    audit["review_transcripts_raw"] = dict(conversations or {})
    audit["completed_finding_audits"] = dict(completed_audits or {})
    audit["approved_row_operations"] = list(approved_row_operations or [])
    audit["approved_column_deletions"] = list(approved_column_deletions or [])
    filename = _name(original_filename)
    stem = filename[:-4]
    files: dict[str, bytes] = {
        f"source/original_{filename}": bytes(original_csv),
        f"cleaned/cleaned_{filename}": bytes(cleaned_csv),
        "audit/reviewer_audit.json": reviewer_audit_json(audit),
        "audit/reviewer_interactions.csv": _csv_bytes(
            ["Finding ID", "Message #", "Speaker", "Exact message"],
            audit.get("conversation", []), ["finding_id", "sequence", "speaker", "message"]),
        "audit/approved_changes.csv": _csv_bytes(
            ["Finding ID", "Finding", "Reviewer", "Operation", "Original position",
             "Final position", "Original source value", "Approved before", "Approved after",
             "Final exported value", "Outcome"],
            audit.get("change_reconciliation", []),
            ["finding_id", "finding", "reviewer", "operation", "original_location",
             "final_location", "original_csv_value", "approved_before", "approved_after",
             "final_csv_value", "outcome"]),
        "audit/finding_decisions.csv": _csv_bytes(
            ["Finding ID", "Finding", "Review status"],
            audit.get("finding_reviews", []), ["finding_id", "title", "status"]),
    }
    if excel_attachment is not None:
        excel_name, excel_data = excel_attachment
        if not isinstance(excel_data, bytes) or excel_data[:2] != b"PK":
            raise ValueError("Refusing to include an invalid Excel attachment")
        # Embed the same audit inside the workbook if the user's existing
        # Excel generator has not done so already. Never duplicate audit sheets.
        with zipfile.ZipFile(io.BytesIO(excel_data)) as existing_workbook:
            workbook_xml = existing_workbook.read("xl/workbook.xml")
        if b'name="Audit trail"' not in workbook_xml:
            from healthcare_data_quality.cleaning.reviewer_audit import embed_reviewer_audit_sheets
            excel_data = embed_reviewer_audit_sheets(excel_data, audit)
        files["cleaned/cleaned_highlighted.xlsx"] = excel_data

    conversation_text = ["REVIEWER–MODEL DISCUSSION HISTORY", ""]
    for entry in audit.get("conversation", []):
        conversation_text.append(
            f"Finding {entry.get('finding_id', '')}, message {entry.get('sequence', '')} "
            f"[{entry.get('speaker', '')}]:")
        conversation_text.append(str(entry.get("message", "")))
        conversation_text.append("")
    files["audit/reviewer_interactions.txt"] = "\n".join(conversation_text).encode("utf-8")
    file_digests = {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                    for name, data in files.items()}
    # Independent verification: the bundle's cleaned payload and the audit's
    # recorded clean SHA-256 must agree. Same for immutable source.
    if (file_digests[f"cleaned/cleaned_{filename}"]["sha256"] != audit["cleaned_sha256"]
            or file_digests[f"source/original_{filename}"]["sha256"] != audit["original_sha256"]):
        raise PermissionError("Reviewer audit checksum does not match its CSV contents")
    files["manifest.json"] = _json_bytes({
        "schema": "hdq-review-bundle-1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "source": filename,
        "dataset_id": dataset_id,
        "finding_count": len(finding_titles),
        "conversation_messages": len(audit.get("conversation", [])),
        "approved_action_events": len(audit.get("approved_operations", [])),
        "files": file_digests,
    })
    files["README.txt"] = (
        "HEALTHCARE DATA QUALITY — REVIEW PACKAGE\n\n"
        "This ZIP contains both the original and the cleaned CSV, plus a reviewer-"
        "linked, source-reconciled audit.\n\n"
        "cleaned/                 Final cleaned CSV; optional styled Excel\n"
        "source/                  Immutable original CSV for comparison\n"
        "audit/reviewer_audit.json Full source-linked approval and interaction history\n"
        "audit/reviewer_interactions.csv Reviewer ↔ model conversation by finding\n"
        "audit/reviewer_interactions.txt Human-readable conversation transcript\n"
        "audit/approved_changes.csv Before/After edits and original/final coordinates\n"
        "audit/finding_decisions.csv Completed review statuses, including no-change\n"
        "manifest.json            SHA-256 and byte count for each archive payload\n\n"
        "Each interaction is preserved within its finding in recorded order.\n"
        "Chronological ordering BETWEEN findings is not assumed unless timestamps exist.\n"
        "A later approved deletion may supersede a previous edit or visual flag.\n"
        "CSV cannot store visual formatting: use optional .xlsx for surviving flags.\n"
        "This archive may contain sensitive dataset and conversation content; store securely.\n"
        "The original upload is never overwritten.\n"
    ).encode("utf-8")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return f"review_package_{stem}.zip", output.getvalue()
