"""Commit an explicitly selected visual flag once, with original-source provenance.

A reviewer's direct flag/highlight instruction is an actionable selection. The
original CSV is never rewritten; red/bold styling is retained for XLSX export.
A model suggestion alone must never call this function.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag

_COMPLETED = {
    "Applied and verified", "Reviewed — no change",
    "Approved row changes for final export",
    "Approved column deletion for final export",
}


def apply_explicit_visual_flag_once(
    state: dict, finding: dict, source: Path, request: str,
    *, previously_displayed: dict | None = None,
) -> dict:
    """Verify original values and atomically record a named reviewer's flag.

    `request` is required to be a real user-selected instruction, never raw
    Agent B/Gemma prose. The caller is responsible for this origin check.
    """
    fid = str(finding["id"])
    ds_id = str(finding["dataset_id"])
    reviewer = str(state.get("cleaning_approved_reviewer") or "").strip()
    if not reviewer or len(reviewer) > 120:
        raise PermissionError("Register your reviewer identifier before applying flags")
    if (state.get("cleaning_review_statuses", {}).get(fid) in _COMPLETED or
            any(event.get("finding_id") == fid for event in state.get("cleaning_history", []))):
        raise PermissionError("Finding already completed; reopen it before revising")
    source = Path(source)
    before_bytes = source.read_bytes()
    verified = requested_visual_flag(request, finding, source)
    if verified is None or not verified.get("flags"):
        raise ValueError("No exact taxonomy flag targets were found in the request")
    digest = hashlib.sha256(before_bytes).hexdigest()
    if verified.get("source_sha256") != digest or verified.get("dataset_id") != ds_id:
        raise PermissionError("Original source/dataset changed since flag preparation")
    if verified.get("finding_id") != fid:
        raise PermissionError("The flag belongs to a different finding")
    # The request submitted through chat is intent, not proof that the reviewer
    # saw the resulting exact scope. Approval is permitted only from an
    # already-rendered Before → After proposal and its selected action.
    if previously_displayed is None:
        raise PermissionError("Show the verified Before → After flag preview before approval")
    if previously_displayed is not None:
        if (previously_displayed.get("source_sha256") != digest or
                previously_displayed.get("flags") != verified["flags"] or
                previously_displayed.get("finding_id") != fid or
                previously_displayed.get("dataset_id") != ds_id):
            raise PermissionError("Displayed Before/After differs from the original source")
    staged = state.get("cleaning_staged_bytes_by_dataset", {}).get(ds_id, before_bytes)
    if not isinstance(staged, bytes):
        raise PermissionError("Staged CSV bytes unavailable")
    # No state mutation before every provenance and authorization check passes.
    new_flags = [
        {**flag, "finding_id": fid, "reviewer": reviewer}
        for flag in verified["flags"]
    ]
    visual_by_ds = state.setdefault("cleaning_visual_flags_by_dataset", {})
    existing = list(visual_by_ds.get(ds_id, []))
    # Do not silently remove another finding's prior annotation history.
    keys = {(f["row"], f["column"]) for f in new_flags}
    replaced = [flag for flag in existing if (flag.get("row"), flag.get("column")) not in keys]
    visual_by_ds[ds_id] = replaced + new_flags
    state.setdefault("cleaning_history", []).append({
        "dataset_id": ds_id, "finding_id": fid, "finding": finding["title"],
        "operation": "flag_cells", "flags": verified["flags"],
        "changes": len(new_flags), "reviewer": reviewer,
        "source_sha256": digest, "staged_sha256": hashlib.sha256(staged).hexdigest(),
        "modified_positions": [], "storage": "session_visual_annotations_no_csv_change",
        "authorization": "explicit_user_instruction_one_click",
    })
    state.setdefault("cleaning_review_statuses", {})[fid] = "Applied and verified"
    state.setdefault("cleaning_decisions", []).append({
        "finding_id": fid, "decision": "visual_flag_applied_by_reviewer_once",
        "reviewer": reviewer, "user_request": request,
        "source_sha256": digest,
        "flags": len(new_flags),
    })
    for name in (
        "cleaning_pending_visual_flags_by_finding", "cleaning_plans_by_finding",
        "cleaning_batch_queue_by_finding", "cleaning_batch_total_by_finding",
        "cleaning_compound_previews_by_finding", "cleaning_pending_gemma_b",
        "cleaning_discussion_mode_by_finding",
    ):
        state.setdefault(name, {}).pop(fid, None)
    return {"approved": len(new_flags), "source_sha256": digest}
