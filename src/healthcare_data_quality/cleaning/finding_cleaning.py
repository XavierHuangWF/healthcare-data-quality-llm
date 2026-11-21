"""Finding-scoped cleaning review and independently verified derived CSV copies.

The local model suggests *data*, never executable Python. One reviewed plan can
change only the physical positions established by its selected finding. Existing
source files are immutable and each accepted plan produces a new version.
"""

import csv
import json
from decimal import Decimal, InvalidOperation
import re
from pathlib import Path
from uuid import uuid4

from .cleaning_authorization import file_sha256
from .cleaning_workspace import validate_header_plan
from healthcare_data_quality.investigation.finding_evidence import dataset_finding_evidence
from healthcare_data_quality.investigation.ingestion import inspect_headers

APPROVAL_PHRASE = "APPROVE THIS FINDING ONLY"
# Agent C's JSON schema remains limited to 32 examples. The host can expand
# a uniform, source-evidenced, researcher-accepted correction to a larger
# validated plan; all positions still must be cited in the selected finding.
MAX_CELL_CHANGES = 2048
MAX_MODEL_EXAMPLE_CHANGES = 32


def build_cleaning_findings(audit: dict, paths: list[Path], names: list[str], runs: list[dict]) -> list[dict]:
    """Use the SAME QC/Gemma evidence powering Data quality findings."""
    baselines = audit.get("dataset_baselines") or {"dataset_1": audit.get("baseline") or {}}
    items = []
    for dataset_id, baseline in baselines.items():
        index = int(dataset_id.rsplit("_", 1)[-1]) - 1
        if not 0 <= index < len(paths):
            continue
        for number, evidence in enumerate(dataset_finding_evidence(
                baseline, paths[index], inspect_headers(paths[index])), 1):
            items.append({"id": f"{dataset_id}:qc:{number}", "dataset_id": dataset_id,
                          "name": names[index], "title": evidence["title"], "source": "QC",
                          "check_key": evidence["check_key"], "method": evidence["method"],
                          "evidence": evidence.get("evidence") or [], "actionable": True})
    for round_number, run in enumerate(runs, 1):
        result = run.get("result") or {}
        proposal = run.get("proposal") or {}
        verification = run.get("verification") or {}
        # Sandbox tests with no new finding are useful to discuss, not to clean.
        findings = result.get("findings") or []
        if not findings:
            # A zero-finding investigation provides NO writable cell evidence.
            # With exactly one uploaded dataset we can still associate its
            # immutable source for read-only review and explicit UI selection.
            check_ds = "dataset_1" if len(paths) == 1 and "dataset_1" in baselines else ""
            items.append({"id": f"gemma:{round_number}:check", "dataset_id": check_ds,
                          "name": names[0] if check_ds else "Gemma",
                          "title": str(result.get("summary") or proposal.get("hypothesis") or "Completed Gemma check"),
                          "source": "Gemma check", "check_key": "gemma_check", "method": proposal.get("method") or "",
                          "evidence": [], "actionable": False})
            continue
        for finding_number, finding in enumerate(findings, 1):
            # Find the dataset from the saved source citations; without an unambiguous
            # dataset the model can discuss, not change, anything.
            evidence_rows = finding.get("evidence") or []
            ds_ids = {str(e.get("dataset_id")) for e in evidence_rows
                      if isinstance(e, dict) and e.get("dataset_id")}
            dataset_id = next(iter(ds_ids)) if len(ds_ids) == 1 else ("dataset_1" if len(paths) == 1 else "")
            confirmed = verification.get("status") == "source_cells_verified_and_output_reproduced"
            items.append({"id": f"gemma:{round_number}:finding:{finding_number}",
                          "dataset_id": dataset_id if dataset_id in baselines else "",
                          "name": "Gemma", "title": str(finding.get("title") or "Candidate finding"),
                          "source": "Gemma finding", "check_key": "gemma_exploration",
                          "method": str(proposal.get("method") or ""),
                          "evidence": evidence_rows, "actionable": confirmed and bool(dataset_id)})
    # Bind QC findings and verified explorations to one authoritative source.
    # This happens downstream of Agent A's evidence recording, before Agent B
    # or Agent C may interpret the finding. Zero-issue Gemma checks remain
    # discussion-only even if they have a source binding.
    from healthcare_data_quality.investigation.evidence_registry import build_finding_registry
    for item in items:
        ds_id = str(item.get("dataset_id") or "")
        if not ds_id.startswith("dataset_"):
            continue
        try:
            index = int(ds_id.rsplit("_", 1)[-1]) - 1
        except ValueError:
            continue
        if 0 <= index < len(paths):
            try:
                item["evidence_registry"] = build_finding_registry(item, paths[index])
            except (OSError, ValueError, UnicodeError, csv.Error) as exc:
                # Malformed CSVs must remain reviewable; an unavailable registry
                # is NOT permission for Gemma to invent summary statistics.
                item["evidence_registry_error"] = f"{type(exc).__name__}: {exc}"[:250]
    return items


def allowed_positions(finding: dict, source: Path) -> tuple[set[tuple[int, int]], set[int]]:
    """Explicit positions observed in the finding; never extrapolate to a whole column."""
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    rows = set()
    headers = set()
    key = finding.get("check_key")
    for record in finding.get("evidence") or []:
        if not isinstance(record, dict):
            continue
        if key in {"inspect_headers", "compare_repeated_headers"}:
            for label in ("Column", "Column A", "Column B"):
                value = record.get(label)
                if type(value) is int and 1 <= value <= len(header):
                    headers.add(value)
            continue
        if key == "find_duplicate_ids":
            # A repeated ID is not sufficient reason to mutate/de-duplicate rows.
            continue
        csv_row = record.get("CSV row", record.get("csv_row"))
        if type(csv_row) is not int or csv_row < 2:
            continue
        column = record.get("Column", record.get("column", record.get("column_position")))
        if type(column) is int and 1 <= column <= len(header):
            rows.add((csv_row, column))
        elif key in {"inspect_taxonomy_structure", "inspect_taxonomy_naming"}:
            rows.add((csv_row, 1))
        elif key == "find_duplicate_conflicts":
            name = record.get("Conflicting field")
            if isinstance(name, str) and header.count(name) == 1:
                rows.add((csv_row, header.index(name) + 1))
    if key == "find_duplicate_taxa":
        # Repeated taxa should be resolved against a biological identifier, not
        # through a guessed label change; offer discussion only.
        rows.clear()
    if not finding.get("actionable", True):
        return set(), set()
    return rows, headers


# An explicit researcher instruction is separate from a QC finding's statistical
# evidence. A row can be deleted even when the current finding concerns missing
# values, but only when the actual user text identifies an exact CSV row.
# This parser is used on the USER message in the host, never on Gemma's reply.
# Parse whole *explicit* row-ID lists, not just the first number.  A command
# such as "delete row 9, row 10 and row 11; keep row 12" identifies three
# deletions; the separate KEEP clause is never a deletion authorization.
_EXPLICIT_DELETE_ROW_RE = re.compile(
    r"\b(?:delete|drop|remove)\s+(?:the\s+)?(?:both\s+)?"
    r"(?:(?:entire|whole|full)\s+)?(?:(?:original|csv)\s+){0,2}"
    r"rows?\s*#?\s*(\d+)\b", re.IGNORECASE,
)
_NEXT_DELETE_ROW_RE = re.compile(
    r"\s*(?:(?:[,;.&]|\band\b)\s*(?:\band\b\s*)?"
    r"(?:(?:original|csv)\s+)?(?:rows?\s*#?\s*)?(\d+)\b"
    r"|(?:(?:original|csv)\s+)?rows?\s*#?\s*(\d+)\b)",
    re.IGNORECASE,
)


def explicitly_requested_row_deletions(user_text: str) -> frozenset[int]:
    """Parse *all* source row IDs explicitly included in deletion clauses.

    Never parse the IDs in a subsequent "keep row", "merge row" or Gemma
    explanation as deletions.  This parser is invoked on the real user's
    request; model-generated coordinates do not count as permission.
    """
    if not isinstance(user_text, str) or len(user_text) > 4000:
        return frozenset()
    value = user_text.strip()
    if re.match(r"I choose option \d+:", value, re.I):
        value = re.sub(r"^I choose option \d+:\s*", "", value, flags=re.I)
        value = re.split(r"\.\s+This clarifies", value, maxsplit=1, flags=re.I)[0]
    if "?" in value or re.search(
        r"\b(?:don't|do not|never|avoid|without)\s+(?:\w+\s+){0,3}"
        r"(?:delete|drop|remove)\b", value, re.I
    ):
        return frozenset()
    ids = []
    for matched in _EXPLICIT_DELETE_ROW_RE.finditer(value):
        # A "keep row N" clause may appear after the first deletion number.
        # Follow only an adjacent list of numbers and optional 'row' labels.
        group = [int(matched.group(1))]
        offset = matched.end()
        while True:
            next_row = _NEXT_DELETE_ROW_RE.match(value, offset)
            if next_row is None:
                break
            group.append(int(next_row.group(1) or next_row.group(2)))
            offset = next_row.end()
        ids.extend(group)
    if not ids or len(ids) > 32 or len(set(ids)) != len(ids) or any(row < 2 for row in ids):
        return frozenset()
    return frozenset(ids)


def propose_explicit_user_row_deletion(user_text: str, finding: dict,
                                       source: Path) -> dict | None:
    """Draft verified row removal without asking Gemma to invent row metadata."""
    row_ids = explicitly_requested_row_deletions(user_text)
    if not row_ids:
        return None
    # This fast path drafts a deletion-only proposal. Combined instructions
    # must reach Gemma as a batch, rather than silently discarding replacements.
    if re.search(r"\b(?:merge|combine|rename|replace|change|update)\b", user_text, re.I):
        return None
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    if (len(rows) < 3 or any(len(r) != len(rows[0]) for r in rows)
            or not rows[0] or any(row > len(rows) for row in row_ids)):
        raise ValueError("Requested CSV row is missing or the source CSV is not rectangular")
    changes = [{"row": row, "before": rows[row-1][0], "after": ""}
               for row in sorted(row_ids)]
    proposal = {"operation": "drop_rows", "dataset_id": finding["dataset_id"],
                "changes": changes}
    checked = validate_scoped_plan(proposal, finding, source,
                                   user_requested_rows=row_ids)
    return {"reply": "Prepared the explicitly requested deletion of original CSV "
                     + ", ".join(f"Row {row}" for row in sorted(row_ids))
                     + ". The entire record will be removed from the FINAL "
                       "download only after separate human approval. The original "
                       "CSV remains unchanged.",
            "operation": "drop_rows", "dataset_id": finding["dataset_id"],
            "changes": checked["changes"]}


# Source-backed reviewer commands are direct *drafts*, never approvals. A
# reviewer need not provide scientific evidence for a selected replacement:
# the host reads the CSV, verifies coordinates and originals, and shows the
# exact changes for one separate named approval.
_REVIEWER_CELL_COMMAND = re.compile(
    r"\b(?:replace|change|set|convert|correct|make|fix)\b", re.I)
_REVIEWER_NUMERIC_REPLACEMENT = re.compile(
    r"(?<![\w.])(?P<before>-?\d+(?:\.\d+)?)\s*"
    r"(?:with|to|->|→)\s*['\"]?(?P<after>-?\d+(?:\.\d+)?|NA)\b",
    re.I,
)
_REVIEWER_COORD_REPLACEMENT = re.compile(
    r"\brow\s*(?P<row>\d+)\s*[,;:]?\s*"
    r"(?:col(?:umn)?)\s*(?P<column>\d+)\s*"
    r"(?:from\s*['\"]?[^\s,;]+\s*)?"
    r"(?:to|with|=|->|→)\s*['\"]?(?P<after>-?\d+(?:\.\d+)?|NA)\b",
    re.I,
)


def propose_reviewer_cell_replacements(user_text: str, finding: dict,
                                       source: Path) -> dict | None:
    """Prepare user-specified cell replacements without an evidence debate.

    Only actual QC finding positions are writable. Every original value is
    fetched from the source, never invented by Gemma or inferred from a text
    sample. Mixed structural requests go to the multi-operation planner so no
    part of the user's instruction is silently dropped.
    """
    if not isinstance(user_text, str) or not 0 < len(user_text) <= 4000:
        return None
    text = user_text.strip()
    if (not _REVIEWER_CELL_COMMAND.search(text)
            or re.search(r"\b(?:delete|drop|remove|merge|rename|combine)\b", text, re.I)
            or re.search(r"\b(?:don't|do not|never|without)\b", text, re.I)
            or "?" in text):
        return None
    allowed, _ = allowed_positions(finding, source)
    if not allowed:
        return None
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.reader(stream, strict=True))
    if not records or any(len(record) != len(records[0]) for record in records):
        raise ValueError("Original CSV cannot be verified")
    changes: dict[tuple[int, int], dict] = {}
    explicit_coords = list(_REVIEWER_COORD_REPLACEMENT.finditer(text))
    for match in explicit_coords:
        row, col = int(match["row"]), int(match["column"])
        if (row, col) not in allowed or row > len(records) or col > len(records[0]):
            raise ValueError(f"Row {row}, column {col} is not in the selected finding")
        before = records[row - 1][col - 1]
        after = match["after"]
        if before == after:
            raise ValueError(f"Row {row}, column {col} already has that value")
        changes[row, col] = {"row": row, "column": col,
                             "before": before, "after": after}
    if not explicit_coords:
        value_matches = list(_REVIEWER_NUMERIC_REPLACEMENT.finditer(text))
        if not value_matches:
            return None
        for match in value_matches:
            before, after = match["before"], match["after"]
            positions = sorted((row, col) for row, col in allowed
                               if row <= len(records) and col <= len(records[0])
                               and records[row - 1][col - 1] == before)
            if len(positions) != 1:
                raise ValueError(
                    f"Value {before} matches {len(positions)} affected cells; "
                    "specify the CSV row and column")
            row, col = positions[0]
            if before == after:
                raise ValueError(f"Row {row}, column {col} already has that value")
            changes[row, col] = {"row": row, "column": col,
                                 "before": before, "after": after}
    if not changes:
        return None
    # A complex sentence that we have only partially parsed must go through
    # the full planner, not silently discard later requested replacements.
    if len(changes) > MAX_CELL_CHANGES:
        raise ValueError("Too many cell replacements in one review")
    proposal = {"operation": "replace_cells", "dataset_id": finding["dataset_id"],
                "changes": list(changes.values())}
    verified = validate_scoped_plan(proposal, finding, source)
    return {"reply": f"{len(verified['changes'])} exact cell replacements ready for review.",
            "operation": "replace_cells", "dataset_id": finding["dataset_id"],
            "changes": verified["changes"], "clarification_options": []}


# A user's dataset-wide predicate is different from a QC finding's small
# collection of examples. Resolve it against the complete immutable CSV,
# never from the language model's illustrative row IDs.
_BULK_EMPTY_ROW_COMMAND = re.compile(
    r"\b(?:delete|drop|remove|discard)\b[^.!?\n]{0,220}"
    r"\b(?:rows?|records?|raws)\b[^.!?\n]{0,125}"
    r"\b(?:empty|blank|missing)\b",
    re.IGNORECASE,
)


def requested_blank_row_scope(message: str, source: Path,
                              *, finding: dict | None = None) -> frozenset[int] | None:
    """Resolve a user's explicit predicate against ALL immutable CSV records.

    'Delete entire CSV rows containing at least one empty cell' is an explicit
    bulk request, even without the literal word 'all'. It is not a request to
    delete every row, only those with a genuinely blank count cell. NA is valid.
    """
    if not isinstance(message, str) or not 0 < len(message) <= 4000:
        return None
    value = message.strip()
    if re.match(r"I choose option \d+:", value, re.I):
        value = re.sub(r"^I choose option \d+:\s*", "", value, flags=re.I)
        value = re.split(r"\.\s+This clarifies", value, maxsplit=1, flags=re.I)[0]
    if (not _BULK_EMPTY_ROW_COMMAND.search(value) or
            re.search(r"\b(?:do not|don't|never|without|avoid)\s+(?:\w+\s+){0,3}"
                      r"(?:delete|drop|remove|discard)\b", value, re.I)):
        return None
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.reader(stream, strict=True))
    if not records or any(len(record) != len(records[0]) for record in records):
        raise ValueError('The source CSV must be rectangular')
    matching = frozenset(
        index for index, record in enumerate(records[1:], start=2)
        if any(cell.strip() == '' for cell in record[1:])
    )
    if re.search(r"\b(?:this|current|selected)\s+finding\b", value, re.I):
        if finding is None:
            raise ValueError('A selected finding is required for finding-scoped deletion')
        allowed, _ = allowed_positions(finding, source)
        matching &= frozenset(
            row for row, col in allowed
            if col > 1 and records[row - 1][col - 1].strip() == ''
        )
    return matching


def propose_bulk_blank_row_deletion(message: str, finding: dict,
                                    source: Path) -> dict | None:
    """Deterministic complete plan for an explicit all-empty-rows request."""
    rows = requested_blank_row_scope(message, source, finding=finding)
    if rows is None:
        return None
    if not rows:
        return {"reply": "No original CSV rows contain empty cells. No deletion needed.",
                "operation": "none", "dataset_id": "", "changes": [],
                "clarification_options": []}
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.reader(stream, strict=True))
    if len(rows) >= len(records) - 1:
        raise ValueError("Every data row contains an empty cell; this would delete all data rows")
    proposal = {
        "operation": "drop_rows", "dataset_id": finding["dataset_id"],
        "changes": [{"row": r, "before": records[r-1][0], "after": ""}
                    for r in sorted(rows)],
    }
    checked = validate_scoped_plan(proposal, finding, source,
                                   user_requested_rows=rows)
    return {"reply": f"Delete {len(rows)} CSV rows containing empty cells. Review and approve below.",
            "operation": "drop_rows", "dataset_id": finding["dataset_id"],
            "changes": checked["changes"], "clarification_options": []}


def _evidenced_taxon_groups(finding: dict, records: list[list[str]]) -> dict[int, str]:
    """Ground row IDs in this finding's duplicate-taxon QC records only."""
    if finding.get("check_key") != "find_duplicate_taxa":
        return {}
    by_row: dict[int, str] = {}
    for item in finding.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        label, positions = item.get("Taxon / feature"), item.get("CSV rows")
        if not isinstance(label, str) or not isinstance(positions, str):
            continue
        ids = [int(value) for value in re.findall(r"\d+", positions)]
        if len(ids) < 2 or len(set(ids)) != len(ids):
            continue
        for csv_row in ids:
            if not (2 <= csv_row <= len(records)) or records[csv_row-1][0] != label:
                raise ValueError("Duplicate-taxon evidence no longer matches the source CSV")
            if csv_row in by_row and by_row[csv_row] != label:
                raise ValueError("Conflicting row evidence")
            by_row[csv_row] = label
    return by_row


def _validate_row_plan(plan: dict, finding: dict, source: Path, *,
                       user_requested_rows: frozenset[int] | None = None) -> dict:
    """Verify immutable source coordinates for structural row edits."""
    operation = plan["operation"]
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.reader(stream, strict=True))
    if len(records) < 3 or any(len(row) != len(records[0]) for row in records):
        raise ValueError("Row operations need a rectangular CSV with data rows")
    evidenced = _evidenced_taxon_groups(finding, records)
    user_rows = (user_requested_rows or frozenset()) if operation == "drop_rows" else frozenset()
    if not evidenced and not user_rows:
        raise ValueError("No duplicate-taxon row evidence or explicitly requested row ID")
    changes = plan.get("changes")
    bulk_requested = (operation == "drop_rows" and bool(user_rows)
                      and all(isinstance(c, dict) and c.get("row") in user_rows
                              for c in (changes if isinstance(changes, list) else [])))
    maximum = (min(len(records) - 2, 20000) if bulk_requested else 32)
    if not isinstance(changes, list) or not 1 <= len(changes) <= maximum:
        raise ValueError(f"Row operations need 1–{maximum} source-verified changes")
    seen: set[int] = set()
    verified = []
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("Invalid row operation")
        row, before, after = change.get("row"), change.get("before"), change.get("after")
        permitted = row in evidenced or row in user_rows
        if type(row) is not int or not (2 <= row <= len(records)) or not permitted or not isinstance(before, str):
            raise ValueError("Row is neither evidenced nor explicitly requested by the user")
        # For an explicit user deletion, the original row's first-column label
        # comes from the immutable CSV, not the unrelated finding evidence.
        if before != records[row - 1][0] or row in seen:
            raise ValueError("Row identity mismatch or duplicate row request")
        seen.add(row)
        if operation == "drop_rows":
            if after != "":
                raise ValueError("Deleting rows requires after='' ")
            verified.append({"row": row, "before": before, "after": ""})
        else:
            other = change.get("source_row")
            if (type(other) is not int or other not in evidenced or other == row
                    or other in seen or evidenced[other] != before
                    or after != "sum_numeric"):
                raise ValueError("Merge requires two distinct, evidenced same-taxon rows and sum_numeric")
            seen.add(other)
            for first, second in zip(records[row-1][1:], records[other-1][1:]):
                for value in (first, second):
                    if value.strip().upper() not in {"", "NA"}:
                        try:
                            if not Decimal(value).is_finite():
                                raise ValueError("Cannot sum non-finite counts")
                        except InvalidOperation as exc:
                            raise ValueError("Cannot sum nonnumeric count values in a row merge") from exc
            verified.append({"row": row, "source_row": other,
                             "before": before, "after": "sum_numeric"})
    if operation == "drop_rows" and evidenced:
        # A saved/model-generated "duplicate removal" must not erase every
        # observation of a taxon. Row 9 and row 11 are both members of the
        # SAME duplicate pair in the example dataset; deleting both is not a
        # valid deduplication strategy. A researcher may explicitly elect to
        # remove both original rows by naming each row in the current request.
        from collections import defaultdict
        groups = defaultdict(set)
        for original_row, label in evidenced.items():
            groups[label].add(original_row)
        for label, group in groups.items():
            if group.issubset(seen) and not group.issubset(user_rows):
                raise ValueError("Proposed deletions remove every row of a duplicate taxon. "
                                 "Keep one row or explicitly request those CSV row IDs")
    if len(records) - 1 <= len(seen) - (len(verified) if operation == "merge_rows" else 0):
        raise ValueError("Row deletion would remove all data rows")
    return {"operation": operation, "dataset_id": finding["dataset_id"],
            "finding_id": finding["id"], "source_sha256": file_sha256(source),
            "changes": verified}


def validate_scoped_plan(plan: dict, finding: dict, source: Path, *,
                         user_requested_rows: frozenset[int] | None = None,
                         user_requested_columns: frozenset[int] | None = None) -> dict:
    """Reject out-of-scope edits and stale plans; never trust model locations."""
    from healthcare_data_quality.investigation.evidence_registry import verify_finding_registry
    verify_finding_registry(finding, source)
    # Explicit column choice must come from the UI's real user selection,
    # never from an LLM response or merely a zero-finding check summary.
    explicit_column = (plan.get("operation") == "drop_column"
                       and isinstance(user_requested_columns, frozenset)
                       and len(user_requested_columns) > 0)
    explicit_rows = (plan.get("operation") == "drop_rows"
                     and isinstance(user_requested_rows, frozenset)
                     and len(user_requested_rows) > 0)
    if not finding.get("actionable", True) and not (explicit_column or explicit_rows):
        raise ValueError("This check is discussion-only; it has no approved affected cells")
    if plan.get("dataset_id") != finding.get("dataset_id"):
        raise ValueError("The proposed dataset differs from the selected finding")
    allowed_cells, allowed_headers = allowed_positions(finding, source)
    operation = plan.get("operation")
    if operation == "rename_headers":
        checked = validate_header_plan(plan, [source]) if plan["dataset_id"] == "dataset_1" else None
        # validate_header_plan is indexed by uploaded dataset, so validate
        # arbitrary dataset IDs against this single working file with a proxy ID.
        if checked is None:
            checked = validate_header_plan({**plan, "dataset_id": "dataset_1"}, [source])
        if not allowed_headers or any(change["position"] not in allowed_headers for change in checked["changes"]):
            raise ValueError("Proposed header is outside this finding's affected columns")
        return {**checked, "dataset_id": finding["dataset_id"], "finding_id": finding["id"]}
    if operation in {"drop_rows", "merge_rows"}:
        return _validate_row_plan(plan, finding, source,
                                  user_requested_rows=user_requested_rows)
    if operation == "drop_column":
        changes = plan.get("changes")
        if not isinstance(changes, list) or len(changes) != 1 or not isinstance(changes[0], dict):
            raise ValueError("A whole-column deletion must name exactly one column")
        item = changes[0]
        pos, before, after = item.get("position"), item.get("before"), item.get("after")
        with Path(source).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream, strict=True))
        if not rows or len(rows[0]) <= 1 or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("A rectangular CSV with at least two columns is required")
        if type(pos) is not int or not 1 <= pos <= len(rows[0]):
            raise ValueError("Column position is invalid")
        if not isinstance(before, str) or rows[0][pos - 1] != before or after != "":
            raise ValueError("Column header mismatch or unsupported deletion payload")
        if pos not in allowed_headers and not any(col == pos for _, col in allowed_cells):
            if not explicit_column or pos not in user_requested_columns:
                raise ValueError("This column is not evidenced in the selected finding or explicitly selected by the researcher")
        return {"operation": "drop_column", "dataset_id": finding["dataset_id"],
                "finding_id": finding["id"], "source_sha256": file_sha256(source),
                "changes": [{"position": pos, "before": before, "after": ""}]}
    if operation != "replace_cells":
        raise ValueError("Only scoped cell replacements, header renames or a whole-column deletion can be staged")
    changes = plan.get("changes")
    if not isinstance(changes, list) or not 1 <= len(changes) <= MAX_CELL_CHANGES:
        raise ValueError(f"Provide 1–{MAX_CELL_CHANGES} exact cell changes")
    if not allowed_cells:
        raise ValueError("No exact affected cell coordinates were verified for this finding")
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.reader(stream, strict=True))
    if not records or any(len(row) != len(records[0]) for row in records):
        raise ValueError("Ragged CSV requires a separate reviewed repair workflow")
    seen = set()
    verified = []
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("Each cell change must be a mapping")
        csv_row, col = change.get("row"), change.get("column")
        before, after = change.get("before"), change.get("after")
        if (type(csv_row) is not int or type(col) is not int or
                (csv_row, col) not in allowed_cells or (csv_row, col) in seen or
                csv_row > len(records) or col > len(records[0])):
            raise ValueError("An edit targets a cell outside this finding or repeats a coordinate")
        if (not isinstance(before, str) or not isinstance(after, str) or len(after) > 2048 or
                before != records[csv_row - 1][col - 1] or before == after or
                "\x00" in after):
            raise ValueError("Original cell mismatch or unsafe/unchanged replacement value")
        # No arbitrary precision/biological transformations are inferred.
        verified.append({"row": csv_row, "column": col, "before": before, "after": after})
        seen.add((csv_row, col))
    return {"operation": operation, "dataset_id": finding["dataset_id"],
            "finding_id": finding["id"], "source_sha256": file_sha256(source),
            "changes": verified}



def expand_uniform_evidence_plan(plan: dict, finding: dict, source: Path, *,
                                 full_finding_accepted: bool,
                                 user_requested_rows: frozenset[int] | None = None,
                                 user_requested_columns: frozenset[int] | None = None) -> tuple[dict, int]:
    """Expand a homogeneous example plan to its complete, evidenced scope.

    Never infer new cell values or extrapolate to a whole column. Expansion is
    allowed only after the researcher accepts an action covering *all* the
    selected finding, all source coordinates are independently in its evidence,
    and every evidenced original value is identical to the verified examples.
    Heterogeneous or ambiguous proposals remain strictly partial.
    """
    checked = validate_scoped_plan(plan, finding, source,
                                   user_requested_rows=user_requested_rows,
                                   user_requested_columns=user_requested_columns)
    seed = checked.get("changes", [])
    if (not full_finding_accepted or checked["operation"] != "replace_cells"
            or not seed or len(seed) > MAX_MODEL_EXAMPLE_CHANGES):
        return checked, 0
    allowed, _ = allowed_positions(finding, source)
    if len(allowed) <= len(seed) or len(allowed) > MAX_CELL_CHANGES:
        return checked, 0
    pairs = {(item["before"], item["after"]) for item in seed}
    if len(pairs) != 1:
        return checked, 0
    before, after = next(iter(pairs))
    # Validate the *complete* CSV independently of the model's partial list.
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        return checked, 0
    if any(r > len(rows) or c > len(rows[0]) or rows[r-1][c-1] != before
           for r, c in allowed):
        return checked, 0
    expanded = {"operation": "replace_cells", "dataset_id": checked["dataset_id"],
                "changes": [{"row": r, "column": c, "before": before, "after": after}
                            for r, c in sorted(allowed)]}
    # This passes through the same unchanged source-value, scope, duplicate,
    # value and SHA-256 checks as a regular plan. No LLM-generated code is run.
    verified = validate_scoped_plan(expanded, finding, source)
    return verified, len(verified["changes"]) - len(seed)


# Bulk instruction normalization belongs to the trusted host, not a generative
# model's JSON token budget. Only the explicitly selected, source-evidenced
# blank COUNT cells may be modified. This never stages data or approves edits.
_ALL_BLANKS_TO_NA_RE = re.compile(
    r"\b(?:set|replace|fill|convert)\s+all\s+(\d+)\s+"
    r"(?:empty|blank)\s+cells?\b[^.!?\n]{0,130}"
    r"\b(?:to|with|as)\s+['\"]?NA\b", re.IGNORECASE,
)
_DIRECT_BLANKS_TO_NA_RE = re.compile(
    r"\b(?:fill|replace|set|convert)\b[^.!?\n]{0,130}"
    r"\b(?:empty|blank|missing)\b[^.!?\n]{0,130}"
    r"\b(?:with|to|as)\s+['\"]?NA\b", re.IGNORECASE,
)
_SHORT_NA_RE = re.compile(
    r"\s*(?:fill\s+(?:(?:all|the)\s+)?(?:(?:empty|blank|missing)\s+)?"
    r"(?:cells?\s+)?(?:in|with|as|to)\s+NA|"
    r"(?:set|replace|change)\s+(?:(?:all|the)\s+)?"
    r"(?:(?:empty|blank|missing)\s+)?cells?\s+(?:to|with|as)\s+NA)"
    r"[.!]?\s*", re.IGNORECASE,
)


def propose_evidenced_blank_fill(
    message: str, recommendation: str, finding: dict, source: Path,
    *, accepted_recommendation: bool = False,
) -> dict | None:
    """Turn a precise blank-to-NA request into a bounded, validated action.

    No call to Gemma is required to enumerate 618 cells. The researcher must
    explicitly request the operation or approve an Agent B recommendation;
    evidence and immutable source values determine every cell coordinate.
    Return None for unrecognized/compound instructions, so the usual Gemma
    discussion handles them instead of silently omitting an action.
    """
    if finding.get("check_key") != "find_invalid_counts" or not finding.get("actionable", True):
        return None
    if not isinstance(message, str) or not isinstance(recommendation, str):
        return None
    # Do not reinterpret refusal, question, or multi-action requests as consent.
    if ("?" in message
            or message.lstrip().lower().startswith(("why ", "how ", "should ", "could ",
                                                   "can ", "would ", "what "))
            or re.search(r"\b(?:don't|do not|never|instead of|not fill)\b", message, re.I)
            or re.search(r"\b(?:delete|drop|remove)\b[^.!?\n]{0,100}\bcolumn\b", message, re.I)):
        return None

    selected = _ALL_BLANKS_TO_NA_RE.search(recommendation)
    direct = _ALL_BLANKS_TO_NA_RE.search(message)
    if selected and accepted_recommendation:
        matched = selected
    elif direct:
        matched = direct
    elif selected and (_SHORT_NA_RE.fullmatch(message) or _DIRECT_BLANKS_TO_NA_RE.search(message)):
        matched = selected
    else:
        return None

    count_expected = int(matched.group(1))
    if not 1 <= count_expected <= MAX_CELL_CHANGES:
        raise ValueError(
            f"Expected {count_expected} blank cells, exceeding the validated "
            f"bulk limit of {MAX_CELL_CHANGES}; no partial edit was prepared"
        )

    allowed, _ = allowed_positions(finding, source)
    if not allowed:
        raise ValueError("This finding has no verified count-cell coordinates")
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.reader(stream, strict=True))
    if not records or any(len(row) != len(records[0]) for row in records):
        raise ValueError("The original CSV is not rectangular")
    cells = sorted((row, col) for row, col in allowed
                   if row <= len(records) and col <= len(records[0])
                   and col > 1 and records[row - 1][col - 1].strip() == "")
    if len(cells) != count_expected:
        raise ValueError(
            f"Agent B proposed {count_expected} blank count cells but only "
            f"{len(cells)} exact blank positions were independently verified "
            "in this finding. No incomplete or inferred bulk edit was prepared"
        )

    plan = {"operation": "replace_cells", "dataset_id": finding["dataset_id"],
            "changes": [{"row": row, "column": col,
                         "before": records[row - 1][col - 1], "after": "NA"}
                        for row, col in cells]}
    checked = validate_scoped_plan(plan, finding, source)
    return {
        "reply": (f"Prepared one evidence-scoped bulk action for all {len(cells)} "
                  "verified empty count cells: replace the literal empty value "
                  "with 'NA'. Coordinates are generated and checked by Python, "
                  "not individually invented by Gemma. No data were changed. "
                  "The existing isolated R test and explicit reviewer approval "
                  "are still required before staging."),
        "operation": "replace_cells", "dataset_id": finding["dataset_id"],
        "changes": checked["changes"],
    }


def reviewed_python_code(plan: dict) -> str:
    """Show an actual reproducible stdlib implementation, not model-supplied code."""
    if plan["operation"] == "rename_headers":
        import inspect
        from .cleaning_executor import execute_authorized_header_cleaning
        return ("# Actual authorized header-only cleaning implementation.\n"
                "# An independent approval record is required by the app.\n"
                "CHANGES = " + repr(plan["changes"]) + "\n\n" +
                inspect.getsource(execute_authorized_header_cleaning))
    return ("# Reference implementation for the *specific* reviewed cell changes.\n"
            "# The app applies and independently verifies these edits only after approval.\n"
            "import csv\n"
            "from pathlib import Path\n"
            "CHANGES = " + repr(plan["changes"]) + "\n"
            "with Path('INPUT.csv').open(encoding='utf-8-sig', newline='') as f:\n"
            "    rows = list(csv.reader(f))\n"
            "for change in CHANGES:\n"
            "    r, c = change['row'] - 1, change['column'] - 1\n"
            "    assert rows[r][c] == change['before']\n"
            "    rows[r][c] = change['after']\n"
            "with Path('REVIEWED_OUTPUT.csv').open('x', encoding='utf-8', newline='') as f:\n"
            "    csv.writer(f).writerows(rows)\n"
            "# The application additionally checks the input SHA-256 and EVERY unchanged cell.\n")


def execute_approved_scoped_plan(plan: dict, finding: dict, source: Path,
                                 output_dir: Path, reviewer: str, confirmation: str) -> dict:
    """Create one verified derived version; never run model-supplied code."""
    if confirmation != APPROVAL_PHRASE or not reviewer.strip() or len(reviewer) > 120:
        raise PermissionError("Explicit per-finding authorization and reviewer identity are required")
    source = Path(source)
    verified = validate_scoped_plan(plan, finding, source)
    if verified != plan:
        raise PermissionError("The reviewed plan is stale or altered")
    if file_sha256(source) != plan["source_sha256"]:
        raise ValueError("Source changed after plan verification")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    output = output_dir / ("reviewed_" + uuid4().hex[:12] + ".csv")
    if output.resolve() == source.resolve():
        raise PermissionError("Cannot overwrite the original source")
    if plan["operation"] != "replace_cells":
        raise ValueError("Header operations must use the existing separate authorization gate")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        original = list(csv.reader(stream, strict=True))
    modified = [list(row) for row in original]
    targets = {(item["row"] - 1, item["column"] - 1): item for item in plan["changes"]}
    for (r, c), item in targets.items():
        if original[r][c] != item["before"]:
            raise PermissionError("Source cell changed since review")
        modified[r][c] = item["after"]
    try:
        with output.open("x", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(modified)
        with output.open(encoding="utf-8", newline="") as stream:
            observed = list(csv.reader(stream, strict=True))
        if len(observed) != len(original) or any(len(a) != len(b) for a, b in zip(original, observed)):
            raise AssertionError("CSV dimensions changed unexpectedly")
        for r, (before_row, after_row) in enumerate(zip(original, observed)):
            for c, (before, after) in enumerate(zip(before_row, after_row)):
                expected = targets[(r, c)]["after"] if (r, c) in targets else before
                if after != expected:
                    raise AssertionError("An unapproved cell was modified")
        if file_sha256(source) != plan["source_sha256"]:
            raise AssertionError("Original file changed during execution")
        record = {"status": "verified", "operation": "replace_cells",
                  "finding_id": finding["id"], "reviewer": reviewer.strip(),
                  "source_sha256": plan["source_sha256"], "output_sha256": file_sha256(output),
                  "output": str(output), "changes_applied": len(targets),
                  "all_other_cells_preserved": True, "changes": plan["changes"]}
        (output_dir / "approved_finding.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record
    except Exception:
        output.unlink(missing_ok=True)
        raise
