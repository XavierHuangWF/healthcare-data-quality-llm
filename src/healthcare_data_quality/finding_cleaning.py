"""Finding-scoped cleaning review and independently verified derived CSV copies.

The local model suggests *data*, never executable Python. One reviewed plan can
change only the physical positions established by its selected finding. Existing
source files are immutable and each accepted plan produces a new version.
"""

import csv
import json
import re
from pathlib import Path
from uuid import uuid4

from .cleaning_authorization import file_sha256
from .cleaning_workspace import validate_header_plan
from .finding_evidence import dataset_finding_evidence
from .ingestion import inspect_headers

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
            items.append({"id": f"gemma:{round_number}:check", "dataset_id": "",
                          "name": "Gemma", "title": str(result.get("summary") or proposal.get("hypothesis") or "Completed Gemma check"),
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


def validate_scoped_plan(plan: dict, finding: dict, source: Path) -> dict:
    """Reject out-of-scope edits and stale plans; never trust model locations."""
    if not finding.get("actionable", True):
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
            raise ValueError("This column is not evidenced in the selected finding")
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
                                 full_finding_accepted: bool) -> tuple[dict, int]:
    """Expand a homogeneous example plan to its complete, evidenced scope.

    Never infer new cell values or extrapolate to a whole column. Expansion is
    allowed only after the researcher accepts an action covering *all* the
    selected finding, all source coordinates are independently in its evidence,
    and every evidenced original value is identical to the verified examples.
    Heterogeneous or ambiguous proposals remain strictly partial.
    """
    checked = validate_scoped_plan(plan, finding, source)
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
