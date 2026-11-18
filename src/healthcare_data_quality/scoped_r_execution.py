"""Two-stage, finding-scoped R cleanup: preview, isolated execution, verified merge.

Gemma supplies *structured proposed changes* only. We compile the approved
positions/values to deterministic R; we never execute model-authored R. The R
sandbox operates on a compact list of affected values, not the original CSV.
An independent writer merges ONLY those R-verified values into a new copy.
"""

import csv
from io import StringIO
from pathlib import Path
import hashlib
import json

from .cleaning_authorization import file_sha256
from .finding_cleaning import validate_scoped_plan
from .r_sandbox import MAX_R_BYTES, MAX_INPUT_BYTES, run_r_sandbox


MINI_HEADERS = ["kind", "csv_row", "csv_column", "value"]
# A fifth column is used ONLY as sandbox input. The immutable R compiler never
# interpolates research values into executable R code; output drops this column.
MINI_INPUT_HEADERS = MINI_HEADERS + ["replacement"]


def _r_literal(value: str) -> str:
    """Quote untrusted CSV content as a single R string literal."""
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("Unsupported NUL or non-text source value")
    # R interprets JSON-style quoted ASCII/control sequences; encode non-ASCII
    # as R Unicode escapes instead of depending on the sandbox's C locale.
    pieces = ['"']
    for char in value:
        codepoint = ord(char)
        if char == '"':
            pieces.append('\\"')
        elif char == "\\":
            pieces.append("\\\\")
        elif char == "\n":
            pieces.append("\\n")
        elif char == "\r":
            pieces.append("\\r")
        elif char == "\t":
            pieces.append("\\t")
        elif codepoint < 32 or codepoint == 127:
            pieces.append(f"\\x{codepoint:02X}")
        elif codepoint > 0xFFFF:
            pieces.append(f"\\U{codepoint:08X}")
        elif codepoint > 126:
            pieces.append(f"\\u{codepoint:04X}")
        else:
            pieces.append(char)
    return "".join(pieces) + '"'


def _affected_changes(plan: dict) -> list[dict]:
    operation = plan["operation"]
    if operation not in {"replace_cells", "rename_headers"}:
        raise ValueError("Only exact cell replacements and header renames have R execution contracts")
    changes = []
    for item in plan["changes"]:
        if operation == "rename_headers":
            changes.append({"kind": "header", "csv_row": "1",
                            "csv_column": str(item["position"]),
                            "before": item["before"], "after": item["after"]})
        else:
            changes.append({"kind": "cell", "csv_row": str(item["row"]),
                            "csv_column": str(item["column"]),
                            "before": item["before"], "after": item["after"]})
    return changes


def _as_csv(changes: list[dict], *, result: bool) -> str:
    """Create the original 4-column output or the isolated 5-column input.

    Replacement strings are CSV data, not executable R literals. csv.writer
    handles quoting/embedded control characters without generating R syntax.
    """
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MINI_HEADERS if result else MINI_INPUT_HEADERS)
    for change in changes:
        fields = [change["kind"], change["csv_row"], change["csv_column"]]
        if result:
            writer.writerow(fields + [change["after"]])
        else:
            writer.writerow(fields + [change["before"], change["after"]])
    return buffer.getvalue()


def _generate_r(changes: list[dict]) -> str:
    """Compile a fixed, data-driven R transform; no user strings in R source.

    This avoids data-dependent parser errors caused by embedding thousands of
    before/after strings inside a shell `-e` expression. The R input is already
    a strict, source-verified finding-scoped table. Output is verified again
    coordinate by coordinate against the exact approved plan.
    """
    if not changes:
        raise ValueError("An R transformation requires at least one affected value")
    for change in changes:
        if change.get("kind") not in {"cell", "header"}:
            raise ValueError("Unsupported affected-value kind")
    # The only interpolated text is a positive integer controlled by Python.
    # The R function body is static, including its CSV header literals.
    return ("# Deterministic R cleaning from independently validated findings.\n"
            "# Original and replacement values are data in the mini CSV input.\n"
            "clean_dataset <- function(df) {\n"
            "  stopifnot(is.data.frame(df))\n"
            '  stopifnot(identical(names(df), c("kind", "csv_row", "csv_column", "value", "replacement")))\n'
            f"  stopifnot(nrow(df) == {len(changes)}L)\n"
            '  stopifnot(all(df$kind %in% c("cell", "header")))\n'
            '  stopifnot(!anyNA(df$kind), !anyNA(df$csv_row), !anyNA(df$csv_column))\n'
            '  stopifnot(!anyDuplicated(paste(df$kind, df$csv_row, df$csv_column, sep="|")))\n'
            '  df$value <- df$replacement\n'
            '  df$replacement <- NULL\n'
            '  df\n'
            '}\n')


def prepare_scoped_r_preview(plan: dict, finding: dict, source: Path) -> dict:
    """After the first human agreement, produce non-executed R and review data."""
    checked = validate_scoped_plan(plan, finding, Path(source))
    if checked != plan:
        raise PermissionError("The edit plan changed since validation")
    if file_sha256(Path(source)) != plan["source_sha256"]:
        raise ValueError("Working CSV changed before R code generation")
    changes = _affected_changes(plan)
    script = _generate_r(changes)
    input_csv = _as_csv(changes, result=False)
    if len(script.encode("utf-8")) > MAX_R_BYTES:
        raise ValueError("Approved changes exceed the isolated R script size limit")
    if len(input_csv.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Affected data exceed the isolated R input size limit")
    fingerprint = hashlib.sha256((script + "\x00" + input_csv + "\x00" +
                                  plan["source_sha256"]).encode("utf-8")).hexdigest()
    return {"finding_id": finding["id"], "plan": checked,
            "r_function": script, "affected_input_csv": input_csv,
            "changes": changes, "preview_sha256": fingerprint,
            "source_sha256": checked["source_sha256"]}


def execute_reviewed_r_preview(preview: dict, plan: dict, finding: dict,
                               source: Path) -> dict:
    """Run isolated R on affected values BEFORE approval; verify exact output.

    This stage cannot modify the source or working CSV. Only a later explicit
    reviewer approval may insert these verified values into a derived copy.
    """
    expected = prepare_scoped_r_preview(plan, finding, source)
    if preview != expected:
        raise PermissionError("Reviewed R preview is stale or altered")
    try:
        output = run_r_sandbox(preview["r_function"], preview["affected_input_csv"],
                               preserve_text=True)
        rows = list(csv.reader(StringIO(output), strict=True))
    except csv.Error as exc:
        raise ValueError("R returned an invalid CSV") from exc
    expected_rows = list(csv.reader(StringIO(_as_csv(preview["changes"], result=True)), strict=True))
    if rows != expected_rows:
        raise ValueError("R changed an unexpected value, coordinate, or row; no cleaning was applied")
    if file_sha256(Path(source)) != plan["source_sha256"]:
        raise ValueError("Source CSV changed during isolated R execution")
    return {"status": "r_execution_verified", "checked_changes": len(preview["changes"]),
            "r_code_sha256": hashlib.sha256(preview["r_function"].encode("utf-8")).hexdigest(),
            "source_sha256": plan["source_sha256"],
            "all_other_positions_excluded_from_r_input": True,
            "verified_output_csv": output,
            "verified_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest()}



def _plan_from_r_output(plan: dict, r_output: str) -> dict:
    """Make the verified R result (not untrusted model prose) the merge input."""
    rows = list(csv.reader(StringIO(r_output), strict=True))
    if not rows or rows[0] != MINI_HEADERS or len(rows) != len(plan["changes"]) + 1:
        raise ValueError("R result does not contain the expected affected positions")
    merged = []
    for original, row in zip(plan["changes"], rows[1:]):
        if plan["operation"] == "rename_headers":
            kind, csv_row, csv_column, value = row
            if (kind, csv_row, csv_column) != ("header", "1", str(original["position"])):
                raise ValueError("R result changed a header coordinate")
        else:
            kind, csv_row, csv_column, value = row
            if (kind, csv_row, csv_column) != ("cell", str(original["row"]), str(original["column"])):
                raise ValueError("R result changed a cell coordinate")
        if value != original["after"]:
            raise ValueError("R result differs from the verified replacement")
        merged.append({**original, "after": value})
    return {**plan, "changes": merged}


def _r_insertion_function(plan: dict) -> str:
    """A compact, replayable R insertion contract for a verified change table.

    Unlike the former code generator, this never emits one R statement per
    affected cell. The *data* (verified coordinates and before/after values)
    live in ``insertion_changes_csv``; the reusable R function is constant-size
    for 1 or 618 changes. No untrusted dataset value is interpolated into code.

    The application still uses its independently verified in-memory writer;
    this R code is an audit/reproduction specification, not executed on the
    user's original or in-memory dataset.
    """
    if not plan.get("changes"):
        raise ValueError("Insertion needs at least one verified change")
    operation = plan.get("operation")
    if operation not in {"replace_cells", "rename_headers"}:
        raise ValueError("Unsupported insertion operation")
    # Strict input shape and character columns preserve lexical CSV contents:
    # literal "NA", blank strings, leading zeros, Unicode, and duplicate names.
    # The user should read the source CSV with check.names=FALSE,
    # colClasses="character", and na.strings=character(0) for reproducibility.
    preamble = (
        "# Compact, data-driven R insertion SPECIFICATION (review only).\n"
        "# verified_changes has columns: kind,csv_row,csv_column,value,replacement\n"
        "# Supply this table from the independently VERIFIED R result.\n"
        "# CSV rows include the header; dataframe rows start at CSV row 2.\n"
        "# For faithful replay, import original CSV with colClasses='character',\n"
        "# check.names=FALSE, na.strings=character(0), stringsAsFactors=FALSE.\n"
        "clean_dataset <- function(df, verified_changes) {\n"
        "  stopifnot(is.data.frame(df), is.data.frame(verified_changes))\n"
        '  stopifnot(identical(names(verified_changes),\n'
        '    c("kind", "csv_row", "csv_column", "value", "replacement")))\n'
        "  stopifnot(all(vapply(verified_changes, is.character, logical(1))))\n"
        "  stopifnot(all(vapply(df, is.character, logical(1))))\n"
        "  stopifnot(nrow(verified_changes) > 0L)\n"
        '  stopifnot(all(grepl("^[1-9][0-9]*$", verified_changes$csv_column)))\n'
        "  cols <- suppressWarnings(as.integer(verified_changes$csv_column))\n"
        "  stopifnot(!anyNA(cols), all(cols <= ncol(df)))\n"
    )
    if operation == "replace_cells":
        body = (
            '  stopifnot(all(verified_changes$kind == "cell"))\n'
            '  stopifnot(all(grepl("^[1-9][0-9]*$", verified_changes$csv_row)))\n'
            "  rows <- suppressWarnings(as.integer(verified_changes$csv_row)) - 1L\n"
            "  stopifnot(!anyNA(rows), all(rows >= 1L), all(rows <= nrow(df)))\n"
            '  stopifnot(!anyDuplicated(paste(rows, cols, sep=":")))\n'
            "  for (col in unique(cols)) {  # batch all affected rows in one column\n"
            "    i <- which(cols == col)\n"
            "    stopifnot(identical(unname(df[[col]][rows[i]]),\n"
            "                        unname(verified_changes$value[i])))\n"
            "    df[[col]][rows[i]] <- verified_changes$replacement[i]\n"
            "  }\n"
        )
    else:
        body = (
            '  stopifnot(all(verified_changes$kind == "header"))\n'
            '  stopifnot(all(verified_changes$csv_row == "1"))\n'
            "  stopifnot(!anyDuplicated(cols))\n"
            "  stopifnot(identical(unname(names(df)[cols]),\n"
            "                      unname(verified_changes$value)))\n"
            "  names(df)[cols] <- verified_changes$replacement\n"
        )
    script = preamble + body + "  df\n}\n"
    if len(script.encode("utf-8")) > MAX_R_BYTES:
        raise ValueError("R insertion code exceeds size limit")
    return script


def prepare_and_execute_scoped_r_preview(plan: dict, finding: dict, source: Path) -> dict:
    """Before final approval, run isolated R ONLY on evidenced affected values.

    No working CSV is modified. Return verified R output and reviewable
    integration code; the researcher separately authorizes inserting that
    exact result into a new working CSV version.
    """
    preview = prepare_scoped_r_preview(plan, finding, source)
    verified_result = execute_reviewed_r_preview(preview, plan, finding, source)
    integration = _plan_from_r_output(plan, verified_result["verified_output_csv"])
    if validate_scoped_plan(integration, finding, source) != plan:
        raise PermissionError("The R output cannot be merged within the finding's scope")
    return {**preview, "r_result": verified_result,
            "insertion_plan": integration,
            "insertion_changes_csv": _as_csv(_affected_changes(integration), result=False),
            "insertion_r_function": _r_insertion_function(integration)}


def verify_prepared_r_insertion(prepared: dict, plan: dict, finding: dict,
                                source: Path) -> dict:
    """Final approval must be tied to the same source, R result, and exact edits."""
    expected = prepare_scoped_r_preview(plan, finding, source)
    if any(prepared.get(key) != expected.get(key) for key in expected):
        raise PermissionError("The R preview is stale or different from the reviewed plan")
    result = prepared.get("r_result") or {}
    output = result.get("verified_output_csv")
    if not isinstance(output, str) or result.get("status") != "r_execution_verified":
        raise ValueError("No verified isolated R result is ready for insertion")
    if hashlib.sha256(output.encode("utf-8")).hexdigest() != result.get("verified_output_sha256"):
        raise PermissionError("The isolated R output has changed since review")
    insertion_plan = _plan_from_r_output(plan, output)
    if insertion_plan != prepared.get("insertion_plan"):
        raise PermissionError("Insertion plan differs from the inspected R result")
    if _r_insertion_function(insertion_plan) != prepared.get("insertion_r_function"):
        raise PermissionError("The displayed R insertion code has changed")
    if _as_csv(_affected_changes(insertion_plan), result=False) != prepared.get("insertion_changes_csv"):
        raise PermissionError("The verified insertion change table has changed")
    if validate_scoped_plan(insertion_plan, finding, source) != plan:
        raise PermissionError("The working CSV or proposed positions changed since preview")
    return insertion_plan


def verify_r_against_current_staged_values(original: bytes, staged: bytes, insertion: dict) -> dict:
    """Run a second isolated R verification when a newer stage overlaps a plan.

    The original finding scope and R output were checked before this call.
    Re-run deterministic R against ONLY current staged affected values, never
    model-authored R or a full working CSV. No dataset bytes are modified here.
    """
    from .in_memory_cleaning import assess_verified_in_memory

    review = assess_verified_in_memory(original, staged, insertion)
    updates = review["ready_to_apply"] + review["conflicts"]
    if not review["conflicts"]:
        raise ValueError("No earlier approved change needs an R rebase")
    affected = []
    for item in updates:
        affected.append({
            "kind": "header" if insertion["operation"] == "rename_headers" else "cell",
            "csv_row": str(item["csv_row"]),
            "csv_column": str(item["column"]),
            "before": item["current"],
            "after": item["proposed"],
        })
    script = _generate_r(affected)
    r_input = _as_csv(affected, result=False)
    if (len(script.encode("utf-8")) > MAX_R_BYTES
            or len(r_input.encode("utf-8")) > MAX_INPUT_BYTES):
        raise ValueError("Rebased R preview exceeds sandbox limits")
    output = run_r_sandbox(script, r_input, preserve_text=True)
    expected = _as_csv(affected, result=True)
    try:
        if list(csv.reader(StringIO(output), strict=True)) != list(
                csv.reader(StringIO(expected), strict=True)):
            raise PermissionError("Rebased R output changes an unexpected coordinate/value")
    except csv.Error as exc:
        raise PermissionError("Rebased R output is invalid CSV") from exc
    # This fingerprint binds the proof to the exact in-memory version and
    # conflict table. It is invalid if a different finding stages first.
    return {
        "source_sha256": review["source_sha256"],
        "staged_sha256": review["staged_sha256"],
        "conflicts": review["conflicts"],
        # Bind approval to the verified logical values. R's CSV writer may
        # quote fields differently while returning identical parsed rows.
        "verified_r_output_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
        "raw_r_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "r_function": script,
        "changes_tested": len(updates),
    }
