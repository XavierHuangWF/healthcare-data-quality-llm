"""Safely preview previously reviewed cleaning RULES against a current CSV.

No edits are made. Every returned operation is independently validated against
current source evidence. Historical coordinates are never copied into a new CSV.
Rules that cannot be inferred unambiguously are deliberately not auto-adapted.
"""
from __future__ import annotations

import csv
import hashlib
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .finding_cleaning import (
    MAX_CELL_CHANGES, allowed_positions, requested_blank_row_scope,
    validate_scoped_plan, _evidenced_taxon_groups,
)


def annotate_source_headers(operations: list[dict], source: Path) -> list[dict]:
    """Add portable header identity when persisting human-approved plans.

    The existing local review store accepts optional change metadata, so the
    store itself and investigation cache fingerprint need not change.
    This does not mutate staged state or the supplied plans.
    """
    headers = _records(source)[0]
    result = []
    for operation in operations:
        new = dict(operation)
        changes = []
        for original in operation.get("changes", []):
            change = dict(original)
            col = change.get("column", change.get("position"))
            if type(col) is int and 1 <= col <= len(headers):
                change["source_header"] = headers[col - 1]
            changes.append(change)
        new["changes"] = changes
        result.append(new)
    return result


def _records(path: Path) -> list[list[str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, strict=True))
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Current CSV must have consistent columns")
    return rows


def _negative_absolute(before: str, after: str) -> bool:
    try:
        a, b = Decimal(before), Decimal(after)
        return a < 0 and b == abs(a) and a.is_finite() and b.is_finite()
    except (InvalidOperation, ValueError, TypeError):
        return False


def _numeric_test(value: str, *, negative: bool = False,
                  fractional: bool = False) -> bool:
    try:
        number = Decimal(value)
        return (number.is_finite() and
                (not negative or number < 0) and
                (not fractional or number % 1 != 0))
    except (ValueError, InvalidOperation, TypeError):
        return False


def _policy(changes: list[dict], user_request: str):
    pairs = [(c.get("before"), c.get("after")) for c in changes]
    if not pairs or any(not isinstance(x, str) or not isinstance(y, str)
                        for x, y in pairs):
        return None
    if all(x.strip() == "" and y == pairs[0][1] for x, y in pairs):
        return ("blank_to", pairs[0][1])
    if all(_negative_absolute(x, y) for x, y in pairs) and (
        len(pairs) > 1 or re.search(r"\b(?:absolute|positive|all|every)\b", user_request, re.I)
    ):
        return ("absolute", "")
    if len({y for x, y in pairs}) == 1:
        value = pairs[0][1]
        if all(_numeric_test(x, negative=True) for x, y in pairs) and (
            len(pairs) > 1 or re.search(r"\bnegative\b", user_request, re.I)
        ):
            return ("negative_to", value)
        if all(_numeric_test(x, fractional=True) for x, y in pairs) and (
            len(pairs) > 1 or re.search(r"\bfractional\b", user_request, re.I)
        ):
            return ("fractional_to", value)
    # Do not infer multiply/divide, or apply a scientific correction to
    # unrelated values. Exact before/after mappings are the only safe fallback.
    mapping = {}
    for before, after in pairs:
        if before in mapping and mapping[before] != after:
            return None
        mapping[before] = after
    return ("mapping", mapping)


def _target_value(value: str, policy):
    kind, arg = policy
    if kind == "blank_to":
        return arg if not value.strip() else None
    if kind == "negative_to":
        return arg if _numeric_test(value, negative=True) else None
    if kind == "fractional_to":
        return arg if _numeric_test(value, fractional=True) else None
    if kind == "absolute":
        try:
            number = Decimal(value)
            return format(abs(number), "f") if number.is_finite() and number < 0 else None
        except (InvalidOperation, ValueError):
            return None
    return arg.get(value)


def adapt_reviewed_rule(record: dict, finding: dict, source: Path,
                        *, exact: bool = False) -> dict:
    """Return validated current plans and an honest applicability explanation.

    All remembered actions must be portable; one unsafe action prevents a
    misleading partial proposal. Exact matches still undergo source validation.
    For new datasets, cell replacements target only current finding evidence.
    """
    operations = record.get("operations") or []
    if not operations or not isinstance(operations, list) or len(operations) > 16:
        raise ValueError("This review has no reusable edit rule")
    rows = _records(source)
    headers = rows[0]
    allowed_cells, allowed_headers = allowed_positions(finding, source)
    row_scope = None
    column_scope = None
    if exact:
        raw = Path(source).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.get("source_sha256"):
            raise ValueError("Saved source fingerprint does not match the current CSV")
        row_scope = frozenset(
            c["row"] for p in operations if p.get("operation") == "drop_rows"
            for c in p.get("changes", [])
        )
        column_scope = frozenset(
            c["position"] for p in operations if p.get("operation") == "drop_column"
            for c in p.get("changes", [])
        )
    plans = []
    for saved in operations:
        if not isinstance(saved, dict):
            raise ValueError("Saved operation is invalid")
        op = saved.get("operation")
        changes = saved.get("changes") or []
        if not changes or not isinstance(changes, list):
            raise ValueError("Saved operation has no verified changes")
        if exact:
            proposed = {"operation": op, "dataset_id": finding["dataset_id"],
                        "changes": changes}
            checked = validate_scoped_plan(
                proposed, finding, source,
                user_requested_rows=row_scope or None,
                user_requested_columns=column_scope or None,
            )
            plans.append(checked)
            continue
        if op == "replace_cells":
            policy = _policy(changes, str(record.get("user_request") or ""))
            if policy is None:
                raise ValueError("Replacement rule has conflicting previous values")
            # Exact-value mappings may be column-specific. New memories
            # pin original headers; legacy memories without bindings rely on
            # current finding evidence and still require human approval.
            bounded_headers = None
            bindings = record.get("source_header_bindings") or {}
            has_header_identity = bool(bindings) or any(
                c.get("source_header") is not None for c in changes)
            if policy[0] == "mapping" and has_header_identity:
                bounded_headers = set()
                for old_change in changes:
                    name = old_change.get("source_header")
                    if name is None:
                        name = bindings.get(str(old_change.get("column")))
                    if name is None or headers.count(name) != 1:
                        raise ValueError("The previously edited column header is absent or ambiguous")
                    bounded_headers.add(name)
            # New CSV locations come ONLY from the current finding, not memory.
            new = []
            for row, col in sorted(allowed_cells):
                if not 2 <= row <= len(rows) or not 1 <= col <= len(headers):
                    continue
                if bounded_headers is not None and headers[col - 1] not in bounded_headers:
                    continue
                old = rows[row-1][col-1]
                after = _target_value(old, policy)
                if after is not None and after != old:
                    new.append({"row": row, "column": col,
                                "before": old, "after": after})
            if not new:
                raise ValueError("No current evidenced cells match this replacement rule")
            if len(new) > MAX_CELL_CHANGES:
                raise ValueError("Too many current matches to review as one action")
            proposed = {"operation": op, "dataset_id": finding["dataset_id"],
                        "changes": new}
        elif op in {"drop_column", "rename_headers"}:
            new = []
            for saved_change in changes:
                header = saved_change.get("before")
                # A remembered column number is NOT portable. Header must
                # uniquely identify a current column evidenced by this finding.
                matches = [idx for idx, name in enumerate(headers, 1) if name == header]
                if not isinstance(header, str) or len(matches) != 1:
                    raise ValueError("A previous column header is missing or ambiguous")
                position = matches[0]
                if op == "rename_headers" and position not in allowed_headers:
                    raise ValueError("The matched header is outside this finding")
                if op == "drop_column":
                    # A *unique* matching header supports a proposed delete,
                    # not approval; user must opt into review and approve it.
                    column_scope = (column_scope or frozenset()) | {position}
                new.append({"position": position, "before": header,
                            "after": saved_change.get("after", "")})
            # Column deletion validator accepts only one column per operation.
            proposed = {"operation": op, "dataset_id": finding["dataset_id"],
                        "changes": new}
        elif op == "merge_rows":
            # Reuse only the previously approved sum-count rule for pairs of
            # duplicate taxa that THIS finding independently verifies.
            if finding.get("check_key") != "find_duplicate_taxa" or any(
                c.get("after") != "sum_numeric" for c in changes
            ):
                raise ValueError("Merge needs source-verified duplicate taxa and sum-count semantics")
            evidenced = _evidenced_taxon_groups(finding, rows)
            groups = {}
            for row, taxon in evidenced.items():
                groups.setdefault(taxon, []).append(row)
            if not groups or any(len(members) != 2 for members in groups.values()):
                raise ValueError("Only evidenced pairs of identical taxa can be merged automatically")
            new = [{"row": min(members), "source_row": max(members),
                    "before": taxon, "after": "sum_numeric"}
                   for taxon, members in sorted(groups.items())]
            if len(new) > 32:
                raise ValueError("Too many duplicate pairs for one reviewed merge")
            proposed = {"operation": op, "dataset_id": finding["dataset_id"],
                        "changes": new}
        elif op == "drop_rows":
            # Portable only if the OLD user explicitly requested a global
            # empty-cell predicate. Never reuse numeric row IDs across files.
            request = str(record.get("user_request") or "")
            row_scope = requested_blank_row_scope(request, source)
            if row_scope is None or not row_scope:
                raise ValueError("Row coordinates cannot be transferred to a new dataset")
            if len(row_scope) >= len(rows)-1:
                raise ValueError("Rule would remove every data row; manual review required")
            new = [{"row": r, "before": rows[r-1][0], "after": ""}
                   for r in sorted(row_scope)]
            proposed = {"operation": "drop_rows", "dataset_id": finding["dataset_id"],
                        "changes": new}
        else:
            raise ValueError(f"{op} needs a new, dataset-specific investigation")
        checked = validate_scoped_plan(
            proposed, finding, source,
            user_requested_rows=frozenset(row_scope) if row_scope else None,
            user_requested_columns=column_scope,
        )
        plans.append(checked)

    dropped = {c["position"] for p in plans if p["operation"] == "drop_column"
               for c in p["changes"]}
    if any(c["column"] in dropped for p in plans if p["operation"] == "replace_cells"
           for c in p["changes"]):
        raise ValueError("A replacement would be discarded by a column deletion")
    if len(dropped) != sum(len(p["changes"]) for p in plans if p["operation"] == "drop_column"):
        raise ValueError("Two saved actions target the same column")
    return {"plans": plans, "row_scope": frozenset(row_scope) if row_scope else frozenset(),
            "column_scope": frozenset(column_scope) if column_scope else frozenset(),
            "exact": exact, "matched_actions": len(plans)}


def current_change_rows(plans: list[dict]) -> list[dict]:
    """Compact before/after preview from CURRENT verified coordinates."""
    result = []
    for index, plan in enumerate(plans, 1):
        op = plan["operation"]
        for change in plan["changes"]:
            if op == "replace_cells":
                target = f"R{change['row']} · C{change['column']}"
                after = change["after"]
                before = change["before"]
            elif op in {"drop_column", "rename_headers"}:
                target = f"Column {change['position']}"
                before = change["before"]
                after = "Delete column" if op == "drop_column" else change["after"]
            elif op == "drop_rows":
                target = f"Row {change['row']}"
                before = change["before"]
                after = "Delete row"
            elif op == "merge_rows":
                target = f"Rows {change['row']} + {change['source_row']}"
                before = change.get("before", "Two original rows")
                after = f"Merged into row {change['row']} (sum counts)"
            else:
                continue
            result.append({"Action": index, "Target": target,
                           "Before": before, "After": after})
    return result
