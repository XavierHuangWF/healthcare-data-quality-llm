"""Deterministic full-source CSV inventory for read-only Gemma context.

Never infer missingness from a single example. "NA" as a taxonomy rank is
kept distinct from an empty count cell. This module never changes source data.
"""

import csv
import re
from pathlib import Path

_TAXON_NA = re.compile(r"^(?:[a-z]__\s*)?(?:NA|N/A)$", re.IGNORECASE)


def source_inventory(path: Path, *, row_limit: int = 500, column_limit: int = 500) -> dict:
    """Scan every CSV record; counts are complete even if display rows are capped."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        headers = next(reader, None)
        if not headers:
            raise ValueError("The source CSV has no header")
        if len(headers) > column_limit:
            raise ValueError("Source has more columns than the bounded inventory supports")
        columns = [{"position": i+1, "header": name, "empty_count": 0,
                    "exact_na_count": 0, "empty_rows": [], "exact_na_rows": []}
                   for i, name in enumerate(headers)]
        taxon_header = headers[0].strip().casefold()
        is_taxonomy_matrix = (taxon_header in {"", "taxon", "taxa", "taxonomy", "feature", "species"}
                              or "taxon" in taxon_header)
        taxonomy_na_rows = []
        taxonomy_na_count = 0
        data_rows = 0
        empty_count_row_ids = set()
        for row_number, cells in enumerate(reader, start=2):
            data_rows += 1
            if len(cells) != len(headers):
                raise ValueError(f"Non-rectangular CSV at row {row_number}")
            tokens = ([value.strip() for value in re.split(r"[;|]", cells[0])]
                      if is_taxonomy_matrix else [])
            if any(_TAXON_NA.fullmatch(token) for token in tokens):
                taxonomy_na_count += 1
                if len(taxonomy_na_rows) < row_limit:
                    taxonomy_na_rows.append(row_number)
            if any(not value.strip() for value in cells[1:]):
                empty_count_row_ids.add(row_number)
            for col, value in zip(columns, cells):
                v = value.strip()
                if not v:
                    col["empty_count"] += 1
                    if len(col["empty_rows"]) < row_limit:
                        col["empty_rows"].append(row_number)
                elif v.casefold() in ("na", "n/a"):
                    col["exact_na_count"] += 1
                    if len(col["exact_na_rows"]) < row_limit:
                        col["exact_na_rows"].append(row_number)
    controls = [col for col in columns if "blank" in col["header"].casefold()]
    return {
        "data_rows": data_rows, "csv_rows_including_header": data_rows + 1,
        "column_count": len(columns), "columns": columns,
        "taxonomy_matrix": is_taxonomy_matrix,
        "blank_control_columns": controls,
        "taxonomy_na_row_count": taxonomy_na_count,
        "taxonomy_na_rows": taxonomy_na_rows,
        "taxonomy_na_rows_truncated": taxonomy_na_count > len(taxonomy_na_rows),
        "empty_cells_total": sum(col["empty_count"] for col in columns),
        "empty_count_cells_total": sum(col["empty_count"] for col in columns[1:]),
        "empty_count_row_count": len(empty_count_row_ids),
        "empty_count_rows": sorted(empty_count_row_ids),
        "exact_na_cells_total": sum(col["exact_na_count"] for col in columns),
    }


def concise_source_context(inventory: dict) -> dict:
    """Bounded evidence for the Gemma response budget, not fabricated coverage."""
    return {
        "all_rows_scanned": True, "data_rows": inventory["data_rows"],
        "csv_header_is_row_1": True,
        "taxonomy_column": 1,
        "taxonomy_na_level_rows_count": inventory["taxonomy_na_row_count"],
        "taxonomy_na_level_rows": inventory["taxonomy_na_rows"][:120],
        "taxonomy_rows_truncated": inventory["taxonomy_na_rows_truncated"] or len(inventory["taxonomy_na_rows"]) > 120,
        "note": "A taxonomy NA rank is a value, not an empty count cell. Do not assume only the first example contains NA.",
        "empty_count_cells_total": inventory["empty_count_cells_total"],
        "empty_count_rows_total": inventory["empty_count_row_count"],
        "empty_count_rows": inventory["empty_count_rows"][:500],
        "exact_na_count_cells_total": inventory["exact_na_cells_total"],
        "blank_control_columns": [
            {"position": c["position"], "header": c["header"],
             "empty_count": c["empty_count"], "empty_rows": c["empty_rows"][:30],
             "exact_na_count": c["exact_na_count"]}
            for c in inventory["blank_control_columns"][:30]
        ],
    }
