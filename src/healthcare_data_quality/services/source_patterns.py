"""Read-only dataset-wide pattern summary for evidence-backed model suggestions.

No patient/taxon names, CSV values, source paths, or identifiers are returned.
The model sees only aggregate normal-format and count-cell patterns, not new
permissions to edit. A bounded scan is explicit about any truncation.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

PREFIXES = ("k__", "p__", "c__", "o__", "f__", "g__", "s__")
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def summarize_source_patterns(source: Path, *, max_data_rows: int = 20000) -> dict:
    """Describe the observed *majority* of ordinary entries before proposing edits."""
    if type(max_data_rows) is not int or not 1 <= max_data_rows <= 100000:
        raise ValueError("Invalid source-profile row limit")
    taxonomy = Counter()
    counts = Counter()
    rows_examined = 0
    truncated = False
    with Path(source).open(encoding="utf-8-sig", newline="") as f:
        source_reader = csv.reader(f, strict=True)
        header = next(source_reader)
        if not header:
            raise ValueError("CSV is missing column headers")
        for index, record in enumerate(source_reader):
            if index >= max_data_rows:
                truncated = True
                break
            if len(record) != len(header):
                raise ValueError("CSV contains inconsistent row widths")
            rows_examined += 1
            first = record[0]
            chunks = re.split(r"[;|]", first)
            if len(chunks) == 7:
                taxonomy["seven_levels"] += 1
                meaningful = [(i, x.strip()) for i, x in enumerate(chunks)
                              if x.strip() and x.strip().upper() != "NA"]
                if meaningful:
                    matched = sum(text.casefold().startswith(PREFIXES[i]) for i, text in meaningful)
                    if matched == len(meaningful):
                        taxonomy["fully_prefixed"] += 1
                    elif matched == 0:
                        taxonomy["unprefixed"] += 1
                    else:
                        taxonomy["mixed_prefixes"] += 1
            if ";" in first:
                taxonomy["semicolon_rows"] += 1
            if "|" in first:
                taxonomy["pipe_rows"] += 1
            for value in record[1:]:
                counts["cells"] += 1
                t = value.strip()
                if not t:
                    counts["empty"] += 1
                elif t.upper() == "NA":
                    counts["NA"] += 1
                elif _NUMBER.fullmatch(t):
                    counts["numeric"] += 1
                    if float(t) == 0:
                        counts["zeros"] += 1
                    if float(t) < 0:
                        counts["negative"] += 1
                    if float(t) % 1:
                        counts["fractional"] += 1
                else:
                    counts["non_numeric"] += 1
    if not rows_examined:
        raise ValueError("Source CSV has no data rows")
    prefix = ("prefixed" if taxonomy["fully_prefixed"] > taxonomy["unprefixed"]
              else "unprefixed" if taxonomy["unprefixed"] > taxonomy["fully_prefixed"]
              else "inconclusive")
    separator = ("semicolon" if taxonomy["semicolon_rows"] > taxonomy["pipe_rows"]
                 else "pipe" if taxonomy["pipe_rows"] > taxonomy["semicolon_rows"]
                 else "inconclusive")
    return {
        "data_rows_examined": rows_examined, "profile_truncated": truncated,
        "column_count": len(header),
        "empty_header_columns": sum(not name.strip() for name in header),
        "first_column_majority_taxonomy": taxonomy["seven_levels"] > rows_examined / 2,
        "normal_taxonomy_separator": separator,
        "normal_taxonomy_prefix_style": prefix,
        "taxonomy_seven_level_rows": taxonomy["seven_levels"],
        "taxonomy_semicolon_rows": taxonomy["semicolon_rows"],
        "taxonomy_pipe_rows": taxonomy["pipe_rows"],
        "count_cells_examined": counts["cells"],
        "count_zero_cells": counts["zeros"],
        "count_empty_cells": counts["empty"],
        "count_literal_NA_cells": counts["NA"],
        "count_negative_cells": counts["negative"],
        "count_fractional_cells": counts["fractional"],
        "count_non_numeric_cells": counts["non_numeric"],
    }
