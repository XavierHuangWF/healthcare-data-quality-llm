
from collections import Counter


def summarize_profile(profile: dict) -> dict:
    columns = profile["columns"]
    names = Counter(c["name"] for c in columns if c["name"])

    return {
        "dataset": profile["dataset"],
        "rows": profile["rows"],
        "total_columns": len(columns),
        "fully_numeric_columns": sum(
            c["nonmissing"] > 0
            and c["numeric"] == c["nonmissing"]
            for c in columns
        ),
        "total_missing_cells": sum(c["missing"] for c in columns),
        "blank_header_positions": [
            c["position"] for c in columns
            if not c["name"].strip()
        ],
        "duplicate_header_positions": [
            c["position"] for c in columns
            if c["name"] and names[c["name"]] > 1
        ],
        "column_preview": [
            {
                "position": c["position"],
                "name": c["name"],
                "examples": c["examples"][:2],
            }
            for c in columns[:3]
        ],
    }

def select_profile(profile: dict) -> dict:
    if len(profile["columns"]) <= 20:
        return profile

    return summarize_profile(profile)