
import csv
from collections import defaultdict
from pathlib import Path
import re

def find_duplicate_taxa(path: Path) -> list[dict]:
    taxa = defaultdict(list)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for record_number, row in enumerate(reader, start=2):
            taxon = row[0]
            counts = tuple(row[1:])

            taxa[taxon].append({
                "record": record_number,
                "counts": counts,
            })

    duplicates = []

    for taxon, records in taxa.items():
        if len(records) < 2:
            continue

        duplicates.append({
            "taxon": taxon,
            "records": [r["record"] for r in records],
            "counts_identical": len({
                r["counts"] for r in records
            }) == 1,
        })

    return duplicates


def inspect_taxonomy_structure(path: Path) -> dict:
    incomplete = []
    alternate_delimiters = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for record_number, row in enumerate(reader, start=2):
            original = row[0]

            if "|" in original:
                alternate_delimiters.append(record_number)

            levels = re.split(r"[;|]", original)

            if len(levels) != 7:
                incomplete.append({
                    "record": record_number,
                    "level_count": len(levels),
                })

    return {
        "incomplete_taxonomy": incomplete,
        "alternate_delimiters": alternate_delimiters,
    }


def find_unclassified_species(path: Path) -> list[dict]:
    findings = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for record_number, row in enumerate(reader, start=2):
            levels = [
                level.strip()
                for level in re.split(r"[;|]", row[0])
            ]

            # Incomplete hierarchies are reported separately.
            if len(levels) != 7:
                continue

            species = re.sub(
                r"^s__\s*", "", levels[6], flags=re.IGNORECASE
            ).strip()

            if species.upper() in {"NA", "N/A", "UNCLASSIFIED", "UNKNOWN", ""}:
                findings.append({
                    "record": record_number,
                    "original_taxonomy": row[0],
                    "species_value": levels[6],
                    "classification": "unclassified_species",
                })

    return findings


def inspect_taxonomy_naming(path: Path) -> dict:
    prefixes = ("k__", "p__", "c__", "o__", "f__", "g__", "s__")

    styles = {
        "fully_prefixed": [],
        "unprefixed": [],
        "mixed": [],
    }

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for record_number, row in enumerate(reader, start=2):
            levels = [
                level.strip()
                for level in re.split(r"[;|]", row[0])
            ]

            # Incomplete taxonomy is handled separately.
            if len(levels) != 7:
                continue

            matches = [
                value.lower().startswith(prefix)
                for value, prefix in zip(levels, prefixes)
            ]

            if all(matches):
                style = "fully_prefixed"
            elif not any(matches):
                style = "unprefixed"
            else:
                style = "mixed"

            styles[style].append(record_number)

    return styles



if __name__ == "__main__":
    import sys

    duplicates = find_duplicate_taxa(Path(sys.argv[1]))

    for item in duplicates:
        print({
            "records": item["records"],
            "counts_identical": item["counts_identical"],
            "taxon": item["taxon"].split(";")[-1].strip(),
        })

    print(f"Duplicate taxonomy groups: {len(duplicates)}")

    unclassified = find_unclassified_species(Path(sys.argv[1]))

    print(f"Unclassified species: {len(unclassified)}")
    print("First 3 records:", [
        item["record"] for item in unclassified[:3]
    ])


    styles = inspect_taxonomy_naming(Path(sys.argv[1]))

    for style, records in styles.items():
        print(f"{style}: {len(records)}")
        if style != "unprefixed":
            print(f"  Records: {records}")
