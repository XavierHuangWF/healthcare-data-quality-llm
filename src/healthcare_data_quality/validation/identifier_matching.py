
import csv
from pathlib import Path
import re

def normalize_sample_id(value: str) -> str:
    return value.strip().upper()


def compare_sample_ids(metadata: Path, microbiome: Path) -> dict:
    with metadata.open("r", encoding="utf-8-sig", newline="") as file:
        metadata_ids = {
            row["SampleID"]
            for row in csv.DictReader(file)
            if row.get("SampleID")
        }

    with microbiome.open("r", encoding="utf-8-sig", newline="") as file:
        microbiome_ids = set(next(csv.reader(file))[1:])

    exact = metadata_ids & microbiome_ids

    normalized_metadata = {
        normalize_sample_id(value) for value in metadata_ids
    }
    normalized_microbiome = {
        normalize_sample_id(value) for value in microbiome_ids
    }

    normalized = normalized_metadata & normalized_microbiome
    additional = normalized - {
        normalize_sample_id(value) for value in exact
    }


    unmatched_metadata = [
        value for value in metadata_ids
        if normalize_sample_id(value) not in normalized_microbiome
    ]

    unmatched_microbiome = [
        value for value in microbiome_ids
        if normalize_sample_id(value) not in normalized_metadata
    ]

    candidates = []

    for micro_id in sorted(unmatched_microbiome):
        micro_match = re.fullmatch(
            r"([A-Z]+)([0-9]+)",
            normalize_sample_id(micro_id),
        )

        if micro_match is None:
            continue

        for meta_id in sorted(unmatched_metadata):
            meta_match = re.fullmatch(
                r"([A-Z]+)([0-9]+)",
                normalize_sample_id(meta_id),
            )

            if meta_match is None:
                continue

            micro_prefix, micro_number = micro_match.groups()
            meta_prefix, meta_number = meta_match.groups()

            if micro_number != meta_number:
                continue

            if len(micro_prefix) != len(meta_prefix):
                continue

            differences = sum(
                a != b
                for a, b in zip(micro_prefix, meta_prefix)
            )

            if differences == 1:
                candidates.append({
                    "microbiome_id": micro_id,
                    "metadata_id": meta_id,
                    "status": "requires_review",
                })


    return {
        "exact_matches": len(exact),
        "normalized_matches": len(normalized),
        "additional_matches": sorted(additional),
        "suspected_matches": candidates,
    }


if __name__ == "__main__":
    import sys

    result = compare_sample_ids(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
    )
    print(result)
