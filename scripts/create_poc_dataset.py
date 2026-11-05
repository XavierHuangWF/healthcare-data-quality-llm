import csv
import hashlib
import json
from pathlib import Path


def main():
    output = Path("data/synthetic/poc_duplicate_columns.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "SubjectID",
            "Score",
            "Site",
            "Score",
        ])

        for i in range(1, 81):
            score = "" if i % 17 == 0 else str(i * 3)

            writer.writerow([
                f"S{i:03d}",
                score,
                "A" if i % 2 else "B",
                score,
            ])


    provenance = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(
            output.read_bytes()
        ).hexdigest(),
        "source_type": "synthetic_fixture",
        "evidence_type": "column_lineage",
        "assertion": {
            "source_position": 2,
            "duplicate_position": 4,
            "column_name": "Score",
            "derivation": (
                "The dataset generator writes both columns "
                "from the same score variable."
            ),
            "intentionally_redundant": True,
        },
        "evidence_scope": "synthetic_poc_only",
        "execution_authorized": False,
    }

    provenance_path = output.with_suffix(
        ".provenance.json"
    )

    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Provenance:", provenance_path)


    print("SYNTHETIC POC DATASET: CREATED")
    print("Path:", output)
    print("Records: 80")
    print("Columns: 4")
    print("Duplicate column pair: 2 and 4")


if __name__ == "__main__":
    main()
