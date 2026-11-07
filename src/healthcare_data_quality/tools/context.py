
import csv
from pathlib import Path


METADATA_CONTEXT = "metadata with SampleID and NPDNum columns"
STRUCTURE_CONTEXT = "CSV header row available"

def verify_contexts(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        headers = next(csv.reader(file), [])

    verified = set()

    if headers:
        verified.add(STRUCTURE_CONTEXT)

    if (
        headers.count("SampleID") == 1
        and headers.count("NPDNum") == 1
    ):
        verified.add(METADATA_CONTEXT)
    if looks_like_taxonomy_matrix(path):
        verified.add(CANDIDATE_TAXONOMY_CONTEXT)
    return verified

CANDIDATE_TAXONOMY_CONTEXT = "candidate taxonomy-by-sample matrix"


def looks_like_taxonomy_matrix(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader, [])

        if len(headers) < 3:
            return False

        examples = []

        for row in reader:
            if not row or not row[0].strip():
                continue

            examples.append(row[0])

            if len(examples) >= 30:
                break

    if len(examples) < 3:
        return False

    taxonomy_like = sum(
        value.count(";") + value.count("|") >= 4
        for value in examples
    )

    return taxonomy_like / len(examples) >= 0.8
