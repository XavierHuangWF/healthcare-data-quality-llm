
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

    return verified
