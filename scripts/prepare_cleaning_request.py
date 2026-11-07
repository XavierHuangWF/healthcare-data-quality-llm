import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rename", action="append", required=True)
    args = parser.parse_args()

    with args.input.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        headers = next(csv.reader(file))

    renames = {}

    for item in args.rename:
        if "=" not in item:
            raise ValueError("Expected OLD=NEW")

        old, new = item.split("=", 1)

        if not old or not new or old == new:
            raise ValueError("Invalid header mapping")

        if old in renames or headers.count(old) != 1:
            raise ValueError(f"Ambiguous source header: {old}")

        if new in headers or new in renames.values():
            raise ValueError(f"Target header already exists: {new}")

        renames[old] = new

    if args.output.resolve() == args.input.resolve():
        raise ValueError("Cannot overwrite original dataset")

    if args.output.exists():
        raise FileExistsError(args.output)

    request = {
        "operation": "rename_headers",
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "draft": str(args.draft.resolve()),
        "draft_sha256": sha256(args.draft),
        "output": str(args.output.resolve()),
        "allowed_changes": [
            {
                "position": headers.index(old) + 1,
                "before": old,
                "after": new,
            }
            for old, new in renames.items()
        ],
        "human_authorization": "pending",
        "execution_status": "not_executed",
    }

    destination = Path(
        "data/local_tests/header_cleaning_request.json"
    )

    if destination.exists():
        raise FileExistsError(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(request, indent=2) + "\n",
        encoding="utf-8",
    )

    print("CLEANING REQUEST: PREPARED")
    print("Source SHA256:", request["source_sha256"])
    print("Draft SHA256:", request["draft_sha256"])
    print("Allowed changes:", request["allowed_changes"])
    print("Authorization: pending")
    print("No data modified.")


if __name__ == "__main__":
    main()