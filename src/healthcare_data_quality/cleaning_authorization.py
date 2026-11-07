
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4


def authorize_cleaning_request(request_path: Path) -> dict | None:
    request = verify_pending_request(request_path)

    print("\nPROPOSED CLEANING OPERATION")
    print("Operation:", request["operation"])
    print("Source:", request["source"])
    print("Source SHA256:", request["source_sha256"])
    print("R draft SHA256:", request["draft_sha256"])
    print("Output:", request["output"])
    print("Allowed changes:")

    for change in request["allowed_changes"]:
        print(
            f"  Column {change['position']}: "
            f"{change['before']} -> {change['after']}"
        )

    print("\nOriginal dataset will remain unchanged.")
    print("Authorization permits only the displayed changes.")
    print("Semantic correctness still requires verification.")

    review = input(
        "\nAccept this exact cleaning plan for authorization review? "
        "[y/N]: "
    ).strip().lower()

    if review != "y":
        print("DEFERRED: Cleaning plan not accepted")
        return None

    reviewer = input("\nReviewer ID: ").strip()

    if not reviewer:
        print("DEFERRED: Reviewer ID required")
        return None

    confirmation_phrase = "AUTHORIZE HEADER CLEANING TO COPY"

    confirmation = input(
        f"Type {confirmation_phrase}: "
    ).strip()

    if confirmation != confirmation_phrase:
        print("DEFERRED: Authorization declined")
        return None

    # Detect changes made during human review.
    verify_pending_request(request_path)

    record = {
        "authorization_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": request["operation"],
        "source": request["source"],
        "source_sha256": request["source_sha256"],
        "draft": request["draft"],
        "draft_sha256": request["draft_sha256"],
        "output": request["output"],
        "allowed_changes": request["allowed_changes"],
        "reviewer": reviewer,
        "human_confirmation": True,
        "execution_status": "not_executed",
        "plan_review": "accepted",
    }

    authorization_path = Path(
        "data/decision_memory/operation_authorizations.jsonl"
    )

    authorization_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with authorization_path.open(
        "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(record) + "\n")

    print("\nCLEANING AUTHORIZATION: RECORDED")
    print("Authorization ID:", record["authorization_id"])
    print("Execution status: not_executed")

    return record


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_pending_request(request_path: Path) -> dict:
    request = json.loads(
        request_path.read_text(encoding="utf-8")
    )

    if request.get("operation") != "rename_headers":
        raise ValueError("Unsupported operation")

    if (
        request.get("human_authorization") != "pending"
        or request.get("execution_status") != "not_executed"
    ):
        raise PermissionError("Request is not pending")

    for field in ("source", "draft"):
        path = Path(request[field])

        if not path.is_file():
            raise FileNotFoundError(path)

        if file_sha256(path) != request[f"{field}_sha256"]:
            raise ValueError(f"{field} fingerprint mismatch")

    source = Path(request["source"])
    output = Path(request["output"])
    draft = Path(request["draft"])

    if output.resolve() in (
        source.resolve(),
        draft.resolve(),
    ):
        raise ValueError("Output conflicts with input")

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    with source.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        headers = next(csv.reader(file))

    changes = request["allowed_changes"]

    if not isinstance(changes, list) or not changes:
        raise ValueError("No allowed changes")

    positions = set()
    new_names = set()

    for change in changes:
        position = change["position"]
        before = change["before"]
        after = change["after"]

        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 1
            or position > len(headers)
            or position in positions
            or headers[position - 1] != before
            or not isinstance(after, str)
            or not after
            or after == before
            or after in headers
            or after in new_names
        ):
            raise ValueError("Invalid header change")

        positions.add(position)
        new_names.add(after)

    return request


def verify_authorized_request(
    request_path: Path,
    authorization_id: str,
) -> dict:
    request = verify_pending_request(request_path)

    authorization_path = Path(
        "data/decision_memory/operation_authorizations.jsonl"
    )

    if not authorization_path.is_file():
        raise PermissionError("Authorization history missing")

    with authorization_path.open(encoding="utf-8") as file:
        matches = [
            json.loads(line)
            for line in file
            if line.strip()
            and json.loads(line).get("authorization_id")
            == authorization_id
        ]

    if len(matches) != 1:
        raise PermissionError(
            "Authorization missing or duplicated"
        )

    record = matches[0]

    required_fields = (
        "operation",
        "source",
        "source_sha256",
        "draft",
        "draft_sha256",
        "output",
        "allowed_changes",
    )

    for field in required_fields:
        if record.get(field) != request[field]:
            raise PermissionError(
                f"Authorization mismatch: {field}"
            )

    if (
        record.get("plan_review") != "accepted"
        or record.get("human_confirmation") is not True
        or record.get("execution_status") != "not_executed"
        or not str(record.get("reviewer", "")).strip()
    ):
        raise PermissionError(
            "Authorization is not valid for execution"
        )

    return record
