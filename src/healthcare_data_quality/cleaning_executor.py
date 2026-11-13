
import csv
import shutil
from pathlib import Path

from .cleaning_authorization import (
    file_sha256,
    verify_authorized_request,
)


def execute_authorized_header_cleaning(
    request_path: Path,
    authorization_id: str,
) -> dict:
    authorization = verify_authorized_request(
        request_path,
        authorization_id,
    )

    source = Path(authorization["source"])
    draft = Path(authorization["draft"])
    output = Path(authorization["output"])
    changes = authorization["allowed_changes"]

    if authorization["operation"] != "rename_headers":
        raise ValueError("Unsupported cleaning operation")

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    expected = {
        "source": authorization["source_sha256"],
        "draft": authorization["draft_sha256"],
    }

    if (
        file_sha256(source) != expected["source"]
        or file_sha256(draft) != expected["draft"]
    ):
        raise PermissionError("Source or draft changed")

    created = False

    try:
        output.parent.mkdir(parents=True, exist_ok=True)

        with source.open("rb") as src:
            original_header = src.readline()

            if original_header.endswith(b"\r\n"):
                ending = b"\r\n"
            elif original_header.endswith(b"\n"):
                ending = b"\n"
            else:
                raise ValueError("Unsupported CSV header ending")

            header_bytes = original_header[:-len(ending)]
            fields = header_bytes.split(b",")

            parsed = next(
                csv.reader(
                    [header_bytes.decode("utf-8-sig")],
                    strict=True,
                )
            )

            if len(fields) != len(parsed):
                raise ValueError(
                    "Complex CSV header requires a different writer"
                )

            expected_headers = list(parsed)

            for change in changes:
                index = change["position"] - 1
                before = change["before"].encode("utf-8")
                after = change["after"].encode("utf-8")
                current = fields[index]

                if current == before:
                    fields[index] = after
                elif current == b'"' + before + b'"':
                    fields[index] = b'"' + after + b'"'
                else:
                    raise ValueError(
                        "Header encoding does not match request"
                    )

                expected_headers[index] = change["after"]

            updated_header = b",".join(fields) + ending

            with output.open("xb") as dst:
                created = True
                dst.write(updated_header)
                shutil.copyfileobj(src, dst, length=1024 * 1024)

        # Verify the produced copy independently of the writer.
        with source.open("rb") as src, output.open("rb") as dst:
            src.readline()
            result_header = dst.readline()

            result_ending = (
                b"\r\n" if result_header.endswith(b"\r\n")
                else b"\n"
            )

            observed_headers = next(
                csv.reader(
                    [
                        result_header[:-len(result_ending)]
                        .decode("utf-8-sig")
                    ],
                    strict=True,
                )
            )

            if observed_headers != expected_headers:
                raise AssertionError("Header verification failed")

            while True:
                before_chunk = src.read(1024 * 1024)
                after_chunk = dst.read(1024 * 1024)

                if before_chunk != after_chunk:
                    raise AssertionError(
                        "Original data records were modified"
                    )

                if not before_chunk:
                    break

        if (
            file_sha256(source) != expected["source"]
            or file_sha256(draft) != expected["draft"]
        ):
            raise AssertionError("Input changed during execution")

    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise

    return {
        "status": "verified",
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "authorization_id": authorization_id,
        "changes_applied": len(changes),
        "data_records_preserved": True,
    }
