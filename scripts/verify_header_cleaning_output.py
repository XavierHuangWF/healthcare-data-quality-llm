
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    request = json.loads(
        Path(
            "data/local_tests/header_cleaning_request.json"
        ).read_text(encoding="utf-8")
    )

    source = Path(request["source"])
    output = Path(request["output"])

    assert source.is_file()
    assert output.is_file()
    assert sha256(source) == request["source_sha256"]

    with source.open("rb") as original, output.open("rb") as cleaned:
        source_header = original.readline()
        output_header = cleaned.readline()

        before = next(
            csv.reader([source_header.decode("utf-8-sig")])
        )
        after = next(
            csv.reader([output_header.decode("utf-8-sig")])
        )

        expected = before.copy()

        for change in request["allowed_changes"]:
            index = change["position"] - 1
            assert expected[index] == change["before"]
            expected[index] = change["after"]

        assert after == expected

        def ending(header):
            if header.endswith(b"\r\n"):
                return b"\r\n"
            if header.endswith(b"\n"):
                return b"\n"
            return b""

        assert ending(source_header) == ending(output_header)

        while True:
            original_chunk = original.read(1024 * 1024)
            cleaned_chunk = cleaned.read(1024 * 1024)

            assert original_chunk == cleaned_chunk, (
                "Data records differ"
            )

            if not original_chunk:
                break

    print("INDEPENDENT VERIFICATION: PASS")
    print("AUTHORIZED HEADER CHANGES: PASS")
    print("DATA RECORDS BYTE-IDENTICAL: PASS")
    print("SOURCE FINGERPRINT: PASS")
    print("Output SHA256:", sha256(output))


if __name__ == "__main__":
    main()
