
from pathlib import Path

from healthcare_data_quality.cleaning_authorization import (
    authorize_cleaning_request,
    verify_authorized_request,
)
from healthcare_data_quality.cleaning_executor import (
    execute_authorized_header_cleaning,
)


def main():
    request_path = Path(
        "data/local_tests/header_cleaning_request.json"
    )

    authorization = authorize_cleaning_request(request_path)

    if authorization is None:
        print("CLEANING CANCELLED")
        print("No data modified.")
        return

    authorization_id = authorization["authorization_id"]

    verify_authorized_request(
        request_path,
        authorization_id,
    )

    print("\nAuthorization independently checked.")
    print("Executing approved header changes on a copy...")

    result = execute_authorized_header_cleaning(
        request_path,
        authorization_id,
    )

    print("\nCLEANING EXECUTION: PASS")
    print("Status:", result["status"])
    print("Output:", result["output"])
    print("Output SHA256:", result["output_sha256"])
    print("Changes applied:", result["changes_applied"])
    print("Data records preserved:", result["data_records_preserved"])
    print("Original dataset unchanged.")


if __name__ == "__main__":
    main()
