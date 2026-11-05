import json
import sys
from pathlib import Path

from healthcare_data_quality.agent_loop import run_investigation
from healthcare_data_quality.ingestion import inspect_headers

from healthcare_data_quality.investigation_report import summarize_investigation

from healthcare_data_quality.tools.context import verify_contexts
from healthcare_data_quality.remediation_proposal import generate_remediation_proposals
from healthcare_data_quality.remediation_validator import validate_remediation_proposals
from healthcare_data_quality.decision_memory import get_decision_context

if __name__ == "__main__":
    dataset_path = Path(sys.argv[1])
    ingestion_report = inspect_headers(dataset_path)

    verified_contexts = verify_contexts(dataset_path)

    result = run_investigation(
        path=dataset_path,
        max_steps=4,
        verified_contexts=verified_contexts,
    )

    result["ingestion"] = ingestion_report

    if result["history"]:

        result["remediation"] = generate_remediation_proposals(
            result["history"],
            decision_memory=get_decision_context(Path(sys.argv[1])),
        )

    else:
        result["remediation"] = {"proposals": []}

    result["remediation"]["proposals"] = validate_remediation_proposals(
        history=result["history"],
        proposals=result["remediation"]["proposals"],
    )

    if result["history"]:
        result["report"] = summarize_investigation(
            result["history"]
        )

    print(json.dumps(result, indent=2))
