from pathlib import Path
import json

from healthcare_data_quality.cleaning_codegen import draft_r_transformation
from healthcare_data_quality.r_verification import verify_duplicate_column_draft

def main():
    report = max(
        Path("data/reports").glob("poc_e2e_20261107_*_audit.json"),
        key=lambda path: path.stat().st_mtime,
    )

    audit = json.loads(report.read_text(encoding="utf-8"))

    evidence = {}

    for step in audit["history"]:
        for index, observation in enumerate(step["observations"], 1):
            evidence_id = f"T{step['step']}.O{index}"
            evidence[evidence_id] = {
                "tool": step["tool"],
                "observation": observation,
            }

    proposal = next(
        (
            p for p in audit["proposals"]
            if p.get("target_ref", {}).get("kind") == "column_pair"
            and set(p["evidence_ids"]).issubset(evidence)
        ),
        None,
    )

    if proposal is None:
        raise RuntimeError("No suitable evidence-grounded proposal found")

    print("Generating R cleaning draft with local Gemma...")

    draft = draft_r_transformation(
        proposal=proposal,
        evidence=evidence,
        reviewer_context="",

    )

    print("\nStatus:", draft["status"])
    print("Reasoning:", draft["reasoning"])
    print("Assumptions:", draft["assumptions"])
    print("Question:", draft["review_question"])
    print("R draft length:", len(draft["r_function"]))
    print("\nGenerated R function:\n", draft["r_function"])
    if draft["status"] != "draft_ready":
        raise RuntimeError(
            f"R draft is not ready: {draft['status']}"
        )

    operation = proposal["operation"]

    verification = verify_duplicate_column_draft(
        r_function=draft["r_function"],
        keep_position=operation["keep_position"],
        drop_position=operation["drop_position"],
    )

    assert verification["status"] == "verified"

    print("R SYNTHETIC REGRESSION: PASS")
    print("Verification:", verification)
    print("Verification plan:", draft["verification_plan"])
    print("\nNO CODE EXECUTED. ORIGINAL DATA PRESERVED.")


if __name__ == "__main__":
    main()