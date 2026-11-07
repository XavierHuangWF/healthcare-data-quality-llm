from pathlib import Path
import json

from healthcare_data_quality.cleaning_codegen import draft_r_transformation


def main():
    report = max(
        Path("data/reports").glob("dirtyL7PVDforXav_*_audit.json"),
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
        reviewer_context=(
            "The reviewer cannot verify the meaning of the "
            "duplicate columns. Their scientific provenance "
            "is currently unknown. Preserve both columns "
            "unless sufficient evidence supports removal."
        ),

    )

    print("\nStatus:", draft["status"])
    print("Reasoning:", draft["reasoning"])
    print("Assumptions:", draft["assumptions"])
    print("Question:", draft["review_question"])
    print("R draft length:", len(draft["r_function"]))
    print("Verification plan:", draft["verification_plan"])
    print("\nNO CODE EXECUTED. ORIGINAL DATA PRESERVED.")


if __name__ == "__main__":
    main()