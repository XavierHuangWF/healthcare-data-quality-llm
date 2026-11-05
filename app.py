
import csv
import hashlib
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from healthcare_data_quality.ingestion import inspect_headers

from healthcare_data_quality.workflow import run_initial_audit

from healthcare_data_quality.decision_memory import (
    record_decision,
    find_latest_decision,
)

st.set_page_config(
    page_title="Healthcare Data Quality AI",
    page_icon="🔬",
    layout="wide",
)

st.title("Healthcare Data Quality AI")
st.caption("Local-first, LLM-assisted data-quality workspace")

uploaded = st.file_uploader(
    "Upload your dataset",
    type=["csv"],
    help="CSV files only for the initial POC.",
)

if uploaded is None:
    st.info("Upload a CSV dataset to begin.")
    st.stop()

if uploaded.size > 20 * 1024 * 1024:
    st.error("Current POC limit: 20 MB per dataset.")
    st.stop()


try:
    upload_hash = hashlib.sha256(
        uploaded.getvalue()
    ).hexdigest()

    if st.session_state.get("upload_hash") != upload_hash:
        if "upload_tempdir" in st.session_state:
            st.session_state.upload_tempdir.cleanup()

        tempdir = TemporaryDirectory(prefix="hdq-upload-")
        path = Path(tempdir.name) / "input.csv"
        path.write_bytes(uploaded.getvalue())

        st.session_state.upload_tempdir = tempdir
        st.session_state.upload_path = path
        st.session_state.upload_hash = upload_hash

    path = st.session_state.upload_path
    report = inspect_headers(path)

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        preview = list(islice(reader, 10))


except (UnicodeError, csv.Error, StopIteration, OSError) as exc:
    st.error(f"Cannot read this CSV: {exc}")
    st.stop()

if not report["headers"]:
    st.error("The uploaded CSV has no header row.")
    st.stop()

st.success(f"Dataset loaded: {uploaded.name}")

col1, col2, col3 = st.columns(3)

col1.metric("Records", report["data_rows"])
col2.metric("Columns", report["total_columns"])
col3.metric(
    "Duplicate headers",
    len(report["duplicate_headers"]),
)

st.subheader("Dataset preview")

width = max(
    [len(headers)] + [len(row) for row in preview]
)

display_columns = [
    f"{i + 1}: {headers[i] if i < len(headers) else '(unnamed)'}"
    for i in range(width)
]

display_rows = [
    row + [""] * (width - len(row))
    for row in preview
]

st.dataframe(
    pd.DataFrame(display_rows, columns=display_columns),
    hide_index=True,
    width="stretch",
)

st.subheader("Structural quality findings")

if report["duplicate_headers"]:
    st.warning("Duplicate column names detected.")
    st.json(report["duplicate_headers"])

if report["ragged_records"]:
    st.warning("Inconsistent CSV record widths detected.")
    st.json(report["ragged_records"][:20])

if report["blank_header_positions"]:
    st.warning("Blank column names detected.")
    st.write(report["blank_header_positions"])

if not (
    report["duplicate_headers"]
    or report["ragged_records"]
    or report["blank_header_positions"]
):
    st.success("No structural header or record-width issues detected.")

st.caption(
    "Read-only inspection. No LLM cleaning or data modification has occurred."
)



st.divider()
st.subheader("AI Data Quality Audit")

# Run once for each uploaded dataset.
if (
    st.session_state.get("audit_dataset_hash")
    != st.session_state.upload_hash
):
    st.session_state.pop("audit_result", None)

    with st.spinner("AI agent is investigating the dataset..."):
        try:
            audit = run_initial_audit(
                st.session_state.upload_path
            )
        except Exception as exc:
            st.error(
                f"Audit failed ({type(exc).__name__}). "
                "Check the local Gemma server and application logs."
            )
            st.stop()

    st.session_state.audit_result = audit
    st.session_state.audit_dataset_hash = (
        st.session_state.upload_hash
    )

audit = st.session_state.audit_result

st.write(f"Investigation status: `{audit['status']}`")

if audit["status"] == "max_steps_reached":
    st.warning(
        "Preliminary investigation completed. "
        "The agent reached its current tool-call limit; "
        "a comprehensive audit is not yet established."
    )

st.markdown("### Evidence-backed findings")

report = audit["report"]

if not report.get("findings"):
    st.info(
        "No verified findings were reported in this run. "
        "This does not mean the dataset is free of problems."
    )

for index, finding in enumerate(
    report.get("findings", []), start=1
):
    with st.container(border=True):
        st.markdown(f"**Finding {index}**")
        st.write(finding["claim"])
        st.caption(
            "Supporting evidence: "
            + ", ".join(finding["evidence_ids"])
        )

if report.get("unresolved_questions"):
    with st.expander("Unresolved questions", expanded=True):
        for question in report["unresolved_questions"]:
            st.write(f"- {question}")

st.markdown("### Proposed actions")

if not audit["proposals"]:
    st.info("No validated remediation proposals were produced.")

for index, proposal in enumerate(audit["proposals"], start=1):
    validation = proposal.get("validation", {})

    with st.container(border=True):
        st.markdown(
            f"**Proposal {index}: "
            f"{proposal.get('issue', 'Unspecified issue')}**"
        )
        st.write(proposal.get("proposed_action", ""))
        st.caption(
            "Validation: "
            + validation.get("status", "unknown")
        )

        if proposal.get("uncertainty"):
            st.write("**Uncertainty:**", proposal["uncertainty"])

        if proposal.get("action_kind") == "data_change":
            st.warning(
                "Data change proposed. Human review and "
                "separate execution authorization are required."
            )
        else:
            st.info("Investigation recommendation only.")


        st.divider()
        st.markdown("**Human review**")

        previous = find_latest_decision(
            st.session_state.upload_path,
            proposal,
        )

        if previous:
            st.info(
                f"Latest decision: {previous['human_decision']} "
                f"by {previous['reviewer']}"
            )
            if previous.get("human_note"):
                st.write(previous["human_note"])

        with st.form(
            key=f"review_{st.session_state.upload_hash}_{index}"
        ):
            reviewer = st.text_input("Reviewer ID")
            decision = st.selectbox(
                "Review decision",
                ["deferred", "accepted", "rejected"],
            )
            note = st.text_area(
                "Reason or clarification (no confidential data)"
            )

            submitted = st.form_submit_button(
                "Record review decision"
            )

        if submitted:
            try:
                record = record_decision(
                    dataset_path=st.session_state.upload_path,
                    proposal=proposal,
                    decision=decision,
                    reviewer=reviewer,
                    note=note,
                )
                st.success(
                    f"Decision recorded: {record['decision_id']}"
                )
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))


with st.expander("Previous decisions and verified outcomes"):
    memory = audit["memory"]

    st.write("Previous review decisions")
    st.json(memory.get("decisions", []))

    st.write("Previously verified outcomes")
    st.json(memory.get("verified_outcomes", []))

with st.expander("Detailed investigation evidence"):
    for step in audit["history"]:
        st.markdown(
            f"**Step {step['step']}: {step['tool']}**"
        )
        st.write(step.get("reason", ""))
        st.json(step.get("observations", []))

st.caption(
    "Read-only investigation. No cleaning has been "
    "authorized or executed."
)

