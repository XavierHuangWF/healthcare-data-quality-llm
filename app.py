"""Read-only Streamlit client rehearsal for the local data-quality agent.

This UI intentionally exposes no human approval, code generation, sandbox
execution, or data-changing actions. It calls the same investigation workflow
and scientific-advisory formatter as the CLI. All uploaded CSVs stay local.
"""

import csv
import hashlib
import json
import sys
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from healthcare_data_quality.ingestion import inspect_headers
from healthcare_data_quality.workflow import run_initial_audit

# Reuse the CLI's existing recommendation assessment, including its provenance
# and partial-coverage rules. No CLI main() or remediation entry point is called.
_SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from run_workflow import build_scientific_recommendations  # noqa: E402


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_DATASETS = 4


def upload_fingerprint(files: list) -> str:
    """Bind displayed results to exact bytes and the selected file ordering."""
    digest = hashlib.sha256()
    for file in files:
        content = file.getvalue()
        name = file.name.encode("utf-8", errors="replace")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def ensure_local_inputs(files: list, fingerprint: str) -> list[Path]:
    """Store uploads in disposable files, never in the customer's repository."""
    if st.session_state.get("input_fingerprint") != fingerprint:
        old = st.session_state.get("input_tempdir")
        if old is not None:
            old.cleanup()
        st.session_state.pop("audit_result", None)
        st.session_state.pop("audit_fingerprint", None)

        folder = TemporaryDirectory(prefix="hdq-investigation-")
        paths = []
        for index, file in enumerate(files, start=1):
            path = Path(folder.name) / f"dataset_{index}.csv"
            path.write_bytes(file.getvalue())
            paths.append(path)
        st.session_state.input_tempdir = folder
        st.session_state.input_paths = paths
        st.session_state.input_fingerprint = fingerprint
        st.session_state.input_names = [file.name for file in files]
    return st.session_state.input_paths


def preview_csv(path: Path, limit: int = 10) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        records = list(islice(reader, limit))
    width = max([len(columns)] + [len(record) for record in records])
    labels = [
        f"{i + 1}: {columns[i] if i < len(columns) and columns[i] else '(unnamed)'}"
        for i in range(width)
    ]
    return pd.DataFrame(
        [record[:width] + [""] * max(0, width - len(record)) for record in records],
        columns=labels,
    )


def render_baseline(dataset_id: str, baseline: dict) -> None:
    checks = baseline.get("checks") or {}
    count_check = checks.get("find_invalid_counts") or {}
    counts = count_check.get("counts_by_type") or {}
    invalid_total = sum(int(v) for v in counts.values())
    duplicate_taxa = len((checks.get("find_duplicate_taxa") or {}).get("observations") or [])
    unclassified = (checks.get("find_unclassified_species") or {}).get("unclassified_count")
    invalid_col, duplicate_col, unclassified_col = st.columns(3)
    invalid_col.metric("Invalid count cells", invalid_total if count_check.get("status") == "completed" else "N/A")
    duplicate_col.metric("Duplicate taxon observations", duplicate_taxa if checks.get("find_duplicate_taxa", {}).get("status") == "completed" else "N/A")
    unclassified_col.metric("Unclassified species rows", unclassified if unclassified is not None else "N/A")
    if count_check.get("downstream_blocked"):
        st.error("Count-dependent preprocessing is blocked until the invalid values and their scientific handling have been reviewed.")
    if counts:
        st.write("Invalid counts by type:", counts)
    rows = []
    for name, check in checks.items():
        status = check.get("status", "unknown")
        rows.append({
            "Check": name,
            "Status": status,
            "Observations": len(check.get("observations") or []),
            "Required context (if skipped)": check.get("required_context", ""),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "A skipped check may be inapplicable to this dataset's verified structure; "
        "it must not automatically be counted as a data-quality failure."
    )


def render_objectives(ledger: dict) -> None:
    counts = ledger.get("counts_by_status") or {}
    st.write("**Objective completion:**", ledger.get("completion", "unknown"))
    st.write("**Status counts:**", counts)
    st.caption(
        "Resolved means an evidence check ran successfully, not that the dataset "
        "is scientifically validated or approved for cleaning."
    )
    objectives = ledger.get("objectives") or []
    if not objectives:
        st.warning("No investigation-objective ledger was returned.")
        return
    rows = [
        {
            "Objective ID": obj.get("objective_id", ""),
            "Status": obj.get("status", "unknown"),
            "Question / check": obj.get("title", ""),
            "Finding": obj.get("finding", ""),
            "Next action": obj.get("next_action", ""),
        }
        for obj in objectives
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pending = [obj for obj in objectives if obj.get("status") == "requires_client_input"]
    if pending:
        st.markdown("#### Questions to resolve with the client")
        for obj in pending:
            with st.expander(obj.get("title", "Untitled question")):
                st.write(obj.get("finding", ""))
                st.write("**Needed:**", obj.get("next_action", ""))
                refs = obj.get("evidence_refs") or []
                if refs:
                    st.caption("Evidence: " + ", ".join(map(str, refs)))
    blocked = [obj for obj in objectives if obj.get("status") in {"blocked", "uninvestigated", "in_progress"}]
    if blocked:
        st.warning(
            f"{len(blocked)} objectives are blocked, not yet investigated, or only partially investigated. "
            "Do not describe this as a complete dataset investigation."
        )


def render_recommendations(recommendations: list[dict]) -> None:
    st.warning(
        "These are provisional methods for discussion. Evidence-reference validity "
        "is not scientific validation. No recommendation here authorizes data changes."
    )
    if not recommendations:
        st.info("No recommendations were generated. This does not mean the dataset is clean.")
        return

    for item in recommendations:
        ident = item.get("recommendation_id", "R?")
        with st.container(border=True):
            st.markdown(f"**{ident}**")
            st.write("**Issue:**", item.get("issue") or "Unspecified issue")
            status = item.get("advisory_status", "unknown")
            st.caption("Recommendation status: " + status)
            st.write("**Suggested method:**", item.get("suggested_method") or "Not specified")
            if item.get("rationale"):
                st.write("**Rationale:**", item["rationale"])
            st.write("**Model uncertainty:**", item.get("uncertainty") or "Not specified")
            evidence_ids = item.get("evidence_ids") or []
            st.caption("Evidence references: " + (", ".join(map(str, evidence_ids)) or "None"))
            if item.get("all_evidence_ids_resolve"):
                st.caption("References resolve; factual and scientific claims still require checking.")
            else:
                st.error("One or more evidence references are absent or invalid.")
            numeric = item.get("numeric_scope") or {}
            if numeric.get("status") == "partial":
                st.error(
                    f"PARTIAL numeric coverage: {numeric.get('covered_invalid_cells', '?')} of "
                    f"{numeric.get('total_invalid_cells', '?')} invalid cells covered."
                )
            elif numeric.get("status") == "complete_for_invalid_cells":
                st.info(
                    f"Numeric condition coverage: {numeric.get('covered_invalid_cells')} of "
                    f"{numeric.get('total_invalid_cells')} invalid cells. This does not validate the method."
                )
            elif numeric.get("status") == "unverified_numeric_coverage":
                st.warning("The number of invalid count cells covered by this method has not been independently established.")
            with st.expander("Required scientific review and execution-safety status"):
                for warning in item.get("scientific_review_flags") or []:
                    st.write("- " + str(warning))
                st.caption(
                    "Separate execution validator: "
                    + str(item.get("execution_validator_status", "unknown"))
                    + "; execution authorized: No"
                )


def render_evidence(audit: dict) -> None:
    report = audit.get("report") or {}
    st.markdown("#### Evidence-linked findings")
    for index, finding in enumerate(report.get("findings") or [], start=1):
        if not isinstance(finding, dict):
            continue
        with st.expander(f"Finding {index}"):
            st.write("Claim:", finding.get("claim", "Unspecified"))
            st.write("Evidence IDs:", finding.get("evidence_ids") or [])
    if not report.get("findings"):
        st.info("No narrative findings returned. Inspect the deterministic baseline and objective ledger instead.")
    questions = report.get("unresolved_questions") or []
    if questions:
        st.markdown("#### Additional unresolved questions")
        for question in questions:
            st.write("- " + str(question))
    st.markdown("#### Read-only investigation history")
    history = audit.get("history") or []
    if not history:
        st.info("No additional investigation tool observations were recorded.")
    for step in history:
        with st.expander(f"Step {step.get('step', '?')}: {step.get('tool', 'unknown')}"):
            st.write("Reason:", step.get("reason") or "Not recorded")
            st.write("Dataset:", step.get("dataset_id") or "dataset_1")
            st.json(step.get("observations") or [])


def render_ui() -> None:
    st.set_page_config(page_title="Healthcare Data Quality AI", page_icon="🔬", layout="wide")
    st.title("Healthcare Data Quality AI")
    st.caption("Local-first · Evidence-driven investigation · Client review rehearsal")
    st.info(
        "Investigation-only mode. No R code is generated or executed, no review "
        "decision is recorded, and no source dataset is modified."
    )

    uploaded = st.file_uploader(
        "Upload one or more CSV datasets",
        type=["csv"],
        accept_multiple_files=True,
        help="Up to four local CSV files, maximum 20 MiB each. The first is the primary dataset.",
    )
    if not uploaded:
        st.info("Upload a CSV dataset to inspect its structure and start the investigation.")
        return
    if len(uploaded) > MAX_DATASETS:
        st.error(f"Limit: {MAX_DATASETS} datasets per investigation.")
        return
    for file in uploaded:
        if file.size > MAX_FILE_BYTES:
            st.error(f"{file.name}: exceeds the 20 MiB per-file limit.")
            return
        if not file.size:
            st.error(f"{file.name}: empty files are not supported.")
            return

    fingerprint = upload_fingerprint(uploaded)
    try:
        paths = ensure_local_inputs(uploaded, fingerprint)
        header_reports = [inspect_headers(path) for path in paths]
    except (UnicodeError, csv.Error, OSError, StopIteration, ValueError) as exc:
        st.error(f"Could not read uploaded CSV(s): {exc}")
        return
    if any(not report.get("headers") for report in header_reports):
        st.error("At least one CSV has an empty header row.")
        return

    st.subheader("Dataset preview")
    options = {f"dataset_{i + 1} — {file.name}": i for i, file in enumerate(uploaded)}
    selected = st.selectbox("Inspect uploaded dataset", list(options))
    index = options[selected]
    details = header_reports[index]
    count_col, col_col, duplicate_col = st.columns(3)
    count_col.metric("Records", details["data_rows"])
    col_col.metric("Columns", details["total_columns"])
    duplicate_col.metric("Repeated header names", len(details["duplicate_headers"]))
    try:
        st.dataframe(preview_csv(paths[index]), hide_index=True, width="stretch")
    except (UnicodeError, csv.Error, OSError) as exc:
        st.error(f"Could not preview this CSV: {exc}")
        return
    if details["duplicate_headers"]:
        st.warning("Repeated headers found. Identical column values do not establish duplicate specimen identity.")
        st.json(details["duplicate_headers"])
    if details["ragged_records"]:
        st.warning("Inconsistent record widths found.")
        st.write(details["ragged_records"][:20])
    if details["blank_header_positions"]:
        st.warning("Blank headers:")
        st.write(details["blank_header_positions"])

    st.divider()
    st.subheader("Agent investigation")
    st.caption("The model must be running locally. Investigation starts only when you click the button below.")
    if st.button("Run read-only investigation", type="primary"):
        status = st.empty()
        try:
            with st.spinner("Gemma and the deterministic QC tools are investigating..."):
                audit = run_initial_audit(
                    paths[0],
                    additional_datasets=paths[1:],
                    progress_callback=lambda message: status.caption("[Agent] " + str(message)),
                )
                audit["scientific_recommendations"] = build_scientific_recommendations(audit)
        except Exception as exc:
            status.empty()
            st.error(
                f"Investigation failed ({type(exc).__name__}). "
                "Check that the local Gemma/vLLM server is running and review the WSL logs. "
                "No cleaning was attempted."
            )
            return
        status.empty()
        st.session_state.audit_result = audit
        st.session_state.audit_fingerprint = fingerprint

    audit = st.session_state.get("audit_result")
    if audit is None or st.session_state.get("audit_fingerprint") != fingerprint:
        st.info("No investigation results yet for this upload.")
        return

    st.success("Investigation run finished; results are available for review.")
    objective_ledger = audit.get("investigation_objectives") or {}
    status = audit.get("status", "unknown")
    st.write("**Agent stop reason:**", f"`{status}`")
    st.write("**Objective-ledger assessment:**", objective_ledger.get("completion", "unknown"))
    if objective_ledger.get("completion") != "evidence_checks_resolved_only":
        st.warning(
            "Investigation remains incomplete. Unresolved scientific questions, "
            "blocked objectives, or unvisited evidence checks require follow-up."
        )
    else:
        st.info("Evidence checks finished; scientific validation and client approval are still outstanding.")
    if audit.get("proposal_generation_error"):
        st.error("Gemma could not complete proposal generation; see the downloaded audit for details.")

    summary_tab, objective_tab, recommendation_tab, evidence_tab = st.tabs(
        ["Baseline QC", "Investigation objectives", "Cleaning recommendations", "Evidence and audit"]
    )
    with summary_tab:
        for dataset_id, baseline in (audit.get("dataset_baselines") or {"dataset_1": audit.get("baseline") or {}}).items():
            label_index = int(dataset_id.rsplit("_", 1)[-1]) - 1
            dataset_name = uploaded[label_index].name if label_index < len(uploaded) else dataset_id
            st.subheader(dataset_id)
            st.caption(dataset_name)
            render_baseline(dataset_id, baseline)
        if len(uploaded) > 1:
            st.warning("Full cross-dataset identifier reconciliation is not yet implemented as an investigation tool.")
    with objective_tab:
        render_objectives(objective_ledger)
    with recommendation_tab:
        render_recommendations(audit.get("scientific_recommendations") or [])
        if len(uploaded) > 1:
            st.caption("These generated proposals are based on the primary dataset; additional dataset baselines appear in the QC tab.")
    with evidence_tab:
        render_evidence(audit)
        st.download_button(
            "Download complete audit JSON (may contain confidential data)",
            data=json.dumps(audit, ensure_ascii=False, indent=2, default=str),
            file_name="healthcare_data_quality_investigation_audit.json",
            mime="application/json",
        )
        st.caption("Keep downloaded audits in approved secure storage. They may contain sample identifiers and detailed source observations.")


if __name__ == "__main__":
    render_ui()
