"""Read-only Streamlit client rehearsal for the local data-quality agent.

This UI intentionally exposes no human approval, code generation, sandbox
execution, or data-changing actions. It calls the same investigation workflow
and scientific-advisory formatter as the CLI. All uploaded CSVs stay local.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from healthcare_data_quality.ingestion import inspect_headers
from healthcare_data_quality.finding_presentation import (
    CHECK_NAMES,
    STATUS_LABELS,
    TOOL_NAMES,
    baseline_issues,
    header_findings,
    header_issues,
    history_observations,
    readable_objective_title,
    readable_objective_summary,
)
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


def preview_csv(path: Path) -> pd.DataFrame:
    """Display every CSV row and field without the old ten-row preview limit.

    Prefix header labels with their original positions so unnamed and repeated
    names remain distinguishable. Preserve all original cell strings.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        records = list(reader)
    width = max([len(columns)] + [len(record) for record in records])
    labels = [
        f"{i + 1}: {columns[i] if i < len(columns) and columns[i].strip() else '(unnamed)'}"
        for i in range(width)
    ]
    return pd.DataFrame(
        [record[:width] + [""] * max(0, width - len(record)) for record in records],
        columns=labels,
    )


def show_finding(title: str, explanation: str, evidence: list[dict], next_step: str = "") -> None:
    """Reader-facing finding; structured evidence stays expandable."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(explanation)
        if next_step:
            st.write("**What to check next:**", next_step)
        if evidence:
            with st.expander(f"View affected rows ({len(evidence):,})"):
                st.dataframe(pd.DataFrame(evidence), hide_index=True, width="stretch")


def show_taxonomy_structure_finding(finding: dict) -> None:
    """Show all original classifications and an explicit rank-by-rank expectation."""
    with st.container(border=True):
        st.markdown(f"**{finding['title']}**")
        st.write(finding["explanation"])
        st.markdown(
            "**Expected seven-rank order:** "
            "Kingdom/Domain → Phylum → Class → Order → Family → Genus → Species"
        )
        st.caption(
            "A taxon may legitimately be identified only to genus or another higher rank. "
            "Seven ranks are the expected file structure, not proof that a species name is known."
        )
        st.write("**Original classifications and missing ranks:**")
        st.dataframe(pd.DataFrame(finding["evidence"]), hide_index=True, width="stretch")


        if finding.get("next_step"):
            st.write("**What to check next:**", finding["next_step"])



def render_header_findings(path: Path, details: dict) -> None:
    repeated = header_findings(path, details)
    missing_and_width = header_issues(details)
    if not repeated and not missing_and_width:
        st.success("No repeated or blank column names, or inconsistent row lengths, were found.")
        return
    st.markdown("#### Data structure findings")
    for finding in repeated:
        evidence = [{
            "Repeated column name": finding["header"],
            "First column": finding["columns"][0],
            "Second column": finding["columns"][1],
            "Matching rows": finding.get("matching_rows", 0),
            "Different rows": finding.get("different_rows", 0),
            "Rows not comparable": finding.get("uncomparable_rows", 0),
        }] if len(finding["columns"]) == 2 else []
        show_finding(finding["title"], finding["explanation"], evidence, finding["next_step"])
    for finding in missing_and_width:
        show_finding(finding["title"], finding["explanation"], finding["rows"], finding["next_step"])


def render_baseline(dataset_id: str, baseline: dict, source_path: Path | None = None) -> None:
    checks = baseline.get("checks") or {}
    invalid = checks.get("find_invalid_counts") or {}
    counts = invalid.get("counts_by_type") or {}
    invalid_total = sum(int(v) for v in counts.values())
    taxa = checks.get("find_duplicate_taxa") or {}
    unclassified = checks.get("find_unclassified_species") or {}
    repeated = checks.get("find_duplicate_ids") or {}

    stats = st.columns(3)
    stats[0].metric("Cells with count issues", f"{invalid_total:,}" if invalid.get("status") == "completed" else "Not applicable")
    stats[1].metric("Repeated taxon labels", f"{len(taxa.get('observations') or []):,}" if taxa.get("status") == "completed" else "Not applicable")
    stats[2].metric("Unclassified species rows", f"{unclassified.get('unclassified_count', 0):,}" if unclassified.get("status") == "completed" else "Not applicable")
    if invalid.get("downstream_blocked"):
        st.warning("Count-based analysis needs scientific review of the invalid values before preprocessing.")
    if repeated.get("status") == "completed" and repeated.get("observations"):
        st.caption(f"Repeated metadata identifiers: {len(repeated['observations']):,} groups")

    findings = baseline_issues(baseline, source_path=source_path)
    if not findings:
        st.info("No issues were identified by the checks applicable to this dataset. This is not a guarantee that the data is error-free.")
    else:
        st.markdown(f"**Issues to review ({len(findings)})**")
        for finding in findings:
            if finding.get("type") == "taxonomy_structure":
                show_taxonomy_structure_finding(finding)
            else:
                show_finding(finding["title"], finding["explanation"],
                             finding["evidence"], finding["next_step"])

    with st.expander("Technical check results (advanced)"):
        check_table = []

        for name, check in checks.items():
            status = check.get("status")
            completed = status == "completed"

            if status == "skipped_unverified_context":
                display_status = "Not assessed"
                if name in {"find_duplicate_ids", "find_duplicate_conflicts"}:
                    reason = (
                        "Requires metadata with SampleID and NPDNum columns."
                    )
                else:
                    reason = (
                            "Required dataset structure not verified: "
                            + str(check.get("required_context", "unknown"))
                    )
            else:
                display_status = STATUS_LABELS.get(
                    status, "Not determined"
                )
                reason = ""

            check_table.append({
                "Check": CHECK_NAMES.get(name, name),
                "Status": display_status,
                "Observations": (
                    len(check.get("observations") or [])
                    if completed else "—"
                ),
                "Explanation": reason,
            })

        st.dataframe(
            pd.DataFrame(check_table),
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "Not assessed means the check did not run. "
            "It does not mean zero problems were found."
        )


def render_objectives(ledger: dict) -> None:
    objectives = ledger.get("objectives") or []
    if not objectives:
        st.info("There are no investigation questions to display yet.")
        return

    st.caption("Evidence collected means an investigation step ran. It does not mean that cleaning was approved.")
    table = [{
        "Question / check": readable_objective_title(obj),
        "Progress": STATUS_LABELS.get(obj.get("status"), "Not determined"),
        "What the investigation found": readable_objective_summary(obj)[0],
        "Next action": readable_objective_summary(obj)[1],
    } for obj in objectives]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    pending = [obj for obj in objectives if obj.get("status") == "requires_client_input"]
    if pending:
        st.markdown("#### Scientific decisions requiring your input")
        for obj in pending:
            with st.expander(readable_objective_title(obj)):
                summary, action = readable_objective_summary(obj)
                st.write(summary or "The available evidence does not resolve this question.")
                st.write("**What is needed:**", action or "Confirm the appropriate scientific interpretation.")
                with st.expander("Technical evidence references"):
                    st.write(obj.get("evidence_refs") or [])
    unresolved = [obj for obj in objectives if obj.get("status") in {"blocked", "uninvestigated", "in_progress"}]
    if unresolved:
        st.warning(f"{len(unresolved)} investigation questions still need additional evidence or tool support.")


def render_recommendations(recommendations: list[dict]) -> None:
    st.warning(
        "These are provisional methods for discussion. Evidence-reference validity "
        "is not scientific validation. No recommendation here authorizes data changes."
    )
    if not recommendations:
        st.info("No recommendations were generated. This does not mean the dataset is clean.")
        return

    for item in recommendations:
        with st.container(border=True):
            st.markdown("**" + str(item.get("issue") or "Issue requiring review") + "**")
            status = item.get("advisory_status", "unknown")
            st.caption("Proposed method — scientific review required")
            st.write("**Suggested method:**", item.get("suggested_method") or "Not specified")
            if item.get("rationale"):
                st.write("**Rationale:**", item["rationale"])
            st.write("**Points to verify:**", item.get("uncertainty") or "Scientific suitability has not been confirmed.")
            if not item.get("all_evidence_ids_resolve"):
                st.error("Supporting evidence could not be fully verified; do not approve this method.")
            numeric = item.get("numeric_scope") or {}
            if numeric.get("status") == "partial":
                st.warning(
                    f"This method addresses only {numeric.get('covered_invalid_cells', '?')} of "
                    f"{numeric.get('total_invalid_cells', '?')} affected count cells. "
                    "The remaining cells need a separate decision."
                )
            elif numeric.get("status") == "complete_for_invalid_cells":
                st.info(
                    f"This method covers all {numeric.get('total_invalid_cells')} identified count issues. "
                    "A complete coverage count does not establish that the method is scientifically appropriate."
                )
            elif numeric.get("status") == "unverified_numeric_coverage":
                st.warning("It has not been verified how many affected cells this method would address.")
            with st.expander("Scientific review and technical details"):
                for warning in item.get("scientific_review_flags") or []:
                    st.write("- " + str(warning))
                st.caption("Technical reference: " + str(item.get("recommendation_id", "")))
                st.caption("Recommendation state: " + str(status))
                evidence_ids = item.get("evidence_ids") or []
                st.caption("Evidence references: " + (", ".join(map(str, evidence_ids)) or "None"))
                st.caption("Execution validator: " + str(item.get("execution_validator_status", "unknown")))
                st.caption("Execution authorized: No")


def render_evidence(audit: dict) -> None:
    report = audit.get("report") or {}
    st.markdown("#### Findings from the investigation")
    findings = [f for f in report.get("findings") or [] if isinstance(f, dict)]
    if not findings:
        st.info("No additional narrative findings were generated. See the baseline results above for independently checked issues.")
    for index, finding in enumerate(findings, start=1):
        with st.expander(f"Finding {index}: {finding.get('claim', 'Finding for review')}"):
            st.write(finding.get("claim") or "No description provided.")
            with st.expander("Technical evidence references"):
                st.write(finding.get("evidence_ids") or [])
    questions = report.get("unresolved_questions") or []
    if questions:
        st.markdown("#### Questions that remain open")
        for question in questions:
            st.write("- " + str(question))
    st.markdown("#### Investigation history")
    st.caption("Steps show what was examined. They are not changes to the uploaded data.")
    history = audit.get("history") or []
    if not history:
        st.info("No additional investigation observations were recorded.")
    for step in history:
        tool = step.get("tool", "")
        label = TOOL_NAMES.get(tool, "Additional investigation")
        observations = step.get("observations") or []
        with st.expander(f"{label} — {len(observations):,} evidence item(s)"):
            if step.get("reason"):
                st.write("**Why this was checked:**", step["reason"])
            evidence = history_observations(tool, observations)
            if evidence:
                st.dataframe(pd.DataFrame(evidence), hide_index=True, width="stretch")
                if tool == "compare_columns":
                    examples = [
                        {"CSV row": example.get("row", 0) + 1,
                         "First column value": example.get("left"),
                         "Second column value": example.get("right")}
                        for obs in observations if isinstance(obs, dict)
                        for example in (obs.get("difference_examples") or [])
                        if isinstance(example, dict) and type(example.get("row")) is int
                    ]
                    if examples:
                        st.markdown("**Examples of rows with different values**")
                        st.dataframe(pd.DataFrame(examples), hide_index=True, width="stretch")
                        st.caption("These are examples from the investigation, not a complete list of differences. All original rows are available in the Full dataset table.")
                if tool in {"inspect_column", "inspect_columns"}:
                    frequency_rows = [
                        {"Column": obs.get("position"), "Column name": obs.get("name"),
                         "Observed value": entry.get("value"), "Times observed": entry.get("count"),
                         "Example CSV rows": ", ".join(map(str, entry.get("csv_rows") or []))}
                        for obs in observations if isinstance(obs, dict)
                        for entry in (obs.get("value_frequencies") or [])
                        if isinstance(entry, dict)
                    ]
                    if frequency_rows:
                        with st.expander("See observed values and where they occur"):
                            st.dataframe(pd.DataFrame(frequency_rows), hide_index=True, width="stretch")
                            st.caption("Common and rare values are sampled for inspection. The Full dataset table contains all original values.")
            elif observations:
                st.write("Supporting observations were recorded. Expand the technical details to inspect the complete source results.")
            else:
                st.write("No observations were returned in this step.")
            with st.expander("Technical details (advanced)"):
                st.json({"tool": tool, "step": step.get("step"),
                         "dataset_id": step.get("dataset_id"),
                         "arguments": step.get("arguments") or {},
                         "observations": observations})


def render_ui() -> None:
    st.set_page_config(page_title="Healthcare Data Quality AI", page_icon="🔬", layout="wide")
    st.title("Healthcare Data Quality AI")
    st.caption("Inspect the data, understand what needs attention, and review supporting evidence.")
    st.info("Review-only demonstration: your original datasets will not be modified or cleaned.")

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

    st.subheader("Full dataset")
    options = {f"dataset_{i + 1} — {file.name}": i for i, file in enumerate(uploaded)}
    selected = st.selectbox("Inspect uploaded dataset", list(options))
    index = options[selected]
    details = header_reports[index]
    count_col, col_col, duplicate_col = st.columns(3)
    count_col.metric("Records", details["data_rows"])
    col_col.metric("Columns", details["total_columns"])
    duplicate_col.metric("Repeated column names", sum(1 for name in details["duplicate_headers"] if name.strip()))
    try:
        full_table = preview_csv(paths[index])
        st.caption(
            f"Displaying all {len(full_table):,} records and {len(full_table.columns):,} columns. "
            "Scroll vertically and horizontally; use the table's full-screen "
            "control for a larger view. No preview rows or columns are omitted."
        )

        full_table.index = pd.RangeIndex(
            start=2,
            stop=len(full_table) + 2,
            name="CSV row"
        )

        st.dataframe(
            full_table,
            hide_index=False,
            width="stretch",
            height=650
        )
    except (UnicodeError, csv.Error, OSError) as exc:
        st.error(f"Could not display the complete CSV: {exc}")
        return
    render_header_findings(paths[index], details)

    st.divider()
    st.subheader("Investigate data quality")
    st.caption("Start a deeper review with the local AI model and automated data-quality checks.")
    if st.button("Run read-only investigation", type="primary"):
        status = st.empty()
        try:
            with st.spinner("Gemma and the deterministic QC tools are investigating..."):
                audit = run_initial_audit(
                    paths[0],
                    additional_datasets=paths[1:],
                    progress_callback=lambda message: status.caption("Reviewing the data and collecting supporting evidence…"),
                )
                audit["scientific_recommendations"] = build_scientific_recommendations(audit)
        except Exception as exc:
            status.empty()
            st.error("The investigation could not finish. Check that the local AI service is running. No data was changed.")
            with st.expander("Technical troubleshooting"):
                st.code(f"{type(exc).__name__}: {exc}")
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
    with st.expander("Investigation run status (advanced)"):
        st.write("Agent stop reason:", status)
        st.write("Objective assessment:", objective_ledger.get("completion", "unknown"))
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
        ["Data quality findings", "Questions and decisions", "Proposed methods", "Supporting evidence"]
    )
    with summary_tab:
        for dataset_id, baseline in (audit.get("dataset_baselines") or {"dataset_1": audit.get("baseline") or {}}).items():
            label_index = int(dataset_id.rsplit("_", 1)[-1]) - 1
            dataset_name = uploaded[label_index].name if label_index < len(uploaded) else dataset_id
            st.subheader(f"Dataset {label_index + 1}: {dataset_name}")
            render_baseline(dataset_id, baseline, source_path=paths[label_index])
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
