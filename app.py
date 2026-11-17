"""Local Streamlit client for biomedical data investigation and reviewed cleaning.

The baseline audit and deterministic QC tools run alongside optional
Gemma-generated exploratory analyses in one investigation. An explicit
full-investigation request authorizes *read-only sandboxed* generated code,
with bounded auto-repair and Gemma-directed stopping. No cleaning or host
code execution is exposed in the investigation. A separate Gemma discussion tab
may propose narrowly supported cleaning, with explicit human authorization
for any verified derived-copy operation. All data and audits stay local.
"""

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from time import monotonic, time
from typing import Callable
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4

import pandas as pd
import streamlit as st

from investigation_progress import InvestigationProgress, format_elapsed, live_timer_html

from history_store import (
    list_history, update_annotations, delete_saved_file, build_history_zip,
    human_size, select_history_rows, latest_checkpoint_view,
    resolve_viewer_timezone, format_saved_time,
    DELETE_CONFIRMATION_TEXT, is_batch_delete_authorized, FILE_RE,
)

from healthcare_data_quality.ingestion import inspect_headers
from healthcare_data_quality.finding_presentation import (
    CHECK_NAMES,
    STATUS_LABELS,
    TOOL_NAMES,
    baseline_issues,
    header_findings,
    header_issues,
    history_observations,
)
from healthcare_data_quality.workflow import run_initial_audit
from healthcare_data_quality.exploratory_agent import MAX_ROUNDS
from healthcare_data_quality.exploratory_combined import run_exploratory_series
from healthcare_data_quality.exploratory_sandbox import sandbox_available, SandboxUnavailable
from healthcare_data_quality.finding_evidence import (
    dataset_finding_evidence,
)
from evidence_view import baseline_evidence_trace, deterministic_detection_code
from healthcare_data_quality.cleaning_workspace import (
    ask_gemma_about_cleaning, validate_header_plan,
    create_header_request, execute_confirmed_header_request,
)

# Reuse the CLI's existing recommendation assessment, including its provenance
# and partial-coverage rules. No CLI main() or remediation entry point is called.
_SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from run_workflow import build_scientific_recommendations  # noqa: E402


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_DATASETS = 4

# Cached audits can include source values and identifying metadata. Keep them
# outside the Git repository, in a private local directory. Each re-run is
# retained as a separate audit instead of overwriting earlier results.
CACHE_SCHEMA_VERSION = 1
AUDIT_CACHE_DIR = (
    Path.home() / ".local" / "share" / "healthcare-data-quality-llm" / "audit-cache"
)
MAX_CACHED_AUDIT_BYTES = 200 * 1024 * 1024


def analysis_fingerprint() -> str:
    """Invalidate old audits when investigation logic/configuration changes.

    The UI file itself is excluded: changing layout should not require a new
    Gemma run. Setting HDQ_MODEL_REVISION invalidates caches if model weights
    or serving configuration change without a source-code change.
    """
    project_root = Path(__file__).resolve().parent
    source_root = project_root / "src" / "healthcare_data_quality"
    # Keep the PRE-EXISTING baseline-audit cache valid when adding the
    # independent exploratory feature. Its implementation has a separate key.
    relevant = sorted(p for p in source_root.rglob("*.py")
                      if not p.name.startswith("exploratory_"))
    relevant.extend([
        project_root / "scripts" / "run_workflow.py",
        project_root / "scripts" / "serve_gemma4.sh",
    ])
    relevant.extend(sorted((project_root / "data" / "decision_memory").glob("*.jsonl")))
    digest = hashlib.sha256(f"analysis-v{CACHE_SCHEMA_VERSION}".encode())
    digest.update(os.environ.get("HDQ_MODEL_REVISION", "").encode("utf-8"))
    for path in relevant:
        if path.is_file():
            digest.update(str(path.relative_to(project_root)).encode("utf-8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def investigation_cache_key(files: list) -> str:
    """Identify matching *content*, preserving order across multiple CSVs.

    File names are deliberately omitted. The workflow uses dataset_1.csv,
    dataset_2.csv etc., so renaming a CSV does not change the investigation.
    """
    digest = hashlib.sha256(bytes.fromhex(analysis_fingerprint()))
    digest.update(len(files).to_bytes(2, "big"))
    for file in files:
        content = file.getvalue()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


# Unlike the baseline audit cache, exploratory results depend on the
# dynamically generated analysis pipeline and its isolated runner.
EXPLORATION_CACHE_DIR = (
    Path.home() / ".local" / "share" / "healthcare-data-quality-llm" / "explorations"
)


def exploration_cache_key(baseline_key: str) -> str:
    digest = hashlib.sha256(("exploration-v1:" + baseline_key).encode("utf-8"))
    root = Path(__file__).resolve().parent / "src" / "healthcare_data_quality"
    for name in ("exploratory_agent.py", "exploratory_dedup.py", "exploratory_combined.py", "exploratory_sandbox.py", "exploratory_runner.py"):
        digest.update((root / name).read_bytes())
    digest.update(os.environ.get("HDQ_MODEL_REVISION", "").encode("utf-8"))
    return digest.hexdigest()


def saved_explorations(key: str) -> list[Path]:
    directory = EXPLORATION_CACHE_DIR / key
    if not directory.is_dir() or directory.is_symlink():
        return []
    return sorted((file for file in directory.glob("discovery-*.json")
                   if file.is_file() and not file.is_symlink()), reverse=True)


def save_explorations(
    key: str, runs: list[dict], *, stop: dict | None = None,
    replace_checkpoint: Path | None = None,
) -> Path:
    """Atomic, private local records; no uploads or Git-backed data."""
    directory = EXPLORATION_CACHE_DIR / key
    if EXPLORATION_CACHE_DIR.is_symlink() or directory.is_symlink():
        raise OSError("Refusing to use a symlinked discovery cache")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    EXPLORATION_CACHE_DIR.chmod(0o700)
    directory.chmod(0o700)
    if replace_checkpoint is None:
        destination = directory / (
            "discovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-" + uuid4().hex[:10] + ".json"
        )
    else:
        # Finalize the last verified-round checkpoint rather than creating
        # another redundant JSON when Gemma decides to stop.
        destination = Path(replace_checkpoint)
        if (destination.parent != directory or destination.is_symlink()
                or not destination.is_file()
                or not FILE_RE["Gemma checkpoint"].fullmatch(destination.name)):
            raise ValueError("Cannot replace a checkpoint outside the current cache")
    record = {"schema": 1, "key": key, "runs": runs, "stop_decision": stop}
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=directory, prefix=".pending-",
                                encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(record, stream, ensure_ascii=False, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_explorations(path: Path, key: str, *, include_stop: bool = False):
    if path.parent != EXPLORATION_CACHE_DIR / key or path.is_symlink():
        raise ValueError("Unexpected discovery cache path")
    if path.stat().st_size > MAX_CACHED_AUDIT_BYTES:
        raise ValueError("Discovery record exceeds the safety size limit")
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if (not isinstance(record, dict) or record.get("schema") != 1
            or record.get("key") != key or not isinstance(record.get("runs"), list)
            or len(record["runs"]) > MAX_ROUNDS):
        raise ValueError("Incompatible saved discovery record")
    stopped = record.get("stop_decision")
    if stopped is not None and (
            not isinstance(stopped, dict)
            or stopped.get("decision") != "stop"
            or not isinstance(stopped.get("reason"), str)
            or not isinstance(stopped.get("remaining_questions"), list)):
        raise ValueError("Incompatible saved exploration stop decision")
    return (record["runs"], stopped) if include_stop else record["runs"]


def saved_investigations(cache_key: str) -> list[Path]:
    """Enumerate matching completed runs without reading large audit JSONs."""
    directory = AUDIT_CACHE_DIR / cache_key
    if not directory.is_dir() or directory.is_symlink():
        return []
    return sorted(
        (path for path in directory.glob("audit-*.json")
         if path.is_file() and not path.is_symlink()),
        key=lambda path: path.name,
        reverse=True,
    )


def browser_viewer_timezone() -> tuple[tzinfo, str]:
    """Resolve each visitor's local browser timezone without changing saved UTC."""
    context = getattr(st, "context", None)
    if context is None:
        return resolve_viewer_timezone(None)
    try:
        name = getattr(context, "timezone", None)
        offset_minutes = getattr(context, "timezone_offset", None)
    except (AttributeError, RuntimeError, ValueError):
        return resolve_viewer_timezone(None)
    return resolve_viewer_timezone(name, offset_minutes)


def cache_label(path: Path, viewer_tz: tzinfo | None = None) -> str:
    """Human-readable local save time from a UTC cache file name."""
    timestamp = path.name.removeprefix("audit-").split("-", 1)[0]
    try:
        saved_utc = datetime.strptime(timestamp, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
        return format_saved_time(saved_utc, viewer_tz or timezone.utc)
    except ValueError:
        return path.name


def save_investigation(cache_key: str, audit: dict) -> Path:
    """Atomically store a JSON audit with user-only directory/file access."""
    if not isinstance(audit, dict):
        raise ValueError("The investigation result must be a JSON object.")
    directory = AUDIT_CACHE_DIR / cache_key
    if AUDIT_CACHE_DIR.is_symlink() or directory.is_symlink():
        raise OSError("Refusing to save audits through a symbolic link.")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Tighten permissions if the cache was created with a permissive umask.
    AUDIT_CACHE_DIR.chmod(0o700)
    directory.chmod(0o700)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = directory / f"audit-{timestamp}-{uuid4().hex[:12]}.json"
    record = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": audit,
    }
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=".pending-", suffix=".json", delete=False
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(record, stream, ensure_ascii=False, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_investigation(path: Path, cache_key: str) -> dict:
    """Load only a compatible audit from this content-addressed directory."""
    if (path.parent != AUDIT_CACHE_DIR / cache_key or
            path.is_symlink() or not path.is_file()):
        raise ValueError("The selected cached audit is not a valid result file.")
    if path.stat().st_size > MAX_CACHED_AUDIT_BYTES:
        raise ValueError("The saved audit exceeds the safety size limit.")
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if (not isinstance(record, dict)
            or record.get("schema_version") != CACHE_SCHEMA_VERSION
            or record.get("cache_key") != cache_key
            or not isinstance(record.get("audit"), dict)
            or not isinstance(record["audit"].get("dataset_baselines"), dict)):
        raise ValueError("The saved audit is incompatible or incomplete.")
    return record["audit"]


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
        st.session_state.pop("audit_cache_key", None)
        st.session_state.pop("audit_source", None)
        st.session_state.pop("active_audit_file", None)

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


def render_saved_file_manager(cache_key: str | None) -> None:
    """Manage multiple saved records without changing automatic audit selection."""

    @st.dialog("Saved investigation files", width="large")
    def show_history_dialog():
        st.markdown("#### Saved investigation files")
        st.caption(
            "Inspect, name, annotate, download, or delete local audits and Gemma "
            "checkpoints. Click rows in the table to select records for review, "
            "ZIP download or batch deletion. "
            "Selection here never changes which investigation is loaded: "
            "Load previous result always chooses the newest matching audit."
        )
        st.code(str(AUDIT_CACHE_DIR.parent), language="text")
        viewer_tz, viewer_zone_name = browser_viewer_timezone()
        st.caption(f"Displayed times: {viewer_zone_name} (detected from your browser; saved files use UTC).")
        try:
            files = list_history(AUDIT_CACHE_DIR, EXPLORATION_CACHE_DIR)
        except OSError as exc:
            st.error(f"Could not list local cache files: {exc}")
            return
        if not files:
            st.info("No saved investigation files were found on this computer.")
            return

        matching_exploration_key = exploration_cache_key(cache_key) if cache_key else None

        def matches_upload(entry) -> bool:
            return bool(
                (entry.kind == "Audit" and entry.key == cache_key)
                or (entry.kind == "Gemma checkpoint" and entry.key == matching_exploration_key)
            )

        audit_count = sum(entry.kind == "Audit" for entry in files)
        checkpoint_count = len(files) - audit_count
        st.caption(
            f"{audit_count} audit(s) and {checkpoint_count} Gemma recovery checkpoint(s) "
            f"· {human_size(sum(f.size_bytes for f in files))} stored"
        )
        st.caption(
            "One full investigation can save both an audit and Gemma progress checkpoints. "
            "These are different records for the same uploaded data, not separate datasets."
        )
        current_only = st.checkbox(
            "Show only files matching uploaded datasets", value=False,
            disabled=cache_key is None, key="history_current_only",
        )
        show_intermediate = st.checkbox(
            "Show intermediate Gemma checkpoints", value=False,
            key="history_show_intermediate_checkpoints",
            help="By default, show one newest Gemma checkpoint per matching cache key. "
                 "Older recovery snapshots are only hidden, never deleted.",
        )
        visible = [entry for entry in files if not current_only or matches_upload(entry)]
        if not show_intermediate:
            collapsed = latest_checkpoint_view(visible)
            hidden = len(visible) - len(collapsed)
            visible = collapsed
            if hidden:
                st.caption(f"{hidden} intermediate checkpoint(s) hidden; enable the option above to manage them.")
        if not visible:
            st.info("No files match this dataset and analysis version.")
            return
        displayed = visible[:250]
        if len(visible) > len(displayed):
            st.caption(f"Showing the newest {len(displayed)} of {len(visible)} saved files.")

        rows = [{
            "Saved (local time)": format_saved_time(entry.created_utc, viewer_tz),
            "Type": entry.kind,
            "Display name": entry.title or "—",
            "Matches uploaded data": "Yes" if matches_upload(entry) else "No",
            "Size": human_size(entry.size_bytes),
            "Cache key": entry.key[:10] + "…",
        } for entry in displayed]
        st.caption(
            "Select files directly in this table: click a Saved (local time) row; "
            "use Ctrl/Cmd-click to select additional rows or Shift-click for a range. "
            "Selection only manages saved files; it never loads a different investigation."
        )
        # Streamlit reports the original dataframe row positions even when users
        # sort the table. Use only the current, validated inventory to resolve
        # those positions; never trust a previously cached row number or name.
        visible_identity = hashlib.sha256(
            "\n".join(str(entry.path) for entry in displayed).encode("utf-8")
        ).hexdigest()[:16]
        reset_key = f"history_table_reset_{visible_identity}"
        table_key = (
            f"history_saved_rows_{visible_identity}_"
            f"{st.session_state.get(reset_key, 0)}"
        )
        table_event = st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            key=table_key, on_select="rerun", selection_mode="multi-row",
        )
        selected = select_history_rows(displayed, table_event.selection.rows)
        selected_paths = [str(entry.path) for entry in selected]
        # Reset BOTH confirmations when the row selection changes, including
        # when all rows are cleared. A confirmation for another file set must
        # never authorize the current selection.
        selection_signature = hashlib.sha256(
            "\n".join(sorted(selected_paths)).encode("utf-8")
        ).hexdigest()
        if st.session_state.get("history_delete_active_selection") != selection_signature:
            st.session_state["history_delete_active_selection"] = selection_signature
            st.session_state["history_delete_acknowledged"] = False
            st.session_state["history_delete_typed_phrase"] = ""
        if st.button(
            "Clear selected rows", disabled=not selected,
            key=f"history_clear_rows_{visible_identity}",
        ):
            # Changing the widget key clears its native selection state without
            # writing to Streamlit's potentially read-only dataframe event state.
            st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
            st.rerun()
        st.caption(f"{len(selected)} file(s) selected")
        feedback = st.session_state.pop("history_batch_feedback", None)
        if feedback:
            if feedback["failed"]:
                st.warning(feedback["message"])
            else:
                st.success(feedback["message"])

        if not selected:
            st.info("Select one file to inspect or edit, or several files for batch actions.")
            return

        if len(selected) == 1:
            entry = selected[0]
            st.markdown("##### Selected record")
            st.caption(
                "Original scientific results are immutable. "
                "Editing changes only this record's private label and notes."
            )
            name = st.text_input(
                "Display name", value=entry.title, max_chars=120,
                key=f"history_name_{entry.path.name}_{entry.key}",
            )
            notes = st.text_area(
                "Notes", value=entry.notes, max_chars=2000,
                key=f"history_notes_{entry.path.name}_{entry.key}",
            )
            if st.button("Save name and notes", key="history_save_metadata"):
                try:
                    update_annotations(
                        entry, name, notes, AUDIT_CACHE_DIR, EXPLORATION_CACHE_DIR,
                    )
                except (OSError, ValueError) as exc:
                    st.error(f"Could not save the note: {exc}")
                else:
                    st.success("Name and notes saved. Scientific results unchanged.")
                    st.rerun()

            with st.expander("Inspect saved JSON (read-only)"):
                st.code(str(entry.path), language="text")
                if entry.size_bytes <= 2 * 1024 * 1024:
                    try:
                        with entry.path.open("r", encoding="utf-8") as stream:
                            st.json(json.load(stream))
                    except (OSError, ValueError, UnicodeError) as exc:
                        st.warning(f"Cannot preview this file: {exc}")
                else:
                    st.caption("This result is too large for an inline preview.")
            if entry.size_bytes <= 12 * 1024 * 1024:
                try:
                    st.download_button(
                        "Download selected JSON", data=entry.path.read_bytes(),
                        file_name=entry.path.name, mime="application/json",
                        key="history_download_json",
                    )
                except OSError as exc:
                    st.warning(f"Could not read the saved file: {exc}")
            else:
                st.caption("Large JSON files must be copied from the local cache directory.")
        else:
            st.caption(
                "For individual display names, notes and JSON previews, select exactly "
                "one record. Batch actions do not modify scientific evidence."
            )

        st.markdown("##### Batch actions")
        selected_size = sum(entry.size_bytes for entry in selected)
        st.caption(
            f"Selected: {len(selected)} file(s) · {human_size(selected_size)} "
            "(ZIP contains only the selected original JSON results)"
        )
        if selected_size <= 48 * 1024 * 1024:
            try:
                archive = build_history_zip(
                    selected, AUDIT_CACHE_DIR, EXPLORATION_CACHE_DIR,
                )
                st.download_button(
                    f"Download {len(selected)} selected file(s) as ZIP",
                    data=archive, file_name="hdq-selected-investigations.zip",
                    mime="application/zip", key="history_download_zip",
                )
            except (OSError, ValueError) as exc:
                st.warning(f"Could not prepare selected files: {exc}")
        else:
            st.warning(
                "Selected records exceed the 48 MiB ZIP limit. "
                "Select a smaller batch or copy the JSON files from the local directory."
            )

        st.warning(
            f"You are about to permanently delete {len(selected)} saved "
            "investigation file(s). This cannot be undone."
        )
        confirmed = st.checkbox(
            f"I understand deleting these {len(selected)} selected file(s) cannot be undone",
            key="history_delete_acknowledged",
        )
        typed_confirmation = st.text_input(
            f"Final confirmation: type `{DELETE_CONFIRMATION_TEXT}` exactly",
            key="history_delete_typed_phrase", max_chars=len(DELETE_CONFIRMATION_TEXT),
            placeholder="Type the exact phrase shown above",
        )
        can_delete = is_batch_delete_authorized(
            len(selected), confirmed, typed_confirmation,
        )
        delete_clicked = st.button(
            f"Permanently delete {len(selected)} selected saved file(s)",
            disabled=not can_delete, key="history_delete_selected",
        )
        # Recheck authorization in the handler, not only in button rendering.
        if delete_clicked and is_batch_delete_authorized(
            len(selected), confirmed, typed_confirmation,
        ):
            deleted = []
            failures = []
            for entry in selected:
                try:
                    delete_saved_file(entry, AUDIT_CACHE_DIR, EXPLORATION_CACHE_DIR)
                except (OSError, ValueError) as exc:
                    failures.append(f"{entry.path.name}: {exc}")
                else:
                    deleted.append(entry)
            if any(
                entry.kind == "Audit" and
                st.session_state.get("active_audit_file") == str(entry.path)
                for entry in deleted
            ):
                for key in (
                    "audit_result", "audit_fingerprint", "audit_cache_key",
                    "audit_source", "active_audit_file",
                ):
                    st.session_state.pop(key, None)
                st.session_state.exploration_runs = []
                st.session_state.exploration_stop = None
                st.session_state.exploration_restored = False
            elif any(
                entry.kind == "Gemma checkpoint" and
                entry.key == matching_exploration_key for entry in deleted
            ):
                st.session_state.exploration_runs = []
                st.session_state.exploration_stop = None
                st.session_state.exploration_restored = False

            message = f"Deleted {len(deleted)} saved file(s)."
            if failures:
                message += f" {len(failures)} failed: " + "; ".join(failures[:3])
            st.session_state["history_batch_feedback"] = {
                "message": message, "failed": bool(failures),
            }
            st.rerun()

    if st.button("Manage saved files", key="open_history_manager"):
        show_history_dialog()


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
            "NA is allowed in any of the seven positions, including multiple positions. "
            "A species name is not required for a structurally valid classification."
        )
        st.write("**Original classifications and taxonomy level counts:**")
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
    structure = checks.get("inspect_taxonomy_structure") or {}
    repeated = checks.get("find_duplicate_ids") or {}

    stats = st.columns(3)
    stats[0].metric("Cells with count issues", f"{invalid_total:,}" if invalid.get("status") == "completed" else "Not applicable")
    stats[1].metric("Repeated taxon labels", f"{len(taxa.get('observations') or []):,}" if taxa.get("status") == "completed" else "Not applicable")
    stats[2].metric("Rows with incorrect taxonomy level count", f"{structure.get('incomplete_count', 0):,}" if structure.get("status") == "completed" else "Not applicable")
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



def render_recommendations(recommendations: list[dict], report: dict | None = None) -> None:
    if not recommendations:
        st.info(
            "Investigation-only mode does not generate remediation methods. "
            "This does not mean the dataset is clean."
        )
        next_step = (report or {}).get("recommended_next_step")
        if next_step:
            st.markdown("**Gemma's suggested next investigation (requires scientific review)**")
            st.write(next_step)
        return
    st.warning(
        "These are provisional methods for discussion. Evidence-reference validity "
        "is not scientific validation. No recommendation here authorizes data changes."
    )

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


def render_technical_history(audit: dict) -> None:
    report = audit.get("report") or {}
    st.markdown("#### Findings from the investigation")
    findings = [f for f in report.get("findings") or [] if isinstance(f, dict)]
    if not findings:
        st.info("No additional narrative findings were generated. See the baseline results above for independently checked issues.")
    for index, finding in enumerate(findings, start=1):
        claim = str(finding.get("claim") or "No description provided.")
        verification = finding.get("claim_verification") or {}
        status = verification.get("status", "requires_scientific_review")
        is_verified = status == "verified_exact_fact"
        label = "Measured fact" if is_verified else "AI interpretation — review required"
        with st.expander(f"Finding {index}: {label}"):
            st.write(claim)
            if is_verified:
                st.caption("This statement matches a fact independently rendered from a recorded read-only tool observation. It does not authorize cleaning.")
            else:
                st.warning("The supporting evidence reference is valid, but the model's interpretation has not been independently verified.")
            facts = verification.get("verified_observation_facts") or []
            if facts:
                st.markdown("**Directly observed measurements**")
                for fact in facts:
                    st.write("- " + str(fact))
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


def render_finding_evidence(
    audit: dict, dataset_paths: list[Path], filenames: list[str],
    header_reports: list[dict], exploratory_runs: list[dict],
) -> None:
    """Show how each result was obtained, never a second findings report.

    This view is explanation-only: the Data quality findings tab retains the
    full original affected rows. It does not re-run Gemma or authorize cleaning.
    """
    st.subheader("How the data quality results were obtained")
    st.caption(
        "Each check shows how the result was calculated from your original CSV. "
        "Expand a check for its rule and measured observations; open 'View Python "
        "detection code' only if you need to audit the actual implementation. "
        "Complete affected records remain in Data quality findings."
    )
    baselines = audit.get("dataset_baselines") or {"dataset_1": audit.get("baseline") or {}}
    for dataset_id, baseline in baselines.items():
        index = int(dataset_id.rsplit("_", 1)[-1]) - 1
        if not 0 <= index < len(dataset_paths):
            continue
        entries = dataset_finding_evidence(baseline, dataset_paths[index], header_reports[index])
        if not entries:
            continue
        st.markdown(f"#### Dataset {index + 1}: {filenames[index]} — check evidence")
        for ordinal, entry in enumerate(entries, start=1):
            trace = baseline_evidence_trace(entry, baseline, dataset_paths[index])
            # Short matching label is for navigation, not a second issue card.
            with st.expander(f"Check {ordinal} · {entry['title']}"):
                st.markdown("**Detection rule**")
                st.write(trace["rule"])
                if trace["measurements"]:
                    st.markdown("**How the finding was calculated**")
                    st.dataframe(pd.DataFrame(trace["measurements"]), hide_index=True, width="stretch")
                st.caption(f"Read-only check: {trace['check_name']} · Source: original CSV")
                with st.expander("View Python detection code (advanced)"):
                    st.caption(
                        "The actual source functions installed in this project, not "
                        "AI-generated or illustrative code. The check reads the CSV; "
                        "the Findings tab groups its recorded observations by issue type. "
                        "For a restored historic run, the current source file may "
                        "differ from the implementation used when the audit was saved."
                    )
                    code_items = deterministic_detection_code(trace["check_key"])
                    if code_items:
                        for item in code_items:
                            st.write(f"**{item['path']} · {item['function']}()**")
                            st.code(item["code"], language="python")
                    else:
                        st.warning(
                            "The installed source for this check is not available. "
                            "No replacement code has been fabricated."
                        )
                examples = trace["examples"]
                if examples:
                    with st.expander(f"Inspect source-data examples ({len(examples)} shown)"):
                        st.dataframe(pd.DataFrame(examples), hide_index=True, width="stretch")
                        if trace["examples_omitted"]:
                            st.caption(
                                f"{trace['examples_omitted']:,} additional observations/rows are "
                                "not repeated here. Open the matching finding for its full source evidence."
                            )
                st.caption("Scientific limitation: " + str(trace["limitation"]))

    # The first tab does not expose the agent's raw narrative as separate QC
    # findings. Do not give it its own evidence list here: the registered tools
    # frequently inspect exactly the cells already covered by deterministic
    # checks (e.g. repeated headers). Those original observations are retained
    # in the downloadable audit, not duplicated in this user-facing view.
    #
    # Only verified exploratory investigations that are ALSO shown in the
    # Data quality findings tab receive evidence entries. This is alignment by
    # the underlying result objects, never hard-coded overlap keywords or an
    # extra, potentially non-deterministic model call on every Streamlit rerun.
    if exploratory_runs:
        st.markdown("#### Gemma investigation evidence")
        st.caption(
            "Evidence for the Gemma investigations shown in Data quality findings, "
            "including completed checks that found no additional issue. "
            "The original executed code and source verification are shown here; "
            "agent tool logs already covered by dataset QC are not repeated."
        )
        # Use the same filtering and numbering as render_exploratory_findings.
        visible_runs = exploratory_runs_with_findings(exploratory_runs)
        checked_runs = exploratory_runs_without_findings(exploratory_runs)
        for round_number, run in [*visible_runs, *checked_runs]:
            proposal = run.get("proposal") or {}
            output = run.get("result") or {}
            verification = run.get("verification") or {}
            findings = output.get("findings") or []
            if findings:
                label = f"Investigation {round_number} · {proposal.get('hypothesis') or 'Additional analysis'}"
            else:
                label = f"{gemma_completed_check_title(round_number, run)} — no additional issue"
            with st.expander(label, expanded=False):
                st.markdown("**How Gemma tested this**")
                st.write("**Question examined:**", proposal.get("hypothesis") or "Not recorded")
                st.write("**Detection method:**", proposal.get("method") or "Not recorded")
                if proposal.get("assumptions"):
                    st.write("Assumptions:", proposal["assumptions"])
                # An accepted result must have already passed sandbox execution;
                # this is its recorded code, never newly generated or executed.
                with st.expander("View executed Gemma Python test (advanced)"):
                    st.caption(
                        "This is the saved code accepted for this check, including "
                        "any successful repair. It was run in the read-only sandbox "
                        "and is displayed here without executing it again."
                    )
                    st.code(str(proposal.get("code") or "# Code not recorded"), language="python")

                st.markdown("**What the executed test returned**")
                if findings:
                    # These candidate findings are exactly the ones visible on
                    # the first tab after exploration-stage redundancy filtering.
                    for number, finding in enumerate(findings, start=1):
                        st.write(f"Candidate {number}: {finding.get('title') or 'Untitled'}")
                        refs = finding.get("evidence") or []
                        if refs:
                            st.dataframe(pd.DataFrame(refs), hide_index=True, width="stretch")
                        else:
                            st.warning("No individual source-cell references were saved for this candidate.")
                else:
                    # Empty candidate lists are still meaningful completed
                    # tests. The current sandbox output schema provides a free-
                    # text summary but not necessarily a per-column metric table.
                    # Show the saved result, not manufactured measurements.
                    st.write(output.get("summary") or "No summary was recorded.")
                    st.caption(
                        "No additional candidate findings were returned. If the "
                        "summary contains no numerical breakdown, inspect the "
                        "executed code; the audit does not provide one."
                    )
                with st.expander("Verification record (advanced)"):
                    st.json(verification)
                if verification.get("status") == "source_cells_verified_and_output_reproduced":
                    st.caption(
                        "Cited source cells (when present) matched the originals "
                        "and the sandbox output was reproduced. This does not "
                        "independently validate the scientific interpretation."
                    )
                else:
                    st.caption(
                        "Check the saved verification record before relying on "
                        "this result; scientific interpretation requires review."
                    )


def render_cleaning_workspace(
    dataset_paths: list[Path], uploaded: list, audit: dict, context_key: str,
) -> None:
    """Discuss potential cleaning with local Gemma; gate every supported change."""
    st.subheader("Discuss data cleaning with Gemma")
    st.caption(
        "Ask about findings, scientific assumptions and possible corrections. "
        "Gemma can suggest a plan, but cannot modify files or authorize an operation. "
        "Currently only simple header renames can be applied to a verified new CSV copy. "
        "All other cleaning methods are discussion-only."
    )
    if st.session_state.get("cleaning_context_key") != context_key:
        st.session_state.cleaning_context_key = context_key
        st.session_state.cleaning_messages = []
        st.session_state.cleaning_plan = None
        st.session_state.cleaning_result = None
    messages = st.session_state.cleaning_messages
    for item in messages:
        with st.chat_message(item["role"]):
            st.write(item["content"])
    plan = st.session_state.get("cleaning_plan")
    if plan:
        st.markdown("#### Proposed change — independent review required")
        st.warning(
            "Gemma proposed this header rename. Confirm the new name against your data dictionary. "
            "The original CSV will not be overwritten; only the listed headers may change."
        )
        st.dataframe(pd.DataFrame([
            {"Column": change["position"], "Original header": change["before"],
             "Proposed header": change["after"]}
            for change in plan["changes"]
        ]), hide_index=True, width="stretch")
        st.caption(f"Dataset: {plan['dataset_id']} · Source SHA-256: {plan['source_sha256']}")
        with st.form("cleaning_authorization_form"):
            reviewer = st.text_input("Reviewer identifier", max_chars=120)
            reviewed = st.checkbox("I verified the exact headers and approve this plan for a new copy")
            phrase = st.text_input("Type AUTHORIZE HEADER CLEANING TO COPY")
            approval = st.form_submit_button(
                "Apply reviewed header renames to a new CSV copy", type="primary"
            )
        if approval:
            if not reviewed or phrase != "AUTHORIZE HEADER CLEANING TO COPY" or not reviewer.strip():
                st.error("Scientific review, reviewer identifier and exact confirmation are required.")
            else:
                try:
                    index = int(plan["dataset_id"].split("_")[-1]) - 1
                    checked = validate_header_plan(plan, dataset_paths)
                    if checked != plan:
                        raise ValueError("The proposed plan has changed; request a fresh review")
                    base = Path.home() / ".local" / "share" / "healthcare-data-quality-llm" / "cleaning-workspace"
                    path = base / uuid4().hex
                    request = create_header_request(plan, dataset_paths[index], path)
                    result = execute_confirmed_header_request(request, reviewer.strip(), phrase)
                    output = Path(result["output"])
                    st.session_state.cleaning_result = {
                        "file_name": f"cleaned_{uploaded[index].name}",
                        "data": output.read_bytes(), "verification": result,
                    }
                    st.session_state.cleaning_plan = None
                    st.success("A new CSV copy was created and verified. Original source remains unchanged.")
                except (OSError, ValueError, PermissionError, AssertionError) as exc:
                    st.error(f"Cleaning was not completed: {exc}")
        if st.button("Discard proposed change", key="discard_cleaning_plan"):
            st.session_state.cleaning_plan = None
            st.rerun()
    completed = st.session_state.get("cleaning_result")
    if completed:
        st.success("Verified derived CSV is available for download.")
        st.download_button("Download cleaned CSV copy", data=completed["data"],
                           file_name=completed["file_name"], mime="text/csv")
        with st.expander("Verification record"):
            st.json(completed["verification"])
    question = st.chat_input("Ask Gemma how to handle a finding or propose a specific header rename")
    if question:
        st.session_state.cleaning_plan = None  # Never carry a stale model plan into a new turn.
        baselines = audit.get("dataset_baselines") or {"dataset_1": audit.get("baseline") or {}}
        facts = []
        for dataset_id, baseline in baselines.items():
            idx = int(dataset_id.rsplit("_", 1)[-1]) - 1
            if idx < len(dataset_paths):
                entries = dataset_finding_evidence(baseline, dataset_paths[idx],
                                                   inspect_headers(dataset_paths[idx]))
                facts.extend({"dataset_id": dataset_id, "finding": e["title"],
                              "method": e["method"], "examples": (e["evidence"] or [])[:2]}
                             for e in entries)
        metadata = []
        for index, source in enumerate(dataset_paths):
            with source.open(encoding="utf-8-sig", newline="") as csv_source:
                headers = next(csv.reader(csv_source))
            metadata.append({"dataset_id": f"dataset_{index + 1}",
                             "name": uploaded[index].name,
                             "columns": [{"position": i + 1, "header": h}
                                         for i, h in enumerate(headers)]})
        st.session_state.cleaning_messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Gemma is reviewing your question (no data will be changed)"):
                response = ask_gemma_about_cleaning(
                    question, messages[:-1], facts[:60], metadata,
                )
            reply = response["reply"]
            if response["operation"] == "rename_headers":
                try:
                    planned = validate_header_plan(response, dataset_paths)
                except ValueError as exc:
                    reply += f"\n\nThe proposed operation is not executable: {exc}. Please clarify the exact header(s)."
                else:
                    st.session_state.cleaning_plan = planned
                    reply += "\n\nA reviewed header-change plan is available below the conversation."
            st.session_state.cleaning_messages.append({"role": "assistant", "content": reply})
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.session_state.cleaning_messages.append({"role": "assistant", "content":
                f"Gemma did not produce a valid response: {exc}. No changes were made."})
        st.rerun()


def prepare_exploration_state(baseline_key: str) -> str:
    """Isolate exploration runs by the uploaded data and source version."""
    key = exploration_cache_key(baseline_key)
    if st.session_state.get("exploration_key") != key:
        st.session_state.exploration_key = key
        st.session_state.exploration_runs = []
        st.session_state.exploration_stop = None
        st.session_state.exploration_error = None
        st.session_state.exploration_cache_warning = False
        st.session_state.exploration_restored = False
    return key


def restore_matching_explorations(baseline_key: str) -> None:
    """Restore latest compatible exploration, when explicitly loading an audit.

    Records match data/analysis version but are not guaranteed to belong to the
    exact same earlier audit instance. We label that distinction in the UI.
    """
    key = prepare_exploration_state(baseline_key)
    earlier = saved_explorations(key)
    if not earlier:
        return
    try:
        runs, stopped = load_explorations(earlier[0], key, include_stop=True)
    except (OSError, ValueError, json.JSONDecodeError):
        st.warning("A previous exploratory record was found but could not be loaded.")
        return
    st.session_state.exploration_runs = runs
    st.session_state.exploration_stop = stopped
    st.session_state.exploration_error = None
    st.session_state.exploration_restored = True


class InvestigationProgressView:
    """Live offline timer plus honest *phase*, not time-estimate, progress."""

    def __init__(self, *, include_exploration: bool, exploration_only: bool = False):
        self.tracker = InvestigationProgress(include_exploration=include_exploration)
        self.started = monotonic()
        self.status = st.status("Investigation running...", expanded=True)
        with self.status:
            self.phase_bar = st.progress(0.0, text=self.tracker.phase_label)
            self.timer_slot = st.empty()
            # A self-ticking, local iframe updates every second even while a
            # synchronous Gemma/vLLM HTTP call blocks the Streamlit script.
            # No patient/CSV data is passed into the HTML; no network requests.
            import streamlit.components.v1 as components
            with self.timer_slot:
                components.html(live_timer_html(int(time() * 1000)), height=40)
            self.round_slot = st.empty()
            self.activity_slot = st.empty()
            st.caption("The progress bar shows workflow phases, not an estimated percentage or time remaining. "
                       "Gemma decides when exploration should stop.")
        if exploration_only:
            self.begin_exploration()
        else:
            self._refresh()

    def _refresh(self) -> None:
        self.phase_bar.progress(self.tracker.fraction, text=self.tracker.phase_label)
        self.activity_slot.caption("Current activity: " + self.tracker.last_activity)
        self.round_slot.caption(
            self.tracker.round_label if self.tracker.phase == 3 else
            "Exploratory rounds: not started yet"
            if self.tracker.include_exploration else "Exploratory analysis disabled for this run"
        )

    def baseline_event(self, message: str) -> None:
        self.tracker.baseline_event(message)
        self._refresh()

    def begin_exploration(self, previous_rounds: int = 0) -> None:
        self.tracker.begin_exploration(previous_rounds=previous_rounds)
        self._refresh()

    def exploration_event(self, message: str) -> None:
        self.tracker.exploration_event(message)
        self._refresh()

    def verified_round(self, count: int) -> None:
        self.tracker.verified_round(count)
        self._refresh()

    def finish(self, *, exploration_error: bool = False) -> None:
        # Incomplete exploratory work is not a successful full investigation.
        if exploration_error:
            self._stop_timer()
            self.status.update(label="Baseline completed; exploratory analysis incomplete",
                               state="error", expanded=True)
            return
        self.tracker.finish()
        self._refresh()
        self._stop_timer()
        self.status.update(label="Investigation finished", state="complete", expanded=False)

    def error(self) -> None:
        self._stop_timer()
        self.status.update(label="Investigation interrupted", state="error", expanded=True)

    def _stop_timer(self) -> None:
        self.timer_slot.caption("Elapsed time: " + format_elapsed(monotonic() - self.started))


def run_combined_exploration(
    dataset_paths: list[Path], baseline_key: str, audit: dict,
    *, resume: bool = False, force_first: bool = False,
    on_progress: Callable[[str], None] | None = None,
    on_verified_round: Callable[[int], None] | None = None,
) -> None:
    """Execute the opted-in exploration in the read-only sandbox, never host Python."""
    key = prepare_exploration_state(baseline_key)
    if not resume:
        st.session_state.exploration_runs = []
        st.session_state.exploration_stop = None
        st.session_state.exploration_error = None
        st.session_state.exploration_restored = False
        st.session_state.exploration_cache_warning = False
    datasets = {f"dataset_{i}": p for i, p in enumerate(dataset_paths, 1)}
    notice = st.empty()

    last_saved_checkpoint: Path | None = None

    def checkpoint(runs: list[dict]) -> None:
        nonlocal last_saved_checkpoint
        st.session_state.exploration_runs = runs
        if on_verified_round is not None:
            on_verified_round(len(runs))
        # Persist each verified round to survive later model/sandbox failure.
        try:
            last_saved_checkpoint = save_explorations(key, runs)
        except (OSError, ValueError, TypeError):
            st.session_state.exploration_cache_warning = True

    def notify(message: str) -> None:
        notice.caption(message)
        if on_progress is not None:
            on_progress(message)

    try:
        report = run_exploratory_series(
            datasets, audit,
            previous=st.session_state.exploration_runs,
            force_first=force_first,
            on_round=checkpoint,
            on_progress=notify,
        )
    except SandboxUnavailable:
        st.session_state.exploration_error = (
            "The read-only Bubblewrap sandbox is unavailable. "
            "No generated Python code was executed."
        )
    except Exception as exc:
        # Baseline and completed/verified exploratory rounds remain visible.
        st.session_state.exploration_error = (
            "Exploratory analysis stopped before completion: "
            + f"{type(exc).__name__}: {exc}"
        )
    else:
        st.session_state.exploration_runs = report["runs"]
        st.session_state.exploration_stop = report["stop"]
        st.session_state.exploration_error = None
        try:
            save_explorations(
                key, report["runs"], stop=report["stop"],
                replace_checkpoint=last_saved_checkpoint,
            )
        except (OSError, ValueError, TypeError):
            st.session_state.exploration_cache_warning = True
    finally:
        notice.empty()


def exploratory_runs_with_findings(runs: list[dict]) -> list[tuple[int, dict]]:
    """Keep original round numbers for verified analyses with new findings."""
    return [
        (index, run)
        for index, run in enumerate(runs, 1)
        if (run.get("result") or {}).get("findings")
    ]


def exploratory_runs_without_findings(runs: list[dict]) -> list[tuple[int, dict]]:
    """Keep evidence-verified checks visible without presenting them as issues."""
    return [
        (index, run)
        for index, run in enumerate(runs, 1)
        if not (run.get("result") or {}).get("findings")
    ]


def investigation_outcome(
    audit: dict, *, header_problems: bool, gemma_runs: list[dict],
    gemma_stop: dict | None, gemma_error: str | None,
    exploration_enabled: bool,
) -> dict[str, str]:
    """Separate run state, detected issues, and scientific certification.

    A model stopping is *not* a clean-data assessment. Only classify the
    observations supported by completed checks; blocked work remains explicit.
    """
    baselines = audit.get("dataset_baselines") or {"dataset_1": audit.get("baseline") or {}}
    deterministic_issues = any(baseline_issues(baseline) for baseline in baselines.values())
    deterministic_checks_run = any(
        (check or {}).get("status") == "completed"
        for baseline in baselines.values()
        for check in (baseline.get("checks") or {}).values()
        if isinstance(check, dict)
    )
    exploratory_issues = any((run.get("result") or {}).get("findings") for run in gemma_runs)
    has_findings = bool(header_problems or deterministic_issues or exploratory_issues)

    controller_paused = bool(
        gemma_stop and (
            gemma_stop.get("stop_source") == "proposal_retry_exhausted"
            or "repeatedly proposed investigations already covered" in
            str(gemma_stop.get("reason", ""))
        )
    )
    exploratory_unfinished = bool(
        exploration_enabled and not gemma_stop and not gemma_error
    )
    no_verified_checks = not deterministic_checks_run and not gemma_runs
    backend_budget_exhausted = audit.get("status") == "incomplete_budget_exhausted"
    if (gemma_error or controller_paused or exploratory_unfinished
            or no_verified_checks or backend_budget_exhausted):
        headline = "Investigation partially completed — review results"
        if gemma_error:
            details = "Exploratory analysis was interrupted. Completed QC checks and verified Gemma results remain available."
        elif controller_paused:
            details = "Gemma's exploration paused because it could not propose another distinct test. Further issues have not been ruled out."
        elif no_verified_checks:
            details = "No completed and verified QC checks are recorded for this run."
        elif backend_budget_exhausted:
            details = "The registered-tool investigation reached its execution budget with objectives still outstanding."
        else:
            details = "A completed Gemma exploration is not available for this run. The displayed results cover only completed checks."
        if has_findings:
            details += " Existing issues still need review."
        return {"state": "partial", "headline": headline, "details": details}

    if has_findings:
        return {
            "state": "review",
            "headline": "Investigation run finished — issues require review",
            "details": "Data-quality issues were identified. No source values were changed, and this dataset has not been certified clean.",
        }
    return {
        "state": "limited_clear",
        "headline": "Investigation run finished — no issues detected in completed checks",
        "details": "No issue was found by the checks performed. This does not establish that the entire dataset is clean or scientifically validated.",
    }


def gemma_completed_check_title(index: int, run: dict) -> str:
    """Identical label for a completed Gemma check in Findings and Evidence."""
    result = run.get("result") or {}
    summary = str(result.get("summary") or "Exploratory check completed").strip()
    title = summary.splitlines()[0].strip()[:160] or "Exploratory check completed"
    return f"Check {index}: {title}"


def render_exploratory_findings(
    dataset_paths: list[Path], baseline_key: str, audit: dict,
) -> None:
    """Show candidate issues and completed zero-finding checks separately."""
    prepare_exploration_state(baseline_key)
    runs = st.session_state.get("exploration_runs") or []
    stopped = st.session_state.get("exploration_stop")
    error = st.session_state.get("exploration_error")
    st.subheader("Gemma investigation results")
    st.caption(
        "Gemma investigates hypotheses not already answered by deterministic checks. "
        "All accepted results were reproduced in the isolated read-only sandbox; "
        "biological validity still needs scientific review."
    )
    if error:
        st.warning(error)
    if st.session_state.get("exploration_cache_warning"):
        st.warning("Exploratory results are available in this session, but local caching failed.")
    if st.session_state.get("exploration_restored"):
        st.caption("Exploratory history restored for the same dataset and code version; it may belong to a different earlier audit run.")
    visible_runs = exploratory_runs_with_findings(runs)
    checked_runs = exploratory_runs_without_findings(runs)
    findings_count = sum(
        len((run.get("result") or {}).get("findings") or [])
        for _index, run in visible_runs
    )
    if runs and findings_count == 0:
        st.caption(
            "No additional candidate issues were identified in the Gemma checks completed. "
            "See the completed checks below for what was actually tested."
        )
    else:
        st.caption(
            f"{findings_count} additional candidate finding(s) across "
            f"{len(runs)} verified exploratory investigation(s)."
        )
    # Empty but verified investigations belong in the audit trail, not in the
    # list of candidate problems. Keep their original round numbering intact.
    for index, run in visible_runs:
        result = run.get("result") or {}
        verification = run.get("verification") or {}
        with st.container(border=True):
            st.markdown(f"**Investigation {index}: {run.get('proposal', {}).get('hypothesis', 'Additional analysis')}**")
            st.write(result.get("summary", ""))
            removed = verification.get("already_reported_findings_suppressed", 0)
            if removed:
                st.caption(f"{removed} finding(s) already covered by QC were omitted.")
            for finding in result.get("findings") or []:
                st.markdown("**Candidate finding: " + str(finding.get("title", "")) + "**")
                st.write(finding.get("description", ""))
                st.caption("How Gemma obtained this result and the supporting source-cell evidence are in Supporting evidence.")
    # Completed checks are valuable evidence of what Gemma actually examined,
    # even when they produced no novel issue. Do not label them as findings.
    if checked_runs:
        st.markdown("**Completed Gemma checks (no additional findings)**")
        st.caption(
            "These investigations were completed and verified, but did not identify "
            "additional candidate issues. They are not evidence that the entire "
            "dataset is free of problems."
        )
        for index, run in checked_runs:
            result = run.get("result") or {}
            verification = run.get("verification") or {}
            summary = str(result.get("summary") or "Exploratory check completed").strip()
            with st.expander(f"{gemma_completed_check_title(index, run)} — no additional issue", expanded=False):
                st.markdown("**Hypothesis examined**")
                st.write(run.get("proposal", {}).get("hypothesis", "Additional analysis"))
                st.markdown("**Result**")
                st.write(summary)
                removed = verification.get("already_reported_findings_suppressed", 0)
                if removed:
                    st.caption(f"{removed} already-reported finding(s) were omitted.")
                st.markdown("**Method and verification (advanced)**")
                st.write("**Method:**", run.get("proposal", {}).get("method", ""))
                st.write("**Assumptions:**", run.get("proposal", {}).get("assumptions", ""))
                st.code(run.get("proposal", {}).get("code", ""), language="python")
                st.json(verification)
    if stopped:
        # Explain the end state once, under the overall investigation status.
        # A controller retry-limit is NOT evidence that the model found the
        # dataset clean, and another exploratory run is never required.
        st.caption("Gemma's exploratory run has ended. Technical stopping details are available in Investigation run status above.")
    elif len(runs) >= MAX_ROUNDS:
        st.caption("The exploratory safety limit was reached. Coverage may be incomplete.")
    elif not runs and not error:
        st.caption("Gemma exploration has not produced a verified check for this run.")


def combined_audit_export(audit: dict, baseline_key: str) -> dict:
    """Return a combined JSON report without mutating the original audit cache."""
    key = exploration_cache_key(baseline_key)
    if st.session_state.get("exploration_key") != key:
        return dict(audit, exploratory_investigations=[], exploratory_stop_decision=None)
    return dict(
        audit,
        exploratory_investigations=st.session_state.get("exploration_runs") or [],
        exploratory_stop_decision=st.session_state.get("exploration_stop"),
        exploratory_execution_error=st.session_state.get("exploration_error"),
        exploratory_result_note=(
            "Only evidence-verified sandbox results are included; detected patterns "
            "are candidate findings subject to scientific review."
        ),
    )


def render_ui() -> None:
    st.set_page_config(page_title="Healthcare Data Quality AI", page_icon="🔬", layout="wide")
    st.title("Healthcare Data Quality AI")
    st.caption("Inspect data, trace findings to evidence, and discuss reviewed cleaning with local Gemma.")
    st.info("Original uploads are never overwritten. Only explicitly authorized, supported operations can create a verified cleaned copy.")

    uploaded = st.file_uploader(
        "Upload one or more CSV datasets",
        type=["csv"],
        accept_multiple_files=True,
        help="Up to four local CSV files, maximum 20 MiB each. The first is the primary dataset.",
    )
    if not uploaded:
        st.info("Upload a CSV dataset to inspect its structure and start the investigation.")
        render_saved_file_manager(None)
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
    st.divider()
    st.subheader("Investigate data quality")
    st.caption("Investigations run locally and do not change your uploaded CSV files.")
    st.caption("A full run also automatically executes generated read-only Python in a no-network sandbox, without per-hypothesis approval. Use only with authorized data.")

    # Check the on-disk cache *before* offering another Gemma run. The key
    # includes all CSV bytes, their order, and the current analysis signature.
    cache_key = investigation_cache_key(uploaded)
    prepare_exploration_state(cache_key)
    include_exploration = st.checkbox(
        "Include Gemma's automatic exploratory analyses (isolated read-only Python)",
        value=True, key="enable_unified_exploration",
        help=("The full-run button explicitly authorizes Gemma-generated standard-library "
              "analysis code to execute in the no-network Bubblewrap sandbox, with up to "
              f"{MAX_ROUNDS} verified investigations. It cannot clean or modify CSV files."),
    )
    if include_exploration and not sandbox_available():
        st.warning("Bubblewrap (`bwrap`) is not available; baseline QC can run, but "
                   "automatic generated-code investigation will be skipped.")
    previous_runs = saved_investigations(cache_key)
    current_audit = st.session_state.get("audit_result")
    current_matches = (
        isinstance(current_audit, dict)
        and st.session_state.get("audit_cache_key") == cache_key
        and st.session_state.get("audit_fingerprint") == fingerprint
    )
    # After upgrading from the previous app.py, an older result may still be
    # held in the active Streamlit session, even though it has no cache key.
    # Offer it explicitly rather than forcing another model invocation.
    legacy_session_result = (
        isinstance(current_audit, dict)
        and st.session_state.get("audit_cache_key") is None
        and st.session_state.get("audit_fingerprint") == fingerprint
    )
    load_requested = False
    load_session_requested = False
    run_requested = False
    selected_run = None

    if previous_runs:
        selected_run = previous_runs[0]
        st.info(
            f"Found {len(previous_runs)} saved investigation(s) with matching dataset "
            "contents and analysis configuration. The most recent result is selected automatically."
        )
        st.caption(f"Latest compatible audit: {cache_label(selected_run, browser_viewer_timezone()[0])} — loaded automatically by content hash.")
        load_column, rerun_column, manager_column = st.columns([1, 1.15, 1])
        with load_column:
            load_requested = st.button(
                "Load previous result", type="primary", key="load_previous_audit"
            )
        with rerun_column:
            run_requested = st.button(
                "Run full investigation again", key="rerun_audit"
            )
    elif legacy_session_result:
        st.info(
            "An investigation from before caching was enabled is still "
            "available in this browser session. Reuse it or run Gemma again."
        )
        st.caption(
            "The earlier run's analysis version cannot be independently "
            "verified. Review its findings if you have changed the code or model."
        )
        load_column, rerun_column, manager_column = st.columns([1, 1.15, 1])
        with load_column:
            load_session_requested = st.button(
                "Load previous result", type="primary", key="load_previous_session_audit"
            )
        with rerun_column:
            run_requested = st.button(
                "Run full investigation again", key="rerun_audit"
            )
    elif current_matches:
        st.info("An investigation is already available in this browser session.")
        load_column, rerun_column, manager_column = st.columns([1, 1.15, 1])
        with rerun_column:
            run_requested = st.button("Run full investigation again", key="rerun_audit")
    else:
        st.caption("No matching saved investigation was found for this dataset and analysis version.")
        load_column, rerun_column, manager_column = st.columns([1, 1.15, 1])
        with rerun_column:
            run_requested = st.button("Run full investigation", type="primary")
    with manager_column:
        render_saved_file_manager(cache_key)

    if load_requested and selected_run is not None:
        try:
            restored = load_investigation(selected_run, cache_key)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error("The saved investigation could not be loaded. You can run a new investigation instead.")
            with st.expander("Technical troubleshooting"):
                st.code(f"{type(exc).__name__}: {exc}")
        else:
            st.session_state.audit_result = restored
            st.session_state.audit_fingerprint = fingerprint
            st.session_state.audit_cache_key = cache_key
            st.session_state.audit_source = "saved"
            st.session_state.active_audit_file = str(selected_run)
            restore_matching_explorations(cache_key)

    if load_session_requested:
        restore_matching_explorations(cache_key)
        st.session_state.audit_cache_key = cache_key
        st.session_state.audit_source = "saved"
        try:
            st.session_state.active_audit_file = str(save_investigation(cache_key, current_audit))
        except (OSError, TypeError, ValueError) as exc:
            st.warning(
                "The existing session result is available now, but it could "
                "not be saved for future sessions."
            )
            with st.expander("Cache troubleshooting"):
                st.code(f"{type(exc).__name__}: {exc}")

    if run_requested:
        progress_view = InvestigationProgressView(include_exploration=include_exploration)
        try:
            new_audit = run_initial_audit(
                paths[0],
                additional_datasets=paths[1:],
                progress_callback=progress_view.baseline_event,
            )
            new_audit["scientific_recommendations"] = build_scientific_recommendations(new_audit)
        except Exception as exc:
            progress_view.error()
            st.error("The investigation could not finish. Check that the local AI service is running. No data was changed.")
            with st.expander("Technical troubleshooting"):
                st.code(f"{type(exc).__name__}: {exc}")
        else:
            # Preserve the newly completed result in the current browser
            # session even if writing the persistent cache is unsuccessful.
            st.session_state.audit_result = new_audit
            st.session_state.audit_fingerprint = fingerprint
            st.session_state.audit_cache_key = cache_key
            st.session_state.audit_source = "new"
            st.session_state.pop("active_audit_file", None)
            try:
                st.session_state.active_audit_file = str(save_investigation(cache_key, new_audit))
            except (OSError, TypeError, ValueError) as exc:
                st.warning(
                    "Investigation completed, but it could not be saved for future sessions. "
                    "You can still review it now or download the audit JSON."
                )
                with st.expander("Cache troubleshooting"):
                    st.code(f"{type(exc).__name__}: {exc}")
            if include_exploration:
                progress_view.begin_exploration()
                run_combined_exploration(
                    paths, cache_key, new_audit,
                    on_progress=progress_view.exploration_event,
                    on_verified_round=progress_view.verified_round,
                )
                progress_view.finish(exploration_error=bool(st.session_state.get("exploration_error")))
            else:
                # An intentionally baseline-only rerun must not show stale
                # discoveries from a previous run as if newly verified.
                prepare_exploration_state(cache_key)
                st.session_state.exploration_runs = []
                st.session_state.exploration_stop = None
                st.session_state.exploration_error = None
                st.session_state.exploration_restored = False
                st.session_state.exploration_cache_warning = False
                progress_view.finish()

    audit = st.session_state.get("audit_result")
    if (not isinstance(audit, dict)
            or st.session_state.get("audit_cache_key") != cache_key
            or st.session_state.get("audit_fingerprint") != fingerprint):
        st.info("Choose one of the options above to view investigation results.")
        return

    if st.session_state.get("audit_source") == "saved":
        st.caption("Showing saved results for these dataset contents; no new Gemma run was started.")
    objective_ledger = audit.get("investigation_objectives") or {}
    status = audit.get("status", "unknown")
    header_problems = any(
        bool(report.get("duplicate_headers") or report.get("blank_header_positions")
             or report.get("ragged_records"))
        for report in header_reports
    )
    outcome = investigation_outcome(
        audit,
        header_problems=header_problems,
        gemma_runs=st.session_state.get("exploration_runs") or [],
        gemma_stop=st.session_state.get("exploration_stop"),
        gemma_error=st.session_state.get("exploration_error"),
        exploration_enabled=include_exploration,
    )
    if outcome["state"] == "limited_clear":
        st.info("**" + outcome["headline"] + "** — " + outcome["details"])
    else:
        st.warning("**" + outcome["headline"] + "** — " + outcome["details"])
    if objective_ledger.get("completion") != "evidence_checks_resolved_only":
        st.caption(
            "Coverage limitation: some scientific questions, blocked objectives, "
            "or evidence checks remain unresolved."
        )
    with st.expander("Investigation run status (advanced)"):
        st.write("Agent stop reason:", status)
        st.write("Objective assessment:", objective_ledger.get("completion", "unknown"))
        stop = st.session_state.get("exploration_stop")
        if stop:
            st.write("Gemma exploration stop source:", stop.get("stop_source", "model_decision"))
            st.write("Why Gemma exploration ended:", stop.get("reason", "No explanation recorded."))
            if stop.get("remaining_questions"):
                st.write("Remaining questions:", stop["remaining_questions"])
        elif include_exploration:
            st.write("Gemma exploration:", "No final stopping decision is available.")
    if audit.get("proposal_generation_error"):
        st.error("Gemma could not complete proposal generation; see the downloaded audit for details.")

    summary_tab, evidence_tab, cleaning_tab = st.tabs(
        ["Data quality findings", "Supporting evidence", "Clean data with Gemma"]
    )
    with summary_tab:
        for dataset_id, baseline in (audit.get("dataset_baselines") or {"dataset_1": audit.get("baseline") or {}}).items():
            label_index = int(dataset_id.rsplit("_", 1)[-1]) - 1
            dataset_name = uploaded[label_index].name if label_index < len(uploaded) else dataset_id
            st.subheader(f"Dataset {label_index + 1}: {dataset_name}")
            render_baseline(dataset_id, baseline, source_path=paths[label_index])
            render_header_findings(paths[label_index], header_reports[label_index])
        if len(uploaded) > 1:
            st.warning("Full cross-dataset identifier reconciliation is not yet implemented as an investigation tool.")
        # No separate exploratory page: render novel, verified Gemma findings
        # alongside deterministic findings in the same results tab.
        st.divider()
        render_exploratory_findings(paths, cache_key, audit)
    with evidence_tab:
        render_finding_evidence(
            audit, paths, [file.name for file in uploaded], header_reports,
            st.session_state.get("exploration_runs") or [],
        )
        st.download_button(
            "Download complete audit JSON (may contain confidential data)",
            data=json.dumps(combined_audit_export(audit, cache_key), ensure_ascii=False, indent=2, default=str),
            file_name="healthcare_data_quality_combined_investigation.json",
            mime="application/json",
        )
        st.caption("Keep downloaded audits in approved secure storage. They may contain sample identifiers and detailed source observations.")
    with cleaning_tab:
        render_cleaning_workspace(paths, uploaded, audit, cache_key)


if __name__ == "__main__":
    render_ui()
