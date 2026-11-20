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
import re
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
from exploration_banner import (
    recommendation_is_complete,
    recommendation_progress_counts,
    render_agent_c_banner_html,
    render_exploration_banner_html,
    render_finished_investigation_banner_html,
    render_idle_investigation_banner_html,
    render_investigation_phase_banner_html,
    render_recommendation_banner_html,
    sticky_banner_css,
)

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
from investigation_jobs import investigation_manager
from healthcare_data_quality.llm_client import model_request_priority
from healthcare_data_quality.exploratory_agent import MAX_ROUNDS
from healthcare_data_quality.exploratory_combined import run_exploratory_series
from healthcare_data_quality.exploratory_sandbox import sandbox_available, SandboxUnavailable
from healthcare_data_quality.finding_evidence import (
    dataset_finding_evidence,
)
from evidence_view import (
    baseline_evidence_trace, deterministic_detection_code, measurement_rows_for_display,
)
from healthcare_data_quality.cleaning_workspace import (
    ask_gemma_about_cleaning, validate_header_plan, agent_c_clarification_options,
)
from healthcare_data_quality.cleaning_authorization import file_sha256
from healthcare_data_quality.source_inventory import source_inventory, concise_source_context

# Reuse the CLI's existing recommendation assessment, including its provenance
# and partial-coverage rules. No CLI main() or remediation entry point is called.
_SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from run_workflow import build_scientific_recommendations  # noqa: E402


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_DATASETS = 1

# Cached audits can include source values and identifying metadata. Keep them
# outside the Git repository, in a private local directory. Each re-run is
# retained as a separate audit instead of overwriting earlier results.
CACHE_SCHEMA_VERSION = 1
AUDIT_CACHE_DIR = (
    Path.home() / ".local" / "share" / "healthcare-data-quality-llm" / "audit-cache"
)
MAX_CACHED_AUDIT_BYTES = 200 * 1024 * 1024


def _agent_c_message_requests_explicit_authorization(message: str) -> bool:
    """Return True when Agent C is asking for an explicit approval click.

    This keeps the UI simple: when Agent C asks the researcher to confirm or
    authorize the exact positions/replacements, show a one-click approval
    button instead of forcing the researcher to type the same authorization
    back into the chat box.
    """
    normalized = " ".join((message or "").lower().split())
    if not normalized:
        return False
    auth_markers = (
        "explicit authorization",
        "authorize the specific",
        "authorize the exact",
        "do you authorize",
        "need you to confirm",
        "confirm the following",
    )
    scope_markers = (
        "structured change",
        "structured changes",
        "specific replacements",
        "affected positions",
        "specified cells",
        "replacement value",
        "exact value",
    )
    return any(marker in normalized for marker in auth_markers) and any(
        marker in normalized for marker in scope_markers
    )


def _agent_c_authorization_message() -> str:
    """Standardized reply for the approval button.

    Agent C still has to validate the scope and generate a structured plan on
    the backend. The button only confirms that the researcher authorizes the
    exact positions and replacement values already listed in Agent C's latest
    message for this finding.
    """
    return (
        "I authorize the exact positions and replacement values you just listed "
        "for this finding only. Prepare the structured change objects now. "
        "Do not execute or apply any data changes yet."
    )


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
                      if not p.name.startswith("exploratory_")
                      and p.name not in {"cleaning_stage_checkpoint.py", "reviewed_rule_replay.py",
                                         "evidence_registry.py"})
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


def numbered_finding_title(title: str, index: int | None = None) -> str:
    """Display-only numbering matching the Supporting evidence check order."""
    return f"Finding {index} · {title}" if index is not None else title


def show_finding(
    title: str, explanation: str, evidence: list[dict],
    next_step: str = "", *, finding_index: int | None = None,
) -> None:
    """Reader-facing finding; structured evidence stays expandable."""
    with st.container(border=True):
        st.markdown(f"**{numbered_finding_title(title, finding_index)}**")
        st.write(explanation)
        if next_step:
            st.write("**What to check next:**", next_step)
        if evidence:
            with st.expander(f"View affected rows ({len(evidence):,})"):
                st.dataframe(pd.DataFrame(evidence), hide_index=True, width="stretch")


def show_taxonomy_structure_finding(finding: dict, *, finding_index: int | None = None) -> None:
    """Show all original classifications and an explicit rank-by-rank expectation."""
    with st.container(border=True):
        st.markdown(f"**{numbered_finding_title(finding['title'], finding_index)}**")
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
        evidence = finding.get("evidence") or []
        if evidence:
            with st.expander(f"View affected rows ({len(evidence):,})"):
                st.write("**Original classifications and taxonomy level counts:**")
                st.dataframe(pd.DataFrame(evidence), hide_index=True, width="stretch")

        if finding.get("next_step"):
            st.write("**What to check next:**", finding["next_step"])



def render_header_findings(path: Path, details: dict, *, start_index: int = 1) -> None:
    repeated = header_findings(path, details)
    missing_and_width = header_issues(details)
    if not repeated and not missing_and_width:
        st.success("No repeated or blank column names, or inconsistent row lengths, were found.")
        return
    st.markdown("#### Data structure findings")
    for finding_index, finding in enumerate(repeated, start=start_index):
        evidence = [{
            "Repeated column name": finding["header"],
            "First column": finding["columns"][0],
            "Second column": finding["columns"][1],
            "Matching rows": finding.get("matching_rows", 0),
            "Different rows": finding.get("different_rows", 0),
            "Rows not comparable": finding.get("uncomparable_rows", 0),
        }] if len(finding["columns"]) == 2 else []
        show_finding(finding["title"], finding["explanation"], evidence, finding["next_step"],
                     finding_index=finding_index)
    for finding_index, finding in enumerate(missing_and_width, start=start_index + len(repeated)):
        show_finding(finding["title"], finding["explanation"], finding["rows"],
                     finding["next_step"], finding_index=finding_index)


def render_baseline(dataset_id: str, baseline: dict, source_path: Path | None = None) -> int:
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
        for finding_index, finding in enumerate(findings, start=1):
            if finding.get("type") == "taxonomy_structure":
                show_taxonomy_structure_finding(finding, finding_index=finding_index)
            else:
                show_finding(finding["title"], finding["explanation"],
                             finding["evidence"], finding["next_step"],
                             finding_index=finding_index)

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

    return len(findings)


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
                    st.dataframe(pd.DataFrame(measurement_rows_for_display(trace["measurements"])),
                                 hide_index=True, width="stretch")
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


def schedule_agent_b_once(context_key: str, audit: dict, paths: list[Path],
                          names: list[str], runs: list[dict]) -> str | None:
    """Start Agent B during Agent A investigation, once per finding and run.

    First call follows deterministic QC; a second call after Gemma exploration
    only appends new checks. No UI action or cleaned version re-runs advice.
    """
    from cleaning_jobs import recommendation_manager
    from healthcare_data_quality.finding_cleaning import build_cleaning_findings
    run_id = audit.get("cleaning_recommendation_run_id")
    if not run_id:
        return None
    findings = build_cleaning_findings(audit, paths, names, runs)
    if not findings:
        return None
    # Historical approvals are inputs to NEW Gemma assessments, never ready
    # recommendations. Reuse only relevant, bounded method summaries, including
    # related finding/check types, before the worker publishes a suggestion.
    from cleaning_jobs import recommendation_manager, reviewed_method_context
    reviewed_methods = {}
    for finding in findings:
        ds_id = str(finding.get("dataset_id") or "")
        try:
            index = int(ds_id.rsplit("_", 1)[-1]) - 1
        except (ValueError, IndexError):
            index = -1
        # A Gemma check can span multiple uploads and have no dataset_id.
        # Assess saved methods against each involved source rather than
        # silently dropping memory for those checks.
        sources = ([paths[index]] if 0 <= index < len(paths) else
                   paths if finding.get("source") != "QC" else [])
        candidates, seen = [], set()
        for candidate_source in sources:
            for method in reviewed_method_context(
                    finding, Path(candidate_source), limit=16):
                fingerprint = json.dumps(method, sort_keys=True, ensure_ascii=False)
                if fingerprint not in seen:
                    candidates.append(method)
                    seen.add(fingerprint)
                if len(candidates) >= 16:
                    break
            if len(candidates) >= 16:
                break
        reviewed_methods[finding["id"]] = candidates
    return recommendation_manager().start(
        context_key, findings, run_id=run_id, reviewed_methods=reviewed_methods,
    )


def _review_action_button_css() -> str:
    """Color only explicit cleaning-review actions, never unrelated app controls."""
    return """<style>
    div[class*="st-key-review_yes_"] button:not(:disabled),
    div[class*="st-key-review_memory_yes_"] button:not(:disabled),
    div[class*="st-key-review_clarify_yes_"] button:not(:disabled),
    div[class*="st-key-review_approve_yes_"] button:not(:disabled) {
        background-color: #15803d !important;
        border-color: #15803d !important;
        color: #fff !important;
    }
    div[class*="st-key-review_yes_"] button:not(:disabled):hover,
    div[class*="st-key-review_memory_yes_"] button:not(:disabled):hover,
    div[class*="st-key-review_clarify_yes_"] button:not(:disabled):hover,
    div[class*="st-key-review_approve_yes_"] button:not(:disabled):hover {
        background-color: #166534 !important;
        border-color: #166534 !important;
        color: #fff !important;
    }
    div[class*="st-key-review_no_"]:not([class*="st-key-review_no_action_"]) button:not(:disabled),
    div[class*="st-key-review_memory_no_"] button:not(:disabled),
    div[class*="st-key-review_discuss_no_"] button:not(:disabled) {
        background-color: #b91c1c !important;
        border-color: #b91c1c !important;
        color: #fff !important;
    }
    div[class*="st-key-review_no_"]:not([class*="st-key-review_no_action_"]) button:not(:disabled):hover,
    div[class*="st-key-review_memory_no_"] button:not(:disabled):hover,
    div[class*="st-key-review_discuss_no_"] button:not(:disabled):hover {
        background-color: #991b1b !important;
        border-color: #991b1b !important;
        color: #fff !important;
    }
    /* Completed review headers are green without changing the expander's
       stable label/key. Changing the label on approval previously caused a
       Streamlit remount and jumped the reviewer back to the top. */
    details:has(.hdq-cleaning-completed-marker) {
        border-color: #15803d !important;
        scroll-margin-top: 22px;
    }
    details:has(.hdq-cleaning-completed-marker) > summary {
        background-color: rgba(22, 163, 74, 0.14) !important;
        color: #4ade80 !important;
    }
    details:has(.hdq-cleaning-completed-marker) > summary p {
        color: #4ade80 !important;
        font-weight: 600 !important;
    }
    details:has(.hdq-cleaning-completed-marker) > summary [data-testid="stMarkdownContainer"] p::after {
        content: " · Completed";
        color: #4ade80;
        font-weight: 700;
    }
    div[data-testid="stMarkdown"]:has(.hdq-cleaning-completed-marker) {
        display: none;
    }
    </style>"""


def _open_agent_c_alternative_chat(fid: str, *, discard_plan: bool = False) -> None:
    """Open the current finding's input box without contacting any model.

    Discard only unapproved proposals. Historical approvals and the original
    CSV are untouched. Expire old clarification widgets so an old answer
    cannot be clicked after the researcher chooses No.
    """
    if discard_plan:
        _cancel_agent_c_batch(fid, status="Discussion open")
        for field in (
            "cleaning_compound_previews_by_finding", "cleaning_r_previews_by_finding",
            "cleaning_r_errors_by_finding", "cleaning_r_diagnostics_by_finding",
            "cleaning_r_selfchecks_by_finding", "cleaning_rebased_previews_by_finding",
            "cleaning_explicit_row_deletions_by_finding",
            "cleaning_explicit_column_deletions_by_finding",
        ):
            st.session_state.setdefault(field, {}).pop(fid, None)
    else:
        _next_agent_c_ui_generation(fid)
    st.session_state.setdefault("cleaning_custom_value_input_by_finding", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_custom_operation_input_by_finding", {}).pop(fid, None)
    st.session_state.cleaning_discussion_mode_by_finding[fid] = "alternative"
    st.session_state.cleaning_review_statuses[fid] = "Discussion open"
    finding_for_memory = st.session_state.get("cleaning_finding_type_lookup", {}).get(fid)
    if finding_for_memory:
        try:
            from healthcare_data_quality.adaptive_skill_memory import record_review
            record_review(finding_for_memory, "declined_proposal", build_skill=False)
        except (OSError, ValueError, RuntimeError, TypeError):
            pass


def _recommendation_preview(text: str, max_chars: int = 220) -> str:
    """Show a short, non-generative preview; preserve the full advice in details."""
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    segment = clean[:max_chars].rsplit(" ", 1)[0]
    return (segment or clean[:max_chars]).rstrip(" ,;:") + "…"


def _next_agent_c_ui_generation(fid: str) -> int:
    """Invalidate all prior clarification/approval widget identities.

    The chat transcript remains available for audit, but cannot keep obsolete
    options active when a newer request or validated proposal takes its place.
    """
    versions = st.session_state.setdefault("cleaning_control_generation_by_finding", {})
    versions[fid] = int(versions.get(fid, 0)) + 1
    for item in st.session_state.get("cleaning_messages_by_finding", {}).get(fid, []):
        if item.get("role") == "assistant":
            item["controls_expired"] = True
    return versions[fid]


def _active_agent_c_control_generation(fid: str) -> int:
    return int(st.session_state.setdefault(
        "cleaning_control_generation_by_finding", {}).get(fid, 0))


def _render_before_after(plans: list[dict]) -> None:
    """Show only exact, independently verified intended edits.

    Model prose and aggregate statistics are not substitutes for each source
    target's actual original value and proposed outcome.
    """
    from healthcare_data_quality.reviewed_rule_replay import current_change_rows
    changes = current_change_rows(plans)
    if changes:
        st.dataframe(pd.DataFrame([
            {"Location": item["Target"], "Before": item["Before"],
             "After": item["After"]} for item in changes
        ]), hide_index=True, width="stretch")


def _render_source_deletion_details(source: Path, plans: list[dict],
                                    *, key: str) -> bool:
    """No second inspection UI: exact before/after and source validators suffice.

    Structural source integrity is rechecked by evaluate_compound (or the
    single-operation validator), both before rendering approval and on commit.
    """
    return True


def _agent_c_control_phase(*, mode: str | None, transcript: list,
                           plan: dict | None, queued_actions: list | None,
                           pending: bool, custom_value: bool = False,
                           custom_operation: bool = False) -> tuple[str, list[str]]:
    """One authoritative interaction state for every finding/Gemma check.

    A validated proposal always supersedes older questions. Reading saved chat
    history must never withdraw an active plan or resurrect its old controls.
    All agent roles use this same per-finding state; a prior button is not a
    live authorization to change data.
    """
    if plan is not None or queued_actions:
        return "approval", []
    if pending:
        return "processing", []
    if mode not in {"accepted", "alternative"}:
        return "idle", []
    if custom_value or custom_operation:
        return "custom_input", []
    latest = transcript[-1] if transcript else None
    if isinstance(latest, dict) and latest.get("role") == "assistant":
        if not latest.get("controls_expired"):
            choices = agent_c_clarification_options(
                str(latest.get("content") or ""), latest.get("clarification_options"))
            if choices:
                return "clarification", choices
    return "discussion", []


def _finding_review_completed(fid: str) -> bool:
    """A final review may collapse; partial/queued approvals must stay open.

    Use the same terminal statuses as the review renderer. A historical
    approval on its own is not enough: a new discussion or pending batch
    must still be available for the reviewer to finish.
    """
    status = st.session_state.get("cleaning_review_statuses", {}).get(fid)
    terminal = {
        "Applied and verified", "Resolved by earlier approved edit",
        "Approved column deletion for final export",
        "Approved row changes for final export", "Reviewed — no change",
    }
    if status not in terminal:
        return False
    if st.session_state.get("cleaning_plans_by_finding", {}).get(fid):
        return False
    if st.session_state.get("cleaning_batch_queue_by_finding", {}).get(fid):
        return False
    if fid in st.session_state.get("cleaning_batch_rejected_by_finding", set()):
        return False
    return status == "Reviewed — no change" or any(
        event.get("finding_id") == fid
        for event in st.session_state.get("cleaning_history", [])
    )


def _scroll_to_completed_finding_once(fid: str) -> None:
    """Restore viewport focus to the just-approved header, only once.

    Native Streamlit fragments preserve position across most reruns, but
    collapsing a tall card can move its header out of view. This optional,
    zero-height browser helper targets only our stable completed marker and
    never reads user data or changes approval state. If a browser disallows
    parent access from the component iframe, normal collapse still works.
    """
    if st.session_state.get("cleaning_scroll_to_completed_finding") != fid:
        return
    st.session_state.pop("cleaning_scroll_to_completed_finding", None)
    try:
        from streamlit.components.v1 import html
    except ImportError:
        return
    marker_id = hashlib.sha256(fid.encode("utf-8")).hexdigest()[:20]
    # The selector embeds only a SHA-256 hex digest, never filename/CSV text.
    html('''<script>
    (() => {
      const markerId = "{marker_id}";
      let attempts = 0;
      const restore = () => {
        try {
          const node = window.parent.document.querySelector(
            '.hdq-cleaning-completed-marker[data-review-focus="' + markerId + '"]');
          const card = node && node.closest('details');
          if (card) {
            // Do not rely on Streamlit accepting the new expanded=False
            // prop after the card was opened manually in the browser.
            if (card.open) card.open = false;
            card.scrollIntoView({ block: 'start', behavior: 'instant' });
            return;
          }
        } catch (error) { return; }
        if (++attempts < 12) window.setTimeout(restore, 75);
      };
      window.requestAnimationFrame(restore);
    })();
    </script>'''.replace("{marker_id}", marker_id), height=0, scrolling=False)


def _rerun_cleaning_fragment(fid: str | None = None) -> None:
    """Refresh cleaning without jumping to another finding.

    A successful final approval closes its card, leaving the green completed
    header visible. When a batch still has pending actions, or the user opens
    discussion, retain that card's expanded state instead.
    """
    if fid:
        open_map = st.session_state.setdefault("cleaning_keep_open_by_finding", {})
        open_map.clear()
        if _finding_review_completed(fid):
            st.session_state["cleaning_scroll_to_completed_finding"] = fid
        else:
            open_map[fid] = True
            st.session_state.pop("cleaning_scroll_to_completed_finding", None)
        st.session_state["cleaning_active_finding"] = fid
    try:
        st.rerun(scope="fragment")
    except TypeError:
        st.rerun()
    except Exception as exc:
        if type(exc).__name__ not in {
            "StreamlitAPIException", "StreamlitInvalidLayoutContextError",
        }:
            raise
        st.rerun()



def _rerun_parent_after_agent_b_choice(fid: str) -> None:
    """Refresh the parent review after clicking a nested Agent B control.

    Agent B is a polling child fragment. A fragment-scoped rerun there only
    refreshes Agent B; the parent's Agent C input/approval cannot appear.
    Force an app rerun for a *user click*, preserving the selected finding's
    open state. Background polling continues to use its normal fragment.
    """
    keep_open = st.session_state.setdefault("cleaning_keep_open_by_finding", {})
    keep_open.clear()
    keep_open[fid] = True
    st.session_state["cleaning_active_finding"] = fid
    try:
        st.rerun(scope="app")
    except TypeError:
        st.rerun()
    except Exception as exc:
        if type(exc).__name__ != "StreamlitAPIException":
            raise
        st.rerun()


def _render_live_finding_assignment(fid: str) -> None:
    """Indicate an in-flight request with Streamlit's native spinner.

    The indicator is rendered *inside the existing cleaning review fragment*.
    Do not create a nested polling fragment here: across Streamlit versions,
    fragment reruns inside an expander/tab can leave stale form blocks mounted.
    The separate background completion poller delivers the final reply.
    """
    from interactive_chat_jobs import interactive_chat_manager

    pending = st.session_state.get("cleaning_pending_gemma_b", {}).get(fid)
    if not isinstance(pending, dict) or not pending.get("job_id"):
        return

    manager = interactive_chat_manager()
    job_id = pending["job_id"]
    role = manager.agent_for(job_id) if hasattr(manager, "agent_for") else None
    assigned = role if role in {"A", "B", "C"} else pending.get("assigned_agent", "C")
    pending["assigned_agent"] = assigned
    phase = manager.phase_for(job_id) if hasattr(manager, "phase_for") else "queued"

    if phase == "queued":
        label = "Queued — waiting for an inference slot"
    elif phase == "generating":
        label = f"Agent {assigned} is processing your request"
    elif phase == "done":
        st.caption("Response ready — opening review")
        return
    elif phase == "failed":
        st.caption("Request finished — opening error details")
        return
    else:
        label = f"Agent {assigned} is preparing your request"

    # This running state animates natively, without HTML escaping/code blocks,
    # fabricated completion percentages, or a second editable form.
    st.status(label, state="running", expanded=False)


def _render_live_agent_b_finding(
    fid: str, finding: dict, source: Path | None, uploaded: list,
    dataset_paths: list[Path], job_id: str | None, ds_id: str,
) -> None:
    """Show live per-finding Agent B advice without rerendering Agent C input.

    This is called from a *small independent polling fragment* nested inside
    the otherwise user-driven cleaning fragment. Only the suggestion and its
    Yes/No buttons are redrawn when Agent B finishes; the chat textarea and
    approval forms elsewhere in the finding stay mounted.
    """
    from cleaning_jobs import recommendation_manager

    entry = {}
    if job_id:
        try:
            snapshot = recommendation_manager().snapshot(job_id) or {}
            entry = ((snapshot.get("entries") or {}).get(fid) or {})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            st.caption(f"Could not refresh Agent B's suggestion: {exc}")

    if entry.get("status") == "ready":
        previous = st.session_state.cleaning_recommendations.get(fid) or {}
        first_ready = fid not in st.session_state.setdefault("cleaning_autoready_seen", set())
        if first_ready:
            st.session_state.cleaning_autoready_seen.add(fid)
        review_status = st.session_state.cleaning_review_statuses.get(fid)
        # A background job must never replace a human-selected recommendation
        # or resurrect a completed/rejected review.
        if (previous.get("_source") not in {"manual", "review_memory"} and
                review_status in {None, "Awaiting review", "Agent B suggestion ready"}):
            st.session_state.cleaning_recommendations[fid] = {
                "suggestion": entry["suggestion"],
                "decision_needed": entry.get("decision_needed", ""),
                "memory_assessed": entry.get("memory_assessed", False),
                "choices": entry.get("choices") or [],
                "_source": "background",
            }
            st.session_state.cleaning_review_statuses[fid] = "Agent B suggestion ready"

    # A parent/card approval or active conversation always owns this finding.
    # The independently polling Agent B fragment must display NOTHING in that
    # state, including old advice and preparation buttons.
    if (st.session_state.cleaning_plans_by_finding.get(fid) is not None
            or st.session_state.cleaning_batch_queue_by_finding.get(fid)
            or fid in st.session_state.cleaning_pending_gemma_b
            or st.session_state.cleaning_discussion_mode_by_finding.get(fid) in {"accepted", "alternative"}
            or any(h.get("finding_id") == fid for h in st.session_state.cleaning_history)
            or st.session_state.cleaning_review_statuses.get(fid) == "Reviewed — no change"):
        return
    suggestion = st.session_state.cleaning_recommendations.get(fid)
    # Memory-backed findings use the single reviewed-solution Yes/No UI in
    # the parent finding, never a second Agent B recommendation widget.
    if suggestion and suggestion.get("_source") == "review_memory":
        return
    if suggestion:
        from recommendation_options import source_backed_choices
        options = source_backed_choices(finding, source, suggestion)
        # Exact Before -> After is the primary interface. Do not turn a
        # plausible model narrative into an unverified operation button.
        if options:
            for option_idx, option in enumerate(options):
                if len(options) > 1:
                    st.markdown('**' + option['label'] + '**')
                st.dataframe(pd.DataFrame(option['rows']), hide_index=True, width="stretch")
            choices = st.columns(len(options))
            for index, option in enumerate(options):
                label = ('Yes — Approve this approach' if len(options) == 1
                         else 'Yes — ' + option['label'])
                if choices[index].button(
                    label, key=f"review_yes_{fid}_{_active_agent_c_control_generation(fid)}_option_{index}"):
                    # User explicitly selected this exact source-grounded
                    # proposal. Validate it through the same Agent C path as
                    # a hand-written request; only a later NAMED Yes executes.
                    op_summary = []
                    for op in option['operations']:
                        if op['operation'] == 'drop_rows':
                            ids = [c['row'] for c in op['changes']]
                            op_summary.append('Delete CSV rows '+', '.join(map(str, ids)))
                            st.session_state.setdefault(
                                'cleaning_explicit_row_deletions_by_finding', {})[fid] = frozenset(ids)
                        if op['operation'] == 'drop_column':
                            ids = [c['position'] for c in op['changes']]
                            op_summary.append('Delete columns '+', '.join(map(str, ids)))
                            st.session_state.setdefault(
                                'cleaning_explicit_column_deletions_by_finding', {})[fid] = frozenset(ids)
                    request = ('; '.join(op_summary) if op_summary else option['label'])
                    st.session_state.cleaning_messages_by_finding.setdefault(fid, []).append({
                        'role': 'user', 'content': request,
                    })
                    st.session_state.cleaning_discussion_mode_by_finding[fid] = 'accepted'
                    st.session_state.cleaning_review_statuses[fid] = 'Preparing verified changes'
                    _record_gemma_b_response(fid, finding, source, {
                        'reply': option['label'] + ' — review exact changes below.',
                        'operation': 'none', 'dataset_id': '', 'changes': [],
                        'operations': option['operations'], 'clarification_options': [],
                    })
                    _rerun_parent_after_agent_b_choice(fid)
        else:
            # Model did not provide exact source-backed actions. No fake
            # before/after or green confirmation of a prose-only suggestion.
            st.write(_recommendation_preview(str(suggestion['suggestion']), 180))
            st.caption('Exact changes require source verification. Discuss the approach.')
        if st.button('No — Discuss Another Idea',
                     key=f"review_no_{fid}_{_active_agent_c_control_generation(fid)}_choices"):
            _open_agent_c_alternative_chat(fid)
            _rerun_parent_after_agent_b_choice(fid)
    else:
        background_status = entry.get('status')
        if background_status == 'running':
            st.caption('Preparing recommendation')
        elif background_status == 'pending':
            st.caption('Recommendation queued')
        elif background_status == 'failed':
            st.caption('Recommendation unavailable — use Agent C to discuss')
    # A child polling fragment cannot expand its parent directly. Rerun the
    # main cleaning fragment once, when this finding first becomes ready.
    if (entry.get("status") == "ready" and first_ready
            and fid not in st.session_state.get("cleaning_parent_ready_ids", set())):
        # A child fragment rerun cannot change its parent's expander state.
        # One full rerun, never continuous polling reruns, opens the card.
        st.rerun()


def _is_blank_control_check(finding: dict) -> bool:
    """Only a Gemma check about blank controls may offer control deletion."""
    if finding.get("check_key") != "gemma_check":
        return False
    description = " ".join(str(finding.get(k) or "") for k in ("title", "method")).casefold()
    return "blank" in description and ("control" in description or "negative control" in description)


def _reconcile_approved_column_review(fid: str, ds_id: str, source: Path) -> bool:
    """Retire redundant column proposals after another finding approved them.

    Never interpret a model suggestion as approval: a match requires an existing
    named approval, original column position/header, and immutable source hash.
    New operations remain pending for ordinary independent verification.
    Returns whether all pending actions were already approved elsewhere.
    """
    plans_by_finding = st.session_state.setdefault("cleaning_plans_by_finding", {})
    current = plans_by_finding.get(fid)
    if not current:
        return False
    plans = [current, *st.session_state.setdefault(
        "cleaning_batch_queue_by_finding", {}).get(fid, [])]
    if not plans:
        return False
    from healthcare_data_quality.cleaning_authorization import file_sha256
    from healthcare_data_quality.in_memory_cleaning import _records
    headers = _records(source.read_bytes())[0]
    source_hash = file_sha256(source)
    approved = {
        int(item["position"]): item
        for item in st.session_state.setdefault(
            "cleaning_approved_columns_by_dataset", {}).get(ds_id, [])
        if isinstance(item, dict) and isinstance(item.get("position"), int)
    }
    remaining, already_approved = [], []
    for plan in plans:
        if (plan.get("operation") == "drop_column" and
                len(plan.get("changes") or []) == 1):
            item = plan["changes"][0]
            position = item.get("position")
            record = approved.get(position)
            if (type(position) is int and 1 <= position <= len(headers) and
                    record and plan.get("dataset_id") == ds_id and
                    plan.get("source_sha256") in (None, "", source_hash) and
                    record.get("source_sha256") == source_hash and
                    record.get("before") == item.get("before") == headers[position - 1] and
                    str(record.get("reviewer") or "").strip()):
                already_approved.append(plan)
                continue
        remaining.append(plan)
    if not already_approved:
        return False
    st.session_state.setdefault("cleaning_compound_previews_by_finding", {}).pop(fid, None)
    if remaining:
        plans_by_finding[fid] = remaining[0]
        st.session_state.cleaning_batch_queue_by_finding[fid] = remaining[1:]
        st.session_state.setdefault("cleaning_batch_total_by_finding", {})[fid] = len(remaining)
        return False
    # All requested edits have already been approved. Never create a second
    # approval event, revert the prior deletion, or show stale chat controls.
    for name in ("cleaning_plans_by_finding", "cleaning_batch_queue_by_finding",
                 "cleaning_batch_total_by_finding", "cleaning_compound_previews_by_finding",
                 "cleaning_memory_candidates", "cleaning_explicit_column_deletions_by_finding"):
        st.session_state.setdefault(name, {}).pop(fid, None)
    st.session_state.setdefault("cleaning_discussion_mode_by_finding", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_pending_gemma_b", {}).pop(fid, None)
    # This finding owns NO second approval event. Record a reviewed-without-
    # additional-change decision, which the recovery ledger permits without
    # inventing a duplicate named approval for this finding.
    st.session_state.setdefault("cleaning_review_statuses", {})[fid] = "Reviewed — no change"
    return True


def _cleaning_export_gate(catalog: list[dict], review_statuses: dict,
                          plans: dict, queues: dict, pending: dict,
                          visual_flags: dict, *, investigation_running: bool) -> tuple[bool, int, int]:
    """No export until every current finding and check reaches terminal review."""
    terminal = {
        "Applied and verified", "Reviewed — no change",
        "Resolved by earlier approved edit", "Approved column deletion for final export",
        "Approved row changes for final export",
    }
    ids = {str(f["id"]) for f in catalog}
    done = sum(
        review_statuses.get(fid) in terminal and
        not plans.get(fid) and not queues.get(fid) and
        fid not in pending and not visual_flags.get(fid)
        for fid in ids
    )
    return bool(ids and done == len(ids) and not investigation_running), done, len(ids)


def render_cleaning_workspace(
    dataset_paths: list[Path], uploaded: list, audit: dict, context_key: str,
    gemma_runs: list[dict] | None = None,
) -> None:
    """Agent C reviews Agent B advice, negotiates scoped changes and R approval."""
    from contextlib import nullcontext
    from cleaning_review import (
        ask_gemma_for_review_options, indexed_review_catalog, review_question,
    )
    from healthcare_data_quality.finding_cleaning import (
        allowed_positions, build_cleaning_findings, validate_scoped_plan,
    )
    from healthcare_data_quality.in_memory_cleaning import (
        apply_verified_in_memory, assess_verified_in_memory, build_final_download,
        review_completion, apply_approved_column_deletions,
        apply_approved_row_operations, _records, sha256_bytes,
    )

    from healthcare_data_quality.scoped_r_execution import (
        prepare_and_execute_scoped_r_preview, verify_prepared_r_insertion,
        verify_r_against_current_staged_values,
    )
    from healthcare_data_quality.r_diagnostics import (
        diagnose_r_preview_failure, run_r_environment_self_test,
    )
    st.subheader("Agent C — Interactive cleaning")
    st.markdown(_review_action_button_css(), unsafe_allow_html=True)
    run_id = audit.get("cleaning_recommendation_run_id")
    session_context = f"{context_key}:{run_id}" if run_id else context_key
    if st.session_state.get("cleaning_context_key") != session_context:
        st.session_state.cleaning_context_key = session_context
        st.session_state.cleaning_messages_by_finding = {}
        st.session_state.cleaning_pending_gemma_b = {}
        st.session_state.cleaning_completed_gemma = {}
        st.session_state.cleaning_plans_by_finding = {}
        st.session_state.cleaning_batch_queue_by_finding = {}
        st.session_state.cleaning_batch_total_by_finding = {}
        st.session_state.cleaning_compound_previews_by_finding = {}
        st.session_state.cleaning_batch_rejected_by_finding = set()
        st.session_state.cleaning_r_previews_by_finding = {}
        st.session_state.cleaning_r_errors_by_finding = {}
        st.session_state.cleaning_r_diagnostics_by_finding = {}
        st.session_state.cleaning_r_selfchecks_by_finding = {}
        st.session_state.cleaning_plan_retry_once = set()
        st.session_state.cleaning_recommendations = {}
        st.session_state.cleaning_review_statuses = {}
        st.session_state.cleaning_discussion_mode_by_finding = {}
        st.session_state.cleaning_custom_value_input_by_finding = {}
        st.session_state.cleaning_custom_operation_input_by_finding = {}
        st.session_state.cleaning_decisions = []
        st.session_state.cleaning_memory_candidates = {}
        st.session_state.cleaning_memory_skipped = set()
        st.session_state.cleaning_reviewed_rule_context_by_finding = {}
        st.session_state.cleaning_cross_source_review_by_finding = {}
        st.session_state.cleaning_result = None
        st.session_state.cleaning_working_paths = {}  # legacy field; no output files created
        st.session_state.cleaning_staged_bytes_by_dataset = {}
        st.session_state.cleaning_approved_columns_by_dataset = {}
        st.session_state.cleaning_approved_rows_by_dataset = {}
        st.session_state.cleaning_explicit_row_deletions_by_finding = {}
        st.session_state.cleaning_explicit_column_deletions_by_finding = {}
        st.session_state.cleaning_history = []
        st.session_state.cleaning_pending_visual_flags_by_finding = {}
        st.session_state.cleaning_visual_flags_by_dataset = {}
        # A user-initiated NEW investigation is a new review, even when the
        # source CSV is byte-identical to a previous run. Keep the saved
        # checkpoint on disk, but never automatically offer/import it here.
        st.session_state.cleaning_recovery_dismissed = bool(
            run_id and st.session_state.get("cleaning_fresh_review_run_id") == run_id
        )
        st.session_state.cleaning_approved_reviewer = ""
        st.session_state.cleaning_rebased_previews_by_finding = {}
        st.session_state.cleaning_completed_audits_by_finding = {}
        st.session_state.cleaning_keep_open_by_finding = {}
        st.session_state.cleaning_active_finding = None
        st.session_state.cleaning_control_generation_by_finding = {}
        st.session_state.cleaning_autoready_seen = set()
        st.session_state.cleaning_parent_ready_ids = set()
        st.session_state.cleaning_finding_type_lookup = {}
    else:
        # Compatible with an already-open Streamlit session from the earlier
        # single-select cleaning UI. Retain its working dataset and change log,
        # while introducing the new per-card review state if absent.
        for key, empty in (
            ("cleaning_messages_by_finding", {}),
            ("cleaning_pending_gemma_b", {}),
            ("cleaning_completed_gemma", {}),
            ("cleaning_plans_by_finding", {}),
            ("cleaning_batch_queue_by_finding", {}),
            ("cleaning_batch_total_by_finding", {}),
            ("cleaning_compound_previews_by_finding", {}),
            ("cleaning_batch_rejected_by_finding", set()),
            ("cleaning_r_previews_by_finding", {}),
            ("cleaning_r_errors_by_finding", {}),
            ("cleaning_r_diagnostics_by_finding", {}),
            ("cleaning_r_selfchecks_by_finding", {}),
            ("cleaning_plan_retry_once", set()),
            ("cleaning_recommendations", {}),
            ("cleaning_review_statuses", {}),
            ("cleaning_discussion_mode_by_finding", {}),
            ("cleaning_custom_value_input_by_finding", {}),
            ("cleaning_custom_operation_input_by_finding", {}),
            ("cleaning_decisions", []),
            ("cleaning_memory_candidates", {}),
            ("cleaning_memory_skipped", set()),
            ("cleaning_reviewed_rule_context_by_finding", {}),
            ("cleaning_cross_source_review_by_finding", {}),
            ("cleaning_working_paths", {}),
            ("cleaning_staged_bytes_by_dataset", {}),
            ("cleaning_approved_columns_by_dataset", {}),
            ("cleaning_approved_rows_by_dataset", {}),
            ("cleaning_explicit_row_deletions_by_finding", {}),
            ("cleaning_explicit_column_deletions_by_finding", {}),
            ("cleaning_history", []),
            ("cleaning_pending_visual_flags_by_finding", {}),
            ("cleaning_visual_flags_by_dataset", {}),
            ("cleaning_recovery_dismissed", False),
            ("cleaning_approved_reviewer", ""),
            ("cleaning_rebased_previews_by_finding", {}),
            ("cleaning_completed_audits_by_finding", {}),
            ("cleaning_keep_open_by_finding", {}),
            ("cleaning_active_finding", None),
            ("cleaning_control_generation_by_finding", {}),
            ("cleaning_autoready_seen", set()),
            ("cleaning_parent_ready_ids", set()),
            ("cleaning_finding_type_lookup", {}),
        ):
            if key not in st.session_state:
                st.session_state[key] = empty

    # Migrate older sessions that already contain saved working CSVs. Read the
    # latest version into memory; NEVER create or delete any CSV here.
    legacy_paths = st.session_state.get("cleaning_working_paths") or {}
    if legacy_paths:
        for ds_id, old_path in legacy_paths.items():
            if ds_id not in st.session_state.cleaning_staged_bytes_by_dataset:
                old_file = Path(old_path)
                if not old_file.is_file():
                    st.error("An earlier saved working CSV is missing. Reload its original "
                             "investigation before exporting a final dataset.")
                    return
                st.session_state.cleaning_staged_bytes_by_dataset[ds_id] = old_file.read_bytes()
        st.session_state.cleaning_working_paths = {}
        st.session_state.cleaning_result = None
        st.info("Earlier working CSV versions have been loaded into session memory. "
                "Existing files were not deleted; this version will not create any more.")

    catalog = build_cleaning_findings(
        audit, dataset_paths, [file.name for file in uploaded],
        gemma_runs if gemma_runs is not None else (st.session_state.get("exploration_runs") or []),
    )
    if not catalog:
        st.info("No findings or checks to review.")

    # A cached investigation and a remembered decision are not a staged edit.
    # Save only approved DELTAS and explicit no-change reviews, never the whole
    # source CSV. Restoration requires one affirmative click and revalidation.
    from healthcare_data_quality.cleaning_stage_checkpoint import (
        save_approved_stage, load_approved_stage,
        verify_and_reconstruct, checkpoint_counts, LegacyCheckpointError,
        inspect_legacy_stage, archive_legacy_stage,
    )
    source_paths = [Path(path) for path in dataset_paths]
    finding_titles = {finding["id"]: str(finding.get("title") or "")
                      for finding in catalog}
    session_completed = (bool(st.session_state.cleaning_history) or
                         any(value == "Reviewed — no change"
                             for value in st.session_state.cleaning_review_statuses.values()))
    if session_completed:
        try:
            save_approved_stage(
                source_paths,
                st.session_state.cleaning_staged_bytes_by_dataset,
                st.session_state.cleaning_approved_columns_by_dataset,
                st.session_state.cleaning_approved_rows_by_dataset,
                st.session_state.cleaning_history,
                st.session_state.cleaning_review_statuses,
                st.session_state.cleaning_approved_reviewer,
                finding_titles,
                completed_audits=st.session_state.cleaning_completed_audits_by_finding,
            )
        except (OSError, ValueError, PermissionError, AssertionError, csv.Error) as exc:
            st.warning("Current approvals are staged, but recovery could not be saved. "
                       "Export before ending this session.")
            with st.expander("Recovery details"):
                st.code(f"{type(exc).__name__}: {exc}")
    elif not st.session_state.get("cleaning_recovery_dismissed"):
        try:
            previous_stage = load_approved_stage(source_paths)
        except LegacyCheckpointError:
            previous_stage = None
            # V1 checkpoint lacks value-level approval proof. Never silently
            # restore it or make old decisions appear as completed.
            st.info("A previous cleaning session was saved by an older version. "
                    "Its edits cannot be automatically restored as approved. "
                    "You can review them, preserve a backup, and start a new review.")
            st.caption("If Agent B has no recommendations for this investigation, "
                       "run a new full investigation to generate them.")
            try:
                legacy_preview = inspect_legacy_stage(source_paths, finding_titles)
            except (OSError, ValueError, PermissionError, AssertionError,
                    csv.Error, json.JSONDecodeError) as exc:
                legacy_preview = None
                st.warning("The older checkpoint cannot be safely previewed. "
                           "No edits have been restored.")
                with st.expander("Recovery details"):
                    st.code(f"{type(exc).__name__}: {exc}")
            if legacy_preview is not None:
                deltas = legacy_preview['cell_changes']
                structural = legacy_preview['structural_operations']
                row_actions = sum(len(v['rows']) for v in structural.values())
                column_actions = sum(len(v['columns']) for v in structural.values())
                st.caption(f"Earlier session: {len(deltas)} cell edits · "
                           f"{row_actions} row actions · {column_actions} column actions · "
                           f"{len(legacy_preview['saved_statuses'])} saved review statuses. "
                           "All are read-only until individually approved again.")
                with st.expander("Review saved changes (read-only)"):
                    if deltas:
                        st.dataframe(pd.DataFrame(deltas), use_container_width=True,
                                     hide_index=True)
                    else:
                        st.caption("The saved checkpoint has no changed cell values.")
                    if row_actions or column_actions:
                        st.markdown("**Earlier row/column removal requests (unrestored)**")
                        st.json(structural)
                    if legacy_preview['unmatched_findings']:
                        st.warning("Some saved finding IDs do not appear in this investigation. "
                                   "Re-investigate before reapplying those edits.")
                    # Streamlit offers a local export. Lightweight test UIs
                    # may not implement this optional download control.
                    if callable(getattr(st, "download_button", None)):
                        st.download_button(
                            "Download original checkpoint JSON",
                            data=legacy_preview['backup_json'],
                            file_name="legacy_cleaning_checkpoint_backup.json",
                            mime="application/json",
                            key="cleaning_download_legacy_checkpoint",
                        )
            else:
                st.caption("The previous file will be archived without importing any edits.")
            if st.button("Archive old checkpoint and start new review",
                         key="cleaning_archive_legacy_stage"):
                try:
                    archived = archive_legacy_stage(
                        source_paths,
                        expected_digest=(legacy_preview['checkpoint_digest']
                                         if legacy_preview else None),
                    )
                except (OSError, ValueError, PermissionError, json.JSONDecodeError) as exc:
                    st.error("Could not archive the previous checkpoint. "
                             "Nothing was changed; download a backup before retrying.")
                    with st.expander("Recovery details"):
                        st.code(f"{type(exc).__name__}: {exc}")
                else:
                    st.session_state.cleaning_recovery_dismissed = True
                    st.session_state.cleaning_legacy_archive_notice = str(archived)
                    _rerun_cleaning_fragment()
        except (OSError, ValueError, PermissionError, json.JSONDecodeError) as exc:
            previous_stage = None
            st.warning("A saved cleaning checkpoint cannot be verified. "
                       "Current data remains unchanged.")
            with st.expander("Recovery details"):
                st.code(f"{type(exc).__name__}: {exc}")
        if previous_stage:
            try:
                recovered = verify_and_reconstruct(previous_stage, source_paths, finding_titles)
            except (OSError, ValueError, PermissionError, AssertionError, csv.Error) as exc:
                st.warning("Previous cleaning edits do not match the current audit. "
                           "Reuse individual saved solutions instead.")
                with st.expander("Recovery details"):
                    st.code(f"{type(exc).__name__}: {exc}")
            else:
                action_count, no_change_count = checkpoint_counts(previous_stage)
                with st.container(border=True):
                    st.markdown("**Previous approved cleaning session available**")
                    st.caption(f"{action_count} approved actions · "
                               f"{no_change_count} no-change reviews · Original CSV verified")
                    restore_col, fresh_col = st.columns(2)
                    with restore_col:
                        restore = st.button("Restore verified stage",
                                            key="cleaning_restore_previous_stage")
                    with fresh_col:
                        fresh = st.button("Start new review",
                                          key="cleaning_skip_previous_stage")
                    if restore:
                        # A second verification catches changes after preview,
                        # before we import any staged bytes or review state.
                        try:
                            confirmed = verify_and_reconstruct(
                                previous_stage, source_paths, finding_titles)
                        except (OSError, ValueError, PermissionError, AssertionError,
                                csv.Error) as exc:
                            st.error("Could not safely restore this checkpoint; nothing was staged.")
                            with st.expander("Recovery details"):
                                st.code(f"{type(exc).__name__}: {exc}")
                        else:
                            for key, value in confirmed.items():
                                st.session_state[key] = value
                            restored_history = confirmed.get("cleaning_history", [])
                            restored_statuses = st.session_state.cleaning_review_statuses
                            for event in restored_history:
                                if event.get("finding_id") in finding_titles:
                                    restored_statuses.setdefault(
                                        event["finding_id"], "Applied and verified")
                            st.session_state.cleaning_recovery_dismissed = True
                            st.session_state.cleaning_restore_notice = (
                                len({event.get("finding_id") for event in restored_history
                                     if event.get("finding_id") in finding_titles}))
                            _rerun_cleaning_fragment()
                    if fresh:
                        st.session_state.cleaning_recovery_dismissed = True
                        _rerun_cleaning_fragment()

    archived_legacy = st.session_state.pop("cleaning_legacy_archive_notice", None)
    if archived_legacy:
        st.success("Previous recovery file was archived, not deleted. "
                   "Start a fresh review; earlier edits were not automatically staged.")
        with st.expander("Archived checkpoint location"):
            st.code(archived_legacy)
    restored_count = st.session_state.pop("cleaning_restore_notice", None)
    if restored_count is not None:
        st.success(f"Stage restored · {restored_count} findings with approved changes")

    # The reviewer identifier is requested inside the FIRST approval form,
    # next to the action it authorizes. Once verified staging succeeds, it is
    # stored for this investigation and never requested again.
    session_reviewer = st.session_state.get("cleaning_approved_reviewer") or ""
    if session_reviewer:
        st.caption(f"Reviewer for this cleaning session: {session_reviewer}")

    # Recall saved methods only as suggestions. Even exact-source matches
    # receive a fresh local Gemma applicability assessment before any plan is
    # offered for independent validation and human approval.
    # Agent B saves one suggestion per finding in the background. Agent C
    # handles interactive discussion, plans and authorization review.
    # Individual Agent B suggestion panels poll automatically.
    from cleaning_jobs import recommendation_manager, job_fingerprint
    jobs = recommendation_manager()
    # Agent B runs once per finding during investigation, not by this tab.
    # Agent C consumes its immutable advice and never reruns Agent B.
    state = None
    job_id = None
    try:
        if run_id:
            job_id = job_fingerprint(context_key, catalog, run_id=run_id)
        else:
            # Backward compatibility: audits from before the once-only update
            # may have completed the former Agent A recommendations in the old queue.
            # Read them if available, but NEVER start an old queue or refresh it.
            job_id = job_fingerprint(context_key, catalog)
        state = jobs.snapshot(job_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not read saved Gemma recommendations: {exc}")
    if state:
        entries = state["entries"]
        ready_count = sum(v["status"] == "ready" for v in entries.values())
        running_count = sum(v["status"] == "running" for v in entries.values())
        failed_count = sum(v["status"] == "failed" for v in entries.values())
        total_count = len(entries)
        if failed_count:
            with st.expander(f"{failed_count} recommendation(s) need retry", expanded=False):
                for result_id, entry in entries.items():
                    if entry["status"] == "failed":
                        st.write(result_id, entry.get("error", "No details returned"))
        for result_id, entry in entries.items():
            if entry["status"] != "ready":
                continue
            if entry.get("origin") == "review_memory":
                # The parent finding provides the only interaction for this
                # memory-backed result; don't display competing Agent B advice.
                continue
            existing = st.session_state.cleaning_recommendations.get(result_id) or {}
            review_status = st.session_state.cleaning_review_statuses.get(result_id)
            if (existing.get("_source") == "manual" or
                    review_status not in {None, "Awaiting review", "Agent B suggestion ready"}):
                continue
            st.session_state.cleaning_recommendations[result_id] = {
                "suggestion": entry["suggestion"],
                "decision_needed": entry.get("decision_needed", ""),
                "memory_assessed": entry.get("memory_assessed", False),
                "choices": entry.get("choices") or [],
                "_source": "background",
            }
            st.session_state.cleaning_review_statuses[result_id] = "Agent B suggestion ready"
    else:
        if run_id and catalog:
            st.warning("No Agent B recommendation job was recorded for this investigation. "
                       "This can happen when the earlier investigation callback failed. "
                       "You can start the missing recommendations without rerunning Agent A.")
            if st.button("Start missing Agent B recommendations",
                         key=f"cleaning_start_missing_agent_b_{session_context}"):
                try:
                    started = schedule_agent_b_once(
                        context_key, audit, source_paths,
                        [item.name for item in uploaded], gemma_runs or [],
                    )
                except (OSError, ValueError, TypeError) as exc:
                    st.error(f"Could not start Agent B recommendations: {exc}")
                else:
                    if started:
                        _rerun_cleaning_fragment()
                    else:
                        st.error("No recommendation job was started. Check the current finding catalog.")
        else:
            st.caption("No Agent B recommendations were saved with this older investigation. "
                       "Run a full investigation to generate them, or discuss a finding "
                       "directly with Agent C below.")

    # Agent B's new queue assesses relevant saved methods with Gemma before
    # marking any suggestion ready. Never promote a historical approval to an
    # actionable suggestion or count it as completed inference here.
    # Intermediate results remain in session memory; no CSV is offered or
    # written while findings are still being reviewed.
    history_all = st.session_state.cleaning_history

    ready_ids = {item_id for item_id, entry in ((state or {}).get("entries") or {}).items()
                 if entry.get("status") == "ready"}

    # Snapshot what the parent expander already knew when it was mounted.
    # Child polling reruns the parent only for genuinely new readiness.
    st.session_state.cleaning_parent_ready_ids = set(ready_ids)
    numbered = indexed_review_catalog(catalog)
    last_group = None
    for label, finding in numbered:
        fid = finding["id"]
        st.session_state.setdefault("cleaning_finding_type_lookup", {})[fid] = {
            "title": finding.get("title", ""),
            "check_key": finding.get("check_key", ""),
        }
        ds_id = finding.get("dataset_id", "")
        index = int(ds_id.rsplit("_", 1)[-1]) - 1 if ds_id.startswith("dataset_") else -1
        # The original upload stays the immutable evidence anchor. Final
        # insertion is reconciled against the latest in-memory dataset; an
        # identical earlier edit is reused, while divergent edits require
        # a fresh isolated R verification and explicit reviewer override.
        source = (Path(dataset_paths[index]) if 0 <= index < len(dataset_paths) else None)
        group = ("Gemma investigations" if finding["source"] != "QC" else
                 f"Dataset {index + 1}: {finding['name']}")
        if group != last_group:
            st.markdown(f"#### {group}")
            last_group = group
        # Bind single-dataset Gemma checks before terminal-state reconciliation.
        # Otherwise an already-approved control deletion in another check can
        # leave a stale, blocked duplicate plan visible here.
        if finding.get("check_key") == "gemma_check" and source is None and len(dataset_paths) == 1:
            source = Path(dataset_paths[0])
            ds_id = "dataset_1"
            finding = {**finding, "dataset_id": ds_id, "name": uploaded[0].name}
        if source is not None and _reconcile_approved_column_review(fid, ds_id, source):
            # Persist the new terminal review status immediately rather than
            # leaving it only in UI state until an unrelated click occurs.
            _rerun_cleaning_fragment(fid)
        status = st.session_state.cleaning_review_statuses.get(fid, "Awaiting review")
        applied = [event for event in history_all if event["finding_id"] == fid]
        if applied:
            if st.session_state.cleaning_review_statuses.get(fid) == "Resolved by earlier approved edit":
                status = "Resolved by earlier approved edit — no additional cells changed"
            else:
                status = f"Staged in memory ({len(applied)} approved edit{'s' if len(applied) != 1 else ''})"
        # Unfinished discussions stay open. After final named approval, close
        # the same card and keep its stable, green Completed heading visible.
        remaining_batch = st.session_state.cleaning_batch_queue_by_finding.get(fid, [])
        current_plan = st.session_state.cleaning_plans_by_finding.get(fid)
        raw_status = st.session_state.cleaning_review_statuses.get(fid)
        # Earlier named approval may already have fixed this finding's exact
        # format defect (e.g. standardizing taxonomy also fixes its '|'
        # delimiter). Do not ask to overwrite the better staged value with
        # an old, less-complete proposal or demand another approval.
        if (source and not applied and finding.get("check_key") in {
                "inspect_taxonomy_structure", "inspect_taxonomy_naming"}
                and any(event.get("dataset_id") == ds_id for event in history_all)):
            from healthcare_data_quality.direct_actions import already_resolved_format_issue
            staged_now = st.session_state.cleaning_staged_bytes_by_dataset.get(ds_id)
            if staged_now and already_resolved_format_issue(finding, source, staged_now):
                for field in (
                    "cleaning_plans_by_finding", "cleaning_batch_queue_by_finding",
                    "cleaning_batch_total_by_finding", "cleaning_r_previews_by_finding",
                    "cleaning_compound_previews_by_finding",
                ):
                    st.session_state.setdefault(field, {}).pop(fid, None)
                st.session_state.cleaning_review_statuses[fid] = "Reviewed — no change"
                st.session_state.cleaning_discussion_mode_by_finding.pop(fid, None)
                current_plan, remaining_batch = None, []
                raw_status = "Reviewed — no change"
        reviewed_no_change = (raw_status == "Reviewed — no change" and not applied)
        # Terminal approval state, not a leftover chat-mode flag, decides when
        # an approval has finished. Earlier versions left mode="alternative"
        # after single-operation approval, resurfacing the Send form despite
        # the user's completed Yes decision (especially findings 2 and 3).
        completed_statuses = {
            "Applied and verified", "Resolved by earlier approved edit",
            "Approved column deletion for final export",
            "Approved row changes for final export", "Reviewed — no change",
        }
        completed = (bool(applied) or reviewed_no_change or
                     raw_status == "Resolved by earlier approved edit") and (
            not current_plan and not remaining_batch and raw_status in completed_statuses)
        # A review decision itself is an active interaction, even before the
        # researcher sends any text. In particular, clicking "No — discuss
        # another action" sets mode="alternative" with an EMPTY transcript.
        # If we only check for messages/jobs/plans, the subsequent Streamlit
        # rerun remounts this expander collapsed and hides the chat form.
        discussion_mode = st.session_state.cleaning_discussion_mode_by_finding.get(fid)
        active_conversation = (not completed and bool(
            discussion_mode in {"alternative", "accepted"}
            or st.session_state.cleaning_messages_by_finding.get(fid)
            or fid in st.session_state.cleaning_pending_gemma_b
            or current_plan or remaining_batch
        ))
        # Never change the expander's identity when approval changes status.
        # The expander title must remain stable through approvals; the green
        # Completed suffix is rendered by a marker-based CSS rule.
        keep_open = st.session_state.setdefault("cleaning_keep_open_by_finding", {}).get(fid, False)
        # A finished review collapses even if an earlier interaction left a
        # stale keep-open flag. Unfinished groups retain their active position.
        if completed:
            _scroll_to_completed_finding_once(fid)
        with st.expander(f"{label} · {finding['title']}",
                         expanded=bool(not completed and (
                             active_conversation or keep_open or fid in ready_ids))):
            if completed and fid not in st.session_state.cleaning_batch_rejected_by_finding:
                # Do not mark a partially rejected batch as fully completed.
                # No extra caption: tint the whole heading green and append
                # " · Completed" to its visible title using the style above.
                marker_id = hashlib.sha256(fid.encode("utf-8")).hexdigest()[:20]
                st.markdown(
                    f'<span class="hdq-cleaning-completed-marker" '
                    f'data-review-focus="{marker_id}" aria-hidden="true">&#8203;</span>',
                    unsafe_allow_html=True)
            if finding.get("check_key") == "gemma_check" and source is None and len(dataset_paths) == 1:
                source = Path(dataset_paths[0])
                ds_id = "dataset_1"
                finding = {**finding, "dataset_id": ds_id, "name": uploaded[0].name}
            if finding.get("check_key") == "gemma_check" and source is None and dataset_paths:
                # A completed Gemma test can inspect several uploaded CSVs.
                # It has no reliable dataset citation when it finds no issue.
                # Let the researcher select the real source, never guess one.
                dataset_choice = st.selectbox(
                    "Source dataset to inspect",
                    options=list(range(len(dataset_paths))), index=None,
                    format_func=lambda i: f"Dataset {i+1}: {uploaded[i].name}",
                    key=f"review_gemma_source_{fid}",
                    placeholder="Choose a dataset to inspect its full CSV evidence",
                )
                if dataset_choice is not None:
                    index = int(dataset_choice)
                    ds_id = f"dataset_{index+1}"
                    source = Path(dataset_paths[index])
                    finding = {**finding, "dataset_id": ds_id,
                               "name": uploaded[index].name}
                    if _reconcile_approved_column_review(fid, ds_id, source):
                        _rerun_cleaning_fragment(fid)
            if completed:
                if st.checkbox("Show completed finding audit",
                               key=f"review_completed_audit_{fid}"):
                    completed_audit = st.session_state.cleaning_completed_audits_by_finding.get(fid) or {}
                    changes = completed_audit.get("changes") or []
                    from healthcare_data_quality.reviewed_rule_replay import current_change_rows
                    operation_types = {event.get("operation") for event in applied}
                    reviewed_plans = []
                    for change in changes:
                        if not isinstance(change, dict):
                            continue
                        if "row" in change and "column" in change:
                            op = "replace_cells"
                        elif "source_row" in change:
                            op = "merge_rows"
                        elif "row" in change:
                            op = "drop_rows"
                        elif "position" in change:
                            op = ("rename_headers" if "rename_headers" in operation_types
                                  and "drop_column" not in operation_types else "drop_column")
                        else:
                            continue
                        reviewed_plans.append({"operation": op, "changes": [change]})
                    # Older approvals and export-only row/column operations may
                    # lack a completed_audit. Recover exact BEFORE values from
                    # the immutable CSV, never infer a value from Gemma prose.
                    if not reviewed_plans and source:
                        from healthcare_data_quality.in_memory_cleaning import _records
                        original_records = _records(source.read_bytes())
                        for event in applied:
                            op = event.get("operation")
                            if op in {"replace_cells", "rename_headers"}:
                                for cell in event.get("approved_cell_changes") or []:
                                    reviewed_plans.append({"operation": "replace_cells", "changes": [
                                        {"row": cell["row"], "column": cell["column"],
                                         "before": cell["before"], "after": cell["after"]}]})
                            elif op == "drop_column":
                                for pos in event.get("modified_positions") or []:
                                    col = pos.get("column")
                                    if type(col) is int and 1 <= col <= len(original_records[0]):
                                        reviewed_plans.append({"operation": op, "changes": [
                                            {"position": col, "before": original_records[0][col-1],
                                             "after": ""}]})
                            elif op in {"drop_rows", "merge_rows"}:
                                for item in event.get("modified_positions") or []:
                                    row = item.get("row")
                                    if type(row) is int and 2 <= row <= len(original_records):
                                        change = {"row": row, "before": original_records[row-1][0],
                                                  "after": ""}
                                        if op == "merge_rows":
                                            other = item.get("source_row")
                                            if type(other) is not int:
                                                continue
                                            change["source_row"] = other
                                            change["after"] = "sum_numeric"
                                        reviewed_plans.append({"operation": op, "changes": [change]})
                    try:
                        rows_for_display = current_change_rows(reviewed_plans)
                    except (ValueError, KeyError, TypeError):
                        rows_for_display = []
                    visual_flag_events = [item for item in applied
                                          if item.get("operation") == "flag_cells"]
                    if visual_flag_events:
                        from healthcare_data_quality.visual_flags import flag_preview_html
                        approved_visual = [item for event in visual_flag_events
                                           for item in event.get("flags", [])]
                        if approved_visual:
                            st.markdown(flag_preview_html(approved_visual),
                                        unsafe_allow_html=True)
                    if rows_for_display:
                        st.dataframe(pd.DataFrame([
                            {"Location": row["Target"], "Before": row["Before"],
                             "After": row["After"]} for row in rows_for_display
                        ]), hide_index=True, width="stretch")
                    elif reviewed_no_change:
                        st.caption("Reviewed without changes")
                    elif raw_status == "Resolved by earlier approved edit":
                        st.caption("Already completed in an earlier approved review")
                    else:
                        st.caption("Approved changes recorded")
                    if st.button("Revise completed action", key=f"review_reopen_{fid}"):
                        if source is None:
                            st.error("Select the original CSV before revising this review.")
                        else:
                            try:
                                from healthcare_data_quality.interaction_memory import revoke_reviewed_solution
                                visual_only = bool(applied) and all(
                                    event.get("operation") == "flag_cells" for event in applied)
                                if visual_only:
                                    snapshot = {"staged": st.session_state.cleaning_staged_bytes_by_dataset.get(
                                                    ds_id, source.read_bytes()),
                                                "columns": st.session_state.cleaning_approved_columns_by_dataset.get(ds_id, []),
                                                "rows": st.session_state.cleaning_approved_rows_by_dataset.get(ds_id, []),
                                                "history": [event for event in st.session_state.cleaning_history
                                                            if event.get("finding_id") != fid]}
                                elif applied:
                                    from healthcare_data_quality.approved_revision import reverse_approved_finding
                                    snapshot = reverse_approved_finding(
                                        source,
                                        st.session_state.cleaning_staged_bytes_by_dataset.get(
                                            ds_id, source.read_bytes()),
                                        finding_id=fid, dataset_id=ds_id,
                                        history=st.session_state.cleaning_history,
                                        columns=st.session_state.cleaning_approved_columns_by_dataset.get(ds_id, []),
                                        rows=st.session_state.cleaning_approved_rows_by_dataset.get(ds_id, []),
                                    )
                                # If this was the last approval, also retire
                                # the old recovery checkpoint; a future session
                                # must not offer to restore an undone action.
                                remaining_history = (snapshot["history"] if applied
                                                     else st.session_state.cleaning_history)
                                remaining_reviews = any(
                                    other_id != fid and review_status == "Reviewed — no change"
                                    for other_id, review_status in
                                    st.session_state.cleaning_review_statuses.items())
                                if not remaining_history and not remaining_reviews:
                                    from healthcare_data_quality.cleaning_stage_checkpoint import invalidate_approved_stage
                                    invalidate_approved_stage(source_paths)
                                revoke_reviewed_solution(finding, source)
                                if visual_only:
                                    flags_by_ds = st.session_state.cleaning_visual_flags_by_dataset.get(ds_id, [])
                                    st.session_state.cleaning_visual_flags_by_dataset[ds_id] = [
                                        flag for flag in flags_by_ds if flag.get("finding_id") != fid]
                                st.session_state.cleaning_pending_visual_flags_by_finding.pop(fid, None)
                                if applied:
                                    st.session_state.cleaning_staged_bytes_by_dataset[ds_id] = snapshot["staged"]
                                    st.session_state.cleaning_approved_columns_by_dataset[ds_id] = snapshot["columns"]
                                    st.session_state.cleaning_approved_rows_by_dataset[ds_id] = snapshot["rows"]
                                    st.session_state.cleaning_history = snapshot["history"]
                                st.session_state.cleaning_review_statuses[fid] = "Discussion open"
                                st.session_state.cleaning_recovery_dismissed = True
                                st.session_state.cleaning_discussion_mode_by_finding[fid] = "alternative"
                                st.session_state.cleaning_memory_skipped.add(fid)
                                st.session_state.cleaning_recommendations.pop(fid, None)
                                st.session_state.cleaning_completed_audits_by_finding.pop(fid, None)
                                st.session_state.cleaning_batch_rejected_by_finding.discard(fid)
                                st.session_state.cleaning_plans_by_finding.pop(fid, None)
                                st.session_state.cleaning_batch_queue_by_finding.pop(fid, None)
                                st.session_state.cleaning_pending_gemma_b.pop(fid, None)
                                _next_agent_c_ui_generation(fid)
                                st.session_state.cleaning_decisions.append({
                                    "finding_id": fid, "decision": "approved_review_reopened_for_revision",
                                    "previous_actions_reversed": len(applied),
                                })
                                _rerun_cleaning_fragment(fid)
                            except (OSError, ValueError, PermissionError, AssertionError,
                                    KeyError, csv.Error) as exc:
                                st.error(f"Cannot safely revise this approval: {exc}")
                # Active approval is terminal until researcher explicitly reopens
                # it; never show old chat input or send controls below this card.
                continue
            evidence = finding.get("evidence") or []
            if evidence and st.checkbox("Show original evidence",
                                        value=True, key=f"review_evidence_{fid}"):
                from healthcare_data_quality.visual_flags import styled_evidence
                current_flags = st.session_state.cleaning_visual_flags_by_dataset.get(ds_id, [])
                st.dataframe(styled_evidence(evidence, current_flags),
                             hide_index=True, width="stretch")

            # Visual annotations are a separate kind of reviewer operation:
            # never replace taxonomy text with markup or force a value-change
            # clarification. This form is the one named approval for all flags.
            flag_plan = st.session_state.cleaning_pending_visual_flags_by_finding.get(fid)
            if flag_plan and source:
                from healthcare_data_quality.visual_flags import flag_preview_html, requested_visual_flag
                st.markdown('**Flag taxonomy — bold red (display only)**')
                st.markdown(flag_preview_html(flag_plan["flags"]), unsafe_allow_html=True)
                with st.form(f"review_visual_flag_{fid}_{_active_agent_c_control_generation(fid)}"):
                    flag_reviewer = st.session_state.get("cleaning_approved_reviewer") or ""
                    if flag_reviewer:
                        st.caption("Reviewer: " + flag_reviewer)
                    else:
                        flag_reviewer = st.text_input(
                            "Reviewer identifier", key=f"visual_flag_reviewer_{fid}",
                            max_chars=120)
                    yes_col, no_col = st.columns(2)
                    with yes_col:
                        accept_flag = st.form_submit_button(
                            "Yes — Approve & Execute",
                            key=f"flag_yes_{fid}_{_active_agent_c_control_generation(fid)}")
                    with no_col:
                        decline_flag = st.form_submit_button(
                            "No — Discuss Another Idea",
                            key=f"flag_no_{fid}_{_active_agent_c_control_generation(fid)}")
                if decline_flag:
                    st.session_state.cleaning_pending_visual_flags_by_finding.pop(fid, None)
                    _open_agent_c_alternative_chat(fid, discard_plan=True)
                    _rerun_cleaning_fragment(fid)
                if accept_flag:
                    reviewer = str(flag_reviewer or "").strip()
                    if not reviewer:
                        st.error("Enter your reviewer identifier to approve this flag.")
                    else:
                        try:
                            # Verify the *current source bytes and coordinates*
                            # immediately before approval. Flags carry no CSV edit.
                            from healthcare_data_quality.visual_flags import requested_visual_flag
                            original_request = flag_plan["request"]
                            checked = requested_visual_flag(original_request, finding, source)
                            if (checked is None or checked["source_sha256"] != flag_plan["source_sha256"]
                                    or checked["flags"] != flag_plan["flags"]):
                                raise PermissionError("Original source or flagged targets changed")
                            existing = st.session_state.cleaning_visual_flags_by_dataset.setdefault(ds_id, [])
                            keys = {(item["row"], item["column"]) for item in checked["flags"]}
                            existing[:] = [item for item in existing
                                           if (item["row"], item["column"]) not in keys]
                            existing.extend({**item, "finding_id": fid, "reviewer": reviewer}
                                            for item in checked["flags"])
                            st.session_state.cleaning_approved_reviewer = reviewer
                            st.session_state.cleaning_history.append({
                                "dataset_id": ds_id, "finding_id": fid, "finding": finding["title"],
                                "operation": "flag_cells", "flags": checked["flags"],
                                "changes": len(checked["flags"]), "reviewer": reviewer,
                                "source_sha256": checked["source_sha256"],
                                "staged_sha256": hashlib.sha256(
                                    st.session_state.cleaning_staged_bytes_by_dataset.get(
                                        ds_id, source.read_bytes())).hexdigest(),
                                "modified_positions": [],  # style-only, no CSV change
                                "storage": "session_visual_annotations_no_csv_change",
                            })
                            st.session_state.cleaning_review_statuses[fid] = "Applied and verified"
                            st.session_state.cleaning_pending_visual_flags_by_finding.pop(fid, None)
                            st.session_state.cleaning_messages_by_finding.get(fid, []).append({
                                "role": "assistant", "content": "Flag applied. CSV values unchanged."})
                            _rerun_cleaning_fragment(fid)
                        except (ValueError, PermissionError, OSError, csv.Error, KeyError) as exc:
                            st.error(f"Flag could not be approved: {exc}")
                continue
            # Even a zero-issue Gemma negative-control check can provide an
            # optional actionable deletion choice. The original column headers
            # and user-selected positions are the only draft source; one named
            # approval follows exact Before -> After review. No prepare step.
            if source and fid in ready_ids and _is_blank_control_check(finding):
                try:
                    whole_source = source_inventory(source)
                except (OSError, ValueError, UnicodeError, csv.Error) as inv_exc:
                    st.error(f"Source unavailable: {inv_exc}")
                else:
                    blank_columns = whole_source["blank_control_columns"]
                    if blank_columns:
                        columns = whole_source["columns"]
                        # Ignore columns already approved for removal by ANY
                        # finding/check. A repeated selector must not generate
                        # duplicate plans and a blocked final approval.
                        approved_positions = {
                            int(entry["position"])
                            for entry in st.session_state.cleaning_approved_columns_by_dataset.get(ds_id, [])
                            if isinstance(entry, dict) and type(entry.get("position")) is int
                        }
                        options = [c["position"] for c in blank_columns
                                   if c["position"] not in approved_positions]
                        if not options:
                            st.session_state.cleaning_review_statuses[fid] = (
                                "Reviewed — no change")
                            st.session_state.cleaning_plans_by_finding.pop(fid, None)
                            st.session_state.cleaning_batch_queue_by_finding.pop(fid, None)
                            st.session_state.cleaning_explicit_column_deletions_by_finding.pop(fid, None)
                            _rerun_cleaning_fragment(fid)
                        lookup = {c["position"]: c for c in columns}
                        selected = st.multiselect(
                            "Remove control columns (optional)", options=options,
                            default=options[:3],
                            format_func=lambda pos: f"Column {pos} · {lookup[pos]['header'] or '(unnamed)'}",
                            key=f"review_check_columns_{fid}",
                        )
                        current = st.session_state.cleaning_plans_by_finding.get(fid)
                        current_group = ([current, *st.session_state.cleaning_batch_queue_by_finding.get(fid, [])]
                                         if current else [])
                        current_positions = sorted(
                            c["position"] for plan_item in current_group
                            if plan_item.get("operation") == "drop_column"
                            for c in plan_item.get("changes", []))
                        if sorted(selected) != current_positions:
                            # A selection is never approval. It supplies the
                            # coordinates against which the final named Yes is
                            # separately validated before atomic export.
                            if selected and len(selected) < len(columns):
                                st.session_state.cleaning_explicit_column_deletions_by_finding[fid] = frozenset(selected)
                                proposals = [{"operation": "drop_column", "dataset_id": ds_id,
                                              "changes": [{"position": p, "before": lookup[p]["header"],
                                                           "after": ""}]}
                                             for p in sorted(selected)]
                                _record_gemma_b_response(fid, finding, source, {
                                    "reply": "Selected control columns are ready for review.",
                                    "operation": "none", "dataset_id": "", "changes": [],
                                    "operations": proposals, "clarification_options": [],
                                })
                            else:
                                st.session_state.cleaning_plans_by_finding.pop(fid, None)
                                st.session_state.cleaning_batch_queue_by_finding.pop(fid, None)
                                st.session_state.cleaning_explicit_column_deletions_by_finding.pop(fid, None)
            cells, headers = allowed_positions(finding, source) if source else (set(), set())
            if source is None:
                st.info("Select a dataset before proposing edits.")
            elif not cells and not headers:
                if finding.get("check_key") == "gemma_check":
                    pass
                elif finding.get("check_key") == "find_duplicate_taxa":
                    pass  # No redundant explanation; proposed edits show exact values.
                else:
                    st.caption("No verified editable targets.")

            # Agent B has already asked Gemma to assess relevant saved methods
            # against the current evidence. Do not enqueue a second implicit
            # Agent C assessment or hide the Yes button behind an auto-chat.

            # Job results are consumed by the UI, never by the worker. A
            # submitted user message is already visible while inference runs.
            _sync_gemma_conversations(finding, source)
            # Obvious, reversible formatting inconsistencies do not need a
            # separate "fix" / "ok" / "draft" conversation. Construct exact
            # original BEFORE and AFTER values in Python, validate scope and R,
            # then present the ordinary single named Yes/No review directly.
            if (source and fid in ready_ids and finding.get("check_key") in {
                    "inspect_taxonomy_structure", "inspect_taxonomy_naming",
                    "compare_repeated_headers"}
                    and not completed and not st.session_state.cleaning_plans_by_finding.get(fid)
                    and not st.session_state.cleaning_messages_by_finding.get(fid)
                    and fid not in st.session_state.cleaning_pending_gemma_b
                    and not st.session_state.cleaning_discussion_mode_by_finding.get(fid)):
                from healthcare_data_quality.direct_actions import (
                    draft_format_correction, draft_identical_column_removal)
                direct = (draft_identical_column_removal(finding, source)
                          if finding.get("check_key") == "compare_repeated_headers"
                          else draft_format_correction(finding, source))
                if direct is not None:
                    st.session_state.cleaning_memory_skipped.add(fid)
                    _record_gemma_b_response(fid, finding, source, direct)
            # The independently refreshed Agent B area may propose an action
            # ONLY while this finding is idle. Once Agent C is discussing,
            # preparing, or approving a newer request, clear the entire older
            # B fragment and its Yes/No widgets. This prevents a late timer
            # rerun from resurrecting obsolete buttons beside an approval.
            b_controls_allowed = (
                st.session_state.cleaning_plans_by_finding.get(fid) is None
                and not st.session_state.cleaning_batch_queue_by_finding.get(fid)
                and fid not in st.session_state.cleaning_pending_gemma_b
                and st.session_state.cleaning_discussion_mode_by_finding.get(fid)
                    not in {"accepted", "alternative"}
                and not any(event.get("finding_id") == fid
                            for event in st.session_state.cleaning_history)
                and st.session_state.cleaning_review_statuses.get(fid)
                    != "Reviewed — no change"
            )
            agent_b_slot = st.empty() if hasattr(st, "empty") else None
            if not b_controls_allowed:
                if agent_b_slot is not None:
                    agent_b_slot.empty()
            else:
                with (agent_b_slot.container() if agent_b_slot is not None
                      else nullcontext()):
                    if hasattr(st, "fragment"):
                        st.fragment(run_every="3s")(_render_live_agent_b_finding)(
                            fid, finding, source, uploaded, dataset_paths, job_id, ds_id,
                        )
                    else:
                        _render_live_agent_b_finding(
                            fid, finding, source, uploaded, dataset_paths, job_id, ds_id,
                        )
            b_pending = fid in st.session_state.cleaning_pending_gemma_b

            # Once Yes or No is clicked the decision controls disappear.
            # The accepted route is automatic unless an essential scientific
            # parameter is genuinely missing; no repeated authorization chat.
            transcript = st.session_state.cleaning_messages_by_finding.get(fid, [])
            mode = st.session_state.cleaning_discussion_mode_by_finding.get(fid)
            plan = st.session_state.cleaning_plans_by_finding.get(fid)
            cross_review = st.session_state.setdefault(
                "cleaning_cross_source_review_by_finding", {}).get(fid) or {}
            clarification_needed = (mode == "accepted" and not b_pending and not plan
                                    and bool(transcript) and transcript[-1]["role"] == "assistant")
            # Prefer structured Gemma choices; also render clearly stated
            # alternatives from older free-form replies without another call.
            last_agent_c_reply = (
                transcript[-1] if transcript and transcript[-1].get("role") == "assistant"
                else None
            )
            # The pending proposal is the current interaction state. Never
            # delete it merely because an OLD assistant message contained
            # clarification options; that message may predate a deterministic
            # user-requested deletion or a verified remembered rule.
            custom_value_input = st.session_state.cleaning_custom_value_input_by_finding.get(fid, False)
            custom_operation_input = st.session_state.cleaning_custom_operation_input_by_finding.get(fid, False)
            control_phase, choice_labels = _agent_c_control_phase(
                mode=mode, transcript=transcript, plan=plan,
                queued_actions=st.session_state.cleaning_batch_queue_by_finding.get(fid),
                pending=b_pending, custom_value=custom_value_input,
                custom_operation=custom_operation_input,
            )
            last_assistant_message = (str(transcript[-1].get("content", ""))
                                      if clarification_needed else "")
            if (clarification_needed and control_phase == "discussion"
                    and not choice_labels
                    and fid not in st.session_state.cleaning_plan_retry_once):
                # Retry once on an accepted action if Gemma answered in prose
                # rather than supplying the structured plan. Do not repeatedly
                # ask the researcher to approve drafting the same plan.
                st.session_state.cleaning_plan_retry_once.add(fid)
                _queue_agent_c_for_finding(
                    finding, source, uploaded, dataset_paths,
                    "My earlier Yes already authorized preparation of the scoped R draft. "
                    "Please return actual structured operation and changes immediately for "
                    "the exact evidenced cells and the previously suggested replacements. "
                    "Do not request duplicate authorization. This does NOT authorize "
                    "execution or insertion.",
                    automatic=True,
                )
                _rerun_cleaning_fragment(fid)
            if False and plan and transcript:
                if False:
                    for item in transcript:
                        if item.get("automatic"):
                            st.caption("Agent C retried the accepted plan automatically.")
                        else:
                            with st.chat_message(item["role"]):
                                if item.get("role") == "assistant" and item.get("assigned_agent") in {"A", "B"}:
                                    st.caption(f"Agent {item['assigned_agent']} assisted while C was busy")
                                st.write(item["content"])
            elif (mode == "alternative" or clarification_needed) and transcript:
                st.markdown("**Discussion**")
                for item in transcript:
                    if not item.get("automatic"):
                        with st.chat_message(item["role"]):
                            st.write(item["content"])
            if b_pending:
                _render_live_finding_assignment(fid)
            if fid not in ready_ids and mode != "alternative" and not plan:
                # The previous implementation required Agent B's ready flag to
                # unlock any Agent C form. A failed/missing B queue therefore
                # blocked all interaction, even though Agent C can independently
                # analyze the current source and request an explicit approval.
                if (source and not applied and not completed and not current_plan
                        and not remaining_batch and not b_pending
                        and st.session_state.cleaning_review_statuses.get(fid)
                            != "Reviewed — no change"):
                    if st.button("Discuss with Agent C now",
                                 key=f"review_start_direct_agent_c_{fid}"):
                        st.session_state.cleaning_discussion_mode_by_finding[fid] = "alternative"
                        st.session_state.cleaning_review_statuses[fid] = "Alternative requested"
                        _next_agent_c_ui_generation(fid)
                        _rerun_cleaning_fragment(fid)
            # Use one replaceable control slot on Streamlit. Test/dummy
            # renderers without st.empty still use the same state guards.
            active_choice_slot = st.empty() if hasattr(st, "empty") else None
            if control_phase == "approval":
                if active_choice_slot is not None:
                    active_choice_slot.empty()
            else:
                with (active_choice_slot.container() if active_choice_slot is not None
                      else nullcontext()):
                    if (control_phase == "clarification" and choice_labels
                            and (fid in ready_ids or mode == "alternative")):
                        st.markdown("**Choose an approach:**")
                        st.caption("Your selection prepares a verified proposal; changes require approval.")
                        # Two buttons per row remain readable with long column/row
                        # descriptions, including a third "other operation" choice.
                        choice_rows = [st.columns(2) for _ in range((len(choice_labels) + 1) // 2)]
                        for option_index, choice_label in enumerate(choice_labels, start=1):
                            choice_column = choice_rows[(option_index - 1) // 2][(option_index - 1) % 2]
                            positive_label = ("Yes — " + choice_label if option_index == 1
                                              else choice_label)
                            if choice_column.button(positive_label,
                                    key=f"review_clarify_yes_{fid}_{_active_agent_c_control_generation(fid)}_{option_index}",
                                    disabled=b_pending):
                                # Ignore clicks from a stale rendered question when a
                                # proposal has since become active. Never queue old work.
                                if (st.session_state.cleaning_plans_by_finding.get(fid) is not None
                                        or st.session_state.cleaning_batch_queue_by_finding.get(fid)
                                        or fid in st.session_state.cleaning_pending_gemma_b):
                                    _rerun_cleaning_fragment(fid)
                                    return
                                # A different user-supplied value requires the actual
                                # numbers, not a redundant Gemma clarification turn.
                                option_text = choice_label.strip().casefold()
                                # Gemma's two structured labels can be worded in many
                                # ways; the equivalent user choice must behave the same.
                                choosing_custom_values = (
                                    "value" in option_text
                                    and any(word in option_text for word in (
                                        "specific", "different", "custom", "my own", "your own"))
                                    and "absolute" not in option_text
                                )
                                if choosing_custom_values:
                                    st.session_state.cleaning_custom_value_input_by_finding[fid] = True
                                    st.session_state.cleaning_custom_operation_input_by_finding.pop(fid, None)
                                    _rerun_cleaning_fragment(fid)
                                # An open-ended third choice does not need another
                                # Gemma inference merely to ask the user to type it.
                                if ("specify another operation" in option_text
                                        or (any(w in option_text for w in (
                                            "another", "different", "other", "alternative", "something else"))
                                            and any(w in option_text for w in (
                                                "operation", "action", "request", "option", "something",
                                                "correction", "edit")))):
                                    st.session_state.cleaning_custom_operation_input_by_finding[fid] = True
                                    st.session_state.cleaning_custom_value_input_by_finding.pop(fid, None)
                                    _rerun_cleaning_fragment(fid)
                                st.session_state.cleaning_custom_value_input_by_finding.pop(fid, None)
                                st.session_state.cleaning_custom_operation_input_by_finding.pop(fid, None)
                                # An explicit absolute-value choice is a user instruction
                                # with precise semantics, not a final edit approval.
                                # The backend can build it from verified source evidence.
                                if ("absolute value" in option_text and
                                        "absolute value" in last_agent_c_reply.get("content", "").casefold()):
                                    answer = (
                                        "Use absolute values for every evidenced negative cell "
                                        "in this finding. Keep the original CSV coordinates "
                                        "and values. Draft a plan for source verification, "
                                        "R testing and separate final approval. Do not edit "
                                        "any data yet."
                                    )
                                else:
                                    # Each other structured choice still replies directly,
                                    # without approving or executing a data modification.
                                    answer = (
                                        f"I choose option {option_index}: {choice_label}. "
                                        "This clarifies the scope of your last question, "
                                        "not permission to change data. Retain my other "
                                        "requested actions and prepare validated plan(s) "
                                        "for separate review and approval."
                                    )
                                _queue_agent_c_for_finding(
                                    finding, source, uploaded, dataset_paths, answer)
                                _rerun_cleaning_fragment(fid)
                        if st.button("No — Discuss Another Idea",
                                     key=f"review_discuss_no_{fid}_{_active_agent_c_control_generation(fid)}_clarify"):
                            _open_agent_c_alternative_chat(fid)
                            _rerun_cleaning_fragment(fid)
                    elif control_phase == "custom_input":
                        st.info(("Enter the exact replacement value for each affected cell below. "
                                 if custom_value_input else "Describe the alternative operation below. ")
                                + "Press Enter to send.")
                        if st.button("No — Discuss Another Idea",
                                     key=f"review_discuss_no_{fid}_{_active_agent_c_control_generation(fid)}_custom"):
                            _open_agent_c_alternative_chat(fid)
                            _rerun_cleaning_fragment(fid)
                    elif control_phase == "discussion" and mode == "alternative":
                        # A new Agent C proposal in prose must have a direct
                        # selection path. It is NOT approval of unverified edits.
                        last_reply = (transcript[-1] if transcript else None)
                        if (isinstance(last_reply, dict)
                                and last_reply.get("role") == "assistant"
                                and str(last_reply.get("content") or "").strip()
                                and not last_reply.get("automatic")
                                and not last_reply.get("controls_expired")
                                and any(word in str(last_reply.get("content") or "").casefold()
                                        for word in ("recommend", "propos", "replace", "delet",
                                                     "remov", "merge", "standardiz", "correct",
                                                     "rename", "clean", "apply", "convert", "pad "))):
                            yes_choice, no_choice = st.columns(2)
                            with yes_choice:
                                if st.button(
                                    "Yes — Prepare Changes",
                                    key=f"review_yes_{fid}_{_active_agent_c_control_generation(fid)}_agent_c",
                                ):
                                    st.session_state.cleaning_discussion_mode_by_finding[fid] = "accepted"
                                    st.session_state.cleaning_review_statuses[fid] = "Preparing verified changes"
                                    _queue_agent_c_for_finding(
                                        finding, source, uploaded, dataset_paths,
                                        "Prepare a structured, executable change plan for your "
                                        "latest recommendation using this finding's verified "
                                        "source positions and exact before/after values. "
                                        "Do not merely promise to draft a plan; return "
                                        "the operations now. Do not execute any change; "
                                        "a single named human approval follows validation.",
                                    )
                                    _rerun_cleaning_fragment(fid)
                            with no_choice:
                                if st.button(
                                    "No — Discuss Another Idea",
                                    key=f"review_discuss_no_{fid}_{_active_agent_c_control_generation(fid)}_agent_c",
                                ):
                                    # Expire this recommendation without sending
                                    # a hidden message to Gemma; chat stays open.
                                    last_reply["controls_expired"] = True
                                    _open_agent_c_alternative_chat(fid)
                                    _rerun_cleaning_fragment(fid)
                    # One conversation form per finding, and none while a request is
                    # in flight. Show the form again when the reply has been collected.
                    # st.chat_input sends with Enter and inserts a newline with
                    # Shift+Enter. It is mounted only for an active discussion;
                    # a pending or approved plan must never display a stale
                    # secondary Send button beneath its final approval.
                    if (control_phase in {"idle", "clarification", "custom_input", "discussion"}
                            and not b_pending and plan is None and not remaining_batch
                            and mode == "alternative"
                            and st.session_state.cleaning_review_statuses.get(fid) != "Reviewed — no change"):
                        placeholder = ("Enter exact replacement values…" if custom_value_input
                                       else "Describe the new action or ask Agent C…")
                        chat_key = f"review_message_{fid}_{_active_agent_c_control_generation(fid)}"
                        message = (st.chat_input(placeholder, key=chat_key,
                                                 max_chars=4000, disabled=b_pending)
                                   if hasattr(st, "chat_input") else None)
                        if message and str(message).strip():
                            if control_phase == "idle":
                                # Direct user feedback is a valid discussion even
                                # without clicking No first; do not auto-approve.
                                st.session_state.cleaning_discussion_mode_by_finding[fid] = "alternative"
                                st.session_state.cleaning_review_statuses[fid] = "Discussion open"
                            _queue_agent_c_for_finding(finding, source, uploaded, dataset_paths,
                                                       str(message).strip())
                            st.session_state.cleaning_custom_value_input_by_finding.pop(fid, None)
                            st.session_state.cleaning_custom_operation_input_by_finding.pop(fid, None)
                            _rerun_cleaning_fragment(fid)

            plan = st.session_state.cleaning_plans_by_finding.get(fid)
            approval_slot = st.empty() if hasattr(st, "empty") else None
            if plan is None:
                if approval_slot is not None:
                    approval_slot.empty()
            else:
                with (approval_slot.container() if approval_slot is not None
                      else nullcontext()):
                    compound_active = bool(plan and st.session_state.cleaning_batch_queue_by_finding.get(fid))
                    if compound_active and source:
                        _render_compound_approval(fid, finding, source, ds_id, catalog)
                    if plan and source and not compound_active and plan.get("operation") == "drop_column":
                        # Gemma only proposes a structural edit; this UI gives the
                        # researcher one independent, human-controlled approval.
                        try:
                            checked = validate_scoped_plan(
                                plan, finding, source,
                                user_requested_columns=st.session_state.cleaning_explicit_column_deletions_by_finding.get(fid))
                            if checked != plan:
                                raise PermissionError("Column deletion plan changed since verification")
                            item = plan["changes"][0]
                            current_bytes = st.session_state.cleaning_staged_bytes_by_dataset.get(
                                ds_id, source.read_bytes())
                            current_rows = _records(current_bytes)
                            column_position = item["position"]
                            approvals = st.session_state.cleaning_approved_columns_by_dataset.get(ds_id, [])
                            if any(entry["position"] == column_position for entry in approvals):
                                raise PermissionError("This column has already been approved for deletion")
                            if len(approvals) + 1 >= len(current_rows[0]):
                                raise PermissionError("At least one column must remain")
                            can_delete = current_rows[0][column_position - 1] is not None
                        except (ValueError, PermissionError, OSError, IndexError, csv.Error) as exc:
                            st.error(f"Column deletion proposal cannot be approved: {exc}")
                            can_delete = False
                        else:
                            _render_before_after([checked])
                            if not _render_source_deletion_details(
                                    source, [checked], key=f"column_{fid}"):
                                can_delete = False
                        with st.form(f"review_drop_column_{fid}_{_active_agent_c_control_generation(fid)}"):
                            reviewer_for_approval = st.session_state.get("cleaning_approved_reviewer") or ""
                            if reviewer_for_approval:
                                st.caption("Reviewer: " + reviewer_for_approval)
                            else:
                                reviewer_for_approval = st.text_input(
                                    "Reviewer identifier", key=f"review_drop_reviewer_{fid}", max_chars=120)
                            yes_col, no_col = st.columns(2)
                            with yes_col:
                                approved = st.form_submit_button(
                                    "Yes — Approve & Execute", disabled=not can_delete,
                                    key=f"review_approve_yes_{fid}_{_active_agent_c_control_generation(fid)}_column")
                            with no_col:
                                decline_column = st.form_submit_button(
                                    "No — Discuss Another Idea",
                                    key=f"review_discuss_no_{fid}_{_active_agent_c_control_generation(fid)}_column")
                        if decline_column:
                            _open_agent_c_alternative_chat(fid, discard_plan=True)
                            _rerun_cleaning_fragment(fid)
                        if approved:
                            reviewer = str(reviewer_for_approval or "").strip()
                            if not reviewer:
                                st.error("Enter a reviewer identifier before approval.")
                            elif not can_delete:
                                st.error("Column deletion validation failed; nothing was staged.")
                            else:
                                try:
                                    # Re-evaluate the entire original/staged relationship
                                    # before committing a pending export-only removal.
                                    fresh = st.session_state.cleaning_staged_bytes_by_dataset.get(
                                        ds_id, source.read_bytes())
                                    checked_again = validate_scoped_plan(
                                        plan, finding, source,
                                        user_requested_columns=st.session_state.cleaning_explicit_column_deletions_by_finding.get(fid))
                                    if checked_again != plan:
                                        raise PermissionError("The original dataset or plan changed")
                                    original_rows, staged_rows = _records(source.read_bytes()), _records(fresh)
                                    if len(original_rows) != len(staged_rows) or any(
                                            len(a) != len(b) for a, b in zip(original_rows, staged_rows)):
                                        raise PermissionError("Previously staged data dimensions changed")
                                    approved_entry = {
                                        "position": column_position, "before": item["before"],
                                        "reviewer": reviewer,
                                        "source_sha256": checked_again["source_sha256"],
                                        "finding_id": fid,
                                    }
                                    updated = [*st.session_state.cleaning_approved_columns_by_dataset.get(ds_id, []),
                                               approved_entry]
                                    # Dry run of final export validates all existing approvals.
                                    apply_approved_column_deletions(source.read_bytes(), fresh, updated)
                                    st.session_state.cleaning_approved_columns_by_dataset[ds_id] = updated
                                    st.session_state.cleaning_approved_reviewer = reviewer
                                    st.session_state.cleaning_history.append({
                                        "dataset_id": ds_id, "finding_id": fid,
                                        "finding": finding["title"], "operation": "drop_column",
                                        "changes": 1, "already_resolved": 0,
                                        "reviewer": reviewer, "storage": "session_memory_export_only",
                                        "modified_positions": [{"column": column_position}],
                                        "source_sha256": checked_again["source_sha256"],
                                        "staged_sha256": sha256_bytes(fresh),
                                    })
                                    st.session_state.cleaning_review_statuses[fid] = "Approved column deletion for final export"
                                    st.session_state.cleaning_decisions.append({
                                        "finding_id": fid, "decision": "whole_column_deletion_approved_export_only",
                                        "column": column_position,
                                        "original_header": item["before"],
                                        "source_sha256": checked_again["source_sha256"],
                                    })
                                    st.session_state.cleaning_plans_by_finding.pop(fid, None)
                                    _advance_agent_c_batch(fid)
                                    _remember_approved_agent_c_solution(fid, finding, source)
                                    _rerun_cleaning_fragment(fid)
                                except (ValueError, PermissionError, OSError, AssertionError, csv.Error) as exc:
                                    st.error(f"Column deletion approval failed; no action staged: {exc}")
                    if plan and source and not compound_active and plan.get("operation") in {"drop_rows", "merge_rows"}:
                        row_op = plan["operation"]
                        is_merge = row_op == "merge_rows"
                        can_approve_rows = False
                        try:
                            checked = validate_scoped_plan(
                                plan, finding, source,
                                user_requested_rows=st.session_state.cleaning_explicit_row_deletions_by_finding.get(fid))
                            if checked != plan:
                                raise PermissionError("Row plan changed since source verification")
                            originals = source.read_bytes()
                            staged_bytes = st.session_state.cleaning_staged_bytes_by_dataset.get(
                                ds_id, originals)
                            existing = st.session_state.cleaning_approved_rows_by_dataset.get(ds_id, [])
                            reviewer_for_approval = st.session_state.get("cleaning_approved_reviewer") or ""
                            preview_entry = {
                                "operation": row_op, "changes": checked["changes"],
                                "reviewer": reviewer_for_approval or "preview_only",
                                "source_sha256": checked["source_sha256"],
                                "finding_id": fid,
                            }
                            # Validate all prior operations and the new one together,
                            # WITHOUT changing the stored approval list.
                            preview = apply_approved_row_operations(
                                originals, staged_bytes, [*existing, preview_entry])
                            before_count = len(_records(staged_bytes)) - 1
                            after_count = len(_records(preview)) - 1
                            if after_count >= before_count:
                                raise PermissionError("Row operation did not remove a source row")
                            can_approve_rows = True
                        except (ValueError, PermissionError, OSError, AssertionError,
                                IndexError, KeyError, csv.Error) as exc:
                            st.error(f"Row operation cannot be approved: {exc}")
                        else:
                            if is_merge:
                                _render_before_after([checked])
                                item = checked["changes"][0]
                                # Show the actual numeric impact of merging before approval.
                                from healthcare_data_quality.in_memory_cleaning import _sum_count_cells
                                staged_rows = _records(staged_bytes)
                                target, source_row = item["row"], item["source_row"]
                                changes_preview = []
                                for col in range(1, len(staged_rows[0])):
                                    left = staged_rows[target-1][col]
                                    right = staged_rows[source_row-1][col]
                                    combined = _sum_count_cells(left, right)
                                    if combined != left:
                                        changes_preview.append({
                                            "Column": col + 1,
                                            "Target before": left,
                                            "Absorbed value": right,
                                            "Merged value": combined,
                                        })
                                if changes_preview:
                                    with st.expander(f"Preview merged counts ({len(changes_preview)} changed columns)"):
                                        st.dataframe(pd.DataFrame(changes_preview), hide_index=True,
                                                     width="stretch")
                            else:
                                row_ids = [item["row"] for item in checked["changes"]]
                                displayed = ", ".join(map(str, row_ids[:16]))
                                if len(row_ids) > 16:
                                    displayed += f", … (+{len(row_ids) - 16} more)"
                                _render_before_after([checked])
                                if len(row_ids) > 16:
                                    with st.expander("All selected row numbers"):
                                        st.dataframe(pd.DataFrame({"CSV row": row_ids}),
                                                     hide_index=True, width="stretch")
                            if not _render_source_deletion_details(
                                    source, [checked], key=f"rows_{fid}"):
                                can_approve_rows = False
                        with st.form(f"review_row_operation_{fid}_{_active_agent_c_control_generation(fid)}"):
                            row_reviewer = st.session_state.get("cleaning_approved_reviewer") or ""
                            if row_reviewer:
                                st.caption("Reviewer: " + row_reviewer)
                            else:
                                row_reviewer = st.text_input(
                                    "Reviewer identifier", key=f"review_row_reviewer_{fid}", max_chars=120)
                            yes_col, no_col = st.columns(2)
                            with yes_col:
                                row_approved = st.form_submit_button(
                                    "Yes — Approve & Execute", disabled=not can_approve_rows,
                                    key=f"review_approve_yes_{fid}_{_active_agent_c_control_generation(fid)}_rows")
                            with no_col:
                                decline_rows = st.form_submit_button(
                                    "No — Discuss Another Idea",
                                    key=f"review_discuss_no_{fid}_{_active_agent_c_control_generation(fid)}_rows")
                        if decline_rows:
                            _open_agent_c_alternative_chat(fid, discard_plan=True)
                            _rerun_cleaning_fragment(fid)
                        if row_approved:
                            if not str(row_reviewer or "").strip():
                                st.error("Enter a reviewer identifier before approval.")
                            elif not can_approve_rows:
                                st.error("Row plan is not verified")
                            else:
                                try:
                                    current_plan = validate_scoped_plan(
                                        plan, finding, source,
                                        user_requested_rows=st.session_state.cleaning_explicit_row_deletions_by_finding.get(fid))
                                    if current_plan != plan:
                                        raise PermissionError("Source or row plan changed")
                                    originals = source.read_bytes()
                                    latest = st.session_state.cleaning_staged_bytes_by_dataset.get(
                                        ds_id, originals)
                                    approved_entry = {
                                        "operation": row_op,
                                        "changes": current_plan["changes"],
                                        "finding_id": fid,
                                        "reviewer": str(row_reviewer).strip(),
                                        "source_sha256": current_plan["source_sha256"],
                                    }
                                    pending = [
                                        *st.session_state.cleaning_approved_rows_by_dataset.get(ds_id, []),
                                        approved_entry,
                                    ]
                                    apply_approved_row_operations(originals, latest, pending)
                                    st.session_state.cleaning_approved_rows_by_dataset[ds_id] = pending
                                    st.session_state.cleaning_approved_reviewer = str(row_reviewer).strip()
                                    st.session_state.cleaning_history.append({
                                        "dataset_id": ds_id, "finding_id": fid,
                                        "finding": finding["title"], "operation": row_op,
                                        "changes": len(current_plan["changes"]),
                                        "already_resolved": 0,
                                        "reviewer": str(row_reviewer).strip(),
                                        "storage": "session_memory_export_only",
                                        "modified_positions": [
                                            {k: value for k, value in item.items()
                                             if k in {"row", "source_row"}}
                                            for item in current_plan["changes"]],
                                        "source_sha256": current_plan["source_sha256"],
                                        "staged_sha256": sha256_bytes(latest),
                                    })
                                    st.session_state.cleaning_review_statuses[fid] = (
                                        "Approved row changes for final export")
                                    st.session_state.cleaning_decisions.append({
                                        "finding_id": fid,
                                        "decision": "structural_row_operation_approved_export_only",
                                        "operation": row_op,
                                        "source_sha256": current_plan["source_sha256"],
                                    })
                                    st.session_state.cleaning_plans_by_finding.pop(fid, None)
                                    _advance_agent_c_batch(fid)
                                    _remember_approved_agent_c_solution(fid, finding, source)
                                    _rerun_cleaning_fragment(fid)
                                except (ValueError, PermissionError, OSError, AssertionError,
                                        IndexError, KeyError, csv.Error) as exc:
                                    st.error(f"Row approval failed; no operation was staged: {exc}")
                    if plan and source and not compound_active and plan.get("operation") not in {"drop_column", "drop_rows", "merge_rows"}:
                        prepared = st.session_state.cleaning_r_previews_by_finding.get(fid)
                        # Earlier UI versions cached code-only previews without an R
                        # result. Re-run the isolated test once instead of treating
                        # that old cache as an approved insertion source.
                        if prepared and ("r_result" not in prepared or "insertion_plan" not in prepared):
                            st.session_state.cleaning_r_previews_by_finding.pop(fid, None)
                            prepared = None
                        previous_error = st.session_state.cleaning_r_errors_by_finding.get(fid)
                        if not prepared and not previous_error:
                            try:
                                # Authorized only to PREVIEW: execute deterministic R on
                                # the exact affected values in the isolated sandbox.
                                # The working CSV is not touched by this step.
                                prepared = prepare_and_execute_scoped_r_preview(plan, finding, source)
                                st.session_state.cleaning_r_previews_by_finding[fid] = prepared
                                st.session_state.cleaning_review_statuses[fid] = "R output verified — insertion awaiting approval"
                                st.session_state.cleaning_decisions.append({
                                    "finding_id": fid, "decision": "scoped_r_preview_tested_no_dataset_change",
                                    "source_sha256": prepared["source_sha256"],
                                    "r_preview_sha256": prepared["preview_sha256"],
                                    "r_result_sha256": prepared["r_result"]["verified_output_sha256"],
                                })
                            except (OSError, ValueError, KeyError, PermissionError, RuntimeError, csv.Error) as exc:
                                diagnosis = diagnose_r_preview_failure(exc, plan, finding)
                                previous_error = ("Isolated R preview failed: " + diagnosis["summary"]
                                                  + " [" + diagnosis["code"] + "]")
                                st.session_state.cleaning_r_errors_by_finding[fid] = previous_error
                                st.session_state.cleaning_r_diagnostics_by_finding[fid] = diagnosis
                        if previous_error:
                            st.error(previous_error + ". No working data were changed.")
                            diagnosis = st.session_state.cleaning_r_diagnostics_by_finding.get(fid)
                            if diagnosis:
                                # A failed operation must explain why it failed without
                                # exposing the source CSV or raw model/R code.
                                with st.expander("R failure details and next steps", expanded=True):
                                    st.write("**Failed stage:**", diagnosis["stage"])
                                    st.write("**Error code:**", diagnosis["code"])
                                    st.write("**Likely cause:**", diagnosis["summary"])
                                    st.write("**What to do:**", diagnosis["action"])
                                    if diagnosis.get("exit_code") is not None:
                                        st.write("**R exit status:**", diagnosis["exit_code"])
                                    if diagnosis.get("stderr_excerpt"):
                                        st.write("**Sanitized R / sandbox error output:**")
                                        st.code(diagnosis["stderr_excerpt"], language="text")
                                    else:
                                        st.caption("No usable R stderr was returned. The error category "
                                                   "and launcher checks above may still identify the cause.")
                                    preflight = diagnosis.get("preflight", {})
                                    st.caption("WSL launcher checks: "
                                               f"bwrap {'found' if preflight.get('bwrap_found') else 'not found'} · "
                                               f"Rscript {'found' if preflight.get('rscript_found') else 'not found'}")
                                    if diagnosis.get("partial_scope"):
                                        st.warning(
                                            f"This plan covers {diagnosis['planned_changes']} of "
                                            f"{diagnosis['affected_cells']} evidenced cells. "
                                            "That is a partial cleaning plan, separate from the R failure. "
                                            "Do not consider the full finding resolved by this batch."
                                        )
                                    if st.button("Run isolated R self-test", key=f"review_r_selftest_{fid}"):
                                        st.session_state.cleaning_r_selfchecks_by_finding[fid] = (
                                            run_r_environment_self_test()
                                        )
                                        _rerun_cleaning_fragment(fid)
                                    selfcheck = st.session_state.cleaning_r_selfchecks_by_finding.get(fid)
                                    if selfcheck:
                                        if selfcheck["passed"]:
                                            st.success(selfcheck["summary"])
                                        else:
                                            st.error("R sandbox self-test failed: " + selfcheck["summary"])
                                            if selfcheck.get("stderr_excerpt"):
                                                st.code(selfcheck["stderr_excerpt"], language="text")
                                        st.caption(selfcheck["action"])
                            if st.button("Retry isolated R preview", key=f"review_retry_r_{fid}"):
                                st.session_state.cleaning_r_errors_by_finding.pop(fid, None)
                                st.session_state.cleaning_r_diagnostics_by_finding.pop(fid, None)
                                st.session_state.cleaning_r_selfchecks_by_finding.pop(fid, None)
                                _rerun_cleaning_fragment(fid)
                        elif prepared:
                            # The researcher sees actual proposed values; R proof
                            # remains available on demand, not a second summary.
                            _render_before_after([prepared["insertion_plan"]])
                            affected = prepared["insertion_plan"]["changes"]
                            evidenced_cells, evidenced_headers = allowed_positions(finding, source)
                            required_positions = (evidenced_headers if
                                                  prepared["insertion_plan"]["operation"] == "rename_headers"
                                                  else evidenced_cells)
                            # Do not treat the first 32 of a large finding as complete.
                            # Smaller findings may legitimately require only selected
                            # coordinates when the researcher's approved plan says so.
                            incomplete_scope = (len(required_positions) > 32
                                                and len(affected) < len(required_positions))
                            if incomplete_scope:
                                st.warning(
                                    f"This plan covers only {len(affected)} of "
                                    f"{len(required_positions)} independently evidenced positions. "
                                    "A partial batch cannot complete this finding; "
                                    "request a full validated plan."
                                )
                            if False:  # technical R proof is verified by the backend, not shown in review
                                st.markdown("**1. R cleaning (already executed in isolation)**")
                                st.code(prepared["r_function"], language="r")
                                st.markdown("**2. R insertion (vectorized, reproducible specification)**")
                                st.code(prepared["insertion_r_function"], language="r")
                                if st.checkbox("Show verified insertion inputs (CSV table)",
                                               key=f"review_r_manifest_{fid}"):
                                    st.code(prepared["insertion_changes_csv"], language="csv")
                            # Reconcile each proposed R value against the most recent
                            # approved in-memory version *before* presenting approval.
                            # Idempotent overlaps do not require another edit. Genuine
                            # differences are displayed and verified by a second R
                            # sandbox run against the current staged cell values.
                            review = None
                            insertion = None
                            approved_rebase = None
                            can_approve = not incomplete_scope
                            original_bytes = source.read_bytes()
                            staged = st.session_state.cleaning_staged_bytes_by_dataset.get(
                                ds_id, original_bytes)
                            try:
                                insertion = verify_prepared_r_insertion(prepared, plan, finding, source)
                                review = assess_verified_in_memory(original_bytes, staged, insertion)
                                if review["already_resolved"]:
                                    st.info(f"{len(review['already_resolved'])} cell(s)/header(s) "
                                            "already have the proposed value from an earlier approval; "
                                            "no second modification will be made to those positions.")
                                if review["conflicts"]:
                                    # Already reconciled for benign taxonomy formatting below;
                                    # all other overlapping edits need explicit override.
                                    prior_by_cell = {}
                                    for past_event in history_all:
                                        if past_event.get("dataset_id") == ds_id:
                                            for coord in past_event.get("modified_positions", []):
                                                if isinstance(coord, dict):
                                                    prior_by_cell[(coord.get("csv_row"), coord.get("column"))] = (
                                                        past_event.get("finding_id") or "Earlier review")
                                    # The single Before → After preview above is the
                                    # complete action summary. Do not display a second
                                    # aggregate/conflict table here.
                                    cached = st.session_state.cleaning_rebased_previews_by_finding.get(fid)
                                    if not cached or (cached.get("staged_sha256") != review["staged_sha256"]
                                                      or cached.get("conflicts") != review["conflicts"]):
                                        cached = verify_r_against_current_staged_values(
                                            original_bytes, staged, insertion)
                                        st.session_state.cleaning_rebased_previews_by_finding[fid] = cached
                                    # Bind the checkbox to the exact current stage
                                    # AND proposed values. A changed stage or revised
                                    # plan requires a new explicit approval.
                                    review_identity = hashlib.sha256((
                                        review["staged_sha256"] + json.dumps(
                                            insertion, sort_keys=True, ensure_ascii=False)
                                    ).encode("utf-8")).hexdigest()[:24]
                                    # When a previous finding changed ONLY
                                    # taxonomy presentation (delimiter/prefixes),
                                    # a second approval checkbox is redundant.
                                    # Both edits still have isolated R proof,
                                    # and the final named Yes validates stage SHA.
                                    from healthcare_data_quality.direct_actions import same_taxonomy_identity
                                    prior_format_changes = {
                                        (pos.get("csv_row"), pos.get("column"))
                                        for event in history_all
                                        if event.get("dataset_id") == ds_id
                                        and event.get("finding_id") != fid
                                        and event.get("finding_id", "").startswith(f"{ds_id}:qc:")
                                        and event.get("operation") == "replace_cells"
                                        for pos in event.get("modified_positions", [])
                                        if isinstance(pos, dict)
                                    }
                                    only_presentation = (
                                        finding.get("check_key") in {
                                            "inspect_taxonomy_structure", "inspect_taxonomy_naming"}
                                        and all((v["csv_row"], v["column"]) in prior_format_changes
                                                and v["column"] == 1
                                                and same_taxonomy_identity(
                                                    v["original"], v["current"], v["proposed"])
                                                for v in review["conflicts"])
                                    )
                                    if only_presentation:
                                        approved_rebase = cached
                                    elif st.checkbox(
                                            "Override previously approved values",
                                            key=f"review_override_conflicts_{fid}_{review_identity}"):
                                        approved_rebase = cached
                                    else:
                                        can_approve = False
                                elif not review["ready_to_apply"]:
                                    st.success("This proposed cleaning is already satisfied by "
                                               "earlier staged edits. Approve to record the finding "
                                               "as resolved without changing any additional cell.")
                            except (OSError, ValueError, PermissionError, RuntimeError,
                                    csv.Error, KeyError) as exc:
                                can_approve = False
                                st.error(f"In-memory insertion failed; no changes were staged: {exc}")
                            with st.form(f"review_execute_r_{fid}_{_active_agent_c_control_generation(fid)}"):
                                # Only the first approval asks for an identifier. It
                                # appears beside the approval button so researchers
                                # never have to scroll back to the top of a long page.
                                reviewer_for_approval = (
                                    st.session_state.get("cleaning_approved_reviewer") or ""
                                )
                                if reviewer_for_approval:
                                    st.caption("Reviewer: " + reviewer_for_approval)
                                elif hasattr(st, "text_input"):
                                    reviewer_for_approval = st.text_input(
                                        "Reviewer identifier (required once for this investigation)",
                                        # More than one finding may have a pending
                                        # cell-edit approval form in the same render.
                                        # The investigation key alone is shared by
                                        # every finding and raises DuplicateElementKey.
                                        key=f"cleaning_reviewer_once_{session_context}_{fid}",
                                        max_chars=120,
                                    )
                                yes_col, no_col = st.columns(2)
                                with yes_col:
                                    execute = st.form_submit_button(
                                        "Yes — Approve & Execute", disabled=not can_approve,
                                        key=f"review_approve_yes_{fid}_{_active_agent_c_control_generation(fid)}_cells")
                                with no_col:
                                    decline_cells = st.form_submit_button(
                                        "No — Discuss Another Idea",
                                        key=f"review_discuss_no_{fid}_{_active_agent_c_control_generation(fid)}_cells")
                            if decline_cells:
                                _open_agent_c_alternative_chat(fid, discard_plan=True)
                                _rerun_cleaning_fragment(fid)
                            if execute:
                                reviewer = str(reviewer_for_approval or "").strip()
                                if not reviewer:
                                    st.error("Enter your reviewer identifier in the field directly "
                                             "above the approval button. This is required only once.")
                                elif not can_approve or review is None or insertion is None:
                                    st.error("Recheck the current data and any conflicts before approval.")
                                else:
                                    try:
                                        # Revalidate the staged version at the instant
                                        # approval is processed. The R proof is bound
                                        # to the same current staged SHA-256.
                                        fresh_staged = st.session_state.cleaning_staged_bytes_by_dataset.get(
                                            ds_id, original_bytes)
                                        if review["staged_sha256"] != hashlib.sha256(
                                                fresh_staged).hexdigest():
                                            raise PermissionError("Another finding changed the staged "
                                                                  "data since review; inspect it again")
                                        new_bytes, result = apply_verified_in_memory(
                                            original_bytes, fresh_staged, insertion, reviewer=reviewer,
                                            approved_rebase=approved_rebase,
                                        )
                                        r_info = {k: v for k, v in prepared["r_result"].items()
                                                  if k != "verified_output_csv"}
                                        result["r_preview_execution"] = r_info
                                        result["r_insertion_code_sha256"] = hashlib.sha256(
                                            prepared["insertion_r_function"].encode("utf-8")).hexdigest()
                                        # Commit session bytes and audit records only
                                        # after all verification has succeeded.
                                        st.session_state.cleaning_staged_bytes_by_dataset[ds_id] = new_bytes
                                        st.session_state.cleaning_approved_reviewer = reviewer
                                        # Retain the exact reviewed R artifacts only in
                                        # session memory for audit display after completion.
                                        st.session_state.cleaning_completed_audits_by_finding[fid] = {
                                            "r_function": prepared["r_function"],
                                            "insertion_r_function": prepared["insertion_r_function"],
                                            "changes": list(prepared["insertion_plan"]["changes"]),
                                        }
                                        st.session_state.cleaning_pending_gemma_b.pop(fid, None)
                                        st.session_state.cleaning_history.append({
                                            "dataset_id": ds_id, "finding_id": fid,
                                            "finding": finding["title"], "operation": insertion["operation"],
                                            "changes": result["changes_applied"],
                                            "already_resolved": result["already_resolved"],
                                            "conflicts_overridden": result["conflicts_overridden"],
                                            "reviewer": reviewer, "storage": "session_memory",
                                            "modified_positions": [
                                                {"csv_row": item["csv_row"], "column": item["column"]}
                                                for item in review["ready_to_apply"] + (
                                                    review["conflicts"] if approved_rebase else [])
                                            ],
                                            "approved_cell_changes": [
                                                {"row": item["csv_row"], "column": item["column"],
                                                 "before": item["current"], "after": item["proposed"]}
                                                for item in review["ready_to_apply"] + (
                                                    review["conflicts"] if approved_rebase else [])
                                            ],
                                            "source_sha256": insertion["source_sha256"],
                                            "staged_sha256": result["staged_sha256"],
                                            "r_code_sha256": r_info["r_code_sha256"],
                                        })
                                        if result["changes_applied"]:
                                            st.session_state.cleaning_review_statuses[fid] = "Applied and verified"
                                        else:
                                            st.session_state.cleaning_review_statuses[fid] = (
                                                "Resolved by earlier approved edit")
                                        st.session_state.cleaning_decisions.append({
                                            "finding_id": fid,
                                            "decision": ("verified_r_result_staged_in_memory"
                                                         if result["changes_applied"] else
                                                         "already_resolved_by_prior_verified_edit"),
                                            "r_result_sha256": r_info["verified_output_sha256"],
                                            "staged_sha256": result["staged_sha256"],
                                        })
                                        for other in catalog:
                                            if other.get("dataset_id") == ds_id:
                                                other_id = other["id"]
                                                # An edit in one finding invalidates
                                                # cached R/current-state previews,
                                                # NOT another finding's approved-
                                                # evidence proposal or pending group.
                                                # Those remain visible and are
                                                # rechecked against fresh staged
                                                # values at their final approval.
                                                if other_id != fid:
                                                    for key in (
                                                        "cleaning_r_previews_by_finding",
                                                        "cleaning_r_errors_by_finding",
                                                        "cleaning_r_diagnostics_by_finding",
                                                        "cleaning_r_selfchecks_by_finding",
                                                        "cleaning_rebased_previews_by_finding",
                                                        "cleaning_compound_previews_by_finding",
                                                    ):
                                                        st.session_state.setdefault(key, {}).pop(other_id, None)
                                        _advance_agent_c_batch(fid)
                                        _remember_approved_agent_c_solution(fid, finding, source)
                                        _rerun_cleaning_fragment(fid)
                                    except (OSError, ValueError, PermissionError, AssertionError,
                                            RuntimeError, csv.Error, KeyError) as exc:
                                        st.error(f"In-memory insertion failed; no changes were staged: {exc}")

            # A partially approved batch may leave valid staged edits plus
            # intentionally rejected remainder. Require an explicit final
            # review decision, not an impossible "all reviewed" download gate.
            if (applied and not plan and not remaining_batch and not b_pending and
                    st.session_state.cleaning_review_statuses.get(fid) ==
                    "Partially approved — remaining action rejected" and
                    st.button("Finish review — keep approved changes",
                              key=f"review_finish_partial_{fid}")):
                st.session_state.cleaning_review_statuses[fid] = "Applied and verified"
                st.session_state.cleaning_batch_rejected_by_finding.discard(fid)
                st.session_state.cleaning_decisions.append({
                    "finding_id": fid,
                    "decision": "remaining_proposal_declined_existing_approvals_retained",
                })
                _rerun_cleaning_fragment(fid)
            if (not plan and not applied and not b_pending and not remaining_batch and
                    st.session_state.cleaning_review_statuses.get(fid) != "Reviewed — no change" and
                    st.button("Mark reviewed (no change)",
                              key=f"review_no_action_{fid}")):
                st.session_state.cleaning_review_statuses[fid] = "Reviewed — no change"
                st.session_state.cleaning_decisions.append({
                    "finding_id": fid, "decision": "no_change_authorized_or_executed",
                })
                if source:
                    from healthcare_data_quality.interaction_memory import save_reviewed_solution
                    try:
                        save_reviewed_solution(
                            finding, source, decision="reviewed_no_change",
                            user_request="Mark reviewed — no change required now",
                        )
                    except (OSError, ValueError, UnicodeError) as exc:
                        st.warning(f"Review completed, but memory could not be saved: {exc}")
                _rerun_cleaning_fragment(fid)


    # Export is intentionally locked until EVERY current finding and Gemma
    # check is finished; no partially reviewed file should look final.
    # A still-running Agent A investigation can append new checks, so it also
    # holds the gate closed until the check catalog is stable.
    from exploration_jobs import exploration_manager
    run_key = (audit.get("cleaning_recommendation_run_id") or
               st.session_state.get("registered_investigation_run_id"))
    active_investigation = False
    if run_key:
        exploration = exploration_manager().snapshot(run_key)
        registered = investigation_manager().snapshot(run_key)
        active_investigation = any(
            isinstance(job, dict) and job.get("status") == "running"
            for job in (exploration, registered)
        )
    can_download, reviewed_count, review_count = _cleaning_export_gate(
        catalog, st.session_state.cleaning_review_statuses,
        st.session_state.cleaning_plans_by_finding,
        st.session_state.cleaning_batch_queue_by_finding,
        st.session_state.cleaning_pending_gemma_b,
        st.session_state.cleaning_pending_visual_flags_by_finding,
        investigation_running=active_investigation,
    )
    # Explicit button size; use Streamlit's native primary (green-themed) button
    # plus narrow, local styling for the download widget only.
    st.markdown("""
    <style>
    div[data-testid="stDownloadButton"] button[kind="primary"] {
        background-color: #15803d !important;
        border-color: #15803d !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        min-height: 3.3rem !important;
        padding: 0.8rem 1.6rem !important;
    }
    div[data-testid="stDownloadButton"] button[kind="primary"]:hover:not(:disabled) {
        background-color: #166534 !important;
        border-color: #166534 !important;
    }
    div[data-testid="stDownloadButton"] button:disabled {
        opacity: 0.46 !important;
        cursor: not-allowed !important;
    }
    </style>
    """, unsafe_allow_html=True)
    if not can_download:
        if active_investigation:
            st.caption(f"Review complete: {reviewed_count}/{review_count} · Investigation still running")
        else:
            st.caption(f"Review complete: {reviewed_count}/{review_count}")
    # Never compute or expose download bytes before the gate is satisfied.
    # The disabled button is a visual indicator; `data` must not be an
    # early-exportable partial output.
    if not can_download:
        if hasattr(st, "download_button"):
            st.download_button(
                "Download cleaned CSV" if len(uploaded) == 1 else "Download cleaned datasets (ZIP)",
                data=b"", file_name="review_required.txt", mime="text/plain",
                key=f"cleaning_download_final_{session_context}",
                type="primary", disabled=True,
            )
    else:
        try:
            datasets_for_download = []
            for i, file in enumerate(uploaded):
                ds_id = f"dataset_{i+1}"
                staged_bytes = st.session_state.cleaning_staged_bytes_by_dataset.get(ds_id)
                original = dataset_paths[i].read_bytes()
                data = staged_bytes if staged_bytes is not None else original
                row_approvals = st.session_state.cleaning_approved_rows_by_dataset.get(ds_id, [])
                data = apply_approved_row_operations(original, data, row_approvals)
                approvals = st.session_state.cleaning_approved_columns_by_dataset.get(ds_id, [])
                data = apply_approved_column_deletions(
                    original, data, approvals,
                    approved_row_operations=row_approvals,
                    staged_before_rows=staged_bytes if staged_bytes is not None else original)
                datasets_for_download.append((ds_id, file.name, data))
            file_name, mime, data = build_final_download(datasets_for_download)
        except (ValueError, PermissionError, AssertionError, OSError, csv.Error) as exc:
            st.error(f"Download unavailable: {exc}")
        else:
            if hasattr(st, "download_button"):
                st.download_button(
                    "Download cleaned CSV" if len(uploaded) == 1
                    else "Download cleaned datasets (ZIP)",
                    data=data, file_name=file_name, mime=mime,
                    key=f"cleaning_download_final_{session_context}",
                    type="primary", disabled=False,
                )


def _prepare_finding_discussion(
    finding: dict, source: Path | None, uploaded: list,
    dataset_paths: list[Path], question: str, *, automatic: bool = False,
):
    """Capture the exact submitted text and immutable context on the UI thread."""
    from healthcare_data_quality.finding_cleaning import allowed_positions
    fid = finding["id"]
    ds_id = finding.get("dataset_id", "")
    index = int(ds_id.rsplit("_", 1)[-1]) - 1 if ds_id.startswith("dataset_") else -1
    evidence = finding.get("evidence") or []
    valid_cells, valid_headers = allowed_positions(finding, source) if source else (set(), set())
    thread = st.session_state.cleaning_messages_by_finding.setdefault(fid, [])
    old_thread = list(thread)
    if not question.strip() or len(question) > 4000:
        raise ValueError("Enter between 1 and 4,000 characters")
    thread.append({"role": "user", "content": question,
                   **({"automatic": True} if automatic else {})})
    st.session_state.cleaning_review_statuses[fid] = "Discussion open"
    st.session_state.cleaning_plans_by_finding.pop(fid, None)
    st.session_state.setdefault("cleaning_compound_previews_by_finding", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_explicit_row_deletions_by_finding", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_explicit_column_deletions_by_finding", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_memory_candidates", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_batch_queue_by_finding", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_batch_total_by_finding", {}).pop(fid, None)
    st.session_state.cleaning_r_previews_by_finding.pop(fid, None)
    st.session_state.cleaning_r_errors_by_finding.pop(fid, None)
    st.session_state.cleaning_r_diagnostics_by_finding.pop(fid, None)
    st.session_state.cleaning_r_selfchecks_by_finding.pop(fid, None)
    from healthcare_data_quality.evidence_registry import (
        build_finding_registry, verify_finding_registry, prompt_evidence,
    )
    registry = (verify_finding_registry(finding, source, enforce_source_hash=True) if source else None)
    if source and registry is None:
        try:
            registry = build_finding_registry(finding, source)
        except (OSError, ValueError, UnicodeError, csv.Error) as exc:
            registry = None
            # It remains possible to discuss a malformed file, but no source
            # totals can be presented as verified until the CSV is parseable.
            finding = {**finding, "evidence_registry_error": str(exc)}
    scoped = {"finding_id": fid, "dataset_id": ds_id,
              "title": finding["title"], "source": finding["source"],
              "method": finding.get("method"), "evidence": evidence[:32],
              "allowed_cell_positions": sorted(valid_cells)[:64],
              "allowed_header_positions": sorted(valid_headers),
              "all_affected_positions_count": len(valid_cells),
              "verified_evidence_registry": prompt_evidence(registry) if registry else None,
              "evidence_registry_unavailable": finding.get("evidence_registry_error"),
              "latest_working_version": bool(st.session_state.cleaning_staged_bytes_by_dataset.get(ds_id))}
    recommendation = st.session_state.cleaning_recommendations.get(fid)
    if recommendation:
        # Batch recommendations are not part of the per-finding conversation.
        # Include the chosen suggestion as untrusted context before asking the
        # model to develop it; otherwise "I agree" has no referent.
        scoped["suggestion_under_discussion"] = {
            "suggestion": str(recommendation.get("suggestion", ""))[:2200],
            "decision_needed": str(recommendation.get("decision_needed", ""))[:1400],
        }
    metadata = []
    if source and 0 <= index < len(dataset_paths):
        from healthcare_data_quality.in_memory_cleaning import _records, sha256_bytes
        original_bytes = source.read_bytes()
        original_records = _records(original_bytes)
        staged_bytes = st.session_state.cleaning_staged_bytes_by_dataset.get(
            ds_id, original_bytes)
        current_records = _records(staged_bytes)
        if (len(current_records) != len(original_records) or any(
                len(a) != len(b) for a, b in zip(current_records, original_records))):
            raise PermissionError("In-memory dataset dimensions changed since investigation")
        sample_positions = sorted(valid_cells)[:32]
        scoped["current_source_cell_examples"] = [
            {"row": row, "column": col, "value": current_records[row - 1][col - 1]}
            for row, col in sample_positions
            if row <= len(current_records) and col <= len(current_records[0])
        ]
        scoped["original_source_cell_examples"] = [
            {"row": row, "column": col, "value": original_records[row - 1][col - 1]}
            for row, col in sample_positions
            if row <= len(original_records) and col <= len(original_records[0])
        ]
        scoped["latest_staged_sha256"] = sha256_bytes(staged_bytes)
        scoped["latest_working_version"] = staged_bytes != original_bytes
        headers = original_records[0]
        # Source-wide counts prevent the model from extrapolating from one
        # example (e.g. assuming taxonomy NA only appears in CSV Row 2).
        trusted_inventory = source_inventory(source)
        scoped["full_source_inventory"] = concise_source_context(trusted_inventory)
        metadata = [{"dataset_id": ds_id, "name": uploaded[index].name,
                     "columns": [{"position": i + 1, "header": h,
                                  "current_header": current_records[0][i]}
                                 for i, h in enumerate(headers)]}]
    return fid, thread, old_thread, scoped, metadata


def _normalize_agent_c_current_before_values(response: dict, finding: dict,
                                              source: Path) -> dict:
    """Bind staged-before values to immutable original evidence for the R plan.

    Agent C is shown the latest approved in-memory values. If it proposes an
    edit from one of those current values, verify that exact current value and
    translate only the `before` field for the pre-existing original-source R
    verifier. The later in-memory R recheck runs against the actual staged
    value. This never changes coordinates, suggested replacements or scope.
    """
    from healthcare_data_quality.in_memory_cleaning import _records
    if response.get("operation") not in {"replace_cells", "rename_headers"}:
        return response
    changes = response.get("changes")
    if not isinstance(changes, list):
        return response
    ds_id = finding.get("dataset_id", "")
    original_bytes = Path(source).read_bytes()
    current_bytes = st.session_state.get("cleaning_staged_bytes_by_dataset", {}).get(
        ds_id, original_bytes)
    if current_bytes == original_bytes:
        return response
    original_records, current_records = _records(original_bytes), _records(current_bytes)
    normalized = []
    for change in changes:
        if not isinstance(change, dict):
            normalized.append(change)
            continue
        is_header = response["operation"] == "rename_headers"
        r = 0 if is_header else change.get("row", 0) - 1 if type(change.get("row")) is int else -1
        c = (change.get("position") if is_header else change.get("column"))
        c = c - 1 if type(c) is int else -1
        item = dict(change)
        if (0 <= r < len(original_records) and 0 <= r < len(current_records)
                and 0 <= c < len(original_records[0]) and 0 <= c < len(current_records[0])):
            original_before, current_before = original_records[r][c], current_records[r][c]
            # Do not normalize guessed or stale values. An attempt to restore
            # the literal original source value needs a separate reversion
            # workflow (the original-source validator rejects no-op edits).
            if (item.get("before") == current_before and current_before != original_before
                    and item.get("after") != original_before):
                item["before"] = original_before
        normalized.append(item)
    return {**response, "changes": normalized}


def _render_compound_approval(fid: str, finding: dict, source: Path,
                              ds_id: str, catalog: list[dict]) -> None:
    """One atomic review/approval for all actions in one Agent C request."""
    from healthcare_data_quality.compound_approval import (
        describe_operation, evaluate_compound, proposal_digest,
    )
    from healthcare_data_quality.in_memory_cleaning import sha256_bytes

    plans = [st.session_state.cleaning_plans_by_finding[fid],
             *st.session_state.cleaning_batch_queue_by_finding.get(fid, [])]
    original = source.read_bytes()
    staged = st.session_state.cleaning_staged_bytes_by_dataset.get(ds_id, original)
    current_cols = st.session_state.cleaning_approved_columns_by_dataset.get(ds_id, [])
    current_rows = st.session_state.cleaning_approved_rows_by_dataset.get(ds_id, [])
    digest = proposal_digest(plans, original, staged, current_cols, current_rows)
    cache_map = st.session_state.setdefault("cleaning_compound_previews_by_finding", {})
    preview = cache_map.get(fid)
    preview_error = (preview.get("_error") if preview and preview.get("digest") == digest
                     else None)
    if not preview or preview.get("digest") != digest:
        try:
            preview = evaluate_compound(
                plans, finding, source, staged, current_cols, current_rows,
                requested_rows=st.session_state.cleaning_explicit_row_deletions_by_finding.get(fid),
                requested_columns=st.session_state.cleaning_explicit_column_deletions_by_finding.get(fid),
            )
            cache_map[fid] = preview
        except (OSError, ValueError, PermissionError, RuntimeError, AssertionError,
                IndexError, KeyError, csv.Error) as exc:
            preview_error = str(exc)
            cache_map[fid] = {"digest": digest, "_error": preview_error}
            preview = None
    if preview_error:
        preview = None

    with st.container(border=True):
        _render_before_after(plans)
        if preview_error:
            st.error(f"Review blocked: {preview_error}")
            if st.button("Retry verification", key=f"review_compound_retry_{fid}_{_active_agent_c_control_generation(fid)}"):
                cache_map.pop(fid, None)
                _rerun_cleaning_fragment(fid)
        elif preview:
            if not _render_source_deletion_details(
                    source, plans, key=f"group_{fid}"):
                preview = None
        with st.form(f"review_compound_{fid}_{_active_agent_c_control_generation(fid)}"):
            reviewer = st.session_state.get("cleaning_approved_reviewer") or ""
            if reviewer:
                st.caption(f"Reviewer: {reviewer}")
            else:
                reviewer = st.text_input("Reviewer", max_chars=120,
                                         key=f"review_compound_reviewer_{fid}")
            approve_col, reject_col = st.columns(2)
            with approve_col:
                approve = st.form_submit_button(
                    "Yes — Approve & Execute", disabled=preview is None,
                    key=f"review_approve_yes_{fid}_{_active_agent_c_control_generation(fid)}_compound")
            with reject_col:
                reject = st.form_submit_button(
                    "No — Discuss Another Idea",
                    key=f"review_discuss_no_{fid}_{_active_agent_c_control_generation(fid)}_compound")

        if reject:
            # Reject only the pending group. Do not submit a Gemma request.
            _open_agent_c_alternative_chat(fid, discard_plan=True)
            st.session_state.cleaning_decisions.append({
                "finding_id": fid, "decision": "compound_plan_rejected_no_change",
                "operations": len(plans),
            })
            _rerun_cleaning_fragment(fid)

        if approve:
            reviewer = str(reviewer or "").strip()
            if not reviewer:
                st.error("Enter a reviewer identifier.")
            elif preview is None:
                st.error("The proposal needs a valid preview.")
            else:
                try:
                    # Revalidate EVERY operation at the moment of approval.
                    # Reuse only the independently verified R outputs, not
                    # cached decisions or the model's unverified claims.
                    approved = evaluate_compound(
                        plans, finding, source,
                        st.session_state.cleaning_staged_bytes_by_dataset.get(ds_id, original),
                        st.session_state.cleaning_approved_columns_by_dataset.get(ds_id, []),
                        st.session_state.cleaning_approved_rows_by_dataset.get(ds_id, []),
                        reviewer=reviewer,
                        requested_rows=st.session_state.cleaning_explicit_row_deletions_by_finding.get(fid),
                        requested_columns=st.session_state.cleaning_explicit_column_deletions_by_finding.get(fid),
                        r_previews=preview["r_previews"],
                    )
                    if (approved["digest"] != digest
                            or approved["export_sha256"] != preview["export_sha256"]):
                        raise PermissionError("Dataset changed; review the proposal again")
                    history_entries = []
                    for event in approved["events"]:
                        history_entries.append({
                            **event,
                            "dataset_id": ds_id, "finding_id": fid,
                            "finding": finding["title"],
                            "reviewer": reviewer,
                            "source_sha256": approved["original_sha256"],
                            "staged_sha256": sha256_bytes(approved["staged_after"]),
                            "storage": ("session_memory_export_only" if event["operation"] in
                                        {"drop_column", "merge_rows", "drop_rows"}
                                        else "session_memory"),
                            "approval_group": digest[:16],
                        })
                    # No state is touched until the complete proposal and
                    # export have passed verification. This is one commit.
                    st.session_state.cleaning_staged_bytes_by_dataset[ds_id] = approved["staged_after"]
                    st.session_state.cleaning_approved_columns_by_dataset[ds_id] = approved["columns_after"]
                    st.session_state.cleaning_approved_rows_by_dataset[ds_id] = approved["rows_after"]
                    st.session_state.cleaning_history.extend(history_entries)
                    st.session_state.cleaning_approved_reviewer = reviewer
                    st.session_state.cleaning_review_statuses[fid] = "Applied and verified"
                    st.session_state.cleaning_decisions.append({
                        "finding_id": fid, "decision": "compound_plan_approved_atomically",
                        "operations": len(plans), "approval_group": digest[:16],
                        "source_sha256": approved["original_sha256"],
                        "export_sha256": approved["export_sha256"],
                    })
                    # Retain the audited R proof and history without requiring
                    # a second confirmation for the structural operations.
                    proofs = approved["r_previews"].values()
                    st.session_state.cleaning_completed_audits_by_finding[fid] = {
                        "r_function": "\n\n".join(p["r_function"] for p in proofs),
                        "insertion_r_function": "\n\n".join(
                            p["insertion_r_function"] for p in approved["r_previews"].values()),
                        "changes": [c for p in plans for c in p["changes"]],
                    }
                    for key in (
                        "cleaning_plans_by_finding", "cleaning_batch_queue_by_finding",
                        "cleaning_batch_total_by_finding", "cleaning_compound_previews_by_finding",
                        "cleaning_r_previews_by_finding", "cleaning_r_errors_by_finding",
                        "cleaning_explicit_row_deletions_by_finding",
                        "cleaning_explicit_column_deletions_by_finding",
                    ):
                        st.session_state.setdefault(key, {}).pop(fid, None)
                    st.session_state.cleaning_discussion_mode_by_finding.pop(fid, None)
                    # Save memory only after approval. Its failure must not
                    # roll back a valid, already-committed transaction.
                    _remember_approved_agent_c_solution(fid, finding, source)
                    _rerun_cleaning_fragment(fid)
                except (OSError, ValueError, PermissionError, RuntimeError,
                        AssertionError, IndexError, KeyError, csv.Error) as exc:
                    st.error(f"Nothing was staged: {exc}")


def _advance_agent_c_batch(fid: str) -> None:
    """Only after explicit approval, expose the next independently checked action."""
    queue = st.session_state.setdefault("cleaning_batch_queue_by_finding", {}).get(fid, [])
    if queue:
        next_plan = queue.pop(0)
        st.session_state.cleaning_plans_by_finding[fid] = next_plan
        st.session_state.cleaning_review_statuses[fid] = "Next action awaiting approval"
        for field in ("cleaning_r_previews_by_finding", "cleaning_r_errors_by_finding",
                      "cleaning_r_diagnostics_by_finding", "cleaning_r_selfchecks_by_finding",
                      "cleaning_rebased_previews_by_finding"):
            st.session_state.setdefault(field, {}).pop(fid, None)
    else:
        st.session_state.setdefault("cleaning_plans_by_finding", {}).pop(fid, None)
        st.session_state.setdefault("cleaning_batch_queue_by_finding", {}).pop(fid, None)
        st.session_state.setdefault("cleaning_batch_total_by_finding", {}).pop(fid, None)
        # Final approval closes discussion for this finding. A new chat must
        # be explicitly reopened via the completed audit's Revise control.
        st.session_state.setdefault("cleaning_discussion_mode_by_finding", {}).pop(fid, None)


def _cancel_agent_c_batch(fid: str, *, status: str) -> None:
    _next_agent_c_ui_generation(fid)
    # A rejected proposal must never later be remembered as approved.
    st.session_state.setdefault("cleaning_memory_candidates", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_explicit_column_deletions_by_finding", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_reviewed_rule_context_by_finding", {}).pop(fid, None)
    """Reject current AND unapproved queued actions without undoing approvals."""
    st.session_state.cleaning_plans_by_finding.pop(fid, None)
    st.session_state.setdefault("cleaning_batch_queue_by_finding", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_batch_total_by_finding", {}).pop(fid, None)
    if any(event.get("finding_id") == fid for event in
           st.session_state.get("cleaning_history", [])):
        st.session_state.setdefault("cleaning_batch_rejected_by_finding", set()).add(fid)
        st.session_state.cleaning_review_statuses[fid] = "Partially approved — remaining action rejected"
    else:
        st.session_state.cleaning_review_statuses[fid] = status
    st.session_state.setdefault("cleaning_decisions", []).append({
        "finding_id": fid, "decision": "remaining_compound_actions_rejected_no_change",
    })


def _remember_approved_agent_c_solution(fid: str, finding: dict,
                                         source: Path | None) -> None:
    """Write persistent history only after every plan received human approval.

    Memory failures must not roll back an already verified and staged edit.
    """
    candidate = st.session_state.setdefault("cleaning_memory_candidates", {}).get(fid)
    if not candidate or source is None:
        return
    if (st.session_state.cleaning_plans_by_finding.get(fid)
            or st.session_state.cleaning_batch_queue_by_finding.get(fid)):
        return
    approvals = [item for item in st.session_state.cleaning_history
                 if item.get("finding_id") == fid]
    if len(approvals) - candidate["approvals_before"] < len(candidate["operations"]):
        return
    try:
        from healthcare_data_quality.interaction_memory import save_reviewed_solution
        if file_sha256(source) != candidate["source_sha256"]:
            raise ValueError("The source changed since the review proposal")
        from healthcare_data_quality.reviewed_rule_replay import annotate_source_headers
        save_reviewed_solution(
            finding, source, operations=annotate_source_headers(candidate["operations"], source),
            decision="approved_edit", user_request=candidate["user_request"],
            agent_response=candidate["agent_response"],
        )
        st.session_state.cleaning_memory_candidates.pop(fid, None)
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        st.warning(f"Edit approved, but its reusable review memory could not be saved: {exc}")


def _record_gemma_b_response(fid: str, finding: dict, source: Path | None,
                             response: dict) -> None:
    """Validate EVERY requested operation before queueing any for human review.

    Multiple operations are independent plans. Approval of the first never
    silently discards the second, and no action executes at proposal time.
    """
    from healthcare_data_quality.finding_cleaning import (
        expand_uniform_evidence_plan, explicitly_requested_row_deletions,
        requested_blank_row_scope,
    )
    # A delayed completion cannot reactivate a terminal finding. An explicit
    # new request first moves the finding back into discussion state.
    if st.session_state.get("cleaning_review_statuses", {}).get(fid) in {
        "Applied and verified", "Resolved by earlier approved edit",
        "Approved column deletion for final export",
        "Approved row changes for final export",
        "Partially approved — remaining action rejected", "Reviewed — no change",
    }:
        return
    st.session_state.setdefault("cleaning_compound_previews_by_finding", {}).pop(fid, None)
    thread = st.session_state.cleaning_messages_by_finding.setdefault(fid, [])
    from healthcare_data_quality.cleaning_workspace import concise_agent_c_reply
    reply = concise_agent_c_reply(str(response["reply"]))
    if source:
        from healthcare_data_quality.evidence_registry import (
            build_finding_registry, evidence_bound_response,
        )
        try:
            verified_evidence = build_finding_registry(finding, source)
        except (OSError, ValueError, UnicodeError, csv.Error):
            verified_evidence = None
        reply = evidence_bound_response(reply, verified_evidence)
    proposals = response.get("operations") or []
    if not proposals and response.get("operation") != "none":
        proposals = [{"operation": response["operation"],
                      "dataset_id": response["dataset_id"],
                      "changes": response["changes"]}]
    # Clarification must precede proposing edits. Some model responses
    # mistakenly contain BOTH a two-choice question and provisional plans.
    # Never queue those plans while the researcher has not chosen a scope.
    choices = agent_c_clarification_options(
        reply, response.get("clarification_options"))
    _next_agent_c_ui_generation(fid)
    if choices:
        st.session_state.cleaning_plans_by_finding.pop(fid, None)
        st.session_state.setdefault("cleaning_batch_queue_by_finding", {}).pop(fid, None)
        st.session_state.setdefault("cleaning_batch_total_by_finding", {}).pop(fid, None)
        st.session_state.setdefault("cleaning_memory_candidates", {}).pop(fid, None)
        st.session_state.cleaning_review_statuses[fid] = "Clarification required"
        thread.append({"role": "assistant", "content": reply,
                       "clarification_options": choices})
        return
    if not proposals:
        st.session_state.cleaning_review_statuses[fid] = "Discussion updated"
        choices = agent_c_clarification_options(
            reply, response.get("clarification_options"))
        message = {"role": "assistant", "content": reply}
        if choices:
            message["clarification_options"] = choices
        thread.append(message)
        return

    try:
        if not source:
            raise ValueError("No attributable dataset is available")
        if len(proposals) > 16:
            raise ValueError("At most 16 operations can be verified in one batch")
        verified_plans = []
        recommendation = st.session_state.get("cleaning_recommendations", {}).get(fid) or {}
        full_scope_action = str(recommendation.get("suggestion") or "")
        accepted_full_scope = (
            st.session_state.cleaning_discussion_mode_by_finding.get(fid) == "accepted"
            and any(word in full_scope_action.casefold().split()
                    for word in ("all", "every", "entire"))
        )
        recent_user_messages = [str(item.get("content") or "") for item in thread
                                if item.get("role") == "user" and not item.get("automatic")]
        latest_explicit_user = recent_user_messages[-1] if recent_user_messages else ""
        explicit_row_ids = explicitly_requested_row_deletions(latest_explicit_user)
        # A brief answer to Gemma's specific-value question (e.g. "679")
        # must not erase the user's prior, explicitly confirmed row deletion.
        # Only carry scope through a numeric continuation; never resurrect
        # older row coordinates for an unrelated new request.
        if (not explicit_row_ids and len(recent_user_messages) >= 2
                and re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)",
                                 latest_explicit_user.strip())):
            for earlier in reversed(recent_user_messages[-5:-1]):
                explicit_row_ids = explicitly_requested_row_deletions(earlier)
                if explicit_row_ids:
                    break
        if latest_explicit_user == "Review source-verified remembered cleaning rule":
            explicit_row_ids = st.session_state.setdefault(
                "cleaning_explicit_row_deletions_by_finding", {}).get(fid, frozenset())
        if source:
            blank_row_scope = requested_blank_row_scope(latest_explicit_user, source, finding=finding)
            if blank_row_scope is not None:
                explicit_row_ids = blank_row_scope
        explicit_column_ids = st.session_state.setdefault(
            "cleaning_explicit_column_deletions_by_finding", {}).get(fid)
        expanded_total = 0
        for raw in proposals:
            if not isinstance(raw, dict):
                raise ValueError("An operation is not a structured plan")
            model_plan = _normalize_agent_c_current_before_values(raw, finding, source)
            checked, expanded_count = expand_uniform_evidence_plan(
                model_plan, finding, source,
                full_finding_accepted=accepted_full_scope and len(proposals) == 1,
                user_requested_rows=explicit_row_ids,
                user_requested_columns=explicit_column_ids)
            verified_plans.append(checked)
            expanded_total += expanded_count
        # Repeating an operation TYPE is safe. Repeating a SOURCE TARGET is
        # not: reject overlaps before the researcher can approve the group.
        from healthcare_data_quality.compound_approval import validate_compound_targets
        validate_compound_targets(verified_plans)
        # Only the current real user message can authorize out-of-finding row
        # deletion. This state is not populated from Gemma JSON or old chats.
        row_scope = st.session_state.setdefault("cleaning_explicit_row_deletions_by_finding", {})
        if explicit_row_ids and any(p["operation"] == "drop_rows" for p in verified_plans):
            row_scope[fid] = explicit_row_ids
        else:
            row_scope.pop(fid, None)
        # Expire all historical clarification controls when a NEW validated
        # proposal becomes the active user interaction. Preserve chat text.
        for past_message in thread:
            if past_message.get("role") == "assistant":
                past_message["controls_expired"] = True
        for state_key in ("cleaning_custom_value_input_by_finding",
                          "cleaning_custom_operation_input_by_finding"):
            st.session_state.setdefault(state_key, {}).pop(fid, None)
        st.session_state.cleaning_plans_by_finding[fid] = verified_plans[0]
        remaining = verified_plans[1:]
        st.session_state.setdefault("cleaning_batch_queue_by_finding", {})[fid] = remaining
        st.session_state.setdefault("cleaning_batch_total_by_finding", {})[fid] = len(verified_plans)
        st.session_state.cleaning_review_statuses[fid] = "Exact plan awaiting approval"
        from healthcare_data_quality.cleaning_authorization import file_sha256
        latest_user = next((str(item.get("content") or "") for item in reversed(thread)
                            if item.get("role") == "user" and not item.get("automatic")), "")
        if not latest_user or latest_user == "Review source-verified remembered cleaning rule":
            # The assessment request was automatic: retain the method rationale
            # only in the eventual memory entry, after a fresh human approval.
            latest_user = st.session_state.setdefault(
                "cleaning_reviewed_rule_context_by_finding", {}).get(fid, {}).get(
                "user_request") or latest_user
        if latest_user.startswith("I accept Agent B's action") and full_scope_action:
            # The actual reviewed action is more useful than the long,
            # generic drafting authorization for future recall.
            latest_user = full_scope_action
        st.session_state.setdefault("cleaning_memory_candidates", {})[fid] = {
            "operations": verified_plans,
            "approvals_before": sum(h.get("finding_id") == fid for h in
                                    st.session_state.cleaning_history),
            "user_request": latest_user,
            "agent_response": reply,
            "source_sha256": file_sha256(source),
        }
        from healthcare_data_quality.compound_approval import describe_operation
        summary = "\n".join(
            f"{index}. {describe_operation(item)}"
            for index, item in enumerate(verified_plans, start=1)
        )
        reply = summary + "\nReview below to approve or reject."
    except (ValueError, KeyError, csv.Error, TypeError) as exc:
        st.session_state.cleaning_plans_by_finding.pop(fid, None)
        st.session_state.setdefault("cleaning_batch_queue_by_finding", {}).pop(fid, None)
        st.session_state.setdefault("cleaning_batch_total_by_finding", {}).pop(fid, None)
        st.session_state.cleaning_review_statuses[fid] = "Plan validation failed"
        st.session_state.setdefault("cleaning_explicit_row_deletions_by_finding", {}).pop(fid, None)
        st.session_state.setdefault("cleaning_memory_candidates", {}).pop(fid, None)
        reply = f"Proposal not ready: {exc}. No changes made."
    thread.append({"role": "assistant", "content": reply})


def _ask_gemma_for_finding(finding: dict, source: Path | None, uploaded: list,
                            dataset_paths: list[Path], question: str) -> None:
    """Legacy synchronous entry point retained for CLI/test compatibility."""
    fid, thread, old_thread, scoped, metadata = _prepare_finding_discussion(
        finding, source, uploaded, dataset_paths, question)
    try:
        with st.spinner("Gemma is reviewing this finding (read-only)"):
            response = ask_gemma_about_cleaning(
                question, old_thread, [scoped], metadata, selected_finding=scoped)
        _record_gemma_b_response(fid, finding, source, response)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        thread.append({"role": "assistant", "content":
                       f"Agent C did not return a valid response: {exc}. No changes were made."})


def _background_agents_idle_for_run(context_key: str, run_id: str) -> tuple[bool, bool]:
    """Thread-safe, process-local availability; never reads Streamlit state.

    The same saved recommendation queue drives the Agent B status card and
    the borrow decision. The probe is safe for queued worker threads to call
    after Agent B finishes; it never starts new background work.
    """
    if not context_key or not run_id:
        return False, False
    from exploration_jobs import exploration_manager
    from cleaning_jobs import recommendation_manager, job_fingerprint
    try:
        a_idle = (not investigation_manager().has_running_jobs()
                  and not exploration_manager().has_running_jobs())
        key = job_fingerprint(context_key, [], run_id=run_id)
        snapshot = recommendation_manager().snapshot(key) or {}
        entries = snapshot.get("entries") or {}
        b_idle = bool(entries) and all(
            isinstance(entry, dict) and entry.get("status") in {"ready", "failed"}
            for entry in entries.values()
        )
        return a_idle, b_idle
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            json.JSONDecodeError):
        return False, False


def _finished_background_agents_for_review() -> tuple[bool, bool]:
    """UI-side adapter for the same background-safe availability check."""
    audit = st.session_state.get("audit_result") or {}
    if not isinstance(audit, dict):
        return False, False
    run_id = (audit.get("cleaning_recommendation_run_id")
              or st.session_state.get("registered_investigation_run_id"))
    context_key = st.session_state.get("audit_cache_key")
    return _background_agents_idle_for_run(context_key, run_id)


def _queue_agent_c_for_finding(finding: dict, source: Path | None, uploaded: list,
                               dataset_paths: list[Path], question: str, *,
                               automatic: bool = False,
                               require_gemma_review: bool = False) -> None:
    """Show the submitted question now, then let Agent C answer in a worker.

    Cross-dataset remembered approaches must go through actual model review;
    text describing an old edit is not a fresh user deletion instruction.
    """
    from interactive_chat_jobs import interactive_chat_manager
    from healthcare_data_quality.cleaning_authorization import file_sha256
    fid = finding["id"]
    # An explicit direct discussion is valid even when Agent B's background
    # recommendation is missing or failed. Treat the finding's verified source
    # evidence as context; NEVER fabricate an Agent B suggestion to pass a gate.
    # Accepted-agent-B actions still require an actual recommendation.
    if (fid not in st.session_state.get("cleaning_recommendations", {})
            and st.session_state.cleaning_discussion_mode_by_finding.get(fid)
                != "alternative"):
        raise ValueError("Choose an Agent B suggestion or start a direct Agent C discussion")
    if fid in st.session_state.cleaning_pending_gemma_b:
        return
    st.session_state.setdefault("cleaning_completed_gemma", {}).pop(fid, None)
    _next_agent_c_ui_generation(fid)
    fid, thread, old_thread, scoped, metadata = _prepare_finding_discussion(
        finding, source, uploaded, dataset_paths, question, automatic=automatic)
    try:
        # Resolve the explicit absolute-value button immediately from the
        # finding's immutable source evidence. No Gemma queue/GPU time is
        # needed; _record_gemma_b_response still enforces scoped validation
        # and the independent R / human-approval workflow.
        if not require_gemma_review:
            # A visual flag is NOT a replacement-cell operation. The host
            # resolves its coordinates and original values deterministically.
            # Never forward this intent to Gemma's numeric-edit-only schema.
            if source:
                from healthcare_data_quality.visual_flags import requested_visual_flag
                try:
                    flag_plan = requested_visual_flag(question, finding, source)
                except ValueError as exc:
                    thread.append({"role": "assistant", "content": f"Flag not prepared: {exc}"})
                    st.session_state.cleaning_review_statuses[fid] = "Discussion open"
                    return
                if flag_plan is not None:
                    for item in thread:
                        if item.get("role") == "assistant":
                            item["controls_expired"] = True
                    st.session_state.cleaning_plans_by_finding.pop(fid, None)
                    st.session_state.cleaning_batch_queue_by_finding.pop(fid, None)
                    st.session_state.cleaning_pending_visual_flags_by_finding[fid] = {
                        **flag_plan, "request": question}
                    st.session_state.cleaning_review_statuses[fid] = "Visual flag awaiting approval"
                    thread.append({"role": "assistant", "content":
                                   f"Flag {len(flag_plan['flags'])} taxonomy names bold red. "
                                   "Values unchanged; review below."})
                    return
            from healthcare_data_quality.finding_cleaning import (
                propose_explicit_user_row_deletion, propose_bulk_blank_row_deletion,
            )
            if (source and question.strip().casefold() in {
                    "fix", "fix it", "ok", "yes", "correct it", "standardize", "standardise"}):
                from healthcare_data_quality.direct_actions import (
                    draft_format_correction, draft_identical_column_removal)
                direct = (draft_identical_column_removal(finding, source)
                          if finding.get("check_key") == "compare_repeated_headers"
                          else draft_format_correction(finding, source))
                if direct is not None:
                    _record_gemma_b_response(fid, finding, source, direct)
                    return
            # A global user-specified condition must be evaluated over the whole
            # original CSV by Python, not guessed from the selected finding's
            # representative evidence or emitted as hundreds of Gemma JSON edits.
            bulk_row_plan = (propose_bulk_blank_row_deletion(question, finding, source)
                             if source else None)
            if bulk_row_plan is not None:
                _record_gemma_b_response(fid, finding, source, bulk_row_plan)
                return
            # Typed 'approval' is not authorization: the prepared proposal always
            # has a dedicated form button after full validation.
            if question.strip().casefold() in {
                "approval", "approve", "approved", "i approve", "yes approve",
            }:
                thread.append({"role": "assistant", "content":
                               "Use Approve on the validated proposal. No edit was staged."})
                return
            user_row_plan = (propose_explicit_user_row_deletion(question, finding, source)
                             if source else None)
            if user_row_plan is not None:
                _record_gemma_b_response(fid, finding, source, user_row_plan)
                return
            # The reviewer supplies the desired operation, not extra scientific
            # evidence. Match unambiguous numeric Before -> After requests to
            # exact affected cells in the immutable original CSV. Ambiguous
            # scope still goes to Gemma, never to an invented operation.
            from healthcare_data_quality.finding_cleaning import (
                propose_reviewer_cell_replacements,
            )
            explicit_cells = (propose_reviewer_cell_replacements(
                question, finding, source) if source else None)
            if explicit_cells is not None:
                _record_gemma_b_response(fid, finding, source, explicit_cells)
                return
            from healthcare_data_quality.cleaning_workspace import _draft_evidenced_absolute_values
            absolute_plan = _draft_evidenced_absolute_values(question, scoped)
            if absolute_plan is not None:
                _record_gemma_b_response(fid, finding, source, absolute_plan)
                return
            # For large, uniform, evidence-backed edits, never ask Gemma to emit
            # hundreds of per-cell JSON changes. Host expands the explicit user's
            # instruction deterministically, then the normal R preview and final
            # human approval gates still run unchanged.
            from healthcare_data_quality.finding_cleaning import propose_evidenced_blank_fill
            recommendation = st.session_state.get("cleaning_recommendations", {}).get(fid) or {}
            bulk = (propose_evidenced_blank_fill(
                question, str(recommendation.get("suggestion") or ""), finding, source,
                accepted_recommendation=(
                    st.session_state.cleaning_discussion_mode_by_finding.get(fid) == "accepted"
                    and question.startswith("I accept Agent B's action for this finding")
                ),
            ) if source else None)
            if bulk is not None:
                _record_gemma_b_response(fid, finding, source, bulk)
                return
        manager = interactive_chat_manager()
        # Explicit Yes/No conversations enter one C lane. A queued job may
        # borrow a newly idle A/B later, not only at click time. This is
        # important when B finishes its recommendations while C is replying.
        mode = st.session_state.cleaning_discussion_mode_by_finding.get(fid)
        user_initiated = (mode in {"accepted", "alternative"} and not automatic)
        if user_initiated and hasattr(manager, "submit_routed"):
            audit = st.session_state.get("audit_result") or {}
            context_key = st.session_state.get("audit_cache_key")
            run_id = (audit.get("cleaning_recommendation_run_id")
                      or st.session_state.get("registered_investigation_run_id"))
            if not context_key or not run_id:
                # The cleaning workspace can be rendered directly in unit
                # tests without the top-level investigation session state.
                context_key = st.session_state.get("cleaning_context_key", "").split(":")[0]
                run_id = audit.get("cleaning_recommendation_run_id")
            idle_probe = (lambda context_key=context_key, run_id=run_id:
                          _background_agents_idle_for_run(context_key, run_id))
            a_idle, b_idle = idle_probe()
            job_id, agent = manager.submit_routed(
                ask_gemma_about_cleaning, question, old_thread, [scoped], metadata,
                selected_finding=scoped, force_model=require_gemma_review,
                agent_a_idle=a_idle, agent_b_idle=b_idle,
                idle_probe=idle_probe,
            )
        else:
            # Automatic retries retain C ownership and are never handed to
            # another role without a new explicit user request.
            job_id = manager.submit(
                ask_gemma_about_cleaning, question, old_thread, [scoped], metadata,
                selected_finding=scoped, force_model=require_gemma_review)
            agent = "C"
        from healthcare_data_quality.in_memory_cleaning import sha256_bytes
        ds_id = finding.get("dataset_id", "")
        original = source.read_bytes() if source else b""
        staged = st.session_state.cleaning_staged_bytes_by_dataset.get(ds_id, original)
        st.session_state.cleaning_pending_gemma_b[fid] = {
            "job_id": job_id, "source_sha256": file_sha256(source) if source else None,
            "staged_sha256": sha256_bytes(staged),
            "ui_generation": _active_agent_c_control_generation(fid),
            "assigned_agent": agent,
        }
        st.session_state.cleaning_review_statuses[fid] = f"Agent {agent} replying"
    except (ValueError, PermissionError, csv.Error) as exc:
        st.session_state.cleaning_review_statuses[fid] = "Bulk proposal validation failed"
        thread.append({"role": "assistant", "content":
                       f"The requested bulk action was not prepared: {exc}. "
                       "No data were changed. Check this finding's exact source evidence."})
    except Exception as exc:
        thread.append({"role": "assistant", "content":
                       f"Could not submit your question: {exc}. No changes were made."})


def _sync_gemma_conversations(finding: dict, source: Path | None) -> None:
    """Collect both agents' results without blocking the Streamlit UI."""
    from interactive_chat_jobs import interactive_chat_manager
    from healthcare_data_quality.cleaning_authorization import file_sha256
    fid = finding["id"]
    manager = interactive_chat_manager()
    pending_b = st.session_state.cleaning_pending_gemma_b.get(fid)
    if pending_b:
        if hasattr(manager, "agent_for"):
            live_agent = manager.agent_for(pending_b["job_id"])
            if live_agent in {"A", "B", "C"}:
                pending_b["assigned_agent"] = live_agent
        # The small completion-only poller can collect the result without
        # rerendering this form. Consume that exact stored outcome once.
        outcome = st.session_state.setdefault("cleaning_completed_gemma", {}).pop(
            fid, None)
        if outcome is None:
            outcome = manager.take(pending_b["job_id"])
        if outcome["status"] != "running":
            st.session_state.cleaning_pending_gemma_b.pop(fid, None)
            if outcome["status"] == "done":
                try:
                    response = outcome["result"]
                    if not isinstance(response, dict):
                        raise ValueError("Agent C did not return a structured reply")
                    from healthcare_data_quality.in_memory_cleaning import sha256_bytes, _records
                    ds_id = finding.get("dataset_id", "")
                    current = (st.session_state.cleaning_staged_bytes_by_dataset.get(
                        ds_id, source.read_bytes()) if source else b"")
                    # Another finding may have approved an unrelated change
                    # while Gemma was thinking. Do not throw away useful work
                    # merely because the staged bytes hash changed: the model
                    # can only DRAFT a plan. Source positions, current staged
                    # conflicts and R output are checked again on approval.
                    # Structural changes to the coordinate-stable staged CSV
                    # are still incompatible with an in-flight answer.
                    coordinate_compatible = True
                    if source and pending_b.get("staged_sha256") != sha256_bytes(current):
                        before_records, latest_records = _records(source.read_bytes()), _records(current)
                        coordinate_compatible = (
                            len(before_records) == len(latest_records)
                            and all(len(a) == len(b) for a, b in
                                    zip(before_records, latest_records))
                        )
                    obsolete = (
                        pending_b.get("ui_generation") != _active_agent_c_control_generation(fid)
                        or not coordinate_compatible
                        or (source and file_sha256(source) != pending_b.get("source_sha256"))
                        or st.session_state.cleaning_review_statuses.get(fid) in {
                            "Applied and verified", "Resolved by earlier approved edit",
                            "Approved column deletion for final export",
                            "Approved row changes for final export",
                            "Partially approved — remaining action rejected", "Reviewed — no change",
                        }
                    )
                    if obsolete:
                        # Preserve terminal approval state, and never revive an old widget.
                        if st.session_state.cleaning_review_statuses.get(fid) not in {
                            "Applied and verified", "Resolved by earlier approved edit",
                            "Approved column deletion for final export",
                            "Approved row changes for final export",
                            "Partially approved — remaining action rejected", "Reviewed — no change",
                        }:
                            _next_agent_c_ui_generation(fid)
                            st.session_state.cleaning_review_statuses[fid] = "Discussion open"
                        st.session_state.cleaning_messages_by_finding[fid].append({
                            "role": "assistant", "content":
                            "This reply became outdated after a review or staged-data change. "
                            "Submit a new request against the current state. "
                            "No cleaning plan was approved or executed."})
                    else:
                        message_count = len(st.session_state.cleaning_messages_by_finding.get(fid, []))
                        _record_gemma_b_response(fid, finding, source, response)
                        cross_review = st.session_state.setdefault(
                            "cleaning_cross_source_review_by_finding", {}).get(fid)
                        if cross_review and cross_review.get("source_sha256") == file_sha256(source):
                            if (st.session_state.cleaning_plans_by_finding.get(fid)
                                    and st.session_state.cleaning_review_statuses.get(fid)
                                        != "Plan validation failed"):
                                cross_review["state"] = "gemma_reviewed_and_source_validated"
                            else:
                                cross_review["state"] = "no_verified_proposal"
                        messages = st.session_state.cleaning_messages_by_finding.get(fid, [])
                        if (len(messages) > message_count and messages[-1].get("role") == "assistant"):
                            messages[-1]["assigned_agent"] = pending_b.get("assigned_agent", "C")
                except (KeyError, ValueError, OSError, csv.Error) as exc:
                    st.session_state.cleaning_messages_by_finding[fid].append({
                        "role": "assistant", "content": f"Could not validate Agent C's response: {exc}"})
            else:
                if st.session_state.cleaning_review_statuses.get(fid) not in {
                    "Applied and verified", "Resolved by earlier approved edit",
                    "Approved column deletion for final export",
                    "Approved row changes for final export",
                    "Partially approved — remaining action rejected", "Reviewed — no change",
                }:
                    _next_agent_c_ui_generation(fid)
                    st.session_state.cleaning_review_statuses[fid] = "Discussion open"
                st.session_state.cleaning_messages_by_finding[fid].append({
                    "role": "assistant", "content":
                    f"Agent C's response was interrupted ({outcome.get('error', 'worker unavailable')}). "
                    "Your question is preserved above; no data were changed."})


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


def start_live_exploration(
    audit: dict, cache_key: str, paths: list[Path], names: list[str],
) -> str:
    """Start sandbox exploration in a worker so Streamlit can render QC now.

    Background callbacks use only plain Python and private file checkpoints;
    never touch st.* or a user's Streamlit Session State.
    """
    from exploration_jobs import exploration_manager

    run_id = audit["cleaning_recommendation_run_id"]
    key = exploration_cache_key(cache_key)
    checkpoint = {"file": None}
    datasets = {f"dataset_{i}": path for i, path in enumerate(paths, 1)}

    def record_round(runs: list[dict]) -> None:
        checkpoint_error = None
        try:
            checkpoint["file"] = save_explorations(key, runs)
        except (OSError, TypeError, ValueError) as exc:
            checkpoint_error = exc
        # Even if a disk write fails, publish a newly verified hypothesis to
        # Agent B; earlier suggestions are never generated again.
        schedule_agent_b_once(cache_key, audit, paths, names, runs)
        if checkpoint_error is not None:
            raise checkpoint_error

    def record_finish(runs: list[dict], stop: dict | None) -> None:
        save_explorations(key, runs, stop=stop,
                          replace_checkpoint=checkpoint["file"])

    exploration_manager().start(
        run_id, datasets, audit, on_round_saved=record_round,
        on_finished=record_finish,
    )
    return run_id


def sync_live_exploration(audit: dict, cache_key: str) -> dict | None:
    """Read the latest verified checks on the UI thread without invoking Gemma."""
    from exploration_jobs import exploration_manager

    run_id = audit.get("cleaning_recommendation_run_id")
    if not run_id:
        return None
    job = exploration_manager().snapshot(run_id)
    if job is None:
        return None
    prepare_exploration_state(cache_key)
    st.session_state.exploration_runs = job["runs"]
    st.session_state.exploration_stop = job["stop"]
    st.session_state.exploration_error = (
        "Exploratory analysis was interrupted: " + job["error"]
        if job.get("error") else None
    )
    st.session_state.exploration_cache_warning = bool(job.get("warning"))
    st.session_state.exploration_restored = False
    return job


class InvestigationProgressView:
    """Live offline timer plus honest *phase*, not time-estimate, progress."""

    def __init__(self, *, include_exploration: bool, exploration_only: bool = False):
        self.tracker = InvestigationProgress(include_exploration=include_exploration)
        self.started = monotonic()
        self.started_epoch_ms = int(time() * 1000)
        # The visual hero is outside the technical panel so it remains highly
        # visible while Gemma's Phase 2 synchronous call blocks Streamlit.
        self.hero_container = st.container(key="hdq_investigation_banner_blocking")
        with self.hero_container:
            self.hero_slot = st.empty()
        self.status = st.status("Technical investigation progress (advanced)", expanded=False)
        with self.status:
            self.phase_bar = st.progress(0.0, text=self.tracker.phase_label)
            self.timer_slot = st.empty()
            # A self-ticking, local iframe updates every second even while a
            # synchronous Gemma/vLLM HTTP call blocks the Streamlit script.
            # No patient/CSV data is passed into the HTML; no network requests.
            import streamlit.components.v1 as components
            with self.timer_slot:
                components.html(live_timer_html(self.started_epoch_ms), height=40)
            self.round_slot = st.empty()
            self.activity_slot = st.empty()
            st.caption("The progress bar shows workflow phases, not an estimated percentage or time remaining. "
                       "Gemma decides when exploration should stop.")
        if exploration_only:
            self.begin_exploration()
        else:
            self._refresh()

    def _refresh(self) -> None:
        # A local iframe timer continues ticking during long blocking model
        # calls; live status is refreshed whenever the backend emits an event.
        import streamlit.components.v1 as components
        with self.hero_slot:
            components.html(render_investigation_phase_banner_html(
                phase=self.tracker.phase,
                phase_count=self.tracker.phase_count,
                activity=self.tracker.last_activity,
                started_epoch_ms=self.started_epoch_ms,
                verified_rounds=self.tracker.verified_rounds,
            ), height=170, scrolling=False)
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
            self.hero_slot.empty()
            self._stop_timer()
            self.status.update(label="Baseline completed; exploratory analysis incomplete",
                               state="error", expanded=True)
            return
        self.tracker.finish()
        self.hero_slot.empty()
        self._stop_timer()
        self.status.update(label="Investigation finished", state="complete", expanded=False)

    def handoff(self) -> None:
        """Baseline is ready; the read-only Gemma stage is nonblocking."""
        self.hero_slot.empty()
        self._stop_timer()
        self.status.update(
            label="QC complete — Gemma exploration continues below",
            state="complete", expanded=False,
        )

    def error(self) -> None:
        self.hero_slot.empty()
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
        headline = "Investigation ended — coverage incomplete"
        if gemma_error:
            details = "Exploration stopped unexpectedly; verified findings remain available."
        elif controller_paused:
            details = "Gemma could not identify another distinct test. Other issues may remain."
        elif no_verified_checks:
            details = "No verified checks were recorded."
        elif backend_budget_exhausted:
            details = "Investigation reached its step limit; objectives remain open."
        else:
            details = "Only the completed checks are available for review."
        if has_findings:
            details += " Review the findings below."
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
    summary = str(result.get("summary") or "").strip()
    # Whitespace-only summaries have no splitlines()[0]. Use a safe title
    # while retaining the completed check in the investigation audit trail.
    title = next(
        (line.strip()[:160] for line in summary.splitlines() if line.strip()),
        "Exploratory check completed",
    )
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
        from exploration_jobs import exploration_manager
        live = exploration_manager().snapshot(audit.get("cleaning_recommendation_run_id", ""))
        if live and live["status"] == "running":
            st.caption("Exploratory tests are in progress. Verified results will appear here automatically.")
        else:
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


def _hide_completed_status(panel: str, run_identity: str) -> None:
    """Dismiss only a completed card for this run; do not touch the saved job."""
    if panel not in {"investigation", "recommendations"}:
        raise ValueError("Unknown status card")
    st.session_state[f"hdq_hidden_{panel}_run"] = run_identity


def _status_card_visible(panel: str, run_identity: str) -> bool:
    """A new run ID automatically restores cards closed for an earlier run."""
    return st.session_state.get(f"hdq_hidden_{panel}_run") != run_identity


STATUS_CARD_AUTO_HIDE_SECONDS = 10.0


def _status_card_auto_visible(
    panel: str, run_identity: str, *, finished: bool, now: float | None = None,
) -> bool:
    """Show finished agent cards for 10 seconds, then release their layout space.

    The two-second status *fragment* calls this on its own reruns, so
    no page rerun, JavaScript timer, or change to the researcher's draft is
    required. Each run has independent completion timers for A, B, and C.
    A/B also support the existing immediate manual close button.

    Agent C starts idle and has no finished-job event yet: leave its card
    visible until it has actually handled a request. A subsequent request
    immediately makes its card visible again.
    """
    if panel not in {"investigation", "recommendations", "agent_c"}:
        raise ValueError("Unknown status card")
    moment = monotonic() if now is None else now
    state_key = f"hdq_auto_hide_{panel}"
    state = st.session_state.get(state_key)
    if not isinstance(state, dict) or state.get("run") != run_identity:
        state = {"run": run_identity, "saw_work": False, "finished_at": None}

    if not finished:
        state["saw_work"] = True
        state["finished_at"] = None
        # A new job in this run supersedes an earlier manual dismissal.
        if st.session_state.get(f"hdq_hidden_{panel}_run") == run_identity:
            st.session_state.pop(f"hdq_hidden_{panel}_run", None)
    elif panel != "agent_c" or state["saw_work"]:
        if state["finished_at"] is None:
            state["finished_at"] = moment

    st.session_state[state_key] = state
    if not _status_card_visible(panel, run_identity):
        return False
    finished_at = state["finished_at"]
    return finished_at is None or moment - finished_at < STATUS_CARD_AUTO_HIDE_SECONDS




def _agent_c_status_counts(recommendation_state: dict | None) -> tuple[int, int, int]:
    """Read-only Agent C status scoped to the current Agent B investigation."""
    counts = recommendation_progress_counts(recommendation_state)
    entries = (recommendation_state or {}).get("entries") or {}
    ready_ids = {
        fid for fid, entry in entries.items()
        if isinstance(entry, dict) and entry.get("status") == "ready"
    }
    pending = st.session_state.get("cleaning_pending_gemma_b") or {}
    active = sum(
        1 for fid, meta in pending.items()
        if fid in ready_ids and isinstance(meta, dict) and meta.get("job_id")
    )
    return counts["ready"], counts["total"], active


def render_live_investigation_status(
    audit: dict, cache_key: str, header_reports: list[dict],
    include_exploration: bool, banner_slot=None,
) -> None:
        run_for_update = audit.get("cleaning_recommendation_run_id")
        if audit.get("partial") and run_for_update:
            snapshot_for_update = investigation_manager().snapshot(run_for_update)
            if snapshot_for_update and snapshot_for_update.get("audit"):
                # Do not rerun the full Streamlit app from a background status
                # fragment: it would reset the active review tab and scroll.
                # Publish the full result; the next ordinary page render uses it.
                audit = snapshot_for_update["audit"]
                st.session_state.audit_result = audit
        objective_ledger = audit.get("investigation_objectives") or {}
        status = audit.get("status", "unknown")
        header_problems = any(
            bool(report.get("duplicate_headers") or report.get("blank_header_positions")
                 or report.get("ragged_records"))
            for report in header_reports
        )
        live = sync_live_exploration(audit, cache_key)
        outcome = investigation_outcome(
            audit,
            header_problems=header_problems,
            gemma_runs=st.session_state.get("exploration_runs") or [],
            gemma_stop=st.session_state.get("exploration_stop"),
            gemma_error=st.session_state.get("exploration_error"),
            exploration_enabled=include_exploration,
        )
        # Render Agent A, Agent B, and Agent C read-only statuses in the SAME floating host.
        # A completed investigation remains visible while the once-only Agent B
        # queue continues; all three cards retain their own live status.
        from cleaning_jobs import recommendation_manager, job_fingerprint
        import streamlit.components.v1 as components
        run_id = audit.get("cleaning_recommendation_run_id")
        recommendation_state = None
        if isinstance(run_id, str) and run_id:
            try:
                recommendation_key = job_fingerprint(cache_key, [], run_id=run_id)
                recommendation_state = recommendation_manager().snapshot(recommendation_key)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # Display an honest unavailable/not-scheduled state rather
                # than block the findings or repeat any model requests.
                recommendation_state = None
        registered = investigation_manager().snapshot(run_id) if run_id else None
        registered_running = bool(registered and registered["status"] == "running")
        if registered_running:
            from datetime import datetime
            started = datetime.fromisoformat(registered["started_at"])
            investigation_html = render_investigation_phase_banner_html(
                phase=int(registered["phase"]),
                phase_count=3 if include_exploration else 2,
                activity=registered["activity"],
                started_epoch_ms=int(started.timestamp() * 1000),
            )
        elif live and live["status"] == "running":
            investigation_html = render_exploration_banner_html(live)
        else:
            investigation_html = render_finished_investigation_banner_html(
                state=outcome["state"],
                verified_checks=len(st.session_state.get("exploration_runs") or []),
            )
        # Native Streamlit close controls are mounted OUTSIDE the iframe:
        # they remain interactive without unsafe cross-frame JavaScript.
        # Neither control is available while its task is running or idle.
        run_identity = str(
            audit.get("cleaning_recommendation_run_id")
            or st.session_state.get("active_audit_file")
            or cache_key
        )
        investigation_complete = not (registered_running or (live and live.get("status") == "running"))
        # B's terminal state is independent of A's exploration: finished B
        # may disappear while A is still running, and reappears if new work
        # joins its recommendation queue.
        recommendations_complete = recommendation_is_complete(recommendation_state)
        ready, total, responding = _agent_c_status_counts(recommendation_state)
        show_a = _status_card_auto_visible(
            "investigation", run_identity, finished=investigation_complete,
        )
        show_b = _status_card_auto_visible(
            "recommendations", run_identity, finished=recommendations_complete,
        )
        show_c = _status_card_auto_visible(
            "agent_c", run_identity, finished=not bool(responding),
        )
        def show_status_cards() -> None:
            if show_a:
                with st.container(key="hdq_investigation_status_card"):
                    components.html(investigation_html, height=170, scrolling=False)
                    if investigation_complete:
                        # The keyed overlay is positioned relative to THIS card,
                        # not the scrolling page or the iframe's content box.
                        with st.container(key="hdq_investigation_close_overlay"):
                            st.button(
                                "×", key="hdq_close_investigation",
                                help="Close completed investigation status",
                                on_click=_hide_completed_status,
                                args=("investigation", run_identity),
                            )
            if show_b:
                with st.container(key="hdq_recommendation_status_card"):
                    components.html(render_recommendation_banner_html(recommendation_state),
                                    height=151, scrolling=False)
                    if recommendations_complete:
                        with st.container(key="hdq_recommendation_close_overlay"):
                            st.button(
                                "×", key="hdq_close_recommendations",
                                help="Close completed Agent B recommendations status",
                                on_click=_hide_completed_status,
                                args=("recommendations", run_identity),
                            )
            # Agent C is on-demand: no false completion percentage and no
            # terminal-state close button. Keep it visible even when A/B are
            # dismissed; the next conversation animates this third card.
            if show_c:
                with st.container(key="hdq_agent_c_status_card"):
                    components.html(
                        render_agent_c_banner_html(
                            ready_findings=ready, total_findings=total,
                            active_conversations=responding, idle=not bool(ready),
                        ),
                        height=151, scrolling=False,
                    )

        if banner_slot is not None:
            with banner_slot.container():
                show_status_cards()
        else:
            show_status_cards()
        if not (registered_running or (live and live["status"] == "running")):
            if outcome["state"] == "limited_clear":
                st.info("**" + outcome["headline"] + "** — " + outcome["details"])
            else:
                st.warning("**" + outcome["headline"] + "** — " + outcome["details"])
        # These are THREE distinct lifecycle states. In particular, an
        # old reviewed solution is never counted as a currently staged edit.
        current_review_context = (f"{cache_key}:{run_id}" if run_id else cache_key)
        if st.session_state.get("cleaning_context_key") == current_review_context:
            staged_events = st.session_state.get("cleaning_history") or []
            review_statuses = st.session_state.get("cleaning_review_statuses") or {}
            no_change_reviews = sum(v == "Reviewed — no change" for v in review_statuses.values())
            clean_summary = (f"{len(staged_events)} approved actions staged in this session"
                             if staged_events else "No changes staged in this session")
            if no_change_reviews:
                clean_summary += f"; {no_change_reviews} no-change reviews"
        else:
            clean_summary = "No changes staged in this session"
        if recommendation_state and (recommendation_state.get("entries") or {}):
            count_info = recommendation_progress_counts(recommendation_state)
            rec_summary = f"{count_info['ready']}/{count_info['total']} ready"
        else:
            rec_summary = "No matching recommendation status"
        coverage_summary = ("Running" if registered_running or (live and live["status"] == "running")
                            else "Incomplete coverage" if outcome["state"] == "partial"
                            else "Completed checks recorded")
        st.caption(f"Investigation: {coverage_summary}  ·  "
                   f"Recommendations: {rec_summary}  ·  Cleaning: {clean_summary}")
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
                st.write("Gemma exploration:", "In progress" if live and live["status"] == "running"
                         else "No final stopping decision is available.")
        if audit.get("proposal_generation_error"):
            st.error("Gemma could not complete proposal generation; see the downloaded audit for details.")


def render_pre_investigation_status(banner_slot, *, state: str = "ready") -> None:
    """Park all three status cards with honest idle states before an investigation.

    This renders only local HTML; no model is called and no stale saved
    conversation is shown as if it belonged to the uploaded dataset.
    """
    import streamlit.components.v1 as components
    with banner_slot.container():
        components.html(render_idle_investigation_banner_html(state=state),
                        height=170, scrolling=False)
        components.html(render_recommendation_banner_html(None, idle=True),
                        height=151, scrolling=False)
        components.html(render_agent_c_banner_html(idle=True),
                        height=151, scrolling=False)


def _toggle_full_dataset_preview(state_key: str) -> None:
    """Toggle the complete CSV table without eagerly loading it while hidden."""
    st.session_state[state_key] = not bool(st.session_state.get(state_key, False))


def render_full_dataset_section(
    paths: list[Path], uploaded: list, header_reports: list[dict], fingerprint: str
) -> bool:
    """Always show small dataset metrics; load the full table only on demand.

    Streamlit expanders execute their contents even when visually collapsed.
    A session-scoped disclosure button avoids reading/rendering wide CSVs until
    someone requests the preview. The selected dataset survives collapsing.
    """
    st.subheader("Full dataset")
    open_key = f"full_dataset_preview_open_{fingerprint}"
    index_key = f"full_dataset_preview_index_{fingerprint}"
    preview_open = bool(st.session_state.get(open_key, False))
    options = {f"dataset_{i + 1} — {file.name}": i for i, file in enumerate(uploaded)}
    previous_index = st.session_state.get(index_key, 0)
    if not isinstance(previous_index, int) or not 0 <= previous_index < len(uploaded):
        previous_index = 0

    if preview_open:
        selected = st.selectbox(
            "Inspect uploaded dataset", list(options), index=previous_index,
            key=f"full_dataset_selector_{fingerprint}",
        )
        index = options[selected]
        st.session_state[index_key] = index
    else:
        index = previous_index

    details = header_reports[index]
    count_col, col_col, duplicate_col = st.columns(3)
    count_col.metric("Records", details["data_rows"])
    col_col.metric("Columns", details["total_columns"])
    duplicate_col.metric(
        "Repeated column names",
        sum(1 for name in details["duplicate_headers"] if name.strip()),
    )
    st.button(
        "▼ Hide dataset preview" if preview_open else "▶ View dataset preview",
        key=f"full_dataset_preview_toggle_{fingerprint}",
        on_click=_toggle_full_dataset_preview,
        args=(open_key,),
    )
    if not preview_open:
        return True

    try:
        full_table = preview_csv(paths[index])
        st.caption(
            f"Displaying all {len(full_table):,} records and {len(full_table.columns):,} columns. "
            "Scroll vertically and horizontally; use the table's full-screen "
            "control for a larger view. No preview rows or columns are omitted."
        )
        full_table.index = pd.RangeIndex(
            start=2, stop=len(full_table) + 2, name="CSV row"
        )
        from healthcare_data_quality.visual_flags import styled_dataset_preview
        flags = st.session_state.get("cleaning_visual_flags_by_dataset", {}).get(
            f"dataset_{index + 1}", [])
        st.dataframe(styled_dataset_preview(full_table, flags),
                     hide_index=False, width="stretch", height=650)
    except (UnicodeError, csv.Error, OSError) as exc:
        st.error(f"Could not display the complete CSV: {exc}")
        return False
    return True


def _poll_agent_c_completion() -> None:
    """Auto-deliver an Agent C answer without periodically remounting its form.

    Agent A/B status is updated by its own fragment. This poller does not
    perform Gemma inference or apply a cleaning operation. An app rerun occurs
    only once, when a submitted Agent C job actually finishes; the active tab
    is preserved by the stateful tab key on supported Streamlit releases.
    """
    from interactive_chat_jobs import interactive_chat_manager

    pending = st.session_state.get("cleaning_pending_gemma_b") or {}
    ready = st.session_state.setdefault("cleaning_completed_gemma", {})
    if not pending:
        return
    manager = interactive_chat_manager()
    for fid, info in list(pending.items()):
        if fid in ready or not isinstance(info, dict) or not info.get("job_id"):
            continue
        if hasattr(manager, "agent_for"):
            live_agent = manager.agent_for(info["job_id"])
            if live_agent in {"A", "B", "C"}:
                info["assigned_agent"] = live_agent
        result = manager.take(info["job_id"])
        if result.get("status") != "running":
            ready[fid] = result
    if any(fid in ready for fid in pending):
        st.rerun()


def _agent_b_first_suggestion_ready(audit: dict, context_key: str) -> bool:
    """Only a fresh model-generated recommendation from THIS run unlocks Agent C.

    Historical review-memory markers, pending and failed queue records are not
    valid suggestions. Agent A can continue exploring while C is unlocked.
    """
    from cleaning_jobs import recommendation_manager, job_fingerprint

    run_id = audit.get("cleaning_recommendation_run_id")
    if not isinstance(run_id, str) or not run_id:
        return False
    try:
        key = job_fingerprint(context_key, [], run_id=run_id)
        snapshot = recommendation_manager().snapshot(key) or {}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return any(isinstance(entry, dict) and entry.get("status") == "ready"
               and entry.get("origin") != "review_memory"
               and bool(entry.get("suggestion"))
               for entry in (snapshot.get("entries") or {}).values())


def render_skills_memory_page() -> None:
    """Review learning capacity, disable historical recall and manage safe skills."""
    from skills_memory_manager import (
        overview, human_bytes, disable_rule, delete_skill, clear_learning_events,
    )

    st.subheader("Manage skills & memory")
    st.caption("Learns from named approvals and review feedback. Saved methods and generated skills are advisory; each new edit still needs source verification and your approval.")
    try:
        stats = overview()
    except (OSError, ValueError, TypeError, PermissionError) as exc:
        st.error(f"Local learning storage is unavailable: {exc}")
        return

    cols = st.columns(4)
    for col, title, amount in zip(cols,
                                  ("Reusable rules", "Learning events", "Generated skills", "Disk usage"),
                                  (stats["active_rules"], stats["learning_events"],
                                   stats["skills"], human_bytes(stats["storage_bytes"]))):
        col.metric(title, amount)
    import shutil
    free_bytes = shutil.disk_usage(Path(__file__).resolve().parent).free
    st.caption(
        f"{stats['stored_reviews']} saved approval record(s) · {human_bytes(free_bytes)} free disk. "
        "No fixed application storage quota. Method recall considers recent review history; "
        "skill generation uses the latest 300 learning events and requires repeated approvals."
    )

    st.markdown("**Generated skills**")
    if not stats["skill_details"]:
        st.caption("No learned skills yet. Repeated approved actions can create advisory skills through local Gemma.")
    else:
        for skill in stats["skill_details"]:
            with st.expander(f"{skill['title']} · {skill['approval_count']} approved examples"):
                st.write("When useful:", skill["when_to_suggest"] or "Not specified")
                st.write("Suggested method:", skill["recommended_method"] or "Not specified")
                st.write("Limit:", skill["do_not_assume"] or "Always validate the new source")
                st.caption(f"Updated {skill['updated_at'] or 'unknown'} · {human_bytes(skill['bytes'])}")
                confirm = st.checkbox("Confirm removal of this skill", key=f"memory_skill_confirm_{skill['family']}")
                if st.button("Remove skill", key=f"memory_skill_delete_{skill['family']}",
                             disabled=not confirm):
                    try:
                        if delete_skill(skill["family"]):
                            st.success("Skill removed. Historical approved decisions are retained.")
                            st.rerun()
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))

    st.markdown("**Previously approved methods**")
    st.caption("These methods are suggestions for future investigations, not pre-approved actions. Disabling recall preserves their audit trail and existing staged changes.")
    rules = stats["rules"]
    if rules:
        names = {r["id"]: (f"{r['title']} · {r['reviewed_at'][:10]} "
                           f"· {r['operation_count']} operation(s)") for r in rules if r["id"]}
        if names:
            selected = st.selectbox("Remembered rule", options=list(names),
                                    format_func=lambda key: names.get(key, key), key="memory_manage_rule")
            if st.checkbox("Confirm disabling this rule for future suggestions", key="memory_confirm_rule"):
                if st.button("Disable rule recall", key="memory_disable_rule"):
                    try:
                        if disable_rule(selected):
                            st.success("Historical rule disabled for future suggestions.")
                            st.rerun()
                        else:
                            st.info("This rule is already disabled or unavailable.")
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
    else:
        st.caption("No active approved methods to recall.")

    with st.expander("Learning-history controls"):
        st.caption("Removes preference-learning events only. It does not delete approvals, current edits, or generated skills.")
        confirm = st.text_input("Type CLEAR LEARNING to confirm", key="memory_clear_learning_confirmation")
        if st.button("Clear learning history", disabled=confirm.strip() != "CLEAR LEARNING",
                     key="memory_clear_learning"):
            try:
                clear_learning_events()
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.success("Learning events cleared; approved decisions preserved.")
                st.rerun()


def render_ui() -> None:
    st.set_page_config(page_title="Healthcare Data Quality AI", page_icon="🔬", layout="wide")
    # Style the named host container: an iframe cannot stick to the page by
    # styling its internal body. No JavaScript or external assets are injected.
    st.markdown(sticky_banner_css(), unsafe_allow_html=True)
    st.title("Healthcare Data Quality AI")
    st.caption("Inspect data, trace findings to evidence, and discuss reviewed cleaning with local Gemma.")
    st.info("Original uploads are never overwritten. Only explicitly authorized, supported operations can create a verified cleaned copy.")

    # Mount the one viewport-following host before the upload/run controls.
    # Without a saved or active audit it displays all three agents' honest idle
    # states, rather than hiding them until the user clicks Investigate.
    with st.container(key="hdq_investigation_banner"):
        live_banner_slot = st.empty()

    uploaded_file = st.file_uploader(
        "Upload one CSV dataset",
        type=["csv"],
        accept_multiple_files=False,
        help="One local CSV file (maximum 20 MiB) per investigation.",
    )
    uploaded = [uploaded_file] if uploaded_file is not None else []
    if not uploaded:
        render_pre_investigation_status(live_banner_slot, state="no_dataset")
        st.info("Upload a CSV dataset to inspect its structure and start the investigation.")
        render_saved_file_manager(None)
        with st.expander("Manage skills & memory"):
            render_skills_memory_page()
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

    if not render_full_dataset_section(paths, uploaded, header_reports, fingerprint):
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
    # An active investigation is idempotent: refuse a second run until this
    # run reaches a terminal status, even if a button rerun slips through.
    current_run_id = st.session_state.get("registered_investigation_run_id")
    current_run_state = investigation_manager().snapshot(current_run_id) if current_run_id else None
    if current_run_state and current_run_state["status"] == "running" and run_requested:
        run_requested = False

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
        run_id = uuid4().hex
        # A fresh run must not inherit Streamlit's previous cleaning context,
        # an old approval checkpoint, or the user's prior opt-in to memory.
        st.session_state.cleaning_fresh_review_run_id = run_id
        st.session_state.pop("cleaning_context_key", None)
        st.session_state.registered_investigation_run_id = run_id
        st.session_state.registered_investigation_context = (cache_key, fingerprint)
        # Clear prior cached audit before presenting this new read-only run.
        st.session_state.pop("audit_result", None)
        st.session_state.pop("active_audit_file", None)
        st.session_state.pop("audit_source", None)
        prepare_exploration_state(cache_key)
        st.session_state.exploration_runs = []
        st.session_state.exploration_stop = None
        st.session_state.exploration_error = None
        names = [item.name for item in uploaded]

        def on_baseline(partial):
            schedule_agent_b_once(cache_key, partial, paths, names, [])

        def on_finished(full):
            full["scientific_recommendations"] = build_scientific_recommendations(full)
            # Saving failure must not prevent the optional exploration worker.
            try:
                save_investigation(cache_key, full)
            except (OSError, TypeError, ValueError):
                pass  # In-memory registered-tool audit remains available.
            if include_exploration:
                start_live_exploration(full, cache_key, paths, names)

        def run_background(*args, **kwargs):
            with model_request_priority("investigation"):
                return run_initial_audit(*args, **kwargs)

        investigation_manager().start(run_id, paths, runner=run_background,
                                      on_baseline=on_baseline, on_finished=on_finished)

    # A worker-owned snapshot is read on the UI thread. The first available
    # baseline opens the three tabs; the full audit replaces it later.
    registered_run = st.session_state.get("registered_investigation_run_id")
    registered_state = None
    if (registered_run and
            st.session_state.get("registered_investigation_context") == (cache_key, fingerprint)):
        registered_state = investigation_manager().snapshot(registered_run)
    if registered_state:
        best_audit = registered_state.get("audit") or registered_state.get("partial_audit")
        current_snapshot = st.session_state.get("audit_result")
        if best_audit and (not isinstance(current_snapshot, dict) or
                          (current_snapshot.get("partial") and not best_audit.get("partial"))):
            st.session_state.audit_result = best_audit
            st.session_state.audit_fingerprint = fingerprint
            st.session_state.audit_cache_key = cache_key
            st.session_state.audit_source = "live"
            if registered_state.get("audit"):
                newer = saved_investigations(cache_key)
                if newer:
                    st.session_state.active_audit_file = str(newer[0])
        if registered_state.get("callback_warning"):
            st.warning(registered_state["callback_warning"])
        if registered_state["status"] == "failed":
            st.error("Gemma's investigation was interrupted. Completed QC findings, if any, remain available.")
            with st.expander("Investigation error details"):
                st.code(registered_state.get("error") or "No error recorded")


    audit = st.session_state.get("audit_result")
    if (not isinstance(audit, dict)
            or st.session_state.get("audit_cache_key") != cache_key
            or st.session_state.get("audit_fingerprint") != fingerprint):
        if registered_state and registered_state["status"] == "running":
            import streamlit.components.v1 as components
            from datetime import datetime
            with live_banner_slot.container():
                components.html(render_investigation_phase_banner_html(
                    phase=registered_state["phase"],
                    phase_count=3 if include_exploration else 2,
                    activity=registered_state["activity"],
                    started_epoch_ms=int(datetime.fromisoformat(
                        registered_state["started_at"]).timestamp() * 1000),
                ), height=170, scrolling=False)
                from cleaning_jobs import recommendation_manager, job_fingerprint
                snapshot = recommendation_manager().snapshot(job_fingerprint(
                    cache_key, [], run_id=registered_run))
                components.html(render_recommendation_banner_html(snapshot, idle=not bool(snapshot)),
                                height=151, scrolling=False)
                ready, total, responding = _agent_c_status_counts(snapshot)
                components.html(render_agent_c_banner_html(
                    ready_findings=ready, total_findings=total,
                    active_conversations=responding, idle=not bool(ready),
                ), height=151, scrolling=False)
            st.info("Agent A is investigating. QC findings and Agent C will become available as soon as verified results and Agent B advice arrive.")
            if hasattr(st, "fragment"):
                @st.fragment(run_every="3s")
                def wait_for_first_qc():
                    latest = investigation_manager().snapshot(registered_run)
                    if latest and (latest.get("partial_audit") or latest.get("audit") or latest["status"] == "failed"):
                        st.rerun()
                    st.caption("Running deterministic checks; the page updates automatically.")
                wait_for_first_qc()
        else:
            render_pre_investigation_status(
                live_banner_slot,
                state="saved_available" if previous_runs or legacy_session_result else "ready",
            )
            st.info("Choose one of the options above to view investigation results.")
        return

    if (registered_state and registered_state.get("audit") and audit.get("partial")):
        # Full QC arrived while Agent A/B were running. Adopt the snapshot
        # in-place instead of resetting the researcher's active tab/draft.
        audit = registered_state["audit"]
        st.session_state.audit_result = audit
    if st.session_state.get("audit_source") == "saved":
        st.caption("Showing saved results for these dataset contents; no new Gemma run was started.")
    # Reuse the same page-level host that displayed the pre-run idle cards.
    # Its Streamlit key must not be mounted twice; refreshing fragments update
    # the existing slot without changing the viewport-following CSS.
    if hasattr(st, "fragment"):
        live_status = st.fragment(run_every="2s")(render_live_investigation_status)
        live_status(audit, cache_key, header_reports, include_exploration, live_banner_slot)
    else:
        render_live_investigation_status(audit, cache_key, header_reports, include_exploration, live_banner_slot)

    results_tab_labels = ["Data quality findings", "Clean data with Gemma", "Supporting evidence", "Manage skills & memory"]
    # On Streamlit releases with stateful tabs, persist the selected tab across
    # *any* unavoidable full-app rerun. Older releases retain the old API;
    # fragment isolation below still prevents routine polling from remounting it.
    try:
        summary_tab, cleaning_tab, evidence_tab, skills_memory_tab = st.tabs(
            results_tab_labels, key="hdq_active_results_tab", on_change="rerun",
        )
    except TypeError:
        summary_tab, cleaning_tab, evidence_tab, skills_memory_tab = st.tabs(results_tab_labels)
    with summary_tab:
        for dataset_id, baseline in (audit.get("dataset_baselines") or {"dataset_1": audit.get("baseline") or {}}).items():
            label_index = int(dataset_id.rsplit("_", 1)[-1]) - 1
            dataset_name = uploaded[label_index].name if label_index < len(uploaded) else dataset_id
            st.subheader(f"Dataset {label_index + 1}: {dataset_name}")
            displayed_findings = render_baseline(dataset_id, baseline, source_path=paths[label_index])
            render_header_findings(paths[label_index], header_reports[label_index],
                                   start_index=displayed_findings + 1)
        if len(uploaded) > 1:
            st.warning("Full cross-dataset identifier reconciliation is not yet implemented as an investigation tool.")
        # No separate exploratory page: render novel, verified Gemma findings
        # alongside deterministic findings in the same results tab.
        st.divider()
        if hasattr(st, "fragment"):
            def refresh_explorations():
                sync_live_exploration(audit, cache_key)
                render_exploratory_findings(paths, cache_key, audit)
            st.fragment(run_every="3s")(refresh_explorations)()
        else:
            sync_live_exploration(audit, cache_key)
            render_exploratory_findings(paths, cache_key, audit)
    with evidence_tab:
        def refresh_evidence():
            sync_live_exploration(audit, cache_key)
            render_finding_evidence(
                audit, paths, [file.name for file in uploaded], header_reports,
                st.session_state.get("exploration_runs") or [],
            )
        if hasattr(st, "fragment"):
            st.fragment(run_every="3s")(refresh_evidence)()
        else:
            refresh_evidence()
        def refresh_export():
            live = sync_live_exploration(audit, cache_key)
            unfinished = bool(live and live["status"] == "running")
            st.download_button(
                "Download partial audit JSON" if unfinished else
                "Download complete audit JSON (may contain confidential data)",
                data=json.dumps(combined_audit_export(audit, cache_key), ensure_ascii=False, indent=2, default=str),
                file_name=("healthcare_data_quality_partial_investigation.json" if unfinished
                           else "healthcare_data_quality_combined_investigation.json"),
                mime="application/json",
            )
            if unfinished:
                st.caption("This is a partial snapshot. Gemma exploratory work is still running.")
        if hasattr(st, "fragment"):
            st.fragment(run_every="3s")(refresh_export)()
        else:
            refresh_export()
        st.caption("Keep downloaded audits in approved secure storage. They may contain sample identifiers and detailed source observations.")
    with skills_memory_tab:
        render_skills_memory_page()
    with cleaning_tab:
        # Lock the main cleaning workflow until Agent B has produced at least
        # one real, source-scoped recommendation for this investigation.
        # Historical remembered rules and queued model jobs never unlock it.
        def _wait_for_first_suggestion():
            current = st.session_state.get("audit_result") or audit
            if _agent_b_first_suggestion_ready(current, cache_key):
                st.rerun()
            st.caption("Agent C is locked until Agent B's first suggestion is ready.")

        if not _agent_b_first_suggestion_ready(st.session_state.get("audit_result") or audit, cache_key):
            st.info("Waiting for the first Agent B suggestion. Cleaning and Gemma discussion are unavailable until then.")
            if hasattr(st, "fragment"):
                st.fragment(run_every="3s")(_wait_for_first_suggestion)()
        else:
            def refresh_cleaning():
                # Render directly into the fragment's OWN root, not an external
                # st.empty/container. Streamlit replaces fragment-owned elements
                # on scoped reruns; explicitly emptying an external tab container
                # unmounts the entire finding and jumps the browser scroll.
                # Never run this editable review on a polling timer.
                current = st.session_state.get("audit_result") or audit
                sync_live_exploration(current, cache_key)
                render_cleaning_workspace(paths, uploaded, current, cache_key)

            if hasattr(st, "fragment"):
                st.fragment(refresh_cleaning)()
                st.fragment(run_every="3s")(_poll_agent_c_completion)()
            else:
                refresh_cleaning()


if __name__ == "__main__":
    render_ui()
