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
    return recommendation_manager().start(context_key, findings, run_id=run_id)


def _recommendation_preview(text: str, max_chars: int = 220) -> str:
    """Show a short, non-generative preview; preserve the full advice in details."""
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    segment = clean[:max_chars].rsplit(" ", 1)[0]
    return (segment or clean[:max_chars]).rstrip(" ,;:") + "…"


def _rerun_cleaning_fragment() -> None:
    """Refresh the cleaning UI without rerendering the entire Streamlit page.

    Cleaning cards live inside ``st.fragment``. A bare ``st.rerun()`` requests
    a *full-app* rerun, which disrupts the browser scroll position and active
    finding during a chat/approval click. If the page runs without fragment
    support (or a direct compatibility test calls this outside a fragment),
    fall back to the normal full rerun.
    """
    try:
        st.rerun(scope="fragment")
    except TypeError:
        # Older Streamlit/mocked ``st.rerun`` without the ``scope`` keyword.
        st.rerun()
    except Exception as exc:
        # Depending on the Streamlit release, a fragment-scoped rerun during
        # the *initial full-app pass* raises StreamlitAPIException or the
        # more specific StreamlitInvalidLayoutContextError. Both indicate a
        # rerun-context mismatch, NOT a failed cleaning/approval operation.
        # Fall back to an app-scoped rerun; do not mask unrelated failures.
        if type(exc).__name__ not in {
            "StreamlitAPIException", "StreamlitInvalidLayoutContextError",
        }:
            raise
        st.rerun()


def _render_live_finding_assignment(fid: str) -> None:
    """Update the small ownership caption without remounting Agent C input."""
    from interactive_chat_jobs import interactive_chat_manager
    pending = st.session_state.get("cleaning_pending_gemma_b", {}).get(fid)
    if not pending:
        return
    manager = interactive_chat_manager()
    role = manager.agent_for(pending["job_id"]) if hasattr(manager, "agent_for") else None
    assigned = role if role in {"A", "B", "C"} else pending.get("assigned_agent", "C")
    pending["assigned_agent"] = assigned
    if assigned in {"A", "B"}:
        st.caption(f"Idle Agent {assigned} is now assisting with this finding "
                   "while C handles another request. Human approval is still required.")
    else:
        st.caption("Agent C is preparing the scoped cleaning plan or waiting "
                   "for its turn. No working CSV is being modified.")


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
        review_status = st.session_state.cleaning_review_statuses.get(fid)
        # A background job must never replace a human-selected recommendation
        # or resurrect a completed/rejected review.
        if (previous.get("_source") != "manual" and
                review_status in {None, "Awaiting review", "Agent B suggestion ready"}):
            st.session_state.cleaning_recommendations[fid] = {
                "suggestion": entry["suggestion"],
                "decision_needed": entry.get("decision_needed", ""),
                "_source": "background",
            }
            st.session_state.cleaning_review_statuses[fid] = "Agent B suggestion ready"

    suggestion = st.session_state.cleaning_recommendations.get(fid)
    if suggestion:
        st.markdown("**Agent B — proposed cleaning action (not approved)**")
        advice = str(suggestion["suggestion"])
        decision = str(suggestion.get("decision_needed") or "")
        needs_expansion = (len(" ".join(advice.split())) > 220 or
                           len(" ".join(decision.split())) > 170)
        show_full = (st.checkbox("Show full recommendation", key=f"review_full_advice_{fid}")
                     if needs_expansion else False)
        st.write(advice if show_full else _recommendation_preview(advice))
        if decision:
            if show_full:
                st.write("**Reason / evidence limit:**", decision)
            else:
                st.caption("Reason / evidence limit: " + _recommendation_preview(decision, 170))
    else:
        background_status = entry.get("status")
        if background_status == "running":
            st.caption("Agent B is preparing this recommendation. "
                       "This section updates automatically when it is ready.")
        elif background_status == "pending":
            st.caption("Agent B has queued this recommendation. "
                       "This section updates automatically when it is ready.")
        elif background_status == "failed":
            st.error("Agent B could not finish this recommendation. "
                     "No automatic retry occurs; inspect its failure details.")
        else:
            st.caption("Waiting for this finding's Agent B recommendation. "
                       "This section updates automatically.")
        return

    if any(h.get("dataset_id") == ds_id
           for h in st.session_state.cleaning_history):
        st.caption("This suggestion uses the original investigation. The next "
                   "verified edit will be checked against the current in-memory stage.")

    b_pending = fid in st.session_state.cleaning_pending_gemma_b
    selected_mode = st.session_state.cleaning_discussion_mode_by_finding.get(fid)
    previously_applied = any(
        event.get("finding_id") == fid for event in st.session_state.cleaning_history
    )
    review_status = st.session_state.cleaning_review_statuses.get(fid)
    if (selected_mode not in {"accepted", "alternative"} and
            not st.session_state.cleaning_plans_by_finding.get(fid) and
            not previously_applied and review_status != "Reviewed — no change"):
        yes, no = st.columns(2)
        if yes.button("Yes — prepare this action", key=f"review_yes_{fid}", disabled=b_pending):
            st.session_state.cleaning_discussion_mode_by_finding[fid] = "accepted"
            st.session_state.cleaning_review_statuses[fid] = "Agreed in principle"
            st.session_state.cleaning_decisions.append({
                "finding_id": fid,
                "decision": "approach_agreed_not_execution",
                "approach": suggestion["suggestion"],
            })
            _queue_agent_c_for_finding(
                finding, source, uploaded, dataset_paths,
                "I accept Agent B's action for this finding and authorize DRAFTING its "
                "exact structured change plan only. Return operation=replace_cells or "
                "rename_headers with source-verified before/after values and positions "
                "immediately if sufficiently supported; do NOT ask for another "
                "authorization to prepare a plan. The app will separately generate "
                "and test R against isolated affected values, and will ask for ONE "
                "final approval before insertion. Do not invent missing scientific "
                "parameters, fabricate data, or edit outside this finding.",
            )
            # Clicking Yes is an explicit action; refresh the parent layout so
            # the Agent C discussion can appear. Do not full-rerun on polling.
            st.rerun()
        if no.button("No — discuss another action", key=f"review_no_{fid}", disabled=b_pending):
            st.session_state.cleaning_discussion_mode_by_finding[fid] = "alternative"
            st.session_state.cleaning_review_statuses[fid] = "Alternative requested"
            st.session_state.cleaning_plans_by_finding.pop(fid, None)
            st.session_state.cleaning_r_previews_by_finding.pop(fid, None)
            st.session_state.cleaning_decisions.append({
                "finding_id": fid,
                "decision": "suggestion_declined_no_edit",
                "approach": suggestion["suggestion"],
            })
            # No Gemma request until the user types or selects a clarification.
            st.rerun()


def render_cleaning_workspace(
    dataset_paths: list[Path], uploaded: list, audit: dict, context_key: str,
    gemma_runs: list[dict] | None = None,
) -> None:
    """Agent C reviews Agent B advice, negotiates scoped changes and R approval."""
    from cleaning_review import (
        ask_gemma_for_review_options, indexed_review_catalog, review_question,
    )
    from healthcare_data_quality.finding_cleaning import (
        allowed_positions, build_cleaning_findings, validate_scoped_plan,
    )
    from healthcare_data_quality.in_memory_cleaning import (
        apply_verified_in_memory, assess_verified_in_memory, build_final_download,
        review_completion, apply_approved_column_deletions, _records, sha256_bytes,
    )

    from healthcare_data_quality.scoped_r_execution import (
        prepare_and_execute_scoped_r_preview, verify_prepared_r_insertion,
        verify_r_against_current_staged_values,
    )
    from healthcare_data_quality.r_diagnostics import (
        diagnose_r_preview_failure, run_r_environment_self_test,
    )
    st.subheader("Agent C — Interactive cleaning")
    st.caption("Review findings one by one. Agent C tests each exact R operation; approval adds its "
               "verified result to session memory only. Download the final cleaned dataset at "
               "the bottom after every finding and completed Gemma check has been reviewed. "
               "Original uploads remain unchanged.")
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
        st.session_state.cleaning_batch_rejected_by_finding = set()
        st.session_state.cleaning_r_previews_by_finding = {}
        st.session_state.cleaning_r_errors_by_finding = {}
        st.session_state.cleaning_r_diagnostics_by_finding = {}
        st.session_state.cleaning_r_selfchecks_by_finding = {}
        st.session_state.cleaning_plan_retry_once = set()
        st.session_state.cleaning_recommendations = {}
        st.session_state.cleaning_review_statuses = {}
        st.session_state.cleaning_discussion_mode_by_finding = {}
        st.session_state.cleaning_decisions = []
        st.session_state.cleaning_result = None
        st.session_state.cleaning_working_paths = {}  # legacy field; no output files created
        st.session_state.cleaning_staged_bytes_by_dataset = {}
        st.session_state.cleaning_approved_columns_by_dataset = {}
        st.session_state.cleaning_history = []
        st.session_state.cleaning_approved_reviewer = ""
        st.session_state.cleaning_rebased_previews_by_finding = {}
        st.session_state.cleaning_completed_audits_by_finding = {}
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
            ("cleaning_batch_rejected_by_finding", set()),
            ("cleaning_r_previews_by_finding", {}),
            ("cleaning_r_errors_by_finding", {}),
            ("cleaning_r_diagnostics_by_finding", {}),
            ("cleaning_r_selfchecks_by_finding", {}),
            ("cleaning_plan_retry_once", set()),
            ("cleaning_recommendations", {}),
            ("cleaning_review_statuses", {}),
            ("cleaning_discussion_mode_by_finding", {}),
            ("cleaning_decisions", []),
            ("cleaning_working_paths", {}),
            ("cleaning_staged_bytes_by_dataset", {}),
            ("cleaning_approved_columns_by_dataset", {}),
            ("cleaning_history", []),
            ("cleaning_approved_reviewer", ""),
            ("cleaning_rebased_previews_by_finding", {}),
            ("cleaning_completed_audits_by_finding", {}),
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
        st.info("No findings or completed Gemma checks require review in this audit. "
                "The final download becomes available after the investigation finishes.")

    # The reviewer identifier is requested inside the FIRST approval form,
    # next to the action it authorizes. Once verified staging succeeds, it is
    # stored for this investigation and never requested again.
    session_reviewer = st.session_state.get("cleaning_approved_reviewer") or ""
    if session_reviewer:
        st.caption(f"Reviewer for this cleaning session: {session_reviewer}")

    # Agent B saves one suggestion per finding in the background. Agent C
    # handles interactive discussion, plans and authorization review.
    # The individual Agent B suggestion panels now poll independently.
    # This manual button is retained only as a compatibility fallback.
    if st.button("Check for new Agent B results", key="hdq_refresh_cleaning_suggestions"):
        _rerun_cleaning_fragment()
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
            existing = st.session_state.cleaning_recommendations.get(result_id) or {}
            review_status = st.session_state.cleaning_review_statuses.get(result_id)
            if (existing.get("_source") == "manual" or
                    review_status not in {None, "Awaiting review", "Agent B suggestion ready"}):
                continue
            st.session_state.cleaning_recommendations[result_id] = {
                "suggestion": entry["suggestion"],
                "decision_needed": entry.get("decision_needed", ""),
                "_source": "background",
            }
            st.session_state.cleaning_review_statuses[result_id] = "Agent B suggestion ready"
    else:
        st.caption("No saved Agent B recommendations belong to this investigation. "
                   "Agent C opens when Agent B produces its first suggestion. "
                   "Recommendations are only generated automatically during a new full investigation.")

    # Intermediate results remain in session memory; no CSV is offered or
    # written while findings are still being reviewed.
    history_all = st.session_state.cleaning_history

    ready_ids = {item_id for item_id, entry in ((state or {}).get("entries") or {}).items()
                 if entry.get("status") == "ready"}
    st.caption("Agent B suggestions appear automatically inside each finding as they finish. "
               "The status cards above show live overall progress; Agent C's input stays stable.")

    numbered = indexed_review_catalog(catalog)
    last_group = None
    for label, finding in numbered:
        fid = finding["id"]
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
        status = st.session_state.cleaning_review_statuses.get(fid, "Awaiting review")
        applied = [event for event in history_all if event["finding_id"] == fid]
        if applied:
            if st.session_state.cleaning_review_statuses.get(fid) == "Resolved by earlier approved edit":
                status = "Resolved by earlier approved edit — no additional cells changed"
            else:
                status = f"Staged in memory ({len(applied)} approved edit{'s' if len(applied) != 1 else ''})"
        # Unfinished discussions stay open so submitted text remains visible.
        # After a verified in-memory stage, remount the expander under a
        # completed label and collapse it. The immutable approval history,
        # conversation and R code remain available on demand below.
        remaining_batch = st.session_state.cleaning_batch_queue_by_finding.get(fid, [])
        current_plan = st.session_state.cleaning_plans_by_finding.get(fid)
        reviewed_no_change = (status == "Reviewed — no change" and not applied)
        completed = (bool(applied) or reviewed_no_change) and not current_plan and not remaining_batch
        active_conversation = (not completed and bool(
            st.session_state.cleaning_messages_by_finding.get(fid)
            or fid in st.session_state.cleaning_pending_gemma_b
            or current_plan or remaining_batch
        ))
        completed_suffix = " · Staged ✓" if completed else ""
        with st.expander(f"{label} · {finding['title']}{completed_suffix}",
                         expanded=active_conversation):
            st.caption(f"{status} · {finding['source']} · {finding.get('name', '')}")
            if completed:
                if reviewed_no_change:
                    st.markdown("**Completed — reviewed, no change required.**")
                    st.caption("Review recorded in session memory. No cell, header, or column "
                               "was changed or queued for deletion; the original data is preserved.")
                    if st.button("Reopen this review", key=f"review_reopen_{fid}"):
                        st.session_state.cleaning_review_statuses.pop(fid, None)
                        st.session_state.cleaning_discussion_mode_by_finding.pop(fid, None)
                        st.session_state.cleaning_decisions.append({
                            "finding_id": fid, "decision": "no_change_review_reopened",
                        })
                        _rerun_cleaning_fragment()
                    # No edit was staged; only the review decision is complete.
                    # Keep the same collapsed, read-only completion treatment as
                    # approved edit findings, without claiming a data mutation.
                    continue
                staged_count = sum(int(event.get("changes", 0) or 0) for event in applied)
                reused_count = sum(int(event.get("already_resolved", 0) or 0)
                                   for event in applied)
                if fid in st.session_state.cleaning_batch_rejected_by_finding:
                    st.warning("Only part of the multi-action request was approved. "
                               "The rejected step was NOT applied; see the audit below.")
                else:
                    st.markdown("**Completed — approved and verified in session memory.**")
                st.caption(f"{staged_count} reviewed change(s)"
                           + (f" · {reused_count} correction(s) reused" if reused_count else "")
                           + " · No intermediate CSV created")
                if any(event.get("operation") == "drop_column" for event in applied):
                    st.caption("Whole-column deletion is approved for final download; "
                               "other in-memory edits retain original column coordinates.")
                if st.checkbox("Show completed finding audit",
                               key=f"review_completed_audit_{fid}"):
                    for event in applied:
                        st.write("**Approved edit:**", {
                            "operation": event.get("operation"),
                            "changes": event.get("changes", 0),
                            "already_resolved": event.get("already_resolved", 0),
                            "reviewer": event.get("reviewer"),
                            "staged_sha256": event.get("staged_sha256"),
                            "r_code_sha256": event.get("r_code_sha256"),
                            "modified_positions": event.get("modified_positions", []),
                        })
                    completed_audit = st.session_state.cleaning_completed_audits_by_finding.get(fid)
                    if completed_audit:
                        changes = completed_audit.get("changes", [])
                        if changes:
                            st.dataframe(pd.DataFrame(changes), hide_index=True, width="stretch")
                        if completed_audit.get("r_function"):
                            st.markdown("**Verified R cleaning code**")
                            st.code(completed_audit["r_function"], language="r")
                        if completed_audit.get("insertion_r_function"):
                            st.markdown("**R insertion specification (not executed)**")
                            st.code(completed_audit["insertion_r_function"], language="r")
                    for message in st.session_state.cleaning_messages_by_finding.get(fid, []):
                        if not message.get("automatic"):
                            with st.chat_message(message["role"]):
                                st.write(message["content"])
                # Do not sync Agent C, offer suggestions, retry generation,
                # or show any editing/approval widgets after a staged action.
                continue
            evidence = finding.get("evidence") or []
            if st.checkbox("Show why this was flagged and original evidence",
                           key=f"review_evidence_{fid}"):
                if finding.get("method"):
                    st.write("**Detection:**", finding["method"])
                st.write("**Decision needed:**", review_question(finding))
                if evidence:
                    st.dataframe(pd.DataFrame(evidence), hide_index=True, width="stretch")
            cells, headers = allowed_positions(finding, source) if source else (set(), set())
            if source:
                version = sum(h["dataset_id"] == ds_id for h in history_all)
                st.caption(f"In-memory cleaning stage v{version} · {len(cells)} evidenced cells"
                           + (f" · {len(headers)} evidenced headers" if headers else ""))
                if version:
                    st.caption("Earlier approved edits are stored in memory. Matching corrections "
                               "are reused; different replacements require explicit review.")
            else:
                st.info("This completed check has no associated editable dataset. "
                        "You can review or discuss it, but it cannot authorize a data change.")
            if not cells and not headers and source:
                st.caption("No editable cell/header positions are evidenced for this result. "
                           "A correction requires a separate validated investigation.")

            # Job results are consumed by the UI, never by the worker. A
            # submitted user message is already visible while inference runs.
            _sync_gemma_conversations(finding, source)
            # Auto-refresh ONLY this finding's Agent B suggestion, not Agent C's
            # chat, form, scroll position, or other findings. Nested fragments
            # have independent timer reruns in supported Streamlit releases.
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
            clarification_needed = (mode == "accepted" and not b_pending and not plan
                                    and bool(transcript) and transcript[-1]["role"] == "assistant")
            # Prefer structured Gemma choices; recognize the earlier saved
            # column-vs-cell prose question without another model call.
            last_agent_c_reply = (
                transcript[-1] if transcript and transcript[-1].get("role") == "assistant"
                else None
            )
            choice_labels = (agent_c_clarification_options(
                last_agent_c_reply.get("content", ""),
                last_agent_c_reply.get("clarification_options"),
            ) if last_agent_c_reply and not plan else [])
            last_assistant_message = (str(transcript[-1].get("content", ""))
                                      if clarification_needed else "")
            if (clarification_needed and not choice_labels
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
                _rerun_cleaning_fragment()
            if plan and transcript:
                if st.checkbox("Show Agent C planning discussion", key=f"review_chat_history_{fid}"):
                    for item in transcript:
                        if item.get("automatic"):
                            st.caption("Agent C retried the accepted plan automatically.")
                        else:
                            with st.chat_message(item["role"]):
                                if item.get("role") == "assistant" and item.get("assigned_agent") in {"A", "B"}:
                                    st.caption(f"Agent {item['assigned_agent']} assisted while C was busy")
                                st.write(item["content"])
            elif (mode == "alternative" or clarification_needed) and transcript:
                st.markdown("**Agent C discussion**")
                for item in transcript:
                    if not item.get("automatic"):
                        with st.chat_message(item["role"]):
                            st.write(item["content"])
            if b_pending:
                if hasattr(st, "fragment"):
                    st.fragment(run_every="3s")(_render_live_finding_assignment)(fid)
                else:
                    _render_live_finding_assignment(fid)
            if fid not in ready_ids:
                st.caption("Agent C unlocks once Agent B's recommendation is ready.")
            if (choice_labels and fid in ready_ids and not plan and not b_pending
                    and mode in {"alternative", "accepted"}):
                st.markdown("**Choose your answer:**")
                st.caption("One click replies to Agent C. It does NOT approve a data edit; "
                           "the reviewed cleaning action still needs its own approval.")
                choice_columns = st.columns(2)
                for option_index, (choice_column, choice_label) in enumerate(
                        zip(choice_columns, choice_labels), start=1):
                    if choice_column.button(choice_label,
                            key=f"review_clarify_{fid}_{option_index}",
                            disabled=b_pending):
                        # The selected label is visible and explicit; do not
                        # send any hidden automatic selection. History retains
                        # the original request so other requested actions are
                        # not inadvertently discarded by this clarification.
                        answer = (
                            f"I choose option {option_index}: {choice_label}. "
                            "This is my clarification of the scope in your "
                            "last question, not an approval to modify data. "
                            "Keep my other previously requested actions from "
                            "this finding when preparing the validated plan(s). "
                            "Do not execute or stage anything before the "
                            "separate explicit review approvals."
                        )
                        _queue_agent_c_for_finding(
                            finding, source, uploaded, dataset_paths, answer)
                        _rerun_cleaning_fragment()
            elif mode == "alternative" and not plan and not b_pending:
                st.info("Tell Agent C what you want instead. Nothing is sent to Gemma until you click Send to Agent C.")
            if (fid in ready_ids and not plan and
                    st.session_state.cleaning_review_statuses.get(fid) != "Reviewed — no change" and
                    (mode == "alternative" or clarification_needed)):
                with st.form(f"review_chat_form_{fid}", clear_on_submit=True):
                    message = st.text_area(
                        "Clarify requested edit" if clarification_needed
                        else "Discuss another action with Agent C",
                        placeholder="Ask a question or specify exact desired changes…",
                        key=f"review_message_{fid}", height=80,
                    )
                    send = st.form_submit_button("Send to Agent C", disabled=b_pending)
                if send and message.strip():
                    _queue_agent_c_for_finding(finding, source, uploaded, dataset_paths,
                                               message.strip())
                    _rerun_cleaning_fragment()
                elif send:
                    st.warning("Enter a question or cleaning decision first.")

            plan = st.session_state.cleaning_plans_by_finding.get(fid)
            if plan:
                queued = len(st.session_state.cleaning_batch_queue_by_finding.get(fid, []))
                total_ops = st.session_state.cleaning_batch_total_by_finding.get(fid, 1)
                if total_ops > 1:
                    step = total_ops - queued
                    st.info(f"Multi-action request: reviewing operation {step} of "
                            f"{total_ops}. {queued} more action(s) remain queued. "
                            "Every action needs independent approval.")
            if plan and source and plan.get("operation") == "drop_column":
                # Gemma only proposes a structural edit; this UI gives the
                # researcher one independent, human-controlled approval.
                try:
                    checked = validate_scoped_plan(plan, finding, source)
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
                    st.warning(
                        f"Agent C proposes removing the ENTIRE original CSV column "
                        f"{column_position} ({item['before']!r}) from all "
                        f"{len(current_rows) - 1} data rows in the FINAL download. "
                        "This is not limited to the cells flagged by QC. "
                        "The original upload and earlier approved cell edits stay unchanged."
                    )
                    st.caption("The physical column position and original header have been "
                               "verified against this finding. Deletion happens only at export, "
                               "so later cell edits retain stable original column numbers.")
                with st.form(f"review_drop_column_{fid}"):
                    reviewer_for_approval = st.session_state.get("cleaning_approved_reviewer") or ""
                    if reviewer_for_approval:
                        st.caption("Reviewer: " + reviewer_for_approval)
                    else:
                        reviewer_for_approval = st.text_input(
                            "Reviewer identifier", key=f"review_drop_reviewer_{fid}", max_chars=120)
                    approved = st.form_submit_button("Approve deletion of entire column",
                                                     disabled=not can_delete)
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
                            checked_again = validate_scoped_plan(plan, finding, source)
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
                            _rerun_cleaning_fragment()
                        except (ValueError, PermissionError, OSError, AssertionError, csv.Error) as exc:
                            st.error(f"Column deletion approval failed; no action staged: {exc}")
                if st.button("Cancel this deletion proposal and discuss another action",
                             key=f"review_cancel_drop_{fid}"):
                    _cancel_agent_c_batch(fid, status="Column deletion rejected")
                    st.session_state.cleaning_discussion_mode_by_finding[fid] = "alternative"
                    _rerun_cleaning_fragment()
            if plan and source and plan.get("operation") != "drop_column":
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
                                _rerun_cleaning_fragment()
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
                        _rerun_cleaning_fragment()
                elif prepared:
                    st.markdown("**R result verified — review before insertion**")
                    st.caption("R has run on ONLY the finding's affected values. Your approval "
                               "will insert its verified result into session memory, NOT create "
                               "a CSV file. Biological suitability still requires your review.")
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
                    st.dataframe(pd.DataFrame(affected), hide_index=True, width="stretch")
                    if st.checkbox("Review R cleaning and insertion code", key=f"review_r_code_{fid}"):
                        st.markdown("**1. R cleaning (already executed in isolation)**")
                        st.code(prepared["r_function"], language="r")
                        st.markdown("**2. R insertion (vectorized, reproducible specification)**")
                        st.code(prepared["insertion_r_function"], language="r")
                        st.caption("The insertion function accepts the verified 5-column "
                                   "change table as data, grouping edits by column instead of "
                                   "generating a command for every cell. The application "
                                   "independently verifies and stages approved changes in "
                                   "memory; this displayed R insertion code is not executed.")
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
                            st.warning("Another approved finding previously changed these positions. "
                                       "Review the current values and proposed replacements before overriding them.")
                            prior_by_cell = {}
                            for past_event in history_all:
                                if past_event.get("dataset_id") == ds_id:
                                    for coord in past_event.get("modified_positions", []):
                                        if isinstance(coord, dict):
                                            prior_by_cell[(coord.get("csv_row"), coord.get("column"))] = (
                                                past_event.get("finding_id") or "Earlier review")
                            st.dataframe(pd.DataFrame([
                                {**conflict, "earlier_finding": prior_by_cell.get(
                                    (conflict["csv_row"], conflict["column"]), "Prior approved edit")}
                                for conflict in review["conflicts"]
                            ]), hide_index=True, width="stretch")
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
                            if st.checkbox("I authorize replacing these previously approved "
                                           "values with the verified new values",
                                           key=f"review_override_conflicts_{fid}_{review_identity}"):
                                approved_rebase = cached
                                if st.session_state.get(f"review_r_code_{fid}"):
                                    st.markdown("**R recheck against latest staged values**")
                                    st.code(cached["r_function"], language="r")
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
                    with st.form(f"review_execute_r_{fid}"):
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
                                key=f"cleaning_reviewer_once_{session_context}",
                                max_chars=120,
                            )
                        execute = st.form_submit_button("Approve & Stage in memory",
                                                        disabled=not can_approve)
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
                                        st.session_state.cleaning_plans_by_finding.pop(other_id, None)
                                        st.session_state.cleaning_r_previews_by_finding.pop(other_id, None)
                                        st.session_state.cleaning_r_errors_by_finding.pop(other_id, None)
                                        st.session_state.cleaning_r_diagnostics_by_finding.pop(other_id, None)
                                        st.session_state.cleaning_r_selfchecks_by_finding.pop(other_id, None)
                                        st.session_state.cleaning_rebased_previews_by_finding.pop(other_id, None)
                                        if other_id != fid:
                                            # A different finding may have generated stale plans.
                                            st.session_state.cleaning_batch_queue_by_finding.pop(other_id, None)
                                _advance_agent_c_batch(fid)
                                _rerun_cleaning_fragment()
                            except (OSError, ValueError, PermissionError, AssertionError,
                                    RuntimeError, csv.Error, KeyError) as exc:
                                st.error(f"In-memory insertion failed; no changes were staged: {exc}")
                if st.button("Reject this proposed insertion", key=f"review_reject_plan_{fid}"):
                    st.session_state.cleaning_discussion_mode_by_finding.pop(fid, None)
                    _cancel_agent_c_batch(fid, status="R insertion rejected")
                    st.session_state.cleaning_r_previews_by_finding.pop(fid, None)
                    st.session_state.cleaning_r_errors_by_finding.pop(fid, None)
                    st.session_state.cleaning_r_diagnostics_by_finding.pop(fid, None)
                    st.session_state.cleaning_r_selfchecks_by_finding.pop(fid, None)
                    st.session_state.cleaning_decisions.append({
                        "finding_id": fid, "decision": "r_result_rejected_no_edit",
                        "changes": len(plan["changes"]),
                    })
                    _rerun_cleaning_fragment()

            if (not plan and not applied and not b_pending and not remaining_batch and
                    st.session_state.cleaning_review_statuses.get(fid) != "Reviewed — no change" and
                    st.button("Mark reviewed — no change required now",
                              key=f"review_no_action_{fid}")):
                st.session_state.cleaning_review_statuses[fid] = "Reviewed — no change"
                st.session_state.cleaning_decisions.append({
                    "finding_id": fid, "decision": "no_change_authorized_or_executed",
                })
                _rerun_cleaning_fragment()


    # This is deliberately the ONLY cleaned-data download control in this
    # workspace. Finalize all QC/Gemma findings AND zero-issue checks first.
    st.markdown("---")
    st.markdown("### Final cleaned dataset")
    reviewed, total = review_completion(catalog, st.session_state.cleaning_review_statuses)
    st.caption(f"{reviewed} of {total} findings/checks reviewed · "
               f"{len(history_all)} verified edit(s) staged in session memory")
    investigation_running = bool(audit.get("partial"))
    if run_id:
        current_run = investigation_manager().snapshot(run_id)
        investigation_running = investigation_running or bool(
            current_run and current_run.get("status") == "running")
        from exploration_jobs import exploration_manager
        exploration_state = exploration_manager().snapshot(run_id)
        investigation_running = investigation_running or bool(
            exploration_state and exploration_state.get("status") == "running")
    recommendation_entries = ((state or {}).get("entries") or {})
    recommendations_running = any(
        item.get("status") in {"running", "pending"}
        for item in recommendation_entries.values() if isinstance(item, dict))
    if reviewed != total or investigation_running or recommendations_running:
        pending_reasons = []
        if reviewed != total:
            pending_reasons.append(f"review the remaining {total - reviewed} finding(s)/check(s)")
        if investigation_running:
            pending_reasons.append("finish the investigation")
        if recommendations_running:
            pending_reasons.append("finish Agent B's recommendations")
        st.info("Final download unlocks when you " + ", ".join(pending_reasons) + ".")
    else:
        try:
            datasets_for_download = []
            for i, file in enumerate(uploaded):
                ds_id = f"dataset_{i+1}"
                staged_bytes = st.session_state.cleaning_staged_bytes_by_dataset.get(ds_id)
                original = dataset_paths[i].read_bytes()
                data = staged_bytes if staged_bytes is not None else original
                approvals = st.session_state.cleaning_approved_columns_by_dataset.get(ds_id, [])
                data = apply_approved_column_deletions(original, data, approvals)
                datasets_for_download.append((ds_id, file.name, data))
            file_name, mime, data = build_final_download(datasets_for_download)
        except (ValueError, PermissionError, AssertionError, OSError, csv.Error) as exc:
            st.error(f"Final in-memory dataset could not be verified: {exc}")
        else:
            if hasattr(st, "download_button"):
                st.download_button(
                    "Download final cleaned dataset" if len(uploaded) == 1
                    else "Download final cleaned datasets (ZIP)",
                    data=data, file_name=file_name, mime=mime,
                    key=f"cleaning_download_final_{session_context}",
                )
            st.caption("Download now: staged changes exist only in this browser session. "
                       "A refresh or server restart may discard them. No working CSV was saved.")
    if history_all:
        with st.expander("Approved in-memory changes and reviewer decisions", expanded=False):
            if hasattr(st, "dataframe"):
                st.dataframe(pd.DataFrame(history_all), hide_index=True, width="stretch")
                if st.session_state.cleaning_decisions:
                    st.dataframe(pd.DataFrame(st.session_state.cleaning_decisions),
                                 hide_index=True, width="stretch")
            else:
                st.write(history_all)


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
    st.session_state.cleaning_plans_by_finding.pop(fid, None)
    st.session_state.setdefault("cleaning_batch_queue_by_finding", {}).pop(fid, None)
    st.session_state.setdefault("cleaning_batch_total_by_finding", {}).pop(fid, None)
    st.session_state.cleaning_r_previews_by_finding.pop(fid, None)
    st.session_state.cleaning_r_errors_by_finding.pop(fid, None)
    st.session_state.cleaning_r_diagnostics_by_finding.pop(fid, None)
    st.session_state.cleaning_r_selfchecks_by_finding.pop(fid, None)
    scoped = {"finding_id": fid, "dataset_id": ds_id,
              "title": finding["title"], "source": finding["source"],
              "method": finding.get("method"), "evidence": evidence[:32],
              "allowed_cell_positions": sorted(valid_cells)[:64],
              "allowed_header_positions": sorted(valid_headers),
              "all_affected_positions_count": len(valid_cells),
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
        st.session_state.setdefault("cleaning_batch_queue_by_finding", {}).pop(fid, None)
        st.session_state.setdefault("cleaning_batch_total_by_finding", {}).pop(fid, None)


def _cancel_agent_c_batch(fid: str, *, status: str) -> None:
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


def _record_gemma_b_response(fid: str, finding: dict, source: Path | None,
                             response: dict) -> None:
    """Validate EVERY requested operation before queueing any for human review.

    Multiple operations are independent plans. Approval of the first never
    silently discards the second, and no action executes at proposal time.
    """
    from healthcare_data_quality.finding_cleaning import expand_uniform_evidence_plan
    if any(event.get("finding_id") == fid
           for event in st.session_state.get("cleaning_history", [])):
        # A late completion must never reopen an already-approved finding.
        return
    thread = st.session_state.cleaning_messages_by_finding.setdefault(fid, [])
    reply = str(response["reply"])
    proposals = response.get("operations") or []
    if not proposals and response.get("operation") != "none":
        proposals = [{"operation": response["operation"],
                      "dataset_id": response["dataset_id"],
                      "changes": response["changes"]}]
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
        if len(proposals) > 4:
            raise ValueError("At most four operations may be reviewed together")
        verified_plans = []
        seen_types = set()
        recommendation = st.session_state.get("cleaning_recommendations", {}).get(fid) or {}
        full_scope_action = str(recommendation.get("suggestion") or "")
        accepted_full_scope = (
            st.session_state.cleaning_discussion_mode_by_finding.get(fid) == "accepted"
            and any(word in full_scope_action.casefold().split()
                    for word in ("all", "every", "entire"))
        )
        expanded_total = 0
        for raw in proposals:
            if not isinstance(raw, dict):
                raise ValueError("An operation is not a structured plan")
            op = raw.get("operation")
            if op in seen_types:
                raise ValueError("Repeated operation types need separate requests")
            seen_types.add(op)
            model_plan = _normalize_agent_c_current_before_values(raw, finding, source)
            checked, expanded_count = expand_uniform_evidence_plan(
                model_plan, finding, source,
                full_finding_accepted=accepted_full_scope and len(proposals) == 1)
            verified_plans.append(checked)
            expanded_total += expanded_count
        # Reject conflicting per-column requests rather than running an edit
        # which disappears when its column is later omitted at export.
        drop_positions = {item["position"] for p in verified_plans
                          if p["operation"] == "drop_column"
                          for item in p["changes"]}
        for p in verified_plans:
            if p["operation"] == "replace_cells" and any(
                    item["column"] in drop_positions for item in p["changes"]):
                raise ValueError("One plan replaces a value in a column also "
                                 "scheduled for deletion; clarify which action to keep")
            if p["operation"] == "rename_headers" and any(
                    item["position"] in drop_positions for item in p["changes"]):
                raise ValueError("One plan renames a column also scheduled for deletion")
        st.session_state.cleaning_plans_by_finding[fid] = verified_plans[0]
        remaining = verified_plans[1:]
        st.session_state.setdefault("cleaning_batch_queue_by_finding", {})[fid] = remaining
        st.session_state.setdefault("cleaning_batch_total_by_finding", {})[fid] = len(verified_plans)
        st.session_state.cleaning_review_statuses[fid] = "Exact plan awaiting approval"
        if expanded_total:
            first = verified_plans[0]
            reply += (f"\n\nThe independently verified evidence expands the uniform "
                      f"correction to all {len(first['changes'])} affected cells "
                      f"({expanded_total} additional coordinates).")
        if len(verified_plans) > 1:
            summary = "; ".join(
                f"{idx}. {p['operation']} ({len(p['changes'])} change(s))"
                for idx, p in enumerate(verified_plans, 1))
            reply += ("\n\nBoth operations independently validated and queued: "
                      + summary + ". Review and approve each in sequence; "
                      "neither has been applied yet.")
        else:
            reply += ("\n\nA validated, finding-scoped edit plan is ready to inspect. "
                      "No changes have been applied.")
    except (ValueError, KeyError, csv.Error, TypeError) as exc:
        st.session_state.cleaning_plans_by_finding.pop(fid, None)
        st.session_state.setdefault("cleaning_batch_queue_by_finding", {}).pop(fid, None)
        st.session_state.setdefault("cleaning_batch_total_by_finding", {}).pop(fid, None)
        st.session_state.cleaning_review_statuses[fid] = "Plan validation failed"
        reply += ("\n\nThe requested edit plan was rejected by the "
                  "independent scope validator: " + str(exc)
                  + ". No partial batch was queued and no data was changed.")
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
                               automatic: bool = False) -> None:
    """Show the submitted question now, then let Agent C answer in a worker."""
    from interactive_chat_jobs import interactive_chat_manager
    from healthcare_data_quality.cleaning_authorization import file_sha256
    fid = finding["id"]
    if fid not in st.session_state.get("cleaning_recommendations", {}):
        raise ValueError("Agent C requires Agent B's saved suggestion for this finding")
    if fid in st.session_state.cleaning_pending_gemma_b:
        return
    st.session_state.setdefault("cleaning_completed_gemma", {}).pop(fid, None)
    fid, thread, old_thread, scoped, metadata = _prepare_finding_discussion(
        finding, source, uploaded, dataset_paths, question, automatic=automatic)
    try:
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
                selected_finding=scoped,
                agent_a_idle=a_idle, agent_b_idle=b_idle,
                idle_probe=idle_probe,
            )
        else:
            # Automatic retries retain C ownership and are never handed to
            # another role without a new explicit user request.
            job_id = manager.submit(
                ask_gemma_about_cleaning, question, old_thread, [scoped], metadata,
                selected_finding=scoped)
            agent = "C"
        st.session_state.cleaning_pending_gemma_b[fid] = {
            "job_id": job_id, "source_sha256": file_sha256(source) if source else None,
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
                    if source and file_sha256(source) != pending_b.get("source_sha256"):
                        # The reply may refer to a previous working file.
                        st.session_state.cleaning_messages_by_finding[fid].append({
                            "role": "assistant", "content":
                            "The working dataset changed while I was replying. "
                            "Please resubmit your request against the new version. "
                            "No cleaning plan was approved or executed."})
                    else:
                        message_count = len(st.session_state.cleaning_messages_by_finding.get(fid, []))
                        _record_gemma_b_response(fid, finding, source, response)
                        messages = st.session_state.cleaning_messages_by_finding.get(fid, [])
                        if (len(messages) > message_count and messages[-1].get("role") == "assistant"):
                            messages[-1]["assigned_agent"] = pending_b.get("assigned_agent", "C")
                except (KeyError, ValueError, OSError, csv.Error) as exc:
                    st.session_state.cleaning_messages_by_finding[fid].append({
                        "role": "assistant", "content": f"Could not validate Agent C's response: {exc}"})
            else:
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
        recommendations_complete = bool(investigation_complete and recommendation_is_complete(recommendation_state))
        def show_status_cards() -> None:
            if (not investigation_complete
                    or _status_card_visible("investigation", run_identity)):
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
            if (not recommendations_complete
                    or _status_card_visible("recommendations", run_identity)):
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
            ready, total, responding = _agent_c_status_counts(recommendation_state)
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
        st.dataframe(full_table, hide_index=False, width="stretch", height=650)
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

    uploaded = st.file_uploader(
        "Upload one or more CSV datasets",
        type=["csv"],
        accept_multiple_files=True,
        help="Up to four local CSV files, maximum 20 MiB each. The first is the primary dataset.",
    )
    if not uploaded:
        render_pre_investigation_status(live_banner_slot, state="no_dataset")
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
        live_status = st.fragment(run_every="3s")(render_live_investigation_status)
        live_status(audit, cache_key, header_reports, include_exploration, live_banner_slot)
    else:
        render_live_investigation_status(audit, cache_key, header_reports, include_exploration, live_banner_slot)

    results_tab_labels = ["Data quality findings", "Supporting evidence", "Clean data with Gemma"]
    # On Streamlit releases with stateful tabs, persist the selected tab across
    # *any* unavoidable full-app rerun. Older releases retain the old API;
    # fragment isolation below still prevents routine polling from remounting it.
    try:
        summary_tab, evidence_tab, cleaning_tab = st.tabs(
            results_tab_labels, key="hdq_active_results_tab", on_change="rerun",
        )
    except TypeError:
        summary_tab, evidence_tab, cleaning_tab = st.tabs(results_tab_labels)
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
    with cleaning_tab:
        def refresh_cleaning():
            # A stable fragment means Agent C's textarea is not destroyed every
            # three seconds while the researcher types. Only a click or an
            # explicit completion notification changes this review surface.
            current = st.session_state.get("audit_result") or audit
            sync_live_exploration(current, cache_key)
            render_cleaning_workspace(paths, uploaded, current, cache_key)
        if hasattr(st, "fragment"):
            st.fragment(refresh_cleaning)()
            # Poll *only* background completion in a tiny sibling fragment.
            # It never re-renders editable forms while Agent A/B are busy.
            st.fragment(run_every="3s")(_poll_agent_c_completion)()
        else:
            refresh_cleaning()


if __name__ == "__main__":
    render_ui()
