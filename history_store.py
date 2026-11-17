"""Local Streamlit cache inventory and user annotations.

Investigation records are immutable evidence. Display labels and notes live in
small .meta sidecars; edits never change an audit/discovery JSON or its SHA key.
No network, dataset loading, or Streamlit dependency is used in this module.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from tempfile import NamedTemporaryFile


KEY_RE = re.compile(r"^[0-9a-f]{64}$")
FILE_RE = {
    "Audit": re.compile(r"^audit-(\d{8}T\d{6}\.\d{6}Z)-[0-9a-f]{12}\.json$"),
    "Gemma checkpoint": re.compile(r"^discovery-(\d{8}T\d{6}\.\d{6}Z)-[0-9a-f]{10}\.json$"),
}
MAX_METADATA_BYTES = 16 * 1024
MAX_TITLE_CHARS = 120
MAX_NOTES_CHARS = 2000

# Explicit, case-sensitive phrase required by the saved-history delete UI.
DELETE_CONFIRMATION_TEXT = "IWANTTODELETEIt"


def is_batch_delete_authorized(
    selected_count: int, acknowledged: bool, typed_confirmation: str,
) -> bool:
    """Never allow a batch delete without an intentional, exact confirmation."""
    return (
        type(selected_count) is int and selected_count > 0
        and acknowledged is True
        and typed_confirmation == DELETE_CONFIRMATION_TEXT
    )


@dataclass(frozen=True)
class HistoryFile:
    kind: str
    path: Path
    key: str
    created_utc: datetime
    size_bytes: int
    title: str = ""
    notes: str = ""


def resolve_viewer_timezone(
    browser_timezone: str | None, browser_offset_minutes: int | None = None,
) -> tuple[tzinfo, str]:
    """Select the browser's timezone; never assume the Streamlit server's zone.

    Streamlit's timezone_offset follows JavaScript getTimezoneOffset() semantics:
    minutes WEST of UTC, hence the minus sign in the fallback offset.
    """
    if isinstance(browser_timezone, str) and browser_timezone:
        try:
            return ZoneInfo(browser_timezone), browser_timezone
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            pass
    if (type(browser_offset_minutes) is int
            and -14 * 60 <= browser_offset_minutes <= 14 * 60):
        offset = -browser_offset_minutes
        tz = timezone(timedelta(minutes=offset))
        sign = "+" if offset >= 0 else "-"
        minutes = abs(offset)
        return tz, f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"
    return timezone.utc, "UTC"


def format_saved_time(saved_utc: datetime, viewer_tz: tzinfo) -> str:
    """Format an absolute UTC instant in the viewer's timezone with DST label."""
    if saved_utc.tzinfo is None:
        raise ValueError("Saved timestamps must carry explicit UTC timezone")
    converted = saved_utc.astimezone(viewer_tz)
    # Include the numeric offset to make repeated local DST clock hours unambiguous.
    return converted.strftime("%Y-%m-%d %H:%M:%S %Z (%z)")


def latest_checkpoint_view(entries: list[HistoryFile]) -> list[HistoryFile]:
    """Keep all audits, but only the newest recovery snapshot per cache key.

    This is strictly a *view*: every checkpoint remains available via the
    Show intermediate checkpoints option for review and manual deletion.
    """
    seen: set[str] = set()
    shown: list[HistoryFile] = []
    for entry in entries:
        if entry.kind == "Gemma checkpoint":
            if entry.key in seen:
                continue
            seen.add(entry.key)
        shown.append(entry)
    return shown


def select_history_rows(
    displayed: list[HistoryFile], selected_positions: list[int],
) -> list[HistoryFile]:
    """Resolve Streamlit row selections against the *displayed* inventory.

    Streamlit returns original dataframe positions even after browser-side
    sorting. Invalid/stale/duplicate positions are ignored; the caller then
    uses the validated HistoryFile paths for preview, export and deletion.
    """
    selected: list[HistoryFile] = []
    seen: set[int] = set()
    for position in selected_positions:
        if (type(position) is not int or position < 0
                or position >= len(displayed) or position in seen):
            continue
        seen.add(position)
        selected.append(displayed[position])
    return selected


def _metadata_path(path: Path) -> Path:
    # .meta is intentionally not .json: do not match audit-*.json glob patterns.
    return path.with_suffix(".meta")


def _read_annotations(path: Path) -> tuple[str, str]:
    sidecar = _metadata_path(path)
    if not sidecar.is_file() or sidecar.is_symlink():
        return "", ""
    try:
        if sidecar.stat().st_size > MAX_METADATA_BYTES:
            return "", ""
        with sidecar.open(encoding="utf-8") as f:
            info = json.load(f)
        if (not isinstance(info, dict) or info.get("record") != path.name
                or not isinstance(info.get("title", ""), str)
                or not isinstance(info.get("notes", ""), str)):
            return "", ""
        return info.get("title", "")[:MAX_TITLE_CHARS], info.get("notes", "")[:MAX_NOTES_CHARS]
    except (OSError, ValueError, UnicodeError):
        return "", ""


def list_history(audit_root: Path, exploration_root: Path) -> list[HistoryFile]:
    """List all recognized saved JSON records; never follow cache symlinks."""
    entries = []
    for kind, root in (("Audit", Path(audit_root)), ("Gemma checkpoint", Path(exploration_root))):
        if root.is_symlink() or not root.is_dir():
            continue
        try:
            directories = list(root.iterdir())
        except OSError:
            continue
        for folder in directories:
            if (not KEY_RE.fullmatch(folder.name) or folder.is_symlink()
                    or not folder.is_dir()):
                continue
            try:
                children = folder.iterdir()
                for path in children:
                    match = FILE_RE[kind].fullmatch(path.name)
                    if not match or path.is_symlink() or not path.is_file():
                        continue
                    try:
                        created = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
                        size = path.stat().st_size
                        title, notes = _read_annotations(path)
                    except (OSError, ValueError):
                        continue
                    entries.append(HistoryFile(kind, path, folder.name, created, size, title, notes))
            except OSError:
                continue
    return sorted(entries, key=lambda entry: (entry.created_utc, entry.path.name), reverse=True)


def _validate_target(entry: HistoryFile, audit_root: Path, exploration_root: Path) -> None:
    root = Path(audit_root if entry.kind == "Audit" else exploration_root)
    if entry.kind not in FILE_RE or not KEY_RE.fullmatch(entry.key):
        raise ValueError("Unrecognized saved investigation")
    if (root.is_symlink() or entry.path.parent != root / entry.key
            or entry.path.parent.is_symlink() or entry.path.is_symlink()
            or not FILE_RE[entry.kind].fullmatch(entry.path.name)
            or not entry.path.is_file()):
        raise ValueError("Saved investigation path is invalid or has changed")


def update_annotations(
    entry: HistoryFile, title: str, notes: str, audit_root: Path, exploration_root: Path,
) -> None:
    """Atomically save cosmetic annotations without touching original evidence."""
    _validate_target(entry, audit_root, exploration_root)
    title = title.strip()
    notes = notes.strip()
    if len(title) > MAX_TITLE_CHARS or len(notes) > MAX_NOTES_CHARS:
        raise ValueError("Display name or notes exceed the supported length")
    sidecar = _metadata_path(entry.path)
    if sidecar.is_symlink():
        raise ValueError("Refusing to modify a symbolic link")
    if not title and not notes:
        sidecar.unlink(missing_ok=True)
        return
    record = {"schema": 1, "record": entry.path.name, "title": title, "notes": notes}
    temp = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=entry.path.parent,
            prefix=".annotation-", suffix=".tmp", delete=False,
        ) as stream:
            temp = Path(stream.name)
            os.chmod(temp, 0o600)
            json.dump(record, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, sidecar)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def delete_saved_file(entry: HistoryFile, audit_root: Path, exploration_root: Path) -> None:
    """Remove only the selected saved snapshot and its annotation, never CSVs."""
    _validate_target(entry, audit_root, exploration_root)
    sidecar = _metadata_path(entry.path)
    if sidecar.is_symlink():
        raise ValueError("Refusing to delete a symbolic-link annotation")
    entry.path.unlink()
    sidecar.unlink(missing_ok=True)


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for suffix in ("B", "KB", "MB", "GB"):
        if size < 1024 or suffix == "GB":
            return f"{size:.0f} B" if suffix == "B" else f"{size:.1f} {suffix}"
        size /= 1024
    return ""


MAX_BATCH_ZIP_BYTES = 48 * 1024 * 1024


def build_history_zip(
    entries: list[HistoryFile], audit_root: Path, exploration_root: Path,
) -> bytes:
    """Return a local ZIP of selected immutable JSONs; no cache mutation.

    Validate each saved path again rather than trusting a stale UI inventory.
    ZIP members are nested under type and cache key, avoiding name collision.
    Sidecar annotations are deliberately excluded from this evidence export.
    """
    from io import BytesIO
    from zipfile import ZipFile, ZIP_DEFLATED

    if not entries:
        raise ValueError("No saved files selected")
    seen: set[Path] = set()
    unique: list[HistoryFile] = []
    total_bytes = 0
    for entry in entries:
        _validate_target(entry, audit_root, exploration_root)
        if entry.path in seen:
            continue
        seen.add(entry.path)
        total_bytes += entry.path.stat().st_size
        if total_bytes > MAX_BATCH_ZIP_BYTES:
            raise ValueError("Selected files exceed the 48 MiB ZIP limit")
        unique.append(entry)

    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in unique:
            folder = "audits" if entry.kind == "Audit" else "gemma-checkpoints"
            archive.write(entry.path, arcname=f"{folder}/{entry.key}/{entry.path.name}")
    return output.getvalue()
