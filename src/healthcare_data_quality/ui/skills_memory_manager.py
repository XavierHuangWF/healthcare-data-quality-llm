"""Conservative, local-only management of reviewer memory and generated skills.

Historical approval logs are not cleaning authorization. Disabling a remembered
rule appends an existing revision marker, preserving the audit trail. Skill files
are advisory JSON/Markdown, never executable code. No raw CSV values are shown
in this manager. All paths are resolved from the current project configuration.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4
import json
import os
import re

from healthcare_data_quality.learning import adaptive_skill_memory
from healthcare_data_quality.learning import interaction_memory

_MANAGE_LOCK = RLock()
_SKILL_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


def _paths(root: Path | None = None) -> tuple[Path, Path, Path]:
    if root is not None:
        project = Path(root)
        return (project / "memory" / "approved_decisions.jsonl",
                project / "memory" / "learning_events.jsonl",
                project / "skills" / "generated")
    return (interaction_memory._location(), adaptive_skill_memory._MEMORY / "learning_events.jsonl",
            adaptive_skill_memory._SKILLS)


def _read_log(path: Path, *, max_lines: int = 50000, max_line_bytes: int = 1_000_000) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        return []
    records: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for count, line in enumerate(source):
            if count >= max_lines:
                break
            if len(line.encode("utf-8")) > max_line_bytes:
                continue
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def _bytes(path: Path) -> int:
    return path.stat().st_size if path.is_file() and not path.is_symlink() else 0


def list_skills(root: Path | None = None) -> list[dict]:
    _, _, folder = _paths(root)
    if not folder.is_dir() or folder.is_symlink():
        return []
    result = []
    for path in sorted(folder.glob("*.json"))[:100]:
        if not _SKILL_NAME.fullmatch(path.stem) or path.is_symlink() or _bytes(path) > 200_000:
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (not isinstance(item, dict) or item.get("schema") != 1 or
                item.get("family") != path.stem or item.get("advisory_only") is not True):
            continue
        result.append({"family": path.stem,
                       "title": str(item.get("title") or path.stem)[:100],
                       "approval_count": int(item.get("approval_count") or 0),
                       "updated_at": str(item.get("updated_at") or ""),
                       "when_to_suggest": str(item.get("when_to_suggest") or "")[:350],
                       "recommended_method": str(item.get("recommended_method") or "")[:350],
                       "do_not_assume": str(item.get("do_not_assume") or "")[:300],
                       "bytes": _bytes(path) + _bytes(path.with_suffix(".md"))})
    return result


def list_rules(root: Path | None = None) -> list[dict]:
    decisions_path, _, _ = _paths(root)
    raw = _read_log(decisions_path)
    # Revocation applies to earlier matching evidence, never to a newly approved method.
    revoked: set[tuple[str, str, str]] = set()
    shown: set[tuple[str, str, str]] = set()
    active = []
    for record in reversed(raw):
        if record.get("schema_version") != interaction_memory._VERSION:
            continue
        identity = interaction_memory.review_memory_identity(record)
        if record.get("decision") == "revision_reset":
            revoked.add(identity)
            continue
        if record.get("decision") != "approved_edit" or identity in revoked or identity in shown:
            continue
        shown.add(identity)
        active.append({"id": str(record.get("memory_id") or ""),
                       "title": str(record.get("finding_title") or "Approved method")[:150],
                       "reviewed_at": str(record.get("reviewed_at_utc") or ""),
                       "operation_count": len(record.get("operations") or [])})
    return active[:200]


def overview(root: Path | None = None) -> dict:
    decisions, events, folder = _paths(root)
    skills = list_skills(root)
    learned = _read_log(events, max_line_bytes=8000)
    history = _read_log(decisions)
    active = list_rules(root)
    size = _bytes(decisions) + _bytes(events)
    if folder.is_dir() and not folder.is_symlink():
        size += sum(_bytes(path) for path in folder.iterdir()
                    if path.suffix in {".json", ".md"} and path.is_file())
    return {
        "stored_reviews": sum(record.get("decision") == "approved_edit" for record in history),
        "active_rules": len(active),
        "learning_events": len(learned),
        "outcomes": dict(Counter(str(item.get("outcome")) for item in learned)),
        "skills": len(skills),
        "storage_bytes": size,
        "rules": active,
        "skill_details": skills,
    }


def disable_rule(memory_id: str, *, root: Path | None = None) -> bool:
    """Append a revocation marker; do not destroy the original audit record."""
    path, _, _ = _paths(root)
    with _MANAGE_LOCK, interaction_memory._LOCK:
        records = _read_log(path)
        valid_ids = {entry["id"] for entry in list_rules(root)}
        match = next((r for r in reversed(records)
                      if r.get("memory_id") == memory_id and r.get("decision") == "approved_edit"), None)
        if not match or memory_id not in valid_ids:
            return False
        marker = {"schema_version": interaction_memory._VERSION,
                  "memory_id": str(uuid4()),
                  "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                  **{key: str(match.get(key) or "") for key in (
                      "source_sha256", "finding_type", "evidence_sha256")},
                  "decision": "revision_reset", "operations": [],
                  "execution_authorized": False}
        if path.is_symlink():
            raise OSError("Memory file cannot be a symlink")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as out:
            out.write(json.dumps(marker, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        return True


def delete_skill(family: str, *, root: Path | None = None) -> bool:
    """Remove just the selected advisory skill (both representations)."""
    if not _SKILL_NAME.fullmatch(family):
        raise ValueError("Invalid skill identifier")
    _, _, folder = _paths(root)
    if folder.is_symlink():
        raise OSError("Skills directory cannot be a symlink")
    with _MANAGE_LOCK:
        if family not in {s["family"] for s in list_skills(root)}:
            return False
        for suffix in (".json", ".md"):
            path = folder / (family + suffix)
            if path.is_symlink():
                raise OSError("Skill file cannot be a symlink")
            path.unlink(missing_ok=True)
        return True


def clear_learning_events(*, root: Path | None = None) -> bool:
    """Clear preference-learning events only; preserve historical approved rules."""
    _, path, _ = _paths(root)
    with _MANAGE_LOCK, adaptive_skill_memory._LOCK:
        if not path.is_file():
            return False
        if path.is_symlink():
            raise OSError("Learning history cannot be a symlink")
        path.unlink()
        return True


def human_bytes(size: int) -> str:
    n = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
