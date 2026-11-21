"""Saved-file manager is non-destructive, content-bound and audit-safe."""

import json
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path

import pytest

from healthcare_data_quality.storage.history_store import (
    delete_saved_file, build_history_zip, human_size, list_history,
    select_history_rows, update_annotations,
    DELETE_CONFIRMATION_TEXT, is_batch_delete_authorized,
)


KEY1 = "a" * 64
KEY2 = "b" * 64


def _records(tmp_path):
    audit_root = tmp_path / "audit-cache"
    discovery_root = tmp_path / "explorations"
    (audit_root / KEY1).mkdir(parents=True)
    (audit_root / KEY2).mkdir(parents=True)
    (discovery_root / KEY1).mkdir(parents=True)
    audit_old = audit_root / KEY1 / ("audit-20261113T131420.000001Z-" + "e" * 12 + ".json")
    audit_new = audit_root / KEY1 / ("audit-20261114T131420.000002Z-" + "f" * 12 + ".json")
    audit_other = audit_root / KEY2 / ("audit-20261112T131420.000003Z-" + "d" * 12 + ".json")
    discovery = discovery_root / KEY1 / ("discovery-20261114T131421.000001Z-" + "c" * 10 + ".json")
    for path in (audit_old, audit_new, audit_other, discovery):
        path.write_text(json.dumps({"important_evidence": "unchanged"}), encoding="utf-8")
    return audit_root, discovery_root, audit_old, audit_new, audit_other, discovery


def test_inventory_shows_all_cache_keys_types_and_sorted_times(tmp_path):
    root_a, root_e, old, new, other, exploration = _records(tmp_path)
    files = list_history(root_a, root_e)
    assert [f.path for f in files] == [exploration, new, old, other]
    assert {f.kind for f in files} == {"Audit", "Gemma checkpoint"}
    assert all(len(f.key) == 64 for f in files)
    assert all(f.created_utc.tzinfo is not None for f in files)


def test_annotation_edits_do_not_touch_evidence_or_hash_directory(tmp_path):
    root_a, root_e, _, newer, _, _ = _records(tmp_path)
    entry = next(f for f in list_history(root_a, root_e) if f.path == newer)
    original = newer.read_bytes()
    update_annotations(entry, "Study QC run", "Checked sample labels", root_a, root_e)
    assert newer.read_bytes() == original
    assert newer.parent.name == KEY1
    renamed = next(f for f in list_history(root_a, root_e) if f.path == newer)
    assert renamed.title == "Study QC run"
    assert renamed.notes == "Checked sample labels"
    sidecar = newer.with_suffix(".meta")
    assert (sidecar.stat().st_mode & 0o077) == 0
    assert len([f for f in list_history(root_a, root_e) if f.kind == "Audit"]) == 3
    update_annotations(renamed, "", "", root_a, root_e)
    assert not sidecar.exists()
    assert newer.read_bytes() == original


def test_deletion_removes_only_selected_record_and_notes(tmp_path):
    root_a, root_e, older, newer, other, discovery = _records(tmp_path)
    entry = next(f for f in list_history(root_a, root_e) if f.path == older)
    update_annotations(entry, "Old", "Obsolete", root_a, root_e)
    delete_saved_file(entry, root_a, root_e)
    assert not older.exists()
    assert not older.with_suffix(".meta").exists()
    assert newer.exists() and other.exists() and discovery.exists()
    assert (root_a / KEY1).exists()


def test_invalid_paths_cannot_be_edited_or_deleted(tmp_path):
    root_a, root_e, _, newer, _, _ = _records(tmp_path)
    entry = next(f for f in list_history(root_a, root_e) if f.path == newer)
    malicious = entry.__class__(entry.kind, tmp_path / "outside.json", entry.key,
                                entry.created_utc, entry.size_bytes)
    with pytest.raises(ValueError):
        update_annotations(malicious, "test", "", root_a, root_e)
    with pytest.raises(ValueError):
        delete_saved_file(malicious, root_a, root_e)
    assert newer.exists()


def test_symlinks_and_unrecognized_json_are_ignored(tmp_path):
    root_a, root_e, _, newer, _, _ = _records(tmp_path)
    (root_a / KEY1 / "audit-20261115T000000.000000Z-000000000000.json").symlink_to(newer)
    (root_a / KEY1 / "not-an-audit.json").write_text("{}")
    (root_a / KEY1 / "audit-20261115T000000.000000Z-000000000001.json.meta").write_text("{}")
    outside = tmp_path / ("audit-20261115T000000.000000Z-000000000002.json")
    outside.write_text("{}")
    (root_a / ("c" * 64)).symlink_to(outside.parent, target_is_directory=True)
    paths = [f.path for f in list_history(root_a, root_e)]
    assert newer in paths
    assert not any("20261115" in path.name for path in paths)


def test_symlink_annotation_is_rejected(tmp_path):
    root_a, root_e, _, newer, _, _ = _records(tmp_path)
    entry = next(f for f in list_history(root_a, root_e) if f.path == newer)
    target = tmp_path / "independent.txt"
    target.write_text("original")
    newer.with_suffix(".meta").symlink_to(target)
    with pytest.raises(ValueError):
        update_annotations(entry, "new", "note", root_a, root_e)
    with pytest.raises(ValueError):
        delete_saved_file(entry, root_a, root_e)
    assert target.read_text() == "original"
    assert newer.exists()


def test_overlong_metadata_rejected(tmp_path):
    root_a, root_e, _, newer, _, _ = _records(tmp_path)
    entry = next(f for f in list_history(root_a, root_e) if f.path == newer)
    with pytest.raises(ValueError):
        update_annotations(entry, "x" * 121, "", root_a, root_e)
    with pytest.raises(ValueError):
        update_annotations(entry, "", "y" * 2001, root_a, root_e)
    assert newer.with_suffix(".meta").exists() is False


def test_displayed_sizes():
    assert human_size(37) == "37 B"
    assert human_size(2048) == "2.0 KB"


def test_ui_does_not_choose_an_investigation_to_load():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "selected_run = previous_runs[0]" in source
    assert '"Choose an earlier investigation"' not in source
    assert '"Manage saved files"' in source
    assert "render_saved_file_manager(cache_key)" in source
    assert 'on_select="rerun", selection_mode="multi-row"' in source
    assert '"Clear selected rows"' in source
    # Skills/memory management may use multiselect; investigation history must not.
    history_ui = source[source.index("def render_saved_file_manager("):source.index("def ", source.index("def render_saved_file_manager(")+4)]
    assert "st.multiselect(" not in history_ui
    assert "build_history_zip(" in source
    assert "for entry in selected:" in source
    assert "saved_investigations(cache_key)" in source


def test_multi_record_zip_contains_only_selected_evidence(tmp_path):
    root_a, root_e, older, newer, other, discovery = _records(tmp_path)
    inventory = list_history(root_a, root_e)
    entries = [e for e in inventory if e.path in {older, discovery}]
    before = {p: p.read_bytes() for p in (older, newer, other, discovery)}
    exported = build_history_zip(entries, root_a, root_e)
    with ZipFile(BytesIO(exported)) as bundle:
        assert len(bundle.namelist()) == 2
        assert any(f"audits/{KEY1}/{older.name}" == name for name in bundle.namelist())
        assert any(f"gemma-checkpoints/{KEY1}/{discovery.name}" == name for name in bundle.namelist())
        assert all(bundle.read(name) == before[older] for name in bundle.namelist())
    assert {p: p.read_bytes() for p in before} == before


def test_batch_export_rejects_invalid_targets_and_oversize(tmp_path, monkeypatch):
    root_a, root_e, _, newer, _, discovery = _records(tmp_path)
    inventory = list_history(root_a, root_e)
    malicious = next(e for e in inventory if e.path == newer).__class__(
        "Audit", tmp_path / "outside.json", KEY1, inventory[0].created_utc, 8
    )
    with pytest.raises(ValueError):
        build_history_zip([malicious], root_a, root_e)
    assert newer.exists()
    from healthcare_data_quality.storage import history_store
    monkeypatch.setattr(history_store, "MAX_BATCH_ZIP_BYTES", 1)
    with pytest.raises(ValueError, match="ZIP limit"):
        build_history_zip(inventory, root_a, root_e)


def test_batch_zip_deduplicates_same_selected_file(tmp_path):
    root_a, root_e, _, newer, _, _ = _records(tmp_path)
    entry = next(e for e in list_history(root_a, root_e) if e.path == newer)
    archive_bytes = build_history_zip([entry, entry], root_a, root_e)
    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert len(archive.namelist()) == 1
        assert archive.read(archive.namelist()[0]) == newer.read_bytes()


def test_multi_selection_ui_keeps_sha_based_loading_unchanged():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'selected_run = previous_runs[0]' in source
    assert 'selected = select_history_rows(displayed, table_event.selection.rows)' in source
    assert 'selected_paths = [str(entry.path) for entry in selected]' in source
    assert 'st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1' in source
    assert 'key="history_delete_acknowledged"' in source
    assert 'key="history_delete_typed_phrase"' in source
    assert 'disabled=not can_delete' in source
    assert 'entry.kind == "Gemma checkpoint"' in source
    assert 'entry.kind == "Audit"' in source


def test_multi_delete_removes_selected_records_only(tmp_path):
    root_a, root_e, older, newer, other, discovery = _records(tmp_path)
    for entry in list_history(root_a, root_e):
        if entry.path in {newer, discovery}:
            delete_saved_file(entry, root_a, root_e)
    assert not newer.exists()
    assert not discovery.exists()
    assert older.exists()
    assert other.exists()


def test_table_selection_maps_original_row_positions_to_correct_files(tmp_path):
    root_a, root_e, older, newer, other, discovery = _records(tmp_path)
    visible = list_history(root_a, root_e)
    assert [entry.path for entry in select_history_rows(visible, [0, 2])] == [
        discovery, older,
    ]
    assert [entry.path for entry in select_history_rows(visible, [1, 3])] == [
        newer, other,
    ]
    # Streamlit row indices do not change after visual sorting in the browser.
    assert select_history_rows(visible, [1])[0].path == newer


def test_table_selection_ignores_stale_or_invalid_positions(tmp_path):
    root_a, root_e, *_ = _records(tmp_path)
    visible = list_history(root_a, root_e)
    assert select_history_rows(visible, [-1, 999, True, "1", 1, 1]) == [visible[1]]
    assert select_history_rows(visible[:1], [1]) == []
    assert select_history_rows([], [0]) == []


def test_selected_table_rows_are_exact_batch_delete_targets(tmp_path):
    root_a, root_e, older, newer, other, discovery = _records(tmp_path)
    visible = list_history(root_a, root_e)
    selected = select_history_rows(visible, [0, 2])
    for entry in selected:
        delete_saved_file(entry, root_a, root_e)
    assert not discovery.exists() and not older.exists()
    assert newer.exists() and other.exists()


def test_table_selection_widget_identity_changes_with_inventory():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert '"\\n".join(str(entry.path) for entry in displayed)' in source
    assert 'table_key = (' in source
    assert 'key=table_key, on_select="rerun", selection_mode="multi-row"' in source
    assert '"Matches uploaded data": "Yes" if matches_upload(entry) else "No"' in source
    assert '"Choose an earlier investigation"' not in source


def test_delete_confirmation_requires_exact_phrase_and_acknowledgement():
    assert DELETE_CONFIRMATION_TEXT == "IWANTTODELETEIt"
    assert is_batch_delete_authorized(1, True, "IWANTTODELETEIt")
    assert is_batch_delete_authorized(4, True, "IWANTTODELETEIt")
    for bad_phrase in (
        "", "IWANTTODELETEIT", "iwanttodeleteit", "IWANTTODELETEit",
        " IWANTTODELETEIt", "IWANTTODELETEIt ", "IWANTTODELETEIt!",
    ):
        assert not is_batch_delete_authorized(2, True, bad_phrase)
    assert not is_batch_delete_authorized(2, False, DELETE_CONFIRMATION_TEXT)
    assert not is_batch_delete_authorized(0, True, DELETE_CONFIRMATION_TEXT)
    assert not is_batch_delete_authorized(-1, True, DELETE_CONFIRMATION_TEXT)
    assert not is_batch_delete_authorized(True, True, DELETE_CONFIRMATION_TEXT)
    assert not is_batch_delete_authorized(1, 1, DELETE_CONFIRMATION_TEXT)


def test_delete_ui_requires_phrase_and_resets_after_row_selection_changes():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'st.session_state.get("history_delete_active_selection") != selection_signature' in source
    assert 'st.session_state["history_delete_acknowledged"] = False' in source
    assert 'st.session_state["history_delete_typed_phrase"] = ""' in source
    assert 'max_chars=len(DELETE_CONFIRMATION_TEXT)' in source
    assert 'disabled=not can_delete' in source
    assert 'if delete_clicked and is_batch_delete_authorized(' in source
    assert 'on_select="rerun", selection_mode="multi-row"' in source
    history_ui = source[source.index('def render_saved_file_manager('):source.index('def ', source.index('def render_saved_file_manager(') + 4)]
    assert 'st.multiselect(' not in history_ui


def test_history_timestamps_follow_browser_timezone_and_dst():
    from datetime import datetime, timezone
    from healthcare_data_quality.storage.history_store import format_saved_time, resolve_viewer_timezone

    pst, name = resolve_viewer_timezone("America/Los_Angeles")
    assert name == "America/Los_Angeles"
    at_night = datetime(2026, 11, 15, 6, 8, 41, tzinfo=timezone.utc)
    assert format_saved_time(at_night, pst) == "2026-11-14 22:08:41 PST (-0800)"
    summer = datetime(2026, 7, 15, 6, 8, 41, tzinfo=timezone.utc)
    assert format_saved_time(summer, pst) == "2026-07-14 23:08:41 PDT (-0700)"
    taipei, name = resolve_viewer_timezone("Asia/Taipei")
    assert name == "Asia/Taipei"
    assert format_saved_time(at_night, taipei) == "2026-11-15 14:08:41 CST (+0800)"


def test_time_display_uses_offset_fallback_without_server_timezone():
    from datetime import datetime, timezone
    from healthcare_data_quality.storage.history_store import format_saved_time, resolve_viewer_timezone

    at_night = datetime(2026, 11, 15, 6, 8, 41, tzinfo=timezone.utc)
    fallback, label = resolve_viewer_timezone("Invalid/Zone", 480)
    assert label == "UTC-08:00"
    assert format_saved_time(at_night, fallback) == "2026-11-14 22:08:41 UTC-08:00 (-0800)"
    utc, label = resolve_viewer_timezone(None)
    assert label == "UTC"
    assert format_saved_time(at_night, utc) == "2026-11-15 06:08:41 UTC (+0000)"
    utc, label = resolve_viewer_timezone(None, 10000)
    assert label == "UTC"
    with pytest.raises(ValueError):
        format_saved_time(datetime(2026, 11, 15), utc)


def test_fall_dst_timestamps_are_unambiguous():
    from datetime import datetime, timezone
    from healthcare_data_quality.storage.history_store import format_saved_time, resolve_viewer_timezone
    la, _ = resolve_viewer_timezone("America/Los_Angeles")
    # The two instants have the same wall-clock hour on fall-back day.
    before = datetime(2026, 11, 1, 8, 30, tzinfo=timezone.utc)
    after = datetime(2026, 11, 1, 9, 30, tzinfo=timezone.utc)
    assert "01:30:00 PDT (-0700)" in format_saved_time(before, la)
    assert "01:30:00 PST (-0800)" in format_saved_time(after, la)


def test_latest_checkpoint_view_keeps_all_audits_and_only_one_snapshot_per_key(tmp_path):
    from healthcare_data_quality.storage.history_store import latest_checkpoint_view
    audit_root, exploration_root, audit_old, audit_new, other, discovery = _records(tmp_path)
    older_checkpoint = exploration_root / KEY1 / (
        "discovery-20261113T131421.000001Z-" + "b" * 10 + ".json"
    )
    older_checkpoint.write_text("{}")
    entries = list_history(audit_root, exploration_root)
    filtered = latest_checkpoint_view(entries)
    assert len(entries) == 5
    assert len(filtered) == 4
    assert discovery in {item.path for item in filtered}
    assert older_checkpoint not in {item.path for item in filtered}
    assert {audit_old, audit_new, other}.issubset({item.path for item in filtered})
    assert latest_checkpoint_view([]) == []


def test_history_timezone_and_checkpoint_ui_are_not_dataset_loading_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'getattr(context, "timezone", None)' in source
    assert 'getattr(context, "timezone_offset", None)' in source
    assert '"Saved (local time)": format_saved_time(entry.created_utc, viewer_tz)' in source
    assert '"Show intermediate Gemma checkpoints"' in source
    assert 'collapsed = latest_checkpoint_view(visible)' in source
    assert 'selected_run = previous_runs[0]' in source
    assert 'on_select="rerun", selection_mode="multi-row"' in source
