"""Non-destructive installer for human-memory options and single-click approval.

Extract this ZIP into the existing project root, then run:
    python tools/apply_verified_human_choices.py

Edits narrowly scoped source regions; aborts if the project's layout differs.
Backups are saved under .git/hdq-patch-backups (or temporary directory).
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import shutil
import sys
import py_compile

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'app.py'
COMPOUND = ROOT / 'src/healthcare_data_quality/cleaning/compound_approval.py'
TEST = ROOT / 'tests/test_cleaning_review_cards.py'
REQUIRED = [ROOT / x for x in (
    'src/healthcare_data_quality/ui/human_choice_replay.py',
    'src/healthcare_data_quality/ui/one_click_choice.py',
    'src/healthcare_data_quality/cleaning/explicit_columns.py',
)]
NEW_CHOICES = '            choices = st.columns(len(options))\n            reviewer = str(st.session_state.get(\'cleaning_approved_reviewer\') or \'\').strip()\n            for index, option in enumerate(options):\n                label = \'Yes — Approve & Execute: \' + option[\'label\']\n                if choices[index].button(\n                        label[:140],\n                        key=f"review_yes_{fid}_{_active_agent_c_control_generation(fid)}_option_{index}",\n                        disabled=not bool(reviewer)):\n                    # The ONE click authorizes this exact Before→After table.\n                    # Host validation, isolated R preview, all outstanding edits,\n                    # and final export integrity are independently verified first.\n                    try:\n                        from healthcare_data_quality.ui.one_click_choice import execute_verified_choice\n                        with st.spinner(\'Verifying and applying your approved changes\'):\n                            outcome = execute_verified_choice(\n                                st.session_state, finding, source, option)\n                        if outcome.get(\'warning\'):\n                            st.warning(outcome[\'warning\'])\n                        _rerun_parent_after_agent_b_choice(fid)\n                    except (OSError, ValueError, PermissionError, RuntimeError,\n                            AssertionError, IndexError, KeyError, csv.Error, TypeError) as exc:\n                        st.error(\'Approval blocked; no changes staged: \' + str(exc))\n'
NEW_LEGACY_TEST = 'def test_accepting_source_backed_suggestion_is_one_named_approval(monkeypatch, tmp_path):\n    """One green click commits the preverified selection, not a second Yes form."""\n    import pytest\n    file = tmp_path / "one.csv"\n    file.write_text("Taxon,Count\\na,0.2\\n", encoding="utf-8")\n    before = file.read_bytes()\n    monkeypatch.setenv("HDQ_APPROVED_STAGE_DIR", str(tmp_path / "checkpoint"))\n    st = StubUI()\n    app = _app(monkeypatch, st)\n    finding = {**_catalog()[0], "evidence": [\n        {"CSV row": 2, "Column": 2, "Original value": "0.2"}]}\n    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",\n                        lambda *args, **kwargs: [finding])\n    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.allowed_positions",\n                        lambda *args: ({(2, 2)}, set()))\n\n    class SavedAgentB:\n        def snapshot(self, key):\n            return {"entries": {finding["id"]: {\n                "status": "ready", "suggestion": "Replace invalid count 0.2 with 0",\n                "decision_needed": "", "choices": [{\n                    "label": "Replace with zero", "operations": [{\n                        "operation": "replace_cells", "changes": [\n                            {"row": 2, "column": 2, "before": "0.2", "after": "0"}]}]}]}}}\n\n    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager",\n                        lambda: SavedAgentB())\n    uploaded = [SimpleNamespace(name="one.csv")]\n    app.render_cleaning_workspace([file], uploaded, {}, "audit-C", [])\n    st.session_state.cleaning_approved_reviewer = \'Reviewer X\'\n    assert any(item[0] == "table" for item in st.comments)\n    called = []\n\n    def execute_once(state, selected_finding, source, option):\n        called.append((selected_finding[\'id\'], option[\'operations\']))\n        state.cleaning_review_statuses[selected_finding[\'id\']] = \'Applied and verified\'\n        state.cleaning_history.append({\'finding_id\': selected_finding[\'id\']})\n        return {\'warning\': None}\n\n    monkeypatch.setattr(\'healthcare_data_quality.ui.one_click_choice.execute_verified_choice\', execute_once)\n    st.buttons_on.add(f"review_yes_{finding[\'id\']}_0_option_0")\n    with pytest.raises(StopIteration, match="rerun"):\n        app.render_cleaning_workspace([file], uploaded, {}, "audit-C", [])\n    assert len(called) == 1\n    assert file.read_bytes() == before\n    assert st.session_state.cleaning_review_statuses[finding[\'id\']] == \'Applied and verified\'\n    assert st.session_state.cleaning_history == [{\'finding_id\': finding[\'id\']}]\n'


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f'{label}: expected exactly one old insertion point; found {count}. No files changed.')
    return text.replace(old, new, 1)


def patch_app(source: str) -> str:
    if 'from healthcare_data_quality.ui.human_choice_replay import prioritize_verified_human_choices' in source:
        if 'from healthcare_data_quality.ui.one_click_choice import execute_verified_choice' not in source:
            raise ValueError('Partial prior installation detected in app.py; inspect manually')
        return source
    source = once(source,
        '        from healthcare_data_quality.services.recommendation_options import source_backed_choices\n'
        '        options = source_backed_choices(finding, source, suggestion)\n',
        '        from healthcare_data_quality.services.recommendation_options import source_backed_choices\n'
        '        from healthcare_data_quality.ui.human_choice_replay import prioritize_verified_human_choices\n'
        '        options = prioritize_verified_human_choices(\n'
        '            finding, source, source_backed_choices(finding, source, suggestion))\n',
        'memory priority')
    start=source.find('            choices = st.columns(len(options))\n', source.find('def _render_live_agent_b_finding('))
    end=source.find('        else:\n            # Model did not provide exact source-backed actions.',start)
    if start<0 or end<0 or end<=start:
        raise ValueError('Agent B choice renderer has changed; refusing to overwrite unknown code')
    existing=source[start:end]
    if ('_record_gemma_b_response(fid, finding, source' not in existing or
            'review_yes_' not in existing or
            'st.session_state.cleaning_discussion_mode_by_finding' not in existing):
        raise ValueError('Agent B option handler has changed; refusing to overwrite unknown code')
    source=source[:start]+NEW_CHOICES+source[end:]
    source=once(source,
        '                st.dataframe(pd.DataFrame(option[\'rows\']), hide_index=True, width="stretch")\n'
        '            choices = st.columns(len(options))',
        '                st.dataframe(pd.DataFrame(option[\'rows\']), hide_index=True, width="stretch")\n'
        '                if option.get(\'provenance\'):\n'
        '                    st.caption(option[\'provenance\'])\n'
        '                if option.get(\'observed_pattern\'):\n'
        "                    st.caption(option['observed_pattern'] + ' — not a rule for other datasets.')\n"
        '            choices = st.columns(len(options))',
        'memory explanation')
    source=once(source,
        '            from healthcare_data_quality.cleaning.finding_cleaning import (\n'
        '                propose_explicit_user_row_deletion, propose_bulk_blank_row_deletion,\n'
        '            )\n',
        '            from healthcare_data_quality.cleaning.finding_cleaning import (\n'
        '                propose_explicit_user_row_deletion, propose_bulk_blank_row_deletion,\n'
        '            )\n'
        '            if source:\n'
        '                from healthcare_data_quality.cleaning.explicit_columns import explicit_column_deletion\n'
        '                direct_columns = explicit_column_deletion(question, finding, source)\n'
        '                if direct_columns is not None:\n'
        '                    st.session_state.setdefault(\n'
        "                        'cleaning_explicit_column_deletions_by_finding', {})[fid] = direct_columns['explicit_columns']\n"
        '                    _record_gemma_b_response(fid, finding, source, direct_columns)\n'
        '                    return\n',
        'unambiguous plural column deletion')
    return source


def patch_compound(source: str) -> str:
    if 'allow_single: bool = False' in source:
        return source
    source=once(source, '    r_previews: dict[int, dict] | None = None,\n) -> dict:\n',
         '    r_previews: dict[int, dict] | None = None,\n    allow_single: bool = False,\n) -> dict:\n',
         'single-action gate')
    return once(source,
        'if not isinstance(plans, list) or not 2 <= len(plans) <= 16:',
        'if not isinstance(plans, list) or not (1 if allow_single else 2) <= len(plans) <= 16:',
        'single-action range')


def patch_legacy_test(source: str) -> str:
    begin=source.find('def test_accepting_source_backed_suggestion_prepares_exact_review_not_execution(')
    if begin < 0:
        return source
    end=source.find('\n\ndef test_existing_open_session_migrates_to_new_card_states(',begin)
    if end < 0:
        raise ValueError('Legacy test layout changed; avoid overwriting newer test code')
    return source[:begin]+NEW_LEGACY_TEST+source[end:]


def main():
    missing=[str(p.relative_to(ROOT)) for p in [APP, COMPOUND, *REQUIRED] if not p.is_file()]
    if missing:
        raise SystemExit('Missing required files after ZIP extraction: '+', '.join(missing))
    original_app=APP.read_text(encoding='utf8')
    original_compound=COMPOUND.read_text(encoding='utf8')
    test_original=TEST.read_text(encoding='utf8') if TEST.is_file() else None
    # PRE-FLIGHT ALL transformations BEFORE writing anything.
    updated={APP:patch_app(original_app), COMPOUND:patch_compound(original_compound)}
    if test_original is not None:
        updated[TEST]=patch_legacy_test(test_original)
    changed=[p for p,src in updated.items() if p.read_text(encoding='utf8')!=src]
    if not changed:
        print('The human-memory and one-click patch is already installed.')
        return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup_root=(ROOT/'.git'/'hdq-patch-backups'/stamp if (ROOT/'.git').is_dir()
                 else Path(tempfile.mkdtemp(prefix='hdq-patch-backups-')))
    backup_root.mkdir(parents=True,exist_ok=True)
    for path in changed:
        dest=backup_root/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    try:
        for path in changed:
            tmp=path.with_name(path.name+'.verified_choice.tmp')
            tmp.write_text(updated[path],encoding='utf8')
            if path.suffix=='.py':
                py_compile.compile(str(tmp),doraise=True)
            tmp.replace(path)
    except Exception:
        for path in changed:
            shutil.copy2(backup_root/path.relative_to(ROOT),path)
        raise
    print('Patched:',', '.join(str(p.relative_to(ROOT)) for p in changed))
    print('Backups:',backup_root)
    print('Run: python -m pytest -q --tb=short')

if __name__=='__main__':
    main()
