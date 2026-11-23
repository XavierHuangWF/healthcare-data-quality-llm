"""Install a single ZIP export containing cleaned CSV and reviewer audit.

Usage: python tools/apply_complete_review_zip.py
- Changes the terminal reviewed export button, not investigation or agent state.
- Creates a backup of app.py.
- Leaves existing reviewer_audit.py alone (installs a fallback only if missing).
- Safe to run twice. Refuses unrecognized layouts instead of guessing.
"""
from __future__ import annotations

import ast
from pathlib import Path
import shutil
import textwrap

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'app.py'
ASSET = ROOT / 'tools' / 'assets' / 'reviewer_audit_fallback.py'
AUDIT_MODULE = ROOT / 'src' / 'healthcare_data_quality' / 'cleaning' / 'reviewer_audit.py'
MARKER = 'from healthcare_data_quality.cleaning.review_zip_bundle import build_review_zip_bundle'


REPLACEMENT = '''\
if hasattr(st, "download_button"):
    {MARKER}
    try:
        if len(datasets_for_download) != 1:
            raise ValueError("Review ZIP export currently supports one uploaded dataset at a time")
        review_zip_name, review_zip_bytes = build_review_zip_bundle(
            original_csv=source_for_export,
            cleaned_csv=datasets_for_download[0][2],
            original_filename=uploaded[0].name,
            reviewer=(st.session_state.get("cleaning_approved_reviewer")
                      or st.session_state.get("hdq_registered_reviewer") or ""),
            history=st.session_state.get("cleaning_history") or [],
            decisions=st.session_state.get("cleaning_decisions") or [],
            review_statuses=st.session_state.get("cleaning_review_statuses") or {},
            conversations=st.session_state.get("cleaning_messages_by_finding") or {},
            approved_row_operations=st.session_state.get(
                "cleaning_approved_rows_by_dataset", {}).get("dataset_1", []),
            approved_column_deletions=st.session_state.get(
                "cleaning_approved_columns_by_dataset", {}).get("dataset_1", []),
            completed_audits=st.session_state.get(
                "cleaning_completed_audits_by_finding") or {},
            finding_titles={item["id"]: item.get("title", "") for item in catalog},
            agent_b_suggestions=st.session_state.get("cleaning_recommendations") or {},
            excel_attachment=(file_name, data) if want_excel else None,
        )
    except (ValueError, PermissionError, OSError, TypeError, KeyError, AssertionError) as exc:
        st.error(f"Reviewer ZIP unavailable: {exc}")
    else:
        st.download_button(
            "Download cleaned CSV + complete reviewer audit (ZIP)",
            data=review_zip_bytes, file_name=review_zip_name,
            mime="application/zip",
            key=f"cleaning_download_final_{session_context}",
            type="primary", disabled=False,
        )
'''


def _contains_active_final_button(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        fn = child.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id == 'st' and fn.attr == 'download_button'):
            continue
        if not any(kw.arg == 'disabled' and isinstance(kw.value, ast.Constant)
                   and kw.value.value is False for kw in child.keywords):
            continue
        if any(kw.arg == 'key' and 'cleaning_download_final' in ast.unparse(kw.value)
               for kw in child.keywords):
            return True
    return False


def transform(src: str) -> str:
    if MARKER in src:
        return src
    tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == 'render_cleaning_workspace']
    if len(funcs) != 1:
        raise RuntimeError('Could not identify the unique cleaning-workspace renderer; no files changed')
    fn = funcs[0]
    candidates = [node for node in ast.walk(fn) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Call)
                  and isinstance(node.test.func, ast.Name)
                  and node.test.func.id == 'hasattr'
                  and _contains_active_final_button(node)]
    # Find the innermost `if hasattr(st, "download_button")` for the active
    # terminal button, never the disabled review gate or other controls.
    matches = [node for node in candidates if any(
        isinstance(arg, ast.Constant) and arg.value == 'download_button'
        for arg in node.test.args)]
    if len(matches) != 1:
        raise RuntimeError('Unable to uniquely match final download control; no files changed')
    node = matches[0]
    lines = src.splitlines(keepends=True)
    indent = ' ' * node.col_offset
    injected = textwrap.indent(REPLACEMENT.replace("{MARKER}", MARKER), indent)
    lines[node.lineno - 1:node.end_lineno] = [injected]
    result = ''.join(lines)
    # Update locked state label, without changing the existing gate semantics.
    old = '"Download cleaned dataset", data=b"", file_name="review_required.txt"'
    new = '"Download cleaned CSV + complete reviewer audit (ZIP)", data=b"", file_name="review_required.txt"'
    if old in result:
        result = result.replace(old, new, 1)
    if result.count(MARKER) != 1:
        raise RuntimeError('ZIP export marker missing or duplicated')
    ast.parse(result)
    return result


def update_legacy_export_test() -> None:
    """Update only the exact old assertion; preserve newer test customizations."""
    test = ROOT / "tests" / "test_final_download_gate.py"
    if not test.exists():
        return
    current = test.read_text(encoding="utf-8")
    old = """    st.session_state.cleaning_review_statuses[_catalog()[2]['id']] = 'Reviewed — no change'
    render()
    assert st.downloads[-1][4] is False
    label, data, name, mime, _ = st.downloads[-1]
    assert label == 'Download cleaned CSV'
    assert data == b'Taxon,Count\\na,0.2\\n'
    assert name == 'dataset_1_cleaned_sample.csv'
    assert file.read_bytes() == b'Taxon,Count\\na,0.2\\n'
"""
    new = """    st.session_state.cleaning_review_statuses[_catalog()[2]['id']] = 'Reviewed — no change'
    st.session_state.cleaning_approved_reviewer = 'Reviewer Test'
    render()
    assert st.downloads[-1][4] is False
    label, data, name, mime, _ = st.downloads[-1]
    assert label == 'Download cleaned CSV + complete reviewer audit (ZIP)'
    assert mime == 'application/zip'
    assert name == 'review_package_sample.zip'
    import io, json, zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.read('cleaned/cleaned_sample.csv') == b'Taxon,Count\\na,0.2\\n'
        assert archive.read('source/original_sample.csv') == b'Taxon,Count\\na,0.2\\n'
        audit = json.loads(archive.read('audit/reviewer_audit.json'))
        assert audit['reviewer'] == 'Reviewer Test'
        assert len(audit['finding_reviews']) == 3
    assert file.read_bytes() == b'Taxon,Count\\na,0.2\\n'
"""
    if old not in current:
        return
    test_backup = test.with_suffix('.before_review_zip.bak')
    if not test_backup.exists():
        shutil.copy2(test, test_backup)
    test.write_text(current.replace(old, new, 1), encoding='utf-8')
    print('Updated legacy final-download regression test')


def main() -> None:
    before = APP.read_text(encoding='utf-8')
    after = transform(before)
    if after == before:
        update_legacy_export_test()
        print('ZIP export already installed; no application changes made')
        return
    if not AUDIT_MODULE.exists() and not ASSET.exists():
        raise RuntimeError('Missing bundled reviewer audit dependency')
    if not AUDIT_MODULE.exists():
        AUDIT_MODULE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ASSET, AUDIT_MODULE)
        print(f'Installed missing reviewer audit module: {AUDIT_MODULE}')
    backup = APP.with_suffix('.before_review_zip.bak')
    if not backup.exists():
        shutil.copy2(APP, backup)
    APP.write_text(after, encoding='utf-8')
    update_legacy_export_test()
    print('Installed single ZIP review export. Original app backed up to', backup)
    print('Run: python -m pytest -q --tb=short')


if __name__ == '__main__':
    main()
