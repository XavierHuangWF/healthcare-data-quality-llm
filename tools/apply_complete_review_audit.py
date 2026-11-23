"""Idempotent patcher: enable review audit by default and attach verifiable exports.

Usage: python tools/apply_complete_review_audit.py
Backs up app.py before modifying; does not replace unrelated UI/agent logic.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


def transform(s: str) -> str:
    marker = 'from healthcare_data_quality.cleaning.reviewer_audit import build_reviewer_audit'
    # Toggle is on by default when reviewer opens a completed finding.
    s = re.sub(
        r'if st\.checkbox\("Show completed finding audit",\s*key=f"review_completed_audit_\{fid\}"\):',
        'if st.checkbox("Show completed finding audit",\n                               value=True, key=f"review_completed_audit_{fid}"):',
        s,
    )
    if marker in s:
        if 'value=True, key=f"review_completed_audit_{fid}"' not in s:
            raise RuntimeError("Could not locate completed audit checkbox")
        return s

    # The cleaned data must be exportable in BOTH formats even when the only
    # approved flags were later removed. Audited Excel does not require flags.
    old = '''    want_excel = False
    if has_visual_flags:
        # Keep CSV available for pipelines, but make formatting-preserving XLSX
        # the first, default option. Neither format gets a second download button.
'''
    new = '''    want_excel = False
    if hasattr(st, "selectbox"):
        # Excel is always available because it includes the complete named
        # reviewer audit, even if all flagged cells were later deleted.
'''
    if old not in s:
        raise RuntimeError("Download format block changed; not patching unknown layout")
    s = s.replace(old, new, 1)
    old = '''            file_name, mime, data = build_final_download(datasets_for_download)
            if want_excel:
'''
    new = '''            file_name, mime, data = build_final_download(datasets_for_download)
            audit_record = None
            audit_data = None
            if len(datasets_for_download) == 1:
                from healthcare_data_quality.cleaning.reviewer_audit import build_reviewer_audit, reviewer_audit_json
                audit_record = build_reviewer_audit(
                    original_csv=source_for_export, cleaned_csv=datasets_for_download[0][2],
                    original_filename=uploaded[0].name,
                    history=st.session_state.get("cleaning_history") or [],
                    decisions=st.session_state.get("cleaning_decisions") or [],
                    review_statuses=st.session_state.get("cleaning_review_statuses") or {},
                    conversations=st.session_state.get("cleaning_messages_by_finding") or {},
                    approved_row_operations=st.session_state.cleaning_approved_rows_by_dataset.get("dataset_1", []),
                    approved_column_deletions=st.session_state.cleaning_approved_columns_by_dataset.get("dataset_1", []),
                    completed_audits=st.session_state.get("cleaning_completed_audits_by_finding") or {},
                    finding_titles={item["id"]: item.get("title", "") for item in catalog},
                    reviewer=st.session_state.get("cleaning_approved_reviewer", ""),
                )
                audit_data = reviewer_audit_json(audit_record)
            if want_excel:
'''
    if old not in s:
        raise RuntimeError("Final download preparation block changed; refusing overwrite")
    s = s.replace(old, new, 1)
    old = '                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"\n'
    new = '''                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if audit_record is not None:
                    from healthcare_data_quality.cleaning.reviewer_audit import embed_reviewer_audit_sheets
                    data = embed_reviewer_audit_sheets(data, audit_record)
'''
    if old not in s:
        raise RuntimeError("Excel MIME anchor changed; refusing overwrite")
    s = s.replace(old,new,1)
    needle = '''                    type="primary", disabled=False,
                )
'''
    new = '''                    type="primary", disabled=False,
                )
                if audit_data is not None:
                    st.download_button(
                        "Download complete reviewer audit (JSON)",
                        data=audit_data,
                        file_name="reviewer_audit_" + uploaded[0].name[:-4] + ".json",
                        mime="application/json",
                        key=f"cleaning_reviewer_audit_{session_context}",
                        type="secondary", disabled=False,
                    )
'''
    # Scoped to the main final download only; do not attach under others.
    start = s.index('    # One export control.')
    end = s.index('\n\ndef _prepare_finding_discussion(',start)
    sub=s[start:end]
    if sub.count(needle)!=1:
        raise RuntimeError("Cannot uniquely locate final download control")
    sub=sub.replace(needle,new,1)
    s=s[:start]+sub+s[end:]
    return s


def main():
    before=APP.read_text(encoding="utf-8")
    after=transform(before)
    if after==before:
        print("Already installed; no changes")
        return
    backup=APP.with_suffix(".pre_review_audit.bak")
    if not backup.exists():
        shutil.copy2(APP,backup)
    APP.write_text(after,encoding="utf-8")
    print("Installed reviewer audit export and completed-review defaults; backup:",backup)


if __name__ == '__main__':
    main()
