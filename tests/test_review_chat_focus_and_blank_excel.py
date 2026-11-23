"""Regression tests for duplicate Agent B controls, keyboard focus, blank Excel headers."""
import csv
import io
from pathlib import Path

import pytest
from openpyxl import load_workbook

from healthcare_data_quality.cleaning.final_export_integrity import verify_final_excel
from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download
from healthcare_data_quality.ui.review_input_focus import (
    focus_reviewer_chat, review_chat_container_key, review_chat_focus_html,
)

ROOT = Path(__file__).resolve().parents[1]


def _csv(*rows):
    buf = io.StringIO(newline='')
    csv.writer(buf).writerows(rows)
    return buf.getvalue().encode('utf-8')


def _excel(csv_data):
    return build_styled_excel_download(
        csv_data, original_csv=csv_data, original_filename='data.csv',
        approved_flags=[], approved_row_operations=[], approved_column_deletions=[],
    )[1]


def test_intentionally_empty_header_is_valid_in_verified_excel():
    source = _csv(['', 'Count', 'Other'], ['Bacteria;A', '0', ''],
                  ['Bacteria;B', '', '3'])
    workbook = _excel(source)
    sheet = load_workbook(io.BytesIO(workbook)).active
    assert sheet['A1'].value is None  # XLSX empty string is read as None.
    assert sheet['B3'].value is None  # Same applies to missing count cells.
    verify_final_excel(workbook, source, original_csv=source,
                       approved_flags=[], approved_rows=[], approved_columns=[])


def test_empty_header_normalization_does_not_hide_wrong_export_values():
    source = _csv(['', 'Count'], ['Bacteria;A', '1'])
    workbook = _excel(source)
    wrong_source = _csv(['Actual Header', 'Count'], ['Bacteria;A', '1'])
    with pytest.raises(PermissionError, match='R1:C1'):
        verify_final_excel(workbook, wrong_source, original_csv=wrong_source,
                           approved_flags=[], approved_rows=[], approved_columns=[])
    wrong_data = _csv(['', 'Count'], ['Bacteria;A', '999'])
    with pytest.raises(PermissionError, match='R2:C2'):
        verify_final_excel(workbook, wrong_data, original_csv=wrong_data,
                           approved_flags=[], approved_rows=[], approved_columns=[])


def test_no_button_focus_target_is_isolated_to_finding():
    a = review_chat_container_key('dataset_1:qc:4')
    b = review_chat_container_key('dataset_1:qc:5')
    assert a != b
    html = review_chat_focus_html('dataset_1:qc:4')
    assert 'st-key-' + a in html
    assert 'st-key-' + b not in html
    assert 'document.activeElement === input' in html
    assert 'MutationObserver' in html
    assert 'input.focus(' in html
    assert '.click(' not in html
    assert 'submit(' not in html


def test_focus_is_requested_only_on_supported_streamlit():
    class Modern:
        def __init__(self):
            self.calls = []
        def html(self, source, **kwargs):
            self.calls.append((source, kwargs))
    modern = Modern()
    assert focus_reviewer_chat(modern, 'finding')
    assert len(modern.calls) == 1
    assert modern.calls[0][1] == {'unsafe_allow_javascript': True}
    class Old:
        def html(self, source):
            pass
    assert not focus_reviewer_chat(Old(), 'finding')
    assert not focus_reviewer_chat(object(), 'finding')


def test_one_render_owner_when_agent_b_becomes_ready():
    text = (ROOT / 'app.py').read_text()
    assert 'st.fragment(run_every=None if fid in ready_ids else "3s")' not in text
    assert 'st.fragment(run_every="3s")(_render_live_agent_b_finding)' not in text
    assert 'st.fragment(run_every="3s")(_refresh_parent_for_new_agent_b_choices)' in text
    assert 'st.session_state.setdefault("cleaning_chat_focus_pending", set()).add(fid)' in text
    assert 'with st.container(key=review_chat_container_key(fid)):' in text
