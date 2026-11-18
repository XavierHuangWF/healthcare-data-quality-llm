"""Explicit structural column removals are export-only and reviewer-authorized."""
import csv
from io import StringIO
from types import SimpleNamespace

import pytest

from healthcare_data_quality.in_memory_cleaning import (
    apply_approved_column_drops, apply_verified_in_memory, authorize_column_drop,
    build_final_download, sha256_bytes,
)
from healthcare_data_quality.cleaning_workspace import ask_gemma_about_cleaning


def _approval(source, column=3):
    return authorize_column_drop(
        source, dataset_id="dataset_1", finding_id="dataset_1:qc:1",
        position=column, reviewer="researcher",
    )


def test_second_flagged_column_is_deleted_from_all_rows_only_on_export():
    original = ("Taxon,ABb100,ABC104,End\n"
                "a,0.2,0.5,x\n"
                "b,1,99,y\n").encode("utf-8")
    approved_cell = {
        "operation": "replace_cells", "dataset_id": "dataset_1",
        "finding_id": "dataset_1:qc:1", "source_sha256": sha256_bytes(original),
        "changes": [{"row": 2, "column": 2, "before": "0.2", "after": "2"}],
    }
    staged, _ = apply_verified_in_memory(original, original, approved_cell, reviewer="researcher")
    assert b"ABC104" in staged and b"0.5" in staged
    approved_drop = _approval(original)
    exported = apply_approved_column_drops(
        original, staged, [approved_drop], dataset_id="dataset_1")
    assert list(csv.reader(StringIO(exported.decode()))) == [
        ["Taxon", "ABb100", "End"], ["a", "2", "x"], ["b", "1", "y"]
    ]
    assert original.endswith(b"b,1,99,y\n")
    assert b"ABC104" in staged  # Evidenced original coordinates remain stable
    filename, _, data = build_final_download([("dataset_1", "counts.csv", exported)])
    assert filename.endswith("counts.csv") and data == exported


def test_stale_source_and_different_dataset_fail_closed():
    original = b"a,b,c\n1,2,3\n"
    approved = _approval(original)
    with pytest.raises(PermissionError, match="stale"):
        apply_approved_column_drops(b"a,b,c\n1,2,4\n", original,
                                    [approved], dataset_id="dataset_1")
    with pytest.raises(PermissionError, match="stale"):
        apply_approved_column_drops(original, original, [approved], dataset_id="dataset_2")
    with pytest.raises(PermissionError, match="stale"):
        apply_approved_column_drops(original, original, [approved, approved], dataset_id="dataset_1")
    with pytest.raises(PermissionError, match="dimensions"):
        apply_approved_column_drops(original, b"a,b,c,d\n1,2,3,4\n",
                                    [approved], dataset_id="dataset_1")


def test_cannot_drop_last_column_or_approve_without_reviewer():
    original = b"Taxon,Count\na,2\n"
    with pytest.raises(ValueError, match="at least one column"):
        authorize_column_drop(b"single\nvalue\n", dataset_id="dataset_1",
                              finding_id="qc", position=1, reviewer="r")
    with pytest.raises(PermissionError, match="Reviewer"):
        authorize_column_drop(original, dataset_id="dataset_1", finding_id="qc",
                              position=2, reviewer="")
    dropped = _approval(original, column=2)
    assert apply_approved_column_drops(original, original, [dropped], dataset_id="dataset_1") == b"Taxon\na\n"


def test_original_columns_are_stable_through_sequential_drops_and_bom():
    original = b"\xef\xbb\xbfA,B,C,D\n0,1,2,3\n"
    first = _approval(original, column=2)
    second = _approval(original, column=4)
    output = apply_approved_column_drops(original, original, [first, second], dataset_id="dataset_1")
    assert output == b"\xef\xbb\xbfA,C\n0,2\n"
    assert apply_approved_column_drops(original, original, [], dataset_id="dataset_1") == original


def test_column_request_is_accurately_described_to_gemma(monkeypatch):
    import json
    observed = []
    monkeypatch.setattr(
        "healthcare_data_quality.cleaning_workspace.query_local_model",
        lambda prompt, **kw: observed.append(prompt) or json.dumps({
            "operation": "none", "reply": "Use the separate final-export deletion control.",
            "dataset_id": "", "changes": [],
        }),
    )
    result = ask_gemma_about_cleaning(
        "change 0.2 to 2 and remove the whole column 11", [], [], [],
    )
    assert "separate final-export deletion" in result["reply"]
    assert "whole-column" in observed[0].lower()
    assert "do not refuse" in observed[0].lower()


def test_streamlit_column_deletion_requires_explicit_click(monkeypatch, tmp_path):
    from test_cleaning_review_cards import StubUI, _app, Context

    class ColumnUI(StubUI):
        def __init__(self):
            super().__init__()
            self.submit_drop = False
            self.chosen_column = 3
            self.success_messages = []

        def checkbox(self, label, *, key=None, **kwargs):
            return key == "review_delete_col_toggle_dataset_1:qc:1"

        def selectbox(self, label, options, **kwargs):
            # After approval the selected column is no longer eligible.
            return self.chosen_column if self.chosen_column in options else 0

        def text_input(self, *args, **kwargs):
            return "researcher"

        def form_submit_button(self, label, *, disabled=False, **kwargs):
            return label.startswith("Approve deletion") and self.submit_drop and not disabled

        def success(self, message):
            self.success_messages.append(message)

    st = ColumnUI()
    app = _app(monkeypatch, st)
    path = tmp_path / "original.csv"
    original = b"Taxon,ABb100,ABC104\na,0.2,0.5\nb,10,11\n"
    path.write_bytes(original)
    finding = {
        "id": "dataset_1:qc:1", "dataset_id": "dataset_1", "name": "original.csv",
        "source": "QC", "title": "Two invalid cells",
        "check_key": "find_invalid_counts", "method": "Inspect source",
        "evidence": [{"CSV row": 2, "Column": 2},
                     {"CSV row": 2, "Column": 3}], "actionable": True,
    }
    monkeypatch.setattr("healthcare_data_quality.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: [finding])

    class AgentB:
        def snapshot(self, _):
            return {"entries": {}}

    monkeypatch.setattr("cleaning_jobs.recommendation_manager", lambda: AgentB())
    def render():
        return app.render_cleaning_workspace(
            [path], [SimpleNamespace(name="original.csv")], {}, "audit-drop", [])

    render()  # No approval means no deletion
    assert st.session_state.cleaning_approved_column_drops_by_dataset["dataset_1"] == []
    st.submit_drop = True
    with pytest.raises(StopIteration, match="rerun"):
        render()
    drops = st.session_state.cleaning_approved_column_drops_by_dataset["dataset_1"]
    assert len(drops) == 1 and drops[0]["position"] == 3
    assert path.read_bytes() == original
    st.submit_drop = False
    render()  # Approval persists without being duplicated
    assert len(drops) == 1
    assert st.success_messages
