"""Two explicit approvals, isolated R delta and exact scoped merge."""
import csv
from io import StringIO
from pathlib import Path

import pytest

from healthcare_data_quality.cleaning.finding_cleaning import (
    APPROVAL_PHRASE, execute_approved_scoped_plan, validate_scoped_plan,
)
from healthcare_data_quality.r_runtime.scoped_r_execution import (
    _r_literal, prepare_scoped_r_preview, execute_reviewed_r_preview,
)


def _file_and_findings(tmp_path):
    source = tmp_path / "research.csv"
    source.write_text('taxon,CountA,CountB\n"NA",0.5,\nA,-1,0002\n', encoding='utf-8')
    fractional = {"id": "dataset_1:qc:1", "dataset_id": "dataset_1",
                  "name": "research.csv", "source": "QC", "actionable": True,
                  "check_key": "find_invalid_counts", "evidence": [
                      {"CSV row": 2, "Column": 2, "Original": "0.5"}]}
    missing = {**fractional, "id": "dataset_1:qc:2", "evidence": [
        {"CSV row": 2, "Column": 3, "Original": ""}]}
    return source, fractional, missing


def _validated(source, finding, row, column, before, after):
    return validate_scoped_plan({"operation": "replace_cells", "dataset_id": "dataset_1",
                                 "changes": [{"row": row, "column": column,
                                              "before": before, "after": after}]}, finding, source)


def fake_r_success(r_function, input_csv, *, preserve_text):
    assert preserve_text is True
    assert "clean_dataset <- function" in r_function
    rows = list(csv.reader(StringIO(input_csv)))
    assert rows[0] == ["kind", "csv_row", "csv_column", "value", "replacement"]
    for row in rows[1:]:
        assert len(row) == 5
        row[3] = row[4]
        row.pop()
    rows[0].pop()
    stream = StringIO(newline="")
    csv.writer(stream).writerows(rows)
    return stream.getvalue()


def test_r_preview_is_not_executed_and_preserves_original_data(tmp_path, monkeypatch):
    source, fractional, _ = _file_and_findings(tmp_path)
    before = source.read_bytes()
    plan = _validated(source, fractional, 2, 2, "0.5", "1")
    monkeypatch.setattr("healthcare_data_quality.r_runtime.scoped_r_execution.run_r_sandbox",
                        lambda *a, **kw: pytest.fail("R ran during preview"))
    preview = prepare_scoped_r_preview(plan, fractional, source)
    # User-provided literals must NOT be inserted into executable R source.
    assert 'df$value <- df$replacement' in preview["r_function"]
    assert '"0.5"' not in preview["r_function"]
    assert '"1"' not in preview["r_function"]
    assert ["cell", "2", "2", "0.5", "1"] in list(csv.reader(StringIO(preview["affected_input_csv"])))
    assert source.read_bytes() == before
    assert len(list(tmp_path.glob("*.csv"))) == 1


def test_execute_verified_r_delta_then_merge_only_approved_cell(tmp_path, monkeypatch):
    source, fractional, missing = _file_and_findings(tmp_path)
    before = source.read_bytes()
    monkeypatch.setattr("healthcare_data_quality.r_runtime.scoped_r_execution.run_r_sandbox", fake_r_success)
    plan = _validated(source, fractional, 2, 2, "0.5", "1")
    preview = prepare_scoped_r_preview(plan, fractional, source)
    result = execute_reviewed_r_preview(preview, plan, fractional, source)
    assert result["status"] == "r_execution_verified"
    # R is sandboxed to affected values. The existing writer alone creates a copy.
    derived = execute_approved_scoped_plan(plan, fractional, source,
                                           tmp_path / "step1", "reviewer", APPROVAL_PHRASE)
    working = Path(derived["output"])
    assert source.read_bytes() == before
    with working.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[1] == ["NA", "1", ""]
    assert rows[2] == ["A", "-1", "0002"]
    # Finding 2 still sees the first cleaned working copy, not original.
    plan2 = _validated(working, missing, 2, 3, "", "0")
    preview2 = prepare_scoped_r_preview(plan2, missing, working)
    execute_reviewed_r_preview(preview2, plan2, missing, working)
    result2 = execute_approved_scoped_plan(plan2, missing, working, tmp_path / "step2",
                                            "reviewer", APPROVAL_PHRASE)
    with Path(result2["output"]).open(newline="") as stream:
        rows2 = list(csv.reader(stream))
    assert rows2[1] == ["NA", "1", "0"]
    assert rows2[2] == ["A", "-1", "0002"]


@pytest.mark.parametrize("fault", ["wrong_value", "wrong_coordinate", "extra_row"])
def test_r_output_can_never_change_unapproved_cell(tmp_path, monkeypatch, fault):
    source, finding, _ = _file_and_findings(tmp_path)
    before = source.read_bytes()
    plan = _validated(source, finding, 2, 2, "0.5", "1")
    preview = prepare_scoped_r_preview(plan, finding, source)

    def fake_bad_r(code, input_csv, *, preserve_text):
        rows = list(csv.reader(StringIO(input_csv)))
        rows[1][3] = "bad" if fault == "wrong_value" else "1"
        if fault == "wrong_coordinate":
            rows[1][2] = "3"
        rows[1].pop()
        rows[0].pop()
        if fault == "extra_row":
            rows.append(list(rows[1]))
        stream = StringIO()
        csv.writer(stream).writerows(rows)
        return stream.getvalue()

    monkeypatch.setattr("healthcare_data_quality.r_runtime.scoped_r_execution.run_r_sandbox", fake_bad_r)
    with pytest.raises(ValueError, match="unexpected value|coordinate|row"):
        execute_reviewed_r_preview(preview, plan, finding, source)
    assert source.read_bytes() == before
    assert not (tmp_path / "step1").exists()


def test_source_changed_or_preview_tampered_is_rejected_without_r(tmp_path, monkeypatch):
    source, finding, _ = _file_and_findings(tmp_path)
    plan = _validated(source, finding, 2, 2, "0.5", "1")
    preview = prepare_scoped_r_preview(plan, finding, source)
    monkeypatch.setattr("healthcare_data_quality.r_runtime.scoped_r_execution.run_r_sandbox",
                        lambda *a, **kw: pytest.fail("R must not run"))
    modified_preview = {**preview, "r_function": "clean_dataset <- function(df) { df }"}
    with pytest.raises(PermissionError, match="stale or altered"):
        execute_reviewed_r_preview(modified_preview, plan, finding, source)
    source.write_bytes(source.read_bytes().replace(b"0.5", b"0.8"))
    with pytest.raises(ValueError, match="Original cell mismatch"):
        execute_reviewed_r_preview(preview, plan, finding, source)


def test_scoped_r_header_rename_preview(tmp_path, monkeypatch):
    source = tmp_path / "headers.csv"
    source.write_text(",sample\nNA,0007\n", encoding="utf-8")
    finding = {"id": "dataset_1:qc:9", "dataset_id": "dataset_1",
               "check_key": "inspect_headers", "actionable": True,
               "evidence": [{"Column": 1}]}
    plan = validate_scoped_plan({"operation": "rename_headers", "dataset_id": "dataset_1",
                                 "changes": [{"position": 1, "before": "", "after": "taxon"}]}, finding, source)
    preview = prepare_scoped_r_preview(plan, finding, source)
    assert '"header"' in preview["r_function"]
    def fake_r(code, input_csv, *, preserve_text):
        rows = list(csv.reader(StringIO(input_csv)))
        rows[1][3] = rows[1][4]
        rows[1].pop()
        rows[0].pop()
        stream = StringIO()
        csv.writer(stream).writerows(rows)
        return stream.getvalue()
    monkeypatch.setattr("healthcare_data_quality.r_runtime.scoped_r_execution.run_r_sandbox", fake_r)
    assert execute_reviewed_r_preview(preview, plan, finding, source)["checked_changes"] == 1


def test_r_literal_quotes_and_unicode_are_safe():
    assert _r_literal('before"; system("bad")') == '"before\\"; system(\\"bad\\")"'
    assert _r_literal("NA") == '"NA"'
    assert _r_literal("é") == '"\\u00E9"'
    assert _r_literal("\n") == '"\\n"'
    assert _r_literal("💡") == '"\\U0001F4A1"'
    with pytest.raises(ValueError):
        _r_literal("bad\x00value")


def test_r_sandbox_uses_string_preservation_only_when_requested(monkeypatch):
    import subprocess
    from healthcare_data_quality.r_runtime import r_sandbox
    observed = []
    class FakeCompleted:
        returncode = 0
    def fake_run(command, input, stdout, stderr, **kwargs):
        observed.append(command[-1])
        stdout.write('"x"\n"ok"\n')
        return FakeCompleted()
    monkeypatch.setattr(subprocess, "run", fake_run)
    r_sandbox.run_r_sandbox('clean_dataset <- function(df) { df }', "x\nok\n")
    r_sandbox.run_r_sandbox('clean_dataset <- function(df) { df }', "x\nok\n",
                            preserve_text=True)
    assert 'colClasses="character"' not in observed[0]
    assert 'colClasses="character"' in observed[1]
    assert 'na.strings=character(0)' in observed[1]

class ReviewUI:
    """Extend the existing fake Streamlit without a live browser."""
    def __init__(self):
        from test_cleaning_review_cards import StubUI
        self._inner = StubUI()
        self.session_state = self._inner.session_state
        self.buttons_on = self._inner.buttons_on
        self.exec_submit = False
        self.code_calls = []
        self.expanders = self._inner.expanders

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def dataframe(self, *args, **kwargs):
        pass

    def code(self, script, *, language=None):
        self.code_calls.append((script, language))

    def text_input(self, *args, **kwargs):
        return "researcher"

    def checkbox(self, label, *args, **kwargs):
        return label == "Review R cleaning and insertion code"

    def form_submit_button(self, label, *args, **kwargs):
        return self.exec_submit and label.startswith("Yes — Apply 1 verified cell change") and not kwargs.get("disabled", False)


def test_ui_automatically_runs_isolated_r_then_merges_only_after_final_approval(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test_cleaning_review_cards import _app
    ui = ReviewUI()
    app = _app(monkeypatch, ui)
    source, finding, _ = _file_and_findings(tmp_path)
    finding.update({"title": "2 fractional count values", "method": "Inspect counts"})
    original = source.read_bytes()
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: [finding])
    r_calls = []
    def tracked_r(code, input_csv, *, preserve_text):
        r_calls.append(input_csv)
        return fake_r_success(code, input_csv, preserve_text=preserve_text)
    monkeypatch.setattr("healthcare_data_quality.r_runtime.scoped_r_execution.run_r_sandbox", tracked_r)
    files = [SimpleNamespace(name=source.name)]
    app.render_cleaning_workspace([source], files, {}, "stable-context", [])
    plan = _validated(source, finding, 2, 2, "0.5", "1")
    ui.session_state.cleaning_plans_by_finding[finding["id"]] = plan
    ui.session_state.cleaning_discussion_mode_by_finding[finding["id"]] = "accepted"
    ui.session_state.cleaning_approved_reviewer = "researcher"
    # The first rerender executes R against affected values, without touching
    # the source or the working copy.
    app.render_cleaning_workspace([source], files, {}, "stable-context", [])
    prepared = ui.session_state.cleaning_r_previews_by_finding[finding["id"]]
    assert len(r_calls) == 1
    assert prepared["r_result"]["status"] == "r_execution_verified"
    assert ui.code_calls == []  # R proof is verified internally, not displayed
    assert not ui.session_state.cleaning_history
    assert source.read_bytes() == original
    # The second and final user action inserts only the verified R output.
    ui.exec_submit = True
    monkeypatch.setattr(app.Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(StopIteration, match="rerun"):
        app.render_cleaning_workspace([source], files, {}, "stable-context", [])
    assert len(r_calls) == 1
    assert len(ui.session_state.cleaning_history) == 1
    assert ui.session_state.cleaning_history[0]["finding_id"] == finding["id"]
    assert ui.session_state.cleaning_working_paths == {}
    assert ui.session_state.cleaning_result is None
    assert not (tmp_path / 'cleaning-workspace').exists()
    records = list(csv.reader(StringIO(
        ui.session_state.cleaning_staged_bytes_by_dataset["dataset_1"].decode('utf-8'))))
    assert records[1] == ["NA", "1", ""]
    assert records[2] == ["A", "-1", "0002"]
    assert source.read_bytes() == original
