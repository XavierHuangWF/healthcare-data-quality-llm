"""Agent C disambiguates cell clearing versus structural row deletion."""

from __future__ import annotations

import json

from healthcare_data_quality.cleaning_workspace import (
    _clarify_ambiguous_row_edit,
    _clarify_ambiguous_column_edit,
    ask_gemma_about_cleaning,
)


def test_ambiguous_delete_and_replace_asks_one_scope_question_without_model(monkeypatch):
    def model_should_not_be_called(*args, **kwargs):
        raise AssertionError("No model call should be needed for unsafe ambiguity")

    monkeypatch.setattr(
        "healthcare_data_quality.cleaning_workspace.query_local_model",
        model_should_not_be_called,
    )
    answer = ask_gemma_about_cleaning(
        "remove entire row 11, column 7 and replace row 11 column 11 with 5",
        [], [], [],
        selected_finding={"id": "dataset_1:qc:1"},
    )
    assert answer["operation"] == "none"
    assert answer["dataset_id"] == ""
    assert answer["changes"] == []
    reply = answer["reply"]
    assert "row 11" in reply and "column 7" in reply
    assert "clear only that cell" in reply.lower()
    assert "delete the entire csv row 11" in reply.lower()
    assert "another cell" in reply.lower()
    assert "empty/NA" in reply
    assert "not permitted" not in reply.lower()
    assert "i cannot" not in reply.lower()


def test_row_removal_and_same_row_update_are_conflicting():
    clarification = _clarify_ambiguous_row_edit(
        "Delete entire CSV row 15 and change row 15 column 11 to 5"
    )
    assert clarification is not None
    assert "deleting row 15 removes all" in clarification.lower()
    assert "keep the row" in clarification.lower()
    assert "delete csv row 15 entirely" in clarification.lower()


def test_not_all_delete_requests_are_ambiguous(monkeypatch):
    model_reply = {
        "reply": "What replacement should be used for the selected cell?",
        "operation": "none",
        "dataset_id": "",
        "changes": [],
    }
    observed = []

    def fake_model(prompt, **kwargs):
        observed.append(prompt)
        return json.dumps(model_reply)

    monkeypatch.setattr(
        "healthcare_data_quality.cleaning_workspace.query_local_model",
        fake_model,
    )
    assert _clarify_ambiguous_row_edit("Clear only cell row 11 column 7 to NA") is None
    result = ask_gemma_about_cleaning(
        "Clear only cell row 11 column 7 to NA", [], [], []
    )
    assert result == model_reply
    assert len(observed) == 1
    assert "Do not start with" in observed[0]
    assert "only rename_headers and replace_cells" in observed[0].lower()


def test_distinct_row_edit_does_not_conflict():
    assert _clarify_ambiguous_row_edit(
        "Delete entire CSV row 11 and replace row 12 column 3 with 5"
    ) is None


def test_column_assignment_with_no_row_asks_scope_without_model_or_rationale(monkeypatch):
    def model_should_not_run(*args, **kwargs):
        raise AssertionError("Ambiguous column-wide edit must not call Gemma")

    monkeypatch.setattr(
        "healthcare_data_quality.cleaning_workspace.query_local_model",
        model_should_not_run,
    )
    result = ask_gemma_about_cleaning(
        "Remove column 7 and make the value of column 11 to 5",
        [], [], [], selected_finding={"id": "dataset_1:qc:1"},
    )
    assert result["operation"] == "none"
    assert result["changes"] == []
    text = result["reply"].lower()
    assert "column 11" in text
    assert "**5**" in text
    assert "every data row" in text and "specific rows" in text
    assert "column 7" in text and "entire column" in text
    assert "row 11 column 11" not in text
    assert "scientific justification" not in text
    assert "structural column deletion" in text


def test_explicit_row_and_all_rows_do_not_trigger_column_scope_clarification():
    assert _clarify_ambiguous_column_edit(
        "Set row 11 column 11 to 5"
    ) is None
    assert _clarify_ambiguous_column_edit(
        "Set every data row in column 11 to 5"
    ) is None
    assert _clarify_ambiguous_column_edit(
        "Change values in column 11 to 5 for all rows"
    ) is None
    assert _clarify_ambiguous_column_edit(
        "Set all values in column 11 to 5"
    ) is None
    assert _clarify_ambiguous_column_edit(
        "Set entire column 11 to 5"
    ) is None
    assert _clarify_ambiguous_column_edit(
        "Remove column 7"
    ) is None


def test_user_provided_numeric_replacement_does_not_require_scientific_rationale(monkeypatch):
    prompts = []
    def fake_model(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({
            "reply": "You specified a replacement of 5 for CSV row 11, column 11. "
                     "I can prepare it for review if that cell is in this finding's evidence.",
            "operation": "none", "dataset_id": "", "changes": [],
        })

    monkeypatch.setattr(
        "healthcare_data_quality.cleaning_workspace.query_local_model", fake_model,
    )
    result = ask_gemma_about_cleaning(
        "Replace row 11 column 11 from 0.5 to 5", [], [], [],
    )
    assert result["operation"] == "none"  # fake output; no executor called
    assert len(prompts) == 1
    assert "do not demand a scientific justification" in prompts[0].lower()
    assert "don't demand a" in prompts[0].lower()
    assert "do not mistake a column number for a row number" in prompts[0].lower()


def test_deleting_whole_other_column_does_not_hide_row_ambiguity():
    text = _clarify_ambiguous_column_edit(
        "Remove the entire column 7 and change column 11 to 5"
    )
    assert text is not None
    assert "every data row" in text
    assert "column 7" in text


def test_decimal_replacement_is_preserved_in_clarifying_question():
    reply = _clarify_ambiguous_column_edit("Set column 11 to 0.5")
    assert "**0.5**" in reply
