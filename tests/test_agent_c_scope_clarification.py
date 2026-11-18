"""Agent C disambiguates cell clearing versus structural row deletion."""

from __future__ import annotations

import json

from healthcare_data_quality.cleaning_workspace import (
    _clarify_ambiguous_row_edit,
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
    assert "drop_column" in observed[0].lower()


def test_distinct_row_edit_does_not_conflict():
    assert _clarify_ambiguous_row_edit(
        "Delete entire CSV row 11 and replace row 12 column 3 with 5"
    ) is None
