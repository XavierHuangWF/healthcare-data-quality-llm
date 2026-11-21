"""Evidence tables have consistent presentation types for Arrow serialization."""
from __future__ import annotations

import pandas as pd

from healthcare_data_quality.ui.evidence_view import measurement_rows_for_display


def test_mixed_observed_values_are_only_coerced_for_display():
    rows = [
        {"Measurement": "Predominant naming style", "Observed": "unprefixed", "How obtained": "Compare taxonomy prefixes"},
        {"Measurement": "Rows classified prefixed", "Observed": 3, "How obtained": "Count taxonomy rows"},
        {"Measurement": "Missing assessment", "Observed": None, "How obtained": "Not available"},
    ]
    rendered = measurement_rows_for_display(rows)
    assert [row["Observed"] for row in rendered] == ["unprefixed", "3", "—"]
    assert all(isinstance(row["Observed"], str) for row in rendered)
    assert rows[1]["Observed"] == 3  # Preserve raw evidence/counting semantics.
    assert rows[2]["Observed"] is None
    assert rendered[0]["Measurement"] == rows[0]["Measurement"]
    df = pd.DataFrame(rendered)
    assert all(isinstance(v, str) for v in df["Observed"])


def test_boolean_float_and_integer_values_remain_readable_in_display():
    rendered = measurement_rows_for_display([
        {"Observed": True}, {"Observed": 0}, {"Observed": 2.75}, {"Observed": "NA"},
    ])
    assert [row["Observed"] for row in rendered] == ["True", "0", "2.75", "NA"]


def test_no_change_to_empty_evidence():
    assert measurement_rows_for_display([]) == []
