"""Reproducible R insertion remains compact and source-scope verified."""
import csv
from io import StringIO
import pytest

from healthcare_data_quality.scoped_r_execution import (
    _r_insertion_function, _as_csv, _affected_changes,
    prepare_and_execute_scoped_r_preview, verify_prepared_r_insertion,
)
from healthcare_data_quality.finding_cleaning import validate_scoped_plan


def _plan(n=618):
    return {
        "operation": "replace_cells", "dataset_id": "dataset_1",
        "changes": [{"row": i + 2, "column": 7, "before": "", "after": "NA"} for i in range(n)]
    }


def test_insertion_uses_compact_column_batches_for_618_positions():
    single = _r_insertion_function(_plan(1))
    large = _r_insertion_function(_plan(618))
    assert single == large
    assert len(large.encode('utf-8')) < 2500
    assert 'for (col in unique(cols))' in large
    assert 'df[[col]][rows[i]] <- verified_changes$replacement[i]' in large
    assert 'stopifnot(identical(unname(df[[col]][rows[i]])' in large
    assert 'df[[7]][[2]]' not in large
    assert '"NA"' not in large
    changes_csv = _as_csv(_affected_changes(_plan(618)), result=False)
    assert len(list(csv.reader(StringIO(changes_csv), strict=True))) == 619


def test_literal_quoted_values_stay_out_of_insertion_source():
    plan = _plan(1)
    plan['changes'][0]['before'] = '"hello" \u2603 \n code'
    plan['changes'][0]['after'] = 'm,complex\nvalue'
    code = _r_insertion_function(plan)
    assert plan['changes'][0]['before'] not in code
    assert plan['changes'][0]['after'] not in code
    assert 'm,complex' in _as_csv(_affected_changes(plan), result=False)


def test_header_renames_are_vectorized_and_checked():
    plan = {"operation": "rename_headers", "changes": [
        {"position": 1, "before": "", "after": "Taxon"},
        {"position": 4, "before": "X", "after": "SampleX"},
    ]}
    code = _r_insertion_function(plan)
    assert 'names(df)[cols] <- verified_changes$replacement' in code
    assert 'names(df)[[1]]' not in code
    assert 'anyDuplicated(cols)' in code


def test_tampered_verified_insertion_manifest_is_rejected(tmp_path, monkeypatch):
    from healthcare_data_quality import scoped_r_execution as scoped
    source = tmp_path/'data.csv'
    source.write_text('Taxon,Count\na,0.2\n', encoding='utf-8')
    finding = {"id":"dataset_1:qc:1", "dataset_id":"dataset_1", "check_key":"find_invalid_counts",
               "actionable":True, "evidence":[{"CSV row":2,"Column":2,"Original":"0.2"}]}
    plan = validate_scoped_plan({"dataset_id":"dataset_1", "operation":"replace_cells",
            "changes":[{"row":2,"column":2,"before":"0.2","after":"NA"}]}, finding, source)
    def fake_r(code, synthetic, *, preserve_text):
        values = list(csv.reader(StringIO(synthetic), strict=True))
        output = StringIO(newline='')
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(['kind','csv_row','csv_column','value'])
        for kind,row,col,before,after in values[1:]:
            writer.writerow([kind,row,col,after])
        return output.getvalue()
    monkeypatch.setattr(scoped, 'run_r_sandbox', fake_r)
    prepared = prepare_and_execute_scoped_r_preview(plan, finding, source)
    assert verify_prepared_r_insertion(prepared, plan, finding, source) == prepared['insertion_plan']
    assert len(prepared['insertion_r_function']) < 2500
    modified = {**prepared, 'insertion_changes_csv': prepared['insertion_changes_csv'].replace('0.2','0.8')}
    with pytest.raises(PermissionError, match='change table'):
        verify_prepared_r_insertion(modified, plan, finding, source)
