"""Regression: a source value can never break generated R syntax."""
import csv
from io import StringIO
import pytest

from healthcare_data_quality.r_runtime.scoped_r_execution import _generate_r, _as_csv


def test_compiler_keeps_all_untrusted_values_out_of_r_code():
    inputs = [
        'NA', '', 'abc\"; stop(\"bad\") #', 'comma,quoted',
        'one\ntwo', 'x\\y', '日本語', '💡', ' leading ',
    ]
    changes = [
        {'kind': 'cell', 'csv_row': str(i+2), 'csv_column': '3',
         'before': before, 'after': before + ' → NA'}
        for i, before in enumerate(inputs)
    ]
    script = _generate_r(changes)
    # The fixed template may legitimately contain R language tokens such as
    # anyNA; exclude those short incidental words from substring comparison.
    for text in inputs:
        if text and len(text) > 3:
            assert text not in script
    assert 'df$value <- df$replacement' in script
    table = list(csv.reader(StringIO(_as_csv(changes, result=False)), strict=True))
    assert table[0] == ['kind','csv_row','csv_column','value','replacement']
    assert [row[3] for row in table[1:]] == inputs
    assert [row[4] for row in table[1:]] == [v + ' → NA' for v in inputs]
    expected = list(csv.reader(StringIO(_as_csv(changes, result=True)), strict=True))
    assert expected[0] == ['kind','csv_row','csv_column','value']
    assert [row[3] for row in expected[1:]] == [v + ' → NA' for v in inputs]


def test_compiler_supports_618_verified_values_without_growing_r_code():
    rows = [dict(kind='cell', csv_row=str(i+2), csv_column='3',
                 before='', after='NA') for i in range(618)]
    code = _generate_r(rows)
    assert '618L' in code
    assert len(code) < 2000
    assert len(list(csv.reader(StringIO(_as_csv(rows,result=False))))) == 619
    # This tests the *compiler* capacity, not the 32-change scoped approval cap.


def test_bad_kind_is_rejected():
    with pytest.raises(ValueError):
        _generate_r([dict(kind='arbitrary', before='x', after='y')])
