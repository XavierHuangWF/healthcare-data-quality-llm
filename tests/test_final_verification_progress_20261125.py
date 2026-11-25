"""Final-export feedback must describe real work without weakening its gate."""
import ast
from pathlib import Path

import pytest
from healthcare_data_quality.ui.investigation_progress import final_verification_label


@pytest.mark.parametrize(('stage', 'phrase'), [
    ('workbook', '1/3 · Preparing and checking Excel'),
    ('gemma', '2/3 · Gemma verifying approved changes'),
    ('seal', '3/3 · Checking final Excel and audit formatting'),
])
def test_actual_stage_and_elapsed(stage, phrase):
    assert phrase in final_verification_label(stage, 65)
    assert final_verification_label(stage, 65).endswith('Elapsed 01:05')
    assert final_verification_label(stage, 71).endswith('Elapsed 01:11')


def test_queue_reconnect_retry_and_long_duration():
    assert 'Queued' in final_verification_label('workbook', 3, queued=True)
    assert 'Waiting for Gemma inference slot' in final_verification_label('gemma', 25, queued=True)
    assert 'Waiting for Gemma inference slot' in final_verification_label('gemma', 25, model_stage='waiting_for_inference_slot')
    label = final_verification_label('gemma', 3661, model_stage='waiting_for_reconnect', attempt=2)
    assert 'Waiting for Gemma connection' in label and 'Retry 1' in label
    assert label.endswith('Elapsed 01:01:01')
    with pytest.raises(ValueError):
        final_verification_label('unknown', 0)


def test_export_status_wiring_and_gate():
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    tree = ast.parse(source)
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'show_final_progress')
    body = ast.get_source_segment(source, fn)
    assert "st.status(label, state='running', expanded=False)" in body
    assert 'monotonic() - started_at' in body
    # All pending workers use the same candidate-scoped timer and retain their
    # early return before the verified artifact and download controls.
    export = source[source.index("candidate = {'ledger_key':ledger_key,"):source.index('def _prepare_finding_discussion(')]
    assert "candidate.setdefault('verification_started_at', monotonic())" in export
    for stage, guard in [('workbook', "if not candidate.get('proof'):"),
                         ('gemma', "if not record.get('result'):"),
                         ('seal', "if 'verified_excel' not in record:")]:
        start = export.index(guard)
        status = export.index(f"show_final_progress('{stage}'", start)
        assert 'return' in export[status:export.index("file_format = st.selectbox", status)]
    assert export.index("hashlib.sha256(record['verified_excel'])") < export.index("record.setdefault('verification_finished_at'") < export.index('st.download_button(')
    assert 'format_elapsed(finished_at - started_at)' in export
