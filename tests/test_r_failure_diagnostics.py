"""R failures must be traceable in the UI without bypassing its sandbox."""
from pathlib import Path
import subprocess

import pytest

from healthcare_data_quality.r_runtime.r_diagnostics import (
    RSandboxFailure, classify_r_process_failure, diagnose_r_preview_failure,
    safe_r_stderr,
)
from healthcare_data_quality.r_runtime.r_sandbox import run_r_sandbox

R_FN = 'clean_dataset <- function(df) { df }'
IN = 'kind,csv_row,csv_column,value\ncell,2,3,0.2\n'


def test_failed_r_captures_actual_exit_code_and_safe_stderr(monkeypatch):
    def fail(*args, stderr=None, **kwargs):
        stderr.write('Error: Source/evidence mismatch at approved change 3\nExecution halted\n')
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(RSandboxFailure) as result:
        run_r_sandbox(R_FN, IN, preserve_text=True)
    assert result.value.code == 'R_EVIDENCE_MISMATCH'
    assert result.value.exit_code == 1
    assert 'Source/evidence mismatch' in result.value.stderr_excerpt


def test_failed_r_timeout_includes_action_and_does_not_claim_syntax_error(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired('bwrap', 15)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(RSandboxFailure) as result:
        run_r_sandbox(R_FN, IN)
    assert result.value.code == 'R_TIMEOUT'
    assert '15-second' in result.value.summary
    assert result.value.exit_code is None


def test_missing_bwrap_is_identified_without_disabling_sandbox(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs:
                        (_ for _ in ()).throw(FileNotFoundError('bwrap')))
    with pytest.raises(RSandboxFailure) as result:
        run_r_sandbox(R_FN, IN)
    assert result.value.code == 'R_LAUNCH_MISSING'
    assert 'Install or verify' in result.value.action


def test_bwrap_namespace_rejection_is_recognized():
    result = classify_r_process_failure(
        stderr='bwrap: Creating new namespace failed: Operation not permitted',
        returncode=1, synthetic_csv=IN)
    assert result.code == 'R_SANDBOX_PERMISSION'
    assert 'namespace' in result.stderr_excerpt


def test_safe_stderr_redacts_source_and_quoted_text():
    private_value = 'PrivatePatient-123-ABC'
    raw = 'Error: unexpected "PrivatePatient-123-ABC" at /home/researcher/private.csv\nExecution halted'
    result = safe_r_stderr(raw, 'id,value\n1,' + private_value + '\n')
    assert private_value not in result
    assert '/home/researcher' not in result
    assert 'Error:' in result


def test_diagnosis_exposes_partial_scope_separately_from_r_failure():
    diag = diagnose_r_preview_failure(
        RSandboxFailure('R_PROCESS_FAILED', 'Isolated R exited unsuccessfully.',
                        'Inspect sandbox stderr.', exit_code=1),
        {'changes': [{'row': i + 2, 'column': 2} for i in range(32)]},
        {'evidence': [{'CSV row': i + 2, 'Column': 2} for i in range(618)]})
    assert diag['code'] == 'R_PROCESS_FAILED'
    assert diag['partial_scope'] is True
    assert (diag['planned_changes'], diag['affected_cells']) == (32, 618)


def test_preview_mismatch_is_treated_as_data_contract_failure():
    diag = diagnose_r_preview_failure(ValueError('R changed an unexpected coordinate'),
                                      {'changes': []}, {'evidence': []})
    assert diag['code'] == 'R_OUTPUT_MISMATCH'
    assert diag['stage'] == 'Preview preparation or output validation'


def test_ui_displays_diagnostic_with_retry(monkeypatch, tmp_path):
    from test_two_click_r_flow import setup_review, FINDING
    import healthcare_data_quality.r_runtime.scoped_r_execution as scoped
    app, st, _, render = setup_review(monkeypatch, tmp_path)
    def failing(*args, **kwargs):
        raise RSandboxFailure('R_SANDBOX_PERMISSION',
                              'bubblewrap could not create its isolated namespace.',
                              'Check WSL permissions; do not disable isolation.',
                              exit_code=1,
                              stderr_excerpt='bwrap: Creating new namespace failed: Operation not permitted')
    monkeypatch.setattr(scoped, 'run_r_sandbox', failing)
    render()
    diag = st.session_state.cleaning_r_diagnostics_by_finding[FINDING['id']]
    assert diag['code'] == 'R_SANDBOX_PERMISSION'
    assert diag['exit_code'] == 1
    assert any('Isolated R preview failed' in e for e in st.errors)


def test_selftest_distinguishes_environment_from_plan(monkeypatch):
    from healthcare_data_quality.r_runtime import r_sandbox
    from healthcare_data_quality.r_runtime.r_diagnostics import run_r_environment_self_test
    monkeypatch.setattr(r_sandbox, 'run_r_sandbox', lambda *args, **kwargs:
                        'kind,csv_row,csv_column,value\ncell,2,3,verified\n')
    status = run_r_environment_self_test()
    assert status['passed'] is True
    assert status['code'] == 'R_SELFTEST_PASSED'


def test_selftest_does_not_mask_r_failure(monkeypatch):
    from healthcare_data_quality.r_runtime import r_sandbox
    from healthcare_data_quality.r_runtime.r_diagnostics import run_r_environment_self_test
    def broken(*args, **kwargs):
        raise RSandboxFailure('R_SANDBOX_PERMISSION', 'Sandbox namespace blocked.',
                              'Enable permitted namespaces.')
    monkeypatch.setattr(r_sandbox, 'run_r_sandbox', broken)
    status = run_r_environment_self_test()
    assert status['passed'] is False
    assert status['code'] == 'R_SANDBOX_PERMISSION'
