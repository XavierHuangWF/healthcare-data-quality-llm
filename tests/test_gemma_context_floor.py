"""Regression tests for the tested 12,288-token vLLM context floor.

Run against a harmless fake vLLM binary; no model or GPU is required.
"""

import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "serve_gemma4.sh"


def _run_launcher(tmp_path: Path, *, model_len: str | None, max_num_seqs: str = "1") -> subprocess.CompletedProcess[str]:
    vllm = tmp_path / "venvs" / "healthcare-vllm" / "bin" / "vllm"
    vllm.parent.mkdir(parents=True)
    vllm.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    vllm.chmod(0o755)
    env = dict(os.environ, HOME=str(tmp_path), HDQ_MAX_NUM_SEQS=max_num_seqs)
    env.pop("HDQ_MAX_MODEL_LEN", None)
    if model_len is not None:
        env["HDQ_MAX_MODEL_LEN"] = model_len
    return subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False)


def _assert_len_arg(stdout: str, expected: str) -> None:
    lines = stdout.splitlines()
    idx = lines.index("--max-model-len")
    assert lines[idx + 1] == expected


def test_default_context_is_12288(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, model_len=None)
    assert result.returncode == 0, result.stderr
    _assert_len_arg(result.stdout, "12288")


@pytest.mark.parametrize("requested", ["8192", "12287", "0", "eight-thousand", "-1"])
def test_rejects_context_below_floor_or_invalid(tmp_path: Path, requested: str) -> None:
    result = _run_launcher(tmp_path, model_len=requested)
    assert result.returncode != 0
    assert "must be an integer >= 12288" in result.stderr
    assert "--max-model-len" not in result.stdout


def test_accepts_explicit_larger_context(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, model_len="16384")
    assert result.returncode == 0, result.stderr
    _assert_len_arg(result.stdout, "16384")


def test_two_sequences_do_not_reduce_context(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, model_len="12288", max_num_seqs="2")
    assert result.returncode == 0, result.stderr
    _assert_len_arg(result.stdout, "12288")
    lines = result.stdout.splitlines()
    assert lines[lines.index("--max-num-seqs") + 1] == "2"
