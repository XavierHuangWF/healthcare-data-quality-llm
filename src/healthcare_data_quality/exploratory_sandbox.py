"""Fail-closed Linux sandbox for model-generated, READ-ONLY Python investigation.

Security boundary: Bubblewrap user/mount/network/PID namespaces, *not* AST
inspection. Never execute a model's code in the Streamlit/host interpreter.
The sandbox sees only ephemeral dataset copies, trusted runner and generated
analysis script. There is no host/home/repository bind or network namespace.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

MAX_CODE_BYTES = 16_000
MAX_DATASET_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_BYTES = 96 * 1024
TIMEOUT_SECONDS = 18


class SandboxUnavailable(RuntimeError):
    """No verified isolated execution environment. Never run on the host."""


class SandboxExecutionError(RuntimeError):
    """The isolated execution did not return a valid, bounded result."""


class SandboxGeneratedCodeError(SandboxExecutionError):
    """A Python traceback came from the isolated generated analysis/runner."""


def sandbox_available() -> bool:
    return bool(shutil.which("bwrap") and Path("/usr/bin/python3").is_file())


def _runtime_mounts() -> list[str]:
    """Permit the system Python's runtime only; no broad host root mount."""
    args = ["--ro-bind", "/usr", "/usr"]
    for target in ("/lib", "/lib64", "/bin", "/sbin"):
        host = Path(target)
        if host.is_symlink():
            args.extend(["--symlink", os.readlink(target), target])
        elif host.is_dir():
            args.extend(["--ro-bind", target, target])
    # A dynamic loader may use this, but the rest of /etc is not exposed.
    if Path("/etc/ld.so.cache").is_file():
        args.extend(["--dir", "/etc", "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"])
    return args


def _bwrap_command(staged_data: Path, staged_code: Path) -> list[str]:
    binary = shutil.which("bwrap")
    if not binary or not Path("/usr/bin/python3").is_file():
        raise SandboxUnavailable(
            "Isolated Python execution requires Bubblewrap (bwrap) and "
            "/usr/bin/python3 in WSL/Linux. Install Bubblewrap and verify "
            "unprivileged user namespaces; no unsandboxed fallback exists."
        )
    args = [
        binary, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
        *_runtime_mounts(),
        "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--dir", "/home",
        "--dir", "/data", "--ro-bind", str(staged_data), "/data",
        "--dir", "/analysis", "--ro-bind", str(staged_code), "/analysis",
        "--chdir", "/tmp",
        "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "HOME", "/tmp",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "PYTHONNOUSERSITE", "1",
        "--setenv", "LC_ALL", "C.UTF-8",
        "/usr/bin/python3", "-I", "-B", "/analysis/runner.py",
    ]
    return args


def run_in_sandbox(code: str, datasets: dict[str, Path]) -> dict:
    """Execute generated code only inside Bubblewrap; return bounded JSON.

    No host execution fallback is permitted. Dataset bytes are copied to a
    disposable staging folder and mounted read-only under /data. Generated
    code cannot see other host files or write to the original datasets.
    """
    if not sandbox_available():
        raise SandboxUnavailable(
            "Bubblewrap (bwrap) and /usr/bin/python3 are required in WSL/Linux. "
            "Generated code was NOT executed. Install Bubblewrap and enable "
            "unprivileged user namespaces; there is no unsafe fallback."
        )
    encoded = code.encode("utf-8")
    if not encoded or len(encoded) > MAX_CODE_BYTES:
        raise ValueError("Generated Python code is empty or exceeds 16 KB.")
    if not 1 <= len(datasets) <= 4:
        raise ValueError("The sandbox accepts 1–4 datasets.")

    with TemporaryDirectory(prefix="hdq-exploration-") as folder:
        root = Path(folder)
        data = root / "data"
        analysis = root / "analysis"
        data.mkdir(mode=0o700)
        analysis.mkdir(mode=0o700)
        for identifier, source in datasets.items():
            if not identifier.startswith("dataset_") or not identifier[8:].isdigit():
                raise ValueError("Dataset identifiers must follow dataset_N.")
            source = Path(source)
            if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_DATASET_BYTES:
                raise ValueError("Invalid or oversized dataset input.")
            shutil.copyfile(source, data / f"{identifier}.csv")
        (analysis / "proposal.py").write_bytes(encoded)
        runner = Path(__file__).with_name("exploratory_runner.py")
        shutil.copyfile(runner, analysis / "runner.py")
        (analysis / "runner.py").chmod(0o400)
        (analysis / "proposal.py").chmod(0o400)
        for staged_file in data.iterdir():
            staged_file.chmod(0o400)

        with TemporaryFile(mode="w+b") as out, TemporaryFile(mode="w+b") as err:
            try:
                completed = subprocess.run(
                    _bwrap_command(data, analysis),
                    stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    timeout=TIMEOUT_SECONDS, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise SandboxExecutionError("Isolated analysis exceeded its time limit.") from exc
            except OSError as exc:
                raise SandboxUnavailable("Cannot launch Bubblewrap; generated code was not run.") from exc
            out.seek(0)
            err.seek(0)
            raw = out.read(MAX_OUTPUT_BYTES + 1)
            diagnostics = err.read(4096).decode("utf-8", errors="replace")
            if completed.returncode:
                # Python tracebacks can be longer than 400 characters. Preserve
                # the final exception line (e.g., NameError: collections) so the
                # LOCAL repair model receives a useful diagnostic. Never reveal
                # this to end users until all bounded attempts are exhausted.
                if diagnostics.lstrip().startswith("Traceback (most recent call last):"):
                    raise SandboxGeneratedCodeError(
                        "Isolated Python analysis failed (exit code "
                        + str(completed.returncode) + "). Python traceback (last lines): "
                        + diagnostics[-950:]
                    )
                # Do not confuse a Bubblewrap/namespace launch error with a
                # Python code error: infrastructure failures must fail closed.
                raise SandboxExecutionError(
                    "Isolated analysis failed (exit code " + str(completed.returncode) + "). "
                    + ("Sandbox infrastructure diagnostic: " + diagnostics[:250]
                       if diagnostics else "")
                )
            if len(raw) > MAX_OUTPUT_BYTES:
                raise SandboxExecutionError("Isolated analysis output exceeded 96 KB.")
            try:
                answer = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise SandboxExecutionError("Isolated analysis did not return valid JSON.") from exc
            if not isinstance(answer, dict):
                raise SandboxExecutionError("Analysis output must be a JSON object.")
            return answer
