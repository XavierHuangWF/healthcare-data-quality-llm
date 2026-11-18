"""Safe, user-facing diagnostics for the isolated R cleaning preview.

Only known operational error categories and a bounded, redacted stderr excerpt
are placed into Streamlit session state. Source CSVs, patient identifiers,
model prompts, and complete R scripts are never included in an error report.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil


class RSandboxFailure(RuntimeError):
    """Execution failed without weakening the read-only sandbox boundary."""

    def __init__(self, code: str, summary: str, action: str, *,
                 exit_code: int | None = None, stderr_excerpt: str = "") -> None:
        super().__init__(f"{summary} ({code})")
        self.code = code
        self.summary = summary
        self.action = action
        self.exit_code = exit_code
        self.stderr_excerpt = stderr_excerpt


# Only display a small sanitized excerpt. The raw stderr could contain source
# values from generated R source, and must not become a persisted log or UI.
_STDIN_LIMIT = 1800


def safe_r_stderr(stderr: str, synthetic_csv: str) -> str:
    """Redact quoted literals, obvious paths, and source values from R stderr."""
    import csv
    from io import StringIO

    result = (stderr or "").replace("\x00", " ")
    result = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", result)
    result = re.sub(r"[\r\t]", " ", result)
    # R parse errors can quote the *literal source string*, including PHI.
    result = re.sub(r"(['\"])(?:\\.|(?!\1).)*?\1", "[quoted text redacted]", result)
    result = re.sub(r"(?i)(?:[a-z]:[\\/]|/(?:home|mnt|users|tmp|var)/)[^\s,:;]+", "[path redacted]", result)
    try:
        rows = list(csv.reader(StringIO(synthetic_csv), strict=True))
        values = {v for row in rows for v in row if len(v) >= 4}
        for value in sorted(values, key=len, reverse=True):
            result = result.replace(value, "[source value redacted]")
    except (csv.Error, ValueError):
        pass
    # Logs from R can also contain arbitrary data. Keep only recognizable
    # diagnostic lines; never include command arguments or generated code.
    safe_lines = []
    allow = ("error", "warning", "bwrap:", "fatal", "execution halted", "cannot allocate",
             "permission denied", "operation not permitted", "no such file", "killed",
             "signal", "unexpected", "namespace", "resource limit")
    for raw in result.splitlines():
        line = raw.strip()
        if line and any(token in line.lower() for token in allow):
            # R may print user content after a colon; never include long lines.
            safe_lines.append(line[:220])
        if len(safe_lines) >= 6:
            break
    return "\n".join(safe_lines)[:_STDIN_LIMIT]


def classify_r_process_failure(*, stderr: str, returncode: int | None,
                               timeout: bool = False, os_error: OSError | None = None,
                               synthetic_csv: str = "") -> RSandboxFailure:
    """Classify the actual sandbox failure without claiming an unproved cause."""
    low = (stderr or "").lower()
    excerpt = safe_r_stderr(stderr, synthetic_csv)
    fields = {"exit_code": returncode, "stderr_excerpt": excerpt}
    if timeout:
        return RSandboxFailure("R_TIMEOUT", "R exceeded the 15-second sandbox wall-time limit.",
                               "Check whether the sandbox is starved or the R input is too large; "
                               "retry once, then inspect sandbox performance.", **fields)
    if os_error is not None:
        if isinstance(os_error, FileNotFoundError):
            return RSandboxFailure("R_LAUNCH_MISSING", "The R sandbox launcher could not be started.",
                                   "Install or verify bubblewrap (bwrap) and /usr/bin/Rscript in WSL.", **fields)
        return RSandboxFailure("R_LAUNCH_ERROR", "The operating system could not start isolated R.",
                               "Check bwrap permissions, mounts and the R installation.", **fields)
    if "namespace" in low or "operation not permitted" in low or "permission denied" in low and "bwrap" in low:
        return RSandboxFailure("R_SANDBOX_PERMISSION", "bubblewrap could not create its isolated namespace.",
                               "Check WSL kernel/user-namespace support and bwrap permissions. "
                               "Do not disable isolation to work around this failure.", **fields)
    if "bwrap:" in low or "no such file or directory" in low and "/usr/bin/rscript" in low:
        return RSandboxFailure("R_SANDBOX_SETUP", "The R sandbox failed to start correctly.",
                               "Verify bwrap, /usr/bin/Rscript and the required read-only mounts.", **fields)
    if "source/evidence mismatch" in low:
        return RSandboxFailure("R_EVIDENCE_MISMATCH", "R rejected a source value that differs from its approved plan.",
                               "Inspect current in-memory values and regenerate the exact scoped plan; "
                               "do not bypass the source-value check.", **fields)
    if "unexpected" in low or "parse error" in low or "syntax error" in low:
        return RSandboxFailure("R_SYNTAX_ERROR", "R could not parse the generated transformation.",
                               "Review the generated R code and its quoting/escaping logic.", **fields)
    if "cannot allocate" in low or "memory exhausted" in low:
        return RSandboxFailure("R_MEMORY_LIMIT", "R exhausted available memory in the isolated sandbox.",
                               "Reduce the affected-value batch size, without changing scope or approval.", **fields)
    if returncode == -24:
        return RSandboxFailure("R_CPU_LIMIT", "R exceeded the 10-second sandbox CPU limit.",
                               "Check the transformation workload or adjust resources only after review.", **fields)
    if returncode in (-25, 25) or "file size limit" in low:
        return RSandboxFailure("R_OUTPUT_LIMIT", "R exceeded the sandbox output file-size limit.",
                               "Inspect generated output size and split the R verification workload.", **fields)
    return RSandboxFailure("R_PROCESS_FAILED", "Isolated R exited unsuccessfully.",
                           "Inspect the sanitized stderr and exit status below to identify the failing step.", **fields)


def diagnose_r_preview_failure(exc: Exception, plan: dict | None = None,
                               finding: dict | None = None) -> dict:
    """Describe a failed preview, including partial-coverage limits if known."""
    if isinstance(exc, RSandboxFailure):
        status = {"stage": "Isolated R execution", "code": exc.code,
                  "summary": exc.summary, "action": exc.action,
                  "exit_code": exc.exit_code, "stderr_excerpt": exc.stderr_excerpt}
    else:
        msg = str(exc).lower()
        if "unexpected value" in msg or "unexpected coordinate" in msg or "unexpected row" in msg:
            code, summary, action = (
                "R_OUTPUT_MISMATCH", "R output failed independent value/coordinate verification.",
                "Inspect the proposed R transformation; no insertion is allowed until output matches the approved cells.")
        elif "source" in msg and ("changed" in msg or "mismatch" in msg):
            code, summary, action = (
                "R_STALE_SOURCE", "The source or current cell values changed after the plan was prepared.",
                "Regenerate the plan from the current reviewed data, then retry verification.")
        elif "limit" in msg or "exceed" in msg or "32" in msg:
            code, summary, action = (
                "R_SIZE_LIMIT", "The proposed R input, code, or plan exceeded a configured limit.",
                "Check batch size and coverage; do not silently apply a partial finding.")
        elif "invalid csv" in msg or "does not contain" in msg:
            code, summary, action = (
                "R_INVALID_OUTPUT", "R returned an unexpected or invalid CSV result.",
                "Inspect the compiled R output contract and input/output row counts.")
        else:
            code, summary, action = (
                "R_PREVIEW_ERROR", "The preview failed while preparing or verifying the R result.",
                "Inspect the scoped plan and the local Streamlit terminal for additional information.")
        status = {"stage": "Preview preparation or output validation", "code": code,
                  "summary": summary, "action": action,
                  "exit_code": None, "stderr_excerpt": ""}
    # This check is informational: the 32-cell plan cap does not establish
    # that R itself failed, but it must never be confused with full remediation.
    changes = (plan or {}).get("changes") or []
    evidence = (finding or {}).get("evidence") or []
    total = len({(e.get("CSV row", e.get("csv_row")), e.get("Column", e.get("column")))
                 for e in evidence if isinstance(e, dict) and
                 isinstance(e.get("CSV row", e.get("csv_row")), int) and
                 isinstance(e.get("Column", e.get("column")), int)})
    status["planned_changes"] = len(changes)
    status["affected_cells"] = total
    status["partial_scope"] = bool(total and len(changes) < total)
    status["preflight"] = {
        "bwrap_found": bool(shutil.which("bwrap")),
        "rscript_found": Path("/usr/bin/Rscript").is_file(),
    }
    return status


def run_r_environment_self_test() -> dict:
    """Use the actual isolated sandbox with fixed, non-research test values.

    Does not read the uploaded dataset, change the working stage, or use Gemma.
    A passing self-test suggests the failure is in the finding-specific plan or
    R output contract; it cannot prove every possible R operation will succeed.
    """
    from .r_sandbox import run_r_sandbox
    import csv
    from io import StringIO

    synthetic = "kind,csv_row,csv_column,value\ncell,2,3,example\n"
    fixed_r = "clean_dataset <- function(df) { df$value[[1]] <- 'verified'; df }"
    try:
        result = run_r_sandbox(fixed_r, synthetic, preserve_text=True)
        parsed = list(csv.reader(StringIO(result), strict=True))
        if parsed != ["kind,csv_row,csv_column,value".split(","),
                      ["cell", "2", "3", "verified"]]:
            raise ValueError("The sandbox self-test returned unexpected CSV content")
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        info = diagnose_r_preview_failure(exc)
        return {"passed": False, "summary": info["summary"], "code": info["code"],
                "action": info["action"], "stderr_excerpt": info["stderr_excerpt"]}
    return {"passed": True,
            "summary": "R and bubblewrap executed a fixed, synthetic transformation successfully.",
            "code": "R_SELFTEST_PASSED",
            "action": "Investigate the specific plan, data shape and generated R contract."}
