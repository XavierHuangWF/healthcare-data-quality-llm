import subprocess

from .r_diagnostics import classify_r_process_failure

import resource
from tempfile import TemporaryFile


MAX_R_BYTES = 64 * 1024
MAX_INPUT_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024


def _limit_resources() -> None:
    resource.setrlimit(
        resource.RLIMIT_AS,
        (2 * 1024**3, 2 * 1024**3),
    )
    resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES),
    )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _build_bwrap_command(r_expression: str) -> list[str]:
    command = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/etc/R", "/etc/R",
        "--ro-bind", "/etc/alternatives", "/etc/alternatives",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--dir", "/home",
        "--chdir", "/tmp",
        "--setenv", "HOME", "/tmp",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "LC_ALL", "C",
        "/usr/bin/Rscript", "--vanilla", "-e",
        r_expression,
    ]
    return command

def run_r_sandbox(r_function: str, synthetic_csv: str, *, preserve_text: bool = False) -> str:


    if not r_function.strip():
        raise ValueError("R function cannot be empty")

    if not synthetic_csv.strip():
        raise ValueError("Synthetic CSV cannot be empty")


    if len(r_function.encode("utf-8")) > MAX_R_BYTES:
        raise ValueError("R function exceeds size limit")

    if len(synthetic_csv.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Synthetic CSV exceeds size limit")



    # The reviewed cleaning workflow must preserve source strings exactly,
    # especially blank values, literal "NA", leading zeroes and duplicate names.
    # Other pre-existing R workflows keep their original inference behavior.
    reader_options = (
        'check.names=FALSE, stringsAsFactors=FALSE, '
        'colClasses="character", na.strings=character(0))'
        if preserve_text else 'check.names=FALSE, stringsAsFactors=FALSE)'
    )
    r_expression = (
        r_function
        + '\ninput_df <- read.csv(file("stdin"), ' + reader_options + '\n'
        + 'output_df <- clean_dataset(input_df)\n'
        + 'stopifnot(is.data.frame(output_df))\n'
        + 'write.csv(output_df, stdout(), row.names=FALSE, na="")\n'
    )


    with (
        TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file,
        TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file,
    ):
        try:
            subprocess.run(
                _build_bwrap_command(r_expression),
                input=synthetic_csv,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                timeout=15,
                check=True,
                preexec_fn=_limit_resources,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            # subprocess writes to bounded, private temp files rather than to
            # Streamlit logs. Capture the actual cause BEFORE those files close.
            # The diagnostic redacts the input values and does not persist raw
            # stdout/stderr or display the generated R expression.
            stderr_file.flush()
            stderr_file.seek(0)
            stderr_text = stderr_file.read(8192)
            failure = classify_r_process_failure(
                stderr=stderr_text,
                returncode=(exc.returncode if isinstance(exc, subprocess.CalledProcessError) else None),
                timeout=isinstance(exc, subprocess.TimeoutExpired),
                os_error=exc if isinstance(exc, OSError) else None,
                synthetic_csv=synthetic_csv,
            )
            raise failure from exc

        stdout_file.seek(0)
        output = stdout_file.read(MAX_OUTPUT_BYTES + 1)

        if len(output.encode("utf-8")) >= MAX_OUTPUT_BYTES:
            raise RuntimeError("R output exceeds size limit (1 MiB sandbox cap)")

    if not output.strip():
        raise RuntimeError("Isolated R returned empty output")

    return output
