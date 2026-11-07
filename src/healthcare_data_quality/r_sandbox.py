import subprocess


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

def run_r_sandbox(r_function: str, synthetic_csv: str) -> str:


    if not r_function.strip():
        raise ValueError("R function cannot be empty")

    if not synthetic_csv.strip():
        raise ValueError("Synthetic CSV cannot be empty")


    if len(r_function.encode("utf-8")) > MAX_R_BYTES:
        raise ValueError("R function exceeds size limit")

    if len(synthetic_csv.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Synthetic CSV exceeds size limit")



    r_expression = (
            r_function
            + '\ninput_df <- read.csv(file("stdin"), '
              'check.names=FALSE, stringsAsFactors=FALSE)\n'
              'output_df <- clean_dataset(input_df)\n'
              'stopifnot(is.data.frame(output_df))\n'
              'write.csv(output_df, stdout(), row.names=FALSE)\n'
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
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
        ) as exc:
            raise RuntimeError("Isolated R execution failed") from exc

        stdout_file.seek(0)
        output = stdout_file.read(MAX_OUTPUT_BYTES + 1)

        if len(output.encode("utf-8")) >= MAX_OUTPUT_BYTES:
            raise RuntimeError("R output exceeds size limit")

    if not output.strip():
        raise RuntimeError("Isolated R returned empty output")

    return output
