import subprocess

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
    (
        'stopifnot(!file.exists("/mnt/c/Users")); '
        'stopifnot(!file.exists("/home/xavier")); '
        'cat("ISOLATED R: PASS\\n")'
    ),
]

result = subprocess.run(
    command,
    capture_output=True,
    text=True,
    timeout=15,
    check=True,
)

assert result.stdout.strip() == "ISOLATED R: PASS"
print(result.stdout, end="")


import csv
import io

from healthcare_data_quality.r_sandbox import run_r_sandbox

r_function = (
    'clean_dataset <- function(df) { '
    'df$status <- "verified"; '
    'return(df) '
    '}'
)

synthetic_csv = "sample_id,value\nS1,10\nS2,20\n"

output = run_r_sandbox(r_function, synthetic_csv)
rows = list(csv.DictReader(io.StringIO(output)))

assert rows == [
    {"sample_id": "S1", "value": "10", "status": "verified"},
    {"sample_id": "S2", "value": "20", "status": "verified"},
]

print("R SANDBOX TRANSFORMATION: PASS")


try:
    run_r_sandbox(
        'clean_dataset <- function(df) { '
        'cat(strrep("x", 2000000)); '
        'return(df) }',
        synthetic_csv,
    )
except RuntimeError:
    print("R OUTPUT LIMIT: PASS")
else:
    raise AssertionError("Oversized R output was accepted")
