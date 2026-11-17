"""Trusted inner runner, invoked ONLY inside the Bubblewrap sandbox.

The generated proposal must define `investigate(datasets)` and return JSON:
  {"summary": "...", "findings": [{"title": "...", "description": "...",
  "evidence": [{"dataset_id":"dataset_1","csv_row":2,"column":1,"value":"raw"}]}]}

Each `datasets[dataset_id]` is {"headers": [...], "rows": [[raw strings], ...]}.
CSV data row numbers are 2-based (the header is row 1); columns are 1-based.
"""

import contextlib
import csv
import io
import json
import resource
import runpy
import sys
from pathlib import Path


def _apply_limits():
    mib = 1024 * 1024
    for kind, soft, hard in (
        (resource.RLIMIT_AS, 768 * mib, 768 * mib),
        (resource.RLIMIT_CPU, 12, 13),
        (resource.RLIMIT_FSIZE, 128 * 1024, 128 * 1024),
        (resource.RLIMIT_CORE, 0, 0),
        (resource.RLIMIT_NOFILE, 48, 48),
    ):
        resource.setrlimit(kind, (soft, hard))


def main():
    _apply_limits()
    datasets = {}
    for path in sorted(Path("/data").glob("dataset_*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            headers = next(reader, [])
            rows = list(reader)
        datasets[path.stem] = {"headers": headers, "rows": rows}
    if not datasets:
        raise ValueError("No datasets supplied to the sandbox")
    # Capture model-generated prints so stdout remains strictly machine-readable.
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = runpy.run_path("/analysis/proposal.py")
        fn = namespace.get("investigate")
        if not callable(fn):
            raise ValueError("The proposal must define investigate(datasets)")
        result = fn(datasets)
    wire = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(wire.encode("utf-8")) > 96 * 1024:
        raise ValueError("Model-generated output too large")
    sys.stdout.write(wire)


if __name__ == "__main__":
    main()
