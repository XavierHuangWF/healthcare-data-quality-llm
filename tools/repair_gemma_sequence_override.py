"""Make the Gemma launcher honor HDQ_MAX_NUM_SEQS without changing GPU tuning.

Usage from the repository root:
    python tools/repair_gemma_sequence_override.py

This makes one targeted edit to scripts/serve_gemma4.sh. It does not change
CUDA graphs, model selection, context length, or memory utilization.
"""

from __future__ import annotations

import re
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "serve_gemma4.sh"
    original = launcher.read_bytes()
    decoded = original.decode("utf-8")

    # Match only the actual vLLM argument line, not comments, status output,
    # or the validation block. Keep all other launcher settings untouched.
    argument = re.compile(
        r'(?m)^(?P<prefix>[ \t]*--max-num-seqs[ \t]+)'
        r'(?P<value>"[^"\r\n]*"|\x27[^\x27\r\n]*\x27|[^ \t\r\n\\]+)'
    )
    matches = list(argument.finditer(decoded))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one --max-num-seqs argument line in {launcher}, "
            f"found {len(matches)}. No changes made."
        )

    replacement = matches[0].group("prefix") + '"${HDQ_MAX_NUM_SEQS:-3}"'
    updated = (
        decoded[: matches[0].start()]
        + replacement
        + decoded[matches[0].end() :]
    )
    if updated == decoded:
        print("Gemma launcher already honors HDQ_MAX_NUM_SEQS. No change needed.")
        return
    launcher.write_bytes(updated.encode("utf-8"))
    print("Updated", launcher)
    print("--max-num-seqs now honors HDQ_MAX_NUM_SEQS (default: 3).")
    print("All other launcher flags remain unchanged.")


if __name__ == "__main__":
    main()
