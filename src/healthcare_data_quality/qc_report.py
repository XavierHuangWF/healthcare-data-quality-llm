import json


SUMMARY_FIELDS = (
    "counts_by_type",
    "downstream_blocked",
    "incomplete_count",
    "alternate_delimiter_count",
    "unclassified_count",
    "counts_by_style",
)


def render_qc_summary(audit: dict) -> str:
    checks = audit["baseline"]["checks"]

    lines = [
        "# Data Quality — Baseline QC Report",
        "",
        f"Investigation status: {audit.get('status', 'unknown')}",
        f"Proposals generated: {len(audit.get('proposals', []))}",
        "",
        "## Validation Coverage",
        "",
        "| Check | Status | Observations |",
        "|---|---|---:|",
    ]

    for name, check in checks.items():
        count = len(check.get("observations", []))

        lines.append(
            f"| `{name}` | {check['status']} | {count} |"
        )

    lines.extend(["", "## Detailed Metrics", ""])

    for name, check in checks.items():
        if check["status"] != "completed":
            continue

        metrics = {
            field: check[field]
            for field in SUMMARY_FIELDS
            if field in check
        }

        if not metrics:
            continue

        lines.extend([
            f"### {name}",
            "",
            "```json",
            json.dumps(metrics, indent=2, ensure_ascii=False),
            "```",
            "",
        ])

    lines.extend([
        "## Limitations",
        "",
        "- Observation counts are not necessarily confirmed errors.",
        "- Skipped checks are not evidence that data is clean.",
        "- Investigation completion does not imply full QC coverage.",
        "- Invalid raw counts block downstream preprocessing.",
        "- Human decisions are stored separately.",
        "- Complete record-level observations remain in the audit JSON.",
    ])

    return "\n".join(lines) + "\n"