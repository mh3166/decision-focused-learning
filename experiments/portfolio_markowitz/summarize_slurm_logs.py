#!/usr/bin/env python3
"""Summarize portfolio SLURM logs.

Scans ``slurm-*.out`` files, classifies each file as:
- ``FAILED``: traceback or explicit runtime error detected
- ``SUCCEEDED``: results/summary write completion detected
- ``INCOMPLETE``: neither success nor failure markers found

For failed logs, also attempts to identify:
- failure mode: ``status=13`` vs ``OTHER``
- likely loss active when the failure occurred

Usage:
    python summarize_slurm_logs.py
    python summarize_slurm_logs.py /path/to/log_dir
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


FAIL_STATUS_13 = re.compile(r"status\s*=\s*13\b")
FAIL_RUNTIME = re.compile(r"RuntimeError: Gurobi failed to solve portfolio problem; status=(\d+)\.")
LOSS_LINE = re.compile(r"Loss number \d+/\d+, on loss function ([A-Za-z0-9_+.-]+)")
JOB_FILE = re.compile(r"slurm-(?P<jobid>\d+)(?:_(?P<taskid>\d+))?\.out$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=".",
        help="Directory containing slurm-*.out files. Defaults to current directory.",
    )
    return parser.parse_args()


def extract_job_info(path: Path) -> tuple[str, str]:
    match = JOB_FILE.match(path.name)
    if not match:
        return path.name, "-"
    return match.group("jobid"), match.group("taskid") or "-"


def infer_faulty_loss(lines: list[str]) -> str:
    last_loss = None
    for line in lines:
        match = LOSS_LINE.search(line)
        if match:
            last_loss = match.group(1)

    joined = "".join(lines)
    if "CILO_lbda" in joined:
        return "CILO"
    if "SPOPlusLossFunc" in joined:
        return "SPOPlus"
    if "PGFunc" in joined:
        return "PG"
    if "FYFunc" in joined:
        return "FY"
    if "DecisionRegretLoss" in joined:
        return "DecisionRegret_Smooth"
    return last_loss or "UNKNOWN"


def classify_log(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    jobid, taskid = extract_job_info(path)

    success = (
        "Wrote summary to" in text
        or "Completed warm-start run" in text
    )
    failed = (
        "Traceback (most recent call last):" in text
        or "RuntimeError:" in text
        or "ImportError:" in text
        or "ModuleNotFoundError:" in text
    )

    if failed:
        status = "FAILED"
    elif success:
        status = "SUCCEEDED"
    else:
        status = "INCOMPLETE"

    failure_mode = "-"
    faulty_loss = "-"
    if status == "FAILED":
        if FAIL_STATUS_13.search(text):
            failure_mode = "status=13"
        else:
            match = FAIL_RUNTIME.search(text)
            failure_mode = f"status={match.group(1)}" if match else "OTHER"
        faulty_loss = infer_faulty_loss(lines)

    last_loss = infer_faulty_loss(lines) if status != "FAILED" else faulty_loss

    return {
        "file": path.name,
        "jobid": jobid,
        "taskid": taskid,
        "status": status,
        "failure_mode": failure_mode,
        "faulty_loss": faulty_loss,
        "last_loss_seen": last_loss if last_loss else "-",
    }


def format_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row[col])))

    def fmt(row: dict[str, str]) -> str:
        return "  ".join(str(row[col]).ljust(widths[col]) for col in columns)

    header = "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    body = "\n".join(fmt(row) for row in rows)
    return f"{header}\n{sep}\n{body}"


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_paths = sorted(log_dir.glob("slurm-*.out"))

    if not log_paths:
        print(f"No slurm-*.out files found in {log_dir}")
        return

    rows = [classify_log(path) for path in log_paths]

    status_counts = Counter(row["status"] for row in rows)
    fail_mode_counts = Counter(row["failure_mode"] for row in rows if row["status"] == "FAILED")
    faulty_loss_counts = Counter(row["faulty_loss"] for row in rows if row["status"] == "FAILED")

    print(f"Directory: {log_dir}")
    print(f"Logs scanned: {len(rows)}")
    print()

    summary_rows = [
        {"category": "status", "value": key, "count": str(status_counts[key])}
        for key in sorted(status_counts)
    ]
    if fail_mode_counts:
        summary_rows.extend(
            {"category": "failure_mode", "value": key, "count": str(fail_mode_counts[key])}
            for key in sorted(fail_mode_counts)
        )
    if faulty_loss_counts:
        summary_rows.extend(
            {"category": "faulty_loss", "value": key, "count": str(faulty_loss_counts[key])}
            for key in sorted(faulty_loss_counts)
        )

    print("Summary")
    print(format_table(summary_rows, ["category", "value", "count"]))
    print()


if __name__ == "__main__":
    main()
