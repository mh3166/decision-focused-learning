from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _gather_files(base: Path, category: str) -> list[Path]:
    root = base / category
    if not root.exists():
        return []
    return [p for p in root.glob("*/*") if p.is_file()]


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    target_run_id = "6922354"
    repo_root = _repo_root()
    base = repo_root / "outputs" / "shortest_path_grid"

    old_categories = ["models", "results", "summary"]
    archived_root = base / "_archived" / "baseline"
    archived_root.mkdir(parents=True, exist_ok=True)

    # Map (category, filename) -> most recent Path
    newest: dict[tuple[str, str], Path] = {}

    for category in old_categories:
        for path in _gather_files(base, category):
            key = (category, path.name)
            cur = newest.get(key)
            if cur is None or path.stat().st_mtime > cur.stat().st_mtime:
                newest[key] = path

    # Move files: newest -> baseline/<run_id>/, older -> _archived/baseline/<run_id>/
    for category in old_categories:
        for path in _gather_files(base, category):
            key = (category, path.name)
            run_id = path.parent.name
            target_dir = base / "baseline" / target_run_id
            target_path = target_dir / path.name
            is_newest = newest.get(key) == path

            if is_newest:
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                if target_path.exists():
                    if path.stat().st_mtime <= target_path.stat().st_mtime:
                        is_newest = False
                    else:
                        if not dry_run:
                            target_path.unlink()

            if is_newest:
                final_path = target_path
                if category in ("results", "summary"):
                    try:
                        df = pd.read_csv(path)
                        row = df.iloc[0]
                        sim = int(row["sim"])
                        n = int(row["n"])
                        ep_type = str(row["ep_type"])
                        trial = int(row["trial"])
                        suffix = "results" if category == "results" else "summary"
                        final_path = target_dir / f"sim{sim}_n{n}_ep{ep_type}_trial{trial}_{suffix}.csv"
                    except Exception:
                        final_path = target_path
                if dry_run:
                    print(f"[DRY RUN] MOVE {path} -> {final_path}")
                else:
                    shutil.move(str(path), str(final_path))
            else:
                archive_dir = archived_root / target_run_id / run_id
                if dry_run:
                    print(f"[DRY RUN] MOVE {path} -> {archive_dir / path.name}")
                else:
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(archive_dir / path.name))

    # Clean up empty legacy directories
    for category in old_categories:
        root = base / category
        if not root.exists():
            continue
        for sub in root.glob("*"):
            if sub.is_dir() and not any(sub.iterdir()):
                if dry_run:
                    print(f"[DRY RUN] RMDIR {sub}")
                else:
                    sub.rmdir()
        if not any(root.iterdir()):
            if dry_run:
                print(f"[DRY RUN] RMDIR {root}")
            else:
                root.rmdir()


if __name__ == "__main__":
    main()
