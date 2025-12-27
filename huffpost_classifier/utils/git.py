"""Git metadata helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_commit_id(repo_root: Path) -> str:
    """Return current git commit id or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
