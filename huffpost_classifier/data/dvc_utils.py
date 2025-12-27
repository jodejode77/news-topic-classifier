"""DVC helpers for pulling data or falling back to download."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from huffpost_classifier.data.download import download_data
from huffpost_classifier.utils.paths import get_repo_root


def _paths_exist(paths: List[Path]) -> bool:
    return all(path.exists() for path in paths)


def ensure_data(cfg: Any) -> None:
    """Ensure data is available locally via DVC or fallback download."""
    required_paths = [
        Path(cfg.data.raw_dir),
        Path(cfg.data.processed_dir),
        Path(cfg.data.splits_path),
        Path(cfg.data.examples_path),
    ]

    repo_root = get_repo_root()
    try:
        from dvc.repo import Repo

        paths_for_dvc = []
        for path in required_paths:
            try:
                paths_for_dvc.append(str(path.relative_to(repo_root)))
            except ValueError:
                paths_for_dvc.append(str(path))

        repo = Repo(str(repo_root))
        repo.pull(paths_for_dvc)
    except Exception:
        download_data(cfg, seed=int(cfg.seed))
        return

    if not _paths_exist(required_paths):
        download_data(cfg, seed=int(cfg.seed))
