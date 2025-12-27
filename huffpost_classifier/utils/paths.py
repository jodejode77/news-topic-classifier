"""Path helpers for repository directories."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def get_repo_root() -> Path:
    """Resolve repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_artifacts_dir(candidate: Path, artifacts_root: Path) -> Optional[Path]:
    """Return candidate if it exists, otherwise pick latest run directory."""
    if candidate.exists():
        return candidate
    if not artifacts_root.exists():
        return None
    run_dirs = [path for path in artifacts_root.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return run_dirs[0]


def resolve_artifacts_dir_for_model(
    candidate: Path, artifacts_root: Path, model_type: str
) -> Optional[Path]:
    """Return an artifacts directory that contains the requested model artifacts."""
    if model_type == "baseline_embedding_bag":
        required_subdir = "baseline"
    elif model_type == "bert_finetune":
        required_subdir = "bert"
    else:
        required_subdir = None

    def has_required(path: Path) -> bool:
        if required_subdir is None:
            return path.exists()
        return (path / required_subdir).exists()

    if candidate.exists() and has_required(candidate):
        return candidate

    if not artifacts_root.exists():
        return None

    run_dirs = [path for path in artifacts_root.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        if has_required(run_dir):
            return run_dir
    return None
