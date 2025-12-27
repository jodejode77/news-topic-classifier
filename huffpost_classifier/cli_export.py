"""Hydra entrypoint for ONNX export."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from huffpost_classifier.export.onnx_export import export_onnx
from huffpost_classifier.utils.paths import resolve_artifacts_dir_for_model

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Export ONNX for the configured model."""
    artifacts_dir = resolve_artifacts_dir_for_model(
        Path(cfg.infer.artifacts_dir),
        Path(cfg.paths.artifacts_dir),
        cfg.model.type,
    )
    if artifacts_dir is None:
        raise FileNotFoundError(
            "No matching artifacts directory found. Train the selected model or set "
            "infer.artifacts_dir=artifacts/<run_name>."
        )

    export_onnx(cfg, artifacts_dir)


if __name__ == "__main__":
    main()
