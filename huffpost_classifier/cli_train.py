"""Hydra entrypoint for training."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from huffpost_classifier.data.dvc_utils import ensure_data
from huffpost_classifier.export.onnx_export import export_onnx
from huffpost_classifier.training.train_baseline import train_baseline
from huffpost_classifier.training.train_bert import train_bert
from huffpost_classifier.utils.seed import seed_everything

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train the configured model."""
    ensure_data(cfg)
    seed_everything(int(cfg.seed))

    if cfg.model.type == "baseline_embedding_bag":
        results = train_baseline(cfg)
    elif cfg.model.type == "bert_finetune":
        results = train_bert(cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    if cfg.export.export_on_train_end:
        export_onnx(cfg, Path(results["artifacts_dir"]))


if __name__ == "__main__":
    main()
