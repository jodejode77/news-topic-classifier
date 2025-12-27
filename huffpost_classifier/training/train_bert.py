"""Training routine for the TinyBERT model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from huffpost_classifier.data.datamodule_bert import BertDataModule
from huffpost_classifier.models.bert_module import BertClassifier
from huffpost_classifier.training.callbacks import MetricsHistoryCallback
from huffpost_classifier.utils.git import get_git_commit_id
from huffpost_classifier.utils.logging import create_mlflow_logger
from huffpost_classifier.utils.paths import ensure_dir, get_repo_root


def _log_artifact(logger, path: Path) -> None:
    logger.experiment.log_artifact(logger.run_id, str(path))


def train_bert(cfg: Any) -> Dict[str, Path]:
    """Train the TinyBERT model and return artifact paths."""
    artifacts_dir = Path(cfg.output_dir)
    bert_dir = ensure_dir(artifacts_dir / "bert")
    plots_dir = ensure_dir(Path(cfg.paths.plots_dir) / cfg.run_name)

    datamodule = BertDataModule(cfg)
    datamodule.setup()
    label_map = datamodule.label_map
    num_classes = len(label_map["id_to_label"])

    class_weights = datamodule.class_weights if cfg.model.use_class_weights else None
    model = BertClassifier(cfg, num_classes=num_classes, class_weights=class_weights)

    mlflow_logger = create_mlflow_logger(
        tracking_uri=cfg.logging.tracking_uri,
        experiment_name=cfg.logging.experiment_name,
        run_name=cfg.logging.run_name,
    )
    git_commit_id = get_git_commit_id(get_repo_root())
    mlflow_logger.log_hyperparams(
        {
            "model_type": cfg.model.type,
            "pretrained_model": cfg.model.pretrained_model_name,
            "max_length": cfg.model.max_length,
            "lr": cfg.model.lr,
            "batch_size": cfg.model.batch_size,
            "git_commit_id": git_commit_id,
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=bert_dir,
        filename="best",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=int(cfg.trainer.early_stopping_patience),
    )
    metrics_callback = MetricsHistoryCallback(plots_dir)

    train_batches = len(datamodule.train_dataloader())
    log_every_n_steps = max(1, int(cfg.trainer.log_every_n_steps))
    if train_batches > 0:
        log_every_n_steps = min(log_every_n_steps, train_batches)

    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=log_every_n_steps,
        precision=cfg.trainer.precision,
        deterministic=cfg.trainer.deterministic,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        callbacks=[checkpoint_callback, early_stopping, metrics_callback],
        logger=mlflow_logger,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    best_checkpoint = Path(checkpoint_callback.best_model_path)
    best_model = BertClassifier.load_from_checkpoint(
        str(best_checkpoint),
        cfg=cfg,
        num_classes=num_classes,
        class_weights=class_weights,
    )

    tokenizer_dir = bert_dir / "tokenizer"
    model_dir = bert_dir / "model"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    datamodule.tokenizer.save_pretrained(tokenizer_dir)
    best_model.model.save_pretrained(model_dir)

    with (bert_dir / "label_map.json").open("w", encoding="utf-8") as file:
        json.dump(label_map, file)
    with (bert_dir / "model_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "pretrained_model_name": cfg.model.pretrained_model_name,
                "max_length": cfg.model.max_length,
                "num_classes": num_classes,
            },
            file,
        )

    for plot_path in plots_dir.glob("*.png"):
        _log_artifact(mlflow_logger, plot_path)

    return {
        "artifacts_dir": artifacts_dir,
        "bert_dir": bert_dir,
        "plots_dir": plots_dir,
        "best_checkpoint": best_checkpoint,
    }
