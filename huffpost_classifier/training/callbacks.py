"""Training callbacks for metric tracking and plotting."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pytorch_lightning import Callback, Trainer


class MetricsHistoryCallback(Callback):
    """Collect metrics per epoch and generate plots at the end of fit."""

    def __init__(self, plots_dir: Path) -> None:
        self.plots_dir = plots_dir
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_macro_f1": [],
            "val_macro_f1": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

    def on_validation_epoch_end(self, trainer: Trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        self._append_metric("train_loss", metrics.get("train_loss"))
        self._append_metric("val_loss", metrics.get("val_loss"))
        self._append_metric("train_macro_f1", metrics.get("train_macro_f1"))
        self._append_metric("val_macro_f1", metrics.get("val_macro_f1"))
        self._append_metric("train_accuracy", metrics.get("train_accuracy"))
        self._append_metric("val_accuracy", metrics.get("val_accuracy"))

    def on_fit_end(self, trainer: Trainer, pl_module) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self._plot_metric(
            "loss",
            self.history["train_loss"],
            self.history["val_loss"],
            self.plots_dir / "loss.png",
        )
        self._plot_metric(
            "macro_f1",
            self.history["train_macro_f1"],
            self.history["val_macro_f1"],
            self.plots_dir / "macro_f1.png",
        )
        self._plot_metric(
            "accuracy",
            self.history["train_accuracy"],
            self.history["val_accuracy"],
            self.plots_dir / "accuracy.png",
        )

    def _append_metric(self, key: str, value: Optional[float]) -> None:
        if value is None:
            return
        self.history[key].append(float(value))

    @staticmethod
    def _plot_metric(
        title: str, train_values: List[float], val_values: List[float], path: Path
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(train_values, label="train")
        plt.plot(val_values, label="val")
        plt.title(title.replace("_", " ").title())
        plt.xlabel("epoch")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
