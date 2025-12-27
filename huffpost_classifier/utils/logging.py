"""Logging helpers."""
from __future__ import annotations

from pytorch_lightning.loggers import MLFlowLogger


def create_mlflow_logger(tracking_uri: str, experiment_name: str, run_name: str) -> MLFlowLogger:
    """Create an MLflow logger configured for Lightning."""
    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
    )
