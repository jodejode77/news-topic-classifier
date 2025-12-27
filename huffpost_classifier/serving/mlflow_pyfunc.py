"""MLflow pyfunc model packaging for serving."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import mlflow.pyfunc
import pandas as pd

from huffpost_classifier.data.preprocess import build_text
from huffpost_classifier.inference.predict_baseline import load_baseline_artifacts, predict_baseline
from huffpost_classifier.inference.predict_bert import load_bert_artifacts, predict_bert


class HuffpostPyFunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for HuffPost classifiers."""

    def __init__(self, model_type: str, top_k: int, text_sep: str, max_length: int) -> None:
        self.model_type = model_type
        self.top_k = top_k
        self.text_sep = text_sep
        self.max_length = max_length
        self.model = None
        self.vocab = None
        self.tokenizer = None
        self.labels: List[str] = []

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        model_dir = Path(context.artifacts["model_dir"])
        if self.model_type == "baseline_embedding_bag":
            self.model, self.vocab, self.labels = load_baseline_artifacts(model_dir)
        elif self.model_type == "bert_finetune":
            self.model, self.tokenizer, self.labels = load_bert_artifacts(model_dir)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame
    ) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("MLflow input must be a pandas DataFrame.")

        headlines = model_input.get("headline", pd.Series([""] * len(model_input)))
        descriptions = model_input.get("short_description", pd.Series([""] * len(model_input)))
        texts = [
            build_text(str(headline), str(description), self.text_sep)
            for headline, description in zip(headlines.tolist(), descriptions.tolist())
        ]

        if self.model_type == "baseline_embedding_bag":
            probabilities = predict_baseline(self.model, self.vocab, texts)
        else:
            probabilities = predict_bert(self.model, self.tokenizer, texts, self.max_length)

        results: List[Dict[str, Any]] = []
        for probs in probabilities:
            sorted_indices = probs.argsort(descending=True)
            top_indices = sorted_indices[: self.top_k].tolist()
            top_k_list = [
                {"label": self.labels[class_id], "probability": float(probs[class_id])}
                for class_id in top_indices
            ]
            results.append(
                {
                    "predicted_label": self.labels[top_indices[0]],
                    "top_k": top_k_list,
                }
            )

        return pd.DataFrame(results)


def package_mlflow_model(cfg: Any, artifacts_dir: Path) -> Path:
    """Package model artifacts as an MLflow pyfunc model for serving."""
    output_dir = Path(cfg.serving.output_dir)
    if cfg.model.type == "baseline_embedding_bag":
        model_dir = artifacts_dir / "baseline"
    elif cfg.model.type == "bert_finetune":
        model_dir = artifacts_dir / "bert"
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    output_dir.mkdir(parents=True, exist_ok=True)
    python_model = HuffpostPyFunc(
        model_type=cfg.model.type,
        top_k=int(cfg.serving.top_k),
        text_sep=str(cfg.data.text_sep),
        max_length=int(getattr(cfg.model, "max_length", 0)),
    )

    mlflow.pyfunc.save_model(
        path=str(output_dir),
        python_model=python_model,
        artifacts={"model_dir": str(model_dir)},
    )
    return output_dir
