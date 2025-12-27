# news-topic-classifier
# HuffPost News Category Classifier (MLOps)

Author: Ticona Perales Guillermo Sebastian

This repository provides a MLOps workflow for classifying HuffPost news categories with two models:

- BASELINE: EmbeddingBag (neural bag-of-words) + small MLP
- MAIN: TinyBERT fine-tuning (huawei-noah/TinyBERT_General_4L_312D)

It includes reproducible training, inference, MLflow logging, Hydra configs, DVC-based data management, and ONNX export.

## Project overview

### Problem statement

Predict the news category (for example POLITICS, ENTERTAINMENT, SPORTS, TECH, TRAVEL) from a HuffPost headline with an optional short description. This enables automatic tagging for search, recommendations, and content analytics.

### Input and output data format

Input (inference):

- `headline`: str (required)
- `short_description`: str (optional, may be empty)

Output:

- `category`: str (one of the predefined classes)
- optional `top_k` probabilities per class

### Metrics

Primary metrics:

- Accuracy
- Macro F1-score (important due to class imbalance)

Expected performance on the full dataset depends on training budget and hardware. The baseline serves as a lower bound, while TinyBERT is expected to outperform it when trained for enough epochs.

### Validation

We split the dataset into train / validation / test = 70% / 15% / 15% with stratification by category and a fixed seed (42). Split indices are saved for reproducibility.

### Data

We use the HuffPost News Category dataset (Rishabh Misra). The implementation downloads from Hugging Face:

- https://huggingface.co/datasets/heegyu/news-category-dataset

The dataset contains ~200k news entries with fields like `headline`, `short_description`, and `category`. The raw files are ~80–90 MB. Known issues include class imbalance and overlapping labels (for example WORLDPOST vs WORLD NEWS).

Alternate source:

- https://www.kaggle.com/datasets/rmisra/news-category-dataset

### Modeling

Baseline:

- Lowercasing + punctuation removal tokenization
- Neural bag-of-words with `nn.EmbeddingBag` + small MLP head

Main:

- TinyBERT fine-tuning (`huawei-noah/TinyBERT_General_4L_312D`)
- Headline + optional short description concatenated with a separator token

Both models are trained with PyTorch Lightning and logged with MLflow.

### Deployment

The project provides CLI-based inference and MLflow Serving packaging for local deployment and integration.

## Setup

```bash
poetry install
poetry run pre-commit install
poetry run pre-commit run -a
```

Notes:

- MLflow tracking defaults to `http://127.0.0.1:8080` and can be changed via `logging.tracking_uri`.
- DVC is optional. If a remote is not configured, the code will fall back to downloading data from Hugging Face.

DVC installation (Windows):

```bash
pip install dvc
```

Initialize DVC in the repo:

```bash
dvc init
```

## Train

Baseline model:

```bash
poetry run python -m huffpost_classifier.cli_train model=baseline_embedding_bag
```

TinyBERT model:

```bash
poetry run python -m huffpost_classifier.cli_train model=bert_finetune
```

GPU training (if available):

```bash
poetry run python -m huffpost_classifier.cli_train model=bert_finetune trainer.accelerator=gpu trainer.devices=1
```

Quick run (small subset):

```bash
poetry run python -m huffpost_classifier.cli_train model=baseline_embedding_bag data.limit_train_samples=20000 data.limit_val_samples=5000 data.limit_test_samples=5000 trainer.max_epochs=5
```

Artifacts are saved under `artifacts/<run_name>/` and plots under `plots/<run_name>/`.
If you have multiple runs, set `infer.artifacts_dir=artifacts/<run_name>` to target a specific run.
The baseline uses lowercase + punctuation removal tokenization, drops rows with empty headlines, and applies class-weighted loss (plus weighted sampling for baseline) to address imbalance.

## Production preparation

Export ONNX for the latest run:

```bash
poetry run python -m huffpost_classifier.cli_export model=baseline_embedding_bag
poetry run python -m huffpost_classifier.cli_export model=bert_finetune
```

If you want to export a specific run:

```bash
poetry run python -m huffpost_classifier.cli_export model=baseline_embedding_bag infer.artifacts_dir=artifacts/<run_name>
```

Artifacts layout:

- `artifacts/<run_name>/baseline/`: `best.ckpt`, `model.pt`, `vocab.json`, `label_map.json`
- `artifacts/<run_name>/bert/`: `best.ckpt`, `model/`, `tokenizer/`, `label_map.json`
- `artifacts/<run_name>/onnx/`: `baseline.onnx`, `bert.onnx`

## Infer

Single example:

```bash
poetry run python -m huffpost_classifier.cli_infer model=baseline_embedding_bag infer.headline="Example headline" infer.short_description="Example short description"
```

TinyBERT example:

```bash
poetry run python -m huffpost_classifier.cli_infer model=bert_finetune infer.headline="Example headline" infer.short_description="Example short description"
```

JSONL input:

```bash
poetry run python -m huffpost_classifier.cli_infer model=baseline_embedding_bag infer.input_path=data/examples/infer_example.jsonl
```

Test-split inference (default when no input is provided):

```bash
poetry run python -m huffpost_classifier.cli_infer model=baseline_embedding_bag infer.max_rows=5
```

Limit rows for quick testing:

```bash
poetry run python -m huffpost_classifier.cli_infer model=baseline_embedding_bag infer.input_path=data/examples/infer_example.jsonl infer.max_rows=5
```

Write predictions to a JSONL file:

```bash
poetry run python -m huffpost_classifier.cli_infer model=baseline_embedding_bag infer.input_path=data/examples/infer_example.jsonl infer.max_rows=5 infer.output_path=outputs/predictions.jsonl
```

stdin JSONL input:

```bash
type data\examples\infer_example.jsonl | poetry run python -m huffpost_classifier.cli_infer model=baseline_embedding_bag infer.use_stdin=true
```

Expected output format:

```
prediction[0]: POLITICS
  top1: POLITICS (0.8123)
  top2: WORLD (0.0644)
  top3: BUSINESS (0.0412)
```

## DVC notes

To use a DVC remote (optional):

```bash
dvc init
```

The training and inference commands first attempt `dvc pull` via the DVC Python API and fall back to a dataset download if the remote is unavailable.

After the first data download, track local data with DVC:

```bash
dvc add data/raw data/processed data/splits data/examples
git add data/*.dvc
```

## Inference server (MLflow Serving)

Package the latest model into an MLflow pyfunc bundle:

```bash
poetry run python -m huffpost_classifier.cli_package_mlflow model=baseline_embedding_bag
```

Serve the model locally using the current environment:

```bash
poetry run mlflow models serve -m artifacts/<run_name>/mlflow_baseline --env-manager local --port 5001
```
