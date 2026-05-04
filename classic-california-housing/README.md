# Classic Data Science Project: California Housing Price Prediction

A production-style, GitHub-ready data science project that predicts median house values using the **California Housing** dataset from scikit-learn.

## Why this project?

This is a classic supervised learning workflow with practical structure:
- reproducible data split
- baseline + stronger model comparison
- metrics tracking
- saved artifacts for reuse/deployment

---

## Project Structure

```text
classic-california-housing/
├── artifacts/                 # trained models + metrics
├── data/
│   ├── raw/                   # raw dataset snapshots
│   └── processed/             # train/valid/test splits
├── notebooks/                 # optional exploration notebook area
├── scripts/
│   └── run_pipeline.sh        # one-command pipeline runner
├── src/
│   ├── data.py                # data ingest + split
│   ├── train.py               # model training + selection
│   └── evaluate.py            # final test-set evaluation
├── .gitignore
├── Makefile
└── requirements.txt
```

---

## Quickstart

### 1) Create venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Run full pipeline

```bash
make run
```

Or:

```bash
bash scripts/run_pipeline.sh
```

---

## Outputs

After running, you get:

- `artifacts/best_model.joblib` → best model pipeline
- `artifacts/train_metrics.json` → validation metrics for candidate models
- `artifacts/test_metrics.json` → final test metrics of selected model
- `data/processed/*.csv` → reproducible data splits

---

## Modeling Approach

Models compared:
1. **Linear Regression** (baseline)
2. **Random Forest Regressor** (non-linear benchmark)

Metrics:
- MAE
- RMSE
- R²

Selection rule:
- Best validation RMSE wins.

---

## Suggested GitHub Additions

Before pushing, you may also add:
- badges (Python, license, CI)
- ✅ a short EDA notebook in `notebooks/` (`01_eda.ipynb`)
- GitHub Actions workflow for lint + train smoke test
- model card (`MODEL_CARD.md`)

---

## License

Use MIT (or your preferred license) before publishing.
