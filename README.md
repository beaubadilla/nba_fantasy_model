# NBA Fantasy Model

An end-to-end machine learning pipeline for forecasting NBA fantasy production. The project ingests multi-season player box score data, engineers basketball-aware features (career trends, lagged stats, coaching/team changes), and trains an explainable XGBoost regressor that exports both metrics and deployment-ready artifacts.

## Why this project stands out
- **Production-quality data stack** – raw CSV ingestion → imputation of missing seasons → advanced feature engineering → model training with traceable artifacts.
- **Basketball-shaped features** – previous-season context, rolling career form, coaching/team stability flags, and fantasy scoring rules turn basic box scores into signal.
- **Reproducible + tested** – deterministic pipelines, type-safe utilities, and `pytest` coverage for every transformation layer.
- **Portfolio-friendly deliverables** – tidy processed dataset, serialized model (`.joblib`), JSON metrics, and prediction exports ready for stakeholder review.

## Repository layout
```
├── data/
│   ├── raw/         # Season folders (e.g., 22_23/players.csv)
│   ├── interim/     # Pipeline checkpoints (imputed, feature engineering, filtered)
│   └── processed/   # Final modeling table (data.csv)
├── outputs/models/  # Saved pipelines, metrics, sample predictions
├── scripts/         # Tools for a variety of purposes (e.g. display feature importance cleanly)
├── src/             # Production code (loading, features, filters, utils)
├── tests/           # Pytest suite covering each transformation
├── run_pipeline.py  # Orchestrates the feature pipeline end-to-end
└── training.py      # Trains & evaluates the XGBoost regressor
```

## Data pipeline
The pipeline is orchestrated by [`run_pipeline.py`](run_pipeline.py) and can be triggered via `python run_pipeline.py --data-root data/raw --output data/interim/players.csv`. Major steps:
1. **Ingestion (`src/data_loading.py`)** – read each season folder (e.g., `22_23/players.csv`), add a `season_start` column, and stack them.
2. **Imputation (`src/data_imputation.py`)** – backfill missing seasons for injured players, fill static attributes (name/position), and recompute age trajectories.
3. **Feature engineering (`src/feature_engineering.py`)**
   - `add_years_in_nba`: experience clock per player.
   - `add_career_per_game_features`: rolling career per-game rates (PPG, APG, etc.).
   - `add_previous_season_lag_features`: previous-season stats aligned to aggregate “TOT/2TM” rows.
   - `add_fantasy_points_per_game`: DraftKings-style fantasy scoring target.
   - `add_new_coach_flag` & `add_new_team_flag`: stability indicators built from curated coaching data.
4. **Filtering (`src/filter.py`)** – remove per-team duplicates for multi-team years, drop zero-year records, and enforce a minimum games-played threshold.
5. **Column hygiene** – friendly feature names via `COLUMN_RENAME_MAP` and rounding helper `round_continuous_features` for clean exports.

The script writes three datasets so you can debug every layer:
- `*_imputed.csv` – gap-filled player seasons.
- `*_feature_engineer.csv` – enriched feature matrix.
- `*_filtered.csv` – final modeling view. A copy is also mirrored to `data/processed/data.csv` for the training step.

## Modeling approach
[`training.py`](training.py) consumes `data/processed/data.csv` and builds a scikit-learn `Pipeline` composed of:
- Column-wise preprocessing (`build_preprocessor`) with median scaling for numeric fields + one-hot encoding for categoricals.
- An `XGBRegressor` tuned for tabular regression (500 estimators, conservative learning rate, subsampling for regularization).
- `KFold` cross-validation to quantify generalization (default 5 folds) before fitting on the full dataset.

Artifacts saved to `outputs/models/` include:
- `xgb_pipeline.joblib` – serialized preprocessing + estimator stack.
- `xgb_metrics.json` – training + CV metrics, e.g.
  ```json
  {
    "cv_rmse_mean": 5.04,
    "cv_rmse_std": 0.11,
    "train_rmse": 2.09,
    "train_mae": 1.60,
    "train_r2": 0.96
  }
  ```
- `xgb_predictions.csv` – per-player predictions, ground truth, and residuals for slicing analyses.

Because identifier columns (`Player-additional`, `season_start`) are validated before merges, the exported predictions can be joined safely back to scouting tools or BI dashboards.

## Getting started
1. **Clone + create a virtual environment**
   ```bash
   git clone <repo-url>
   cd nba_fantasy_model
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run unit tests**
   ```bash
   make test  # wraps `pytest -v`
   ```

## Reproducing the pipeline
1. **Prepare raw data** – place `players.csv` files into season-named folders inside `data/raw/` (e.g., `data/raw/23_24/players.csv`).
2. **Generate features**
   ```bash
   python run_pipeline.py --data-root data/raw --output data/interim/players.csv
   ```
3. **Train + evaluate the model**
   ```bash
   python training.py \
     --data-path data/processed/data.csv \
     --model-output outputs/models/xgb_pipeline.joblib \
     --metrics-output outputs/models/xgb_metrics.json \
     --predictions-output outputs/models/xgb_predictions.csv
   ```
4. **Inspect artifacts** – review `outputs/models/xgb_metrics.json` for performance, open the CSV of predictions to audit players by team, coach, or experience level, and compare lift versus baseline heuristics.

## Quality checklist
- ✅ Deterministic preprocessing with explicit schema checks.
- ✅ Comprehensive unit tests for loaders, imputers, feature builders, and filters.
- ✅ Reusable logging utilities (`src/utils/logging_utils.py`) for pipeline observability.
- ✅ Make targets for testing, formatting, pipeline execution, and clean-up.

## Extending the project
1. **Feature sandbox** – incorporate advanced tracking metrics (e.g., usage rate, on/off splits) by adding columns to `add_previous_season_lag_features` and `COLUMN_RENAME_MAP`.
2. **Model experimentation** – plug alternative regressors (LightGBM, CatBoost, linear baselines) into `build_pipeline` and compare via the same CV harness.
3. **Deployment** – expose `xgb_pipeline.joblib` through a FastAPI endpoint or scheduled batch job that re-trains as soon as a new season’s data drops.
4. **Visualization suite** – pair `xgb_predictions.csv` with a Streamlit dashboard to communicate forecasts to product managers and coaches.

---
_This repository highlights the full spectrum of a sports analytics engagement: data engineering discipline, modeling rigor, and stakeholder-ready outputs._
