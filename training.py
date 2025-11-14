"""Train an XGBoost regressor on the processed NBA fantasy dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    # mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

IDENTIFIER_COLUMNS = ("Player-additional", "season_start")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model on the processed NBA fantasy dataset."
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/data.csv",
        help="Path to the processed dataset CSV.",
    )
    parser.add_argument(
        "--target-col",
        default="fantasy_points_per_game",
        help="Target column to predict. Defaults to 'fantasy_points_per_game' until a target is available.",
    )
    parser.add_argument(
        "--model-output",
        default="outputs/models/xgb_pipeline.joblib",
        help="Where to persist the trained pipeline.",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/models/xgb_metrics.json",
        help="Where to write evaluation metrics.",
    )
    parser.add_argument(
        "--predictions-output",
        default="outputs/models/xgb_predictions.csv",
        help="Where to export per-row predictions for analysis.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for cross-validation scoring (use -1 for all cores).",
    )
    return parser.parse_args()


def load_data(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"Dataset at {data_path} is empty.")
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str, drop_cols: Iterable[str] | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. Available columns: {sorted(df.columns)}"
        )

    drop_cols = list(drop_cols or [])
    if target_col in drop_cols:
        raise ValueError("Target column cannot be part of the columns to drop.")

    feature_df = df.drop(columns=drop_cols, errors="ignore")
    y = feature_df[target_col]
    X = feature_df.drop(columns=[target_col])
    if X.empty:
        raise ValueError("Feature matrix is empty after dropping the target column.")
    return X, y


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found for preprocessing.")

    return ColumnTransformer(transformers=transformers)


def build_pipeline(feature_df: pd.DataFrame, random_state: int) -> Pipeline:
    preprocessor = build_preprocessor(feature_df)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=random_state,
        n_jobs=1,  # avoid nested parallelism when cross_val_score uses multiple workers
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
    n_jobs: int,
) -> Dict[str, float | list[float]]:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs,
    )
    rmse_scores = (-scores).tolist()
    return {
        "fold_rmse": rmse_scores,
        "rmse_mean": float(pd.Series(rmse_scores).mean()),
        "rmse_std": float(pd.Series(rmse_scores).std(ddof=0)),
    }


def fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipeline.fit(X, y)
    return pipeline


def compute_training_metrics(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> Tuple[Dict[str, float], pd.Series]:
    predictions = pipeline.predict(X)
    # rmse = mean_squared_error(y, predictions, squared=False)  # TODO: replace with root_mean_squared_error below?
    rmse = root_mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    return metrics, pd.Series(predictions, index=y.index, name="prediction")


def save_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, float | list[float]],
    model_output: str | Path,
    metrics_output: str | Path,
) -> None:
    model_path = Path(model_output)
    metrics_path = Path(metrics_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def save_predictions(
    df: pd.DataFrame,
    predictions: pd.Series,
    target_col: str,
    output_path: str | Path,
) -> pd.DataFrame:
    pred_path = Path(output_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    result = df.copy()
    result["prediction"] = predictions
    if target_col in result.columns:
        result["residual"] = result[target_col] - result["prediction"]
    result.to_csv(pred_path, index=False)
    return result


def validate_identifier_columns(df: pd.DataFrame) -> None:
    missing = [col for col in IDENTIFIER_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Identifier columns missing from dataset: {', '.join(missing)}")
    if df.duplicated(subset=IDENTIFIER_COLUMNS).any():
        raise ValueError(
            "Identifier columns do not uniquely identify each row; cannot safely merge predictions."
        )


def main() -> None:
    args = parse_args()
    df = load_data(args.data_path)
    validate_identifier_columns(df)
    identifier_df = df[list(IDENTIFIER_COLUMNS)].reset_index(drop=True)
    X, y = split_features_target(df, args.target_col, drop_cols=IDENTIFIER_COLUMNS)

    print(f"Loaded dataset with {len(df)} rows and {X.shape[1]} feature columns.")

    pipeline = build_pipeline(X, args.random_state)
    cv_results = run_cross_validation(
        pipeline, X, y, args.cv_folds, args.random_state, args.n_jobs
    )

    print(
        f"Cross-validated RMSE: {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f} "
        f"over {args.cv_folds} folds."
    )

    trained_pipeline = fit_pipeline(pipeline, X, y)
    training_metrics, predictions = compute_training_metrics(trained_pipeline, X, y)
    pred_df = identifier_df.assign(
        y_true=y.reset_index(drop=True),
        y_pred=predictions.reset_index(drop=True),
    )

    all_metrics: Dict[str, float | list[float]] = {
        **{f"cv_{k}": v for k, v in cv_results.items()},
        **{f"train_{k}": v for k, v in training_metrics.items()},
    }

    save_artifacts(
        trained_pipeline, all_metrics, args.model_output, args.metrics_output
    )
    prediction_export_df = save_predictions(
        df, predictions, args.target_col, args.predictions_output
    )
    pred_df = prediction_export_df.merge(
        pred_df,
        on=list(IDENTIFIER_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    pred_df.to_csv("pred_df.csv", index=False)

    print(f"Model saved to {args.model_output}")
    print(f"Metrics saved to {args.metrics_output}")
    print(f"Predictions saved to {args.predictions_output}")


if __name__ == "__main__":
    main()
