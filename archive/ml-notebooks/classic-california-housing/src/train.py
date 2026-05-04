import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COL = "MedHouseVal"


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def score_model(model, X_valid, y_valid) -> dict:
    pred = model.predict(X_valid)
    return {
        "mae": float(mean_absolute_error(y_valid, pred)),
        "rmse": float(rmse(y_valid, pred)),
        "r2": float(r2_score(y_valid, pred)),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed_dir / "train.csv")
    valid_df = pd.read_csv(processed_dir / "valid.csv")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_valid = valid_df.drop(columns=[TARGET_COL])
    y_valid = valid_df[TARGET_COL]

    candidates = {
        "linear_regression": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest": Pipeline(
            steps=[
                ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
            ]
        ),
    }

    train_metrics = {}
    best_name = None
    best_model = None
    best_rmse = float("inf")

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics = score_model(model, X_valid, y_valid)
        train_metrics[name] = metrics

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_name = name
            best_model = model

    with open(artifacts_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_metric": "rmse",
                "best_model": best_name,
                "metrics": train_metrics,
            },
            f,
            indent=2,
        )

    joblib.dump(best_model, artifacts_dir / "best_model.joblib")

    print(f"Best model: {best_name}")
    print(f"Validation RMSE: {best_rmse:.4f}")
    print("Saved model + training metrics.")


if __name__ == "__main__":
    main()
