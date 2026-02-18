import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_COL = "MedHouseVal"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"

    test_df = pd.read_csv(processed_dir / "test.csv")
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    model = joblib.load(artifacts_dir / "best_model.joblib")
    pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2": float(r2_score(y_test, pred)),
    }

    with open(artifacts_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
