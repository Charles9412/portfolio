from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    data_bunch = fetch_california_housing(as_frame=True)
    df = data_bunch.frame.copy()

    # Persist raw snapshot
    df.to_csv(raw_dir / "california_housing.csv", index=False)

    target_col = "MedHouseVal"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    # Final split ratio: 60/20/20

    train_df = X_train.copy()
    train_df[target_col] = y_train

    valid_df = X_valid.copy()
    valid_df[target_col] = y_valid

    test_df = X_test.copy()
    test_df[target_col] = y_test

    train_df.to_csv(processed_dir / "train.csv", index=False)
    valid_df.to_csv(processed_dir / "valid.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    print("Saved raw + processed datasets.")


if __name__ == "__main__":
    main()
