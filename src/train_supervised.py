import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    LOGISTIC_REGRESSION_MODEL_PATH,
    MODELS_DIR,
    NORMAL_SAMPLE_SIZE,
    RANDOM_FOREST_ESTIMATORS,
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_STATE,
    SUPERVISED_FEATURES_PATH,
    SUPERVISED_SCALER_PATH,
    TRAINING_DATA_PATH,
)
from src.data_utils import get_feature_columns, validate_training_columns


def build_training_sample(df: pd.DataFrame) -> pd.DataFrame:
    fraud_df = df[df["Class"] == 1]
    normal_df = df[df["Class"] == 0].sample(n=NORMAL_SAMPLE_SIZE, random_state=RANDOM_STATE)

    return pd.concat([fraud_df, normal_df], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )


def train_supervised_models() -> None:
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")

    df = pd.read_csv(TRAINING_DATA_PATH)
    validate_training_columns(df)

    train_df = build_training_sample(df)
    feature_columns = get_feature_columns(train_df)

    X_train = train_df[feature_columns]
    y_train = train_df["Class"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logistic_regression = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    random_forest = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logistic_regression.fit(X_train_scaled, y_train)
    random_forest.fit(X_train_scaled, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SUPERVISED_SCALER_PATH)
    joblib.dump(feature_columns, SUPERVISED_FEATURES_PATH)
    joblib.dump(logistic_regression, LOGISTIC_REGRESSION_MODEL_PATH)
    joblib.dump(random_forest, RANDOM_FOREST_MODEL_PATH)

    print(f"Saved supervised scaler: {SUPERVISED_SCALER_PATH}")
    print(f"Saved supervised feature columns: {SUPERVISED_FEATURES_PATH}")
    print(f"Saved Logistic Regression model: {LOGISTIC_REGRESSION_MODEL_PATH}")
    print(f"Saved Random Forest model: {RANDOM_FOREST_MODEL_PATH}")


def main() -> None:
    train_supervised_models()


if __name__ == "__main__":
    main()
