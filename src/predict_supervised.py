import joblib
import pandas as pd

from src.config import (
    LOGISTIC_REGRESSION_MODEL_PATH,
    NEW_TRANSACTIONS_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_FOREST_MODEL_PATH,
    SUPERVISED_FEATURES_PATH,
    SUPERVISED_PREDICTIONS_PATH,
    SUPERVISED_SCALER_PATH,
)
from src.data_utils import validate_feature_order, validate_prediction_columns


def label_predictions(predictions) -> list[str]:
    return ["Fraud" if prediction == 1 else "Normal" for prediction in predictions]


def predict_supervised_models() -> pd.DataFrame:
    missing_files = [
        path
        for path in [
            SUPERVISED_SCALER_PATH,
            SUPERVISED_FEATURES_PATH,
            LOGISTIC_REGRESSION_MODEL_PATH,
            RANDOM_FOREST_MODEL_PATH,
            NEW_TRANSACTIONS_PATH,
        ]
        if not path.exists()
    ]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Supervised prediction requires these files, but some are missing:\n"
            f"{missing}\n\n"
            "Run 'python -m src.train_supervised' first."
        )

    scaler = joblib.load(SUPERVISED_SCALER_PATH)
    feature_columns = joblib.load(SUPERVISED_FEATURES_PATH)
    logistic_regression = joblib.load(LOGISTIC_REGRESSION_MODEL_PATH)
    random_forest = joblib.load(RANDOM_FOREST_MODEL_PATH)

    new_data = pd.read_csv(NEW_TRANSACTIONS_PATH)
    validate_prediction_columns(new_data)
    validate_feature_order(new_data, feature_columns)

    X_new = new_data[feature_columns]
    X_new_scaled = scaler.transform(X_new)

    lr_predictions = logistic_regression.predict(X_new_scaled)
    rf_predictions = random_forest.predict(X_new_scaled)

    result_df = new_data.copy()
    result_df["Logistic_Regression_Prediction"] = label_predictions(lr_predictions)
    result_df["Random_Forest_Prediction"] = label_predictions(rf_predictions)
    result_df["Logistic_Regression_Fraud_Probability"] = logistic_regression.predict_proba(
        X_new_scaled
    )[:, 1].round(4)
    result_df["Random_Forest_Fraud_Probability"] = random_forest.predict_proba(X_new_scaled)[
        :, 1
    ].round(4)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(SUPERVISED_PREDICTIONS_PATH, index=False)

    print(f"Saved supervised predictions: {SUPERVISED_PREDICTIONS_PATH}")
    print("\nLogistic Regression prediction counts:")
    print(result_df["Logistic_Regression_Prediction"].value_counts().to_string())
    print("\nRandom Forest prediction counts:")
    print(result_df["Random_Forest_Prediction"].value_counts().to_string())

    return result_df


def main() -> None:
    predict_supervised_models()


if __name__ == "__main__":
    main()
