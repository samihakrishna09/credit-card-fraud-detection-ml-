import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    LOGISTIC_REGRESSION_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    MODELS_DIR,
    NEW_TRANSACTIONS_PATH,
    NORMAL_SAMPLE_SIZE,
    PREDICTIONS_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_FOREST_ESTIMATORS,
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_STATE,
    SUPERVISED_SCALER_PATH,
    SUPERVISED_PREDICTIONS_PATH,
    TRAINING_DATA_PATH,
)
from src.data_utils import get_feature_columns, validate_feature_order, validate_training_columns


def build_training_sample(df: pd.DataFrame) -> pd.DataFrame:
    fraud_df = df[df["Class"] == 1]
    normal_df = df[df["Class"] == 0].sample(n=NORMAL_SAMPLE_SIZE, random_state=RANDOM_STATE)
    return pd.concat([fraud_df, normal_df], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )


def calculate_metrics(model_name: str, y_true: pd.Series, y_pred: pd.Series) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    return {
        "Model": model_name,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
    }


def compare_models() -> pd.DataFrame:
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")

    if not NEW_TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(f"Evaluation data not found: {NEW_TRANSACTIONS_PATH}")

    train_df = pd.read_csv(TRAINING_DATA_PATH)
    eval_df = pd.read_csv(NEW_TRANSACTIONS_PATH)

    validate_training_columns(train_df)

    if "Class" not in eval_df.columns:
        raise ValueError("new_transactions.csv must contain Class to compare model performance.")

    train_sample = build_training_sample(train_df)
    feature_columns = get_feature_columns(train_sample)
    validate_feature_order(eval_df, feature_columns)

    X_train = train_sample[feature_columns]
    y_train = train_sample["Class"]
    X_eval = eval_df[feature_columns]
    y_eval = eval_df["Class"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=RANDOM_FOREST_ESTIMATORS,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    prediction_df = eval_df.copy()
    comparison_rows = []

    if PREDICTIONS_PATH.exists():
        dbscan_predictions = pd.read_csv(PREDICTIONS_PATH)
        y_dbscan = dbscan_predictions["Fraud_Prediction"].map({"Normal": 0, "Fraud": 1})
        prediction_df["DBSCAN_Prediction"] = y_dbscan
        comparison_rows.append(calculate_metrics("DBSCAN Cluster Matching", y_eval, y_dbscan))

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_eval_scaled)
        prediction_df[f"{model_name}_Prediction"] = predictions
        comparison_rows.append(calculate_metrics(model_name, y_eval, predictions))

    comparison_df = pd.DataFrame(comparison_rows).sort_values("F1 Score", ascending=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SUPERVISED_SCALER_PATH)
    joblib.dump(models["Logistic Regression"], LOGISTIC_REGRESSION_MODEL_PATH)
    joblib.dump(models["Random Forest"], RANDOM_FOREST_MODEL_PATH)
    prediction_df.to_csv(SUPERVISED_PREDICTIONS_PATH, index=False)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    print("Model Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved supervised scaler: {SUPERVISED_SCALER_PATH}")
    print(f"Saved Logistic Regression model: {LOGISTIC_REGRESSION_MODEL_PATH}")
    print(f"Saved Random Forest model: {RANDOM_FOREST_MODEL_PATH}")
    print(f"\nSaved supervised predictions: {SUPERVISED_PREDICTIONS_PATH}")
    print(f"Saved model comparison: {MODEL_COMPARISON_PATH}")

    return comparison_df


def main() -> None:
    compare_models()


if __name__ == "__main__":
    main()
