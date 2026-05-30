import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.config import (
    CLUSTER_PROFILES_PATH,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    MATCH_DISTANCE_MULTIPLIER,
    NEW_TRANSACTIONS_PATH,
    PREDICTIONS_PATH,
    PROCESSED_DATA_DIR,
    SCALER_PATH,
)
from src.data_utils import validate_feature_order, validate_prediction_columns


def get_fraud_cluster_ids(cluster_profiles: dict) -> list[int]:
    return [
        cluster_id
        for cluster_id, profile in cluster_profiles["clusters"].items()
        if profile["is_fraud_dominant"]
    ]


def match_to_historical_cluster(row: np.ndarray, cluster_profiles: dict) -> tuple[int | None, float]:
    best_cluster_id = None
    best_distance = float("inf")

    for cluster_id, profile in cluster_profiles["clusters"].items():
        distance = float(np.linalg.norm(row - profile["centroid"]))

        if distance < best_distance:
            best_cluster_id = cluster_id
            best_distance = distance

    return best_cluster_id, best_distance


def predict_transactions() -> pd.DataFrame:
    missing_files = [
        path
        for path in [SCALER_PATH, CLUSTER_PROFILES_PATH, NEW_TRANSACTIONS_PATH]
        if not path.exists()
    ]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Prediction requires these files, but some are missing:\n"
            f"{missing}\n\n"
            "Run 'python -m src.generate_sample_data' and 'python -m src.train' first."
        )

    scaler = joblib.load(SCALER_PATH)
    cluster_profiles = joblib.load(CLUSTER_PROFILES_PATH)
    new_data = pd.read_csv(NEW_TRANSACTIONS_PATH)

    validate_prediction_columns(new_data)
    feature_columns = cluster_profiles["feature_columns"]
    validate_feature_order(new_data, feature_columns)

    X_new = new_data[feature_columns]
    X_new_scaled = scaler.transform(X_new)

    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    new_data["New_DBSCAN_Cluster"] = dbscan.fit_predict(X_new_scaled)

    fraud_clusters = get_fraud_cluster_ids(cluster_profiles)
    matches = [match_to_historical_cluster(row, cluster_profiles) for row in X_new_scaled]
    new_data["Matched_Historical_Cluster"] = [match[0] for match in matches]
    new_data["Distance_To_Match"] = [round(match[1], 4) for match in matches]

    def label_prediction(row: pd.Series) -> str:
        matched_cluster = row["Matched_Historical_Cluster"]

        if pd.isna(matched_cluster) or int(matched_cluster) not in fraud_clusters:
            return "Normal"

        profile = cluster_profiles["clusters"][int(matched_cluster)]
        allowed_distance = max(profile["radius"] * MATCH_DISTANCE_MULTIPLIER, 1e-9)
        return "Fraud" if row["Distance_To_Match"] <= allowed_distance else "Normal"

    new_data["Fraud_Prediction"] = new_data.apply(label_prediction, axis=1)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_data.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Fraud-dominant historical clusters: {fraud_clusters}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print("\nPrediction counts:")
    print(new_data["Fraud_Prediction"].value_counts().to_string())

    return new_data


def main() -> None:
    predict_transactions()


if __name__ == "__main__":
    main()
