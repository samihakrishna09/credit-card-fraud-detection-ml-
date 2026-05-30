import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLUSTER_PROFILES_PATH,
    CLUSTERED_DATA_PATH,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    FRAUD_CLUSTER_MIN_COUNT,
    FRAUD_CLUSTER_MIN_RATE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SCALER_PATH,
    TRAINING_DATA_PATH,
)
from src.data_utils import get_feature_columns, validate_training_columns


def build_cluster_profiles(
    df: pd.DataFrame, scaled_features: np.ndarray, feature_columns: list[str]
) -> dict:
    profiles = {
        "feature_columns": feature_columns,
        "fraud_cluster_min_count": FRAUD_CLUSTER_MIN_COUNT,
        "fraud_cluster_min_rate": FRAUD_CLUSTER_MIN_RATE,
        "clusters": {},
    }

    scaled_df = pd.DataFrame(scaled_features, columns=feature_columns, index=df.index)

    for cluster_id, cluster_rows in df[df["Cluster"] != -1].groupby("Cluster"):
        row_index = cluster_rows.index
        cluster_points = scaled_df.loc[row_index].to_numpy()
        centroid = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        fraud_count = int(cluster_rows["Class"].sum())
        total_count = int(len(cluster_rows))
        fraud_rate = fraud_count / total_count if total_count else 0.0

        profiles["clusters"][int(cluster_id)] = {
            "centroid": centroid,
            "radius": float(np.percentile(distances, 95)) if len(distances) else 0.0,
            "total_count": total_count,
            "fraud_count": fraud_count,
            "fraud_rate": fraud_rate,
            "is_fraud_dominant": (
                fraud_count > FRAUD_CLUSTER_MIN_COUNT and fraud_rate >= FRAUD_CLUSTER_MIN_RATE
            ),
        }

    return profiles


def train_cluster_model() -> pd.DataFrame:
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAINING_DATA_PATH}. "
            "Run 'python -m src.generate_sample_data' or add your real creditcard.csv file."
        )

    df = pd.read_csv(TRAINING_DATA_PATH)

    fraud_df = df[df["Class"] == 1]
    normal_df = df[df["Class"] == 0].sample(n=5000, random_state=42)
    df = pd.concat([fraud_df, normal_df], ignore_index=True)

    feature_columns = get_feature_columns(df)
    X = df[feature_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    df["Cluster"] = dbscan.fit_predict(X_scaled)
    cluster_profiles = build_cluster_profiles(df, X_scaled, feature_columns)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(cluster_profiles, CLUSTER_PROFILES_PATH)
    df.to_csv(CLUSTERED_DATA_PATH, index=False)

    cluster_summary = (
        df.groupby("Cluster")["Class"]
        .agg(total_transactions="count", fraud_transactions="sum")
        .reset_index()
        .sort_values(["fraud_transactions", "total_transactions"], ascending=False)
    )

    print(f"Saved scaler: {SCALER_PATH}")
    print(f"Saved cluster profiles: {CLUSTER_PROFILES_PATH}")
    print(f"Saved clustered data: {CLUSTERED_DATA_PATH}")
    print("\nCluster summary:")
    print(cluster_summary.to_string(index=False))

    return df


def main() -> None:
    train_cluster_model()


if __name__ == "__main__":
    main()
