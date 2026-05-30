from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAINING_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
NEW_TRANSACTIONS_PATH = RAW_DATA_DIR / "new_transactions.csv"

CLUSTERED_DATA_PATH = PROCESSED_DATA_DIR / "creditcard_clustered.csv"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "fraud_predictions.csv"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
CLUSTER_PROFILES_PATH = MODELS_DIR / "cluster_profiles.pkl"

DBSCAN_EPS = 2
DBSCAN_MIN_SAMPLES = 5
FRAUD_CLUSTER_MIN_COUNT = 10
FRAUD_CLUSTER_MIN_RATE = 0.50
MATCH_DISTANCE_MULTIPLIER = 1.25
