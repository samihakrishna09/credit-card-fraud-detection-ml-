from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TRAINING_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
NEW_TRANSACTIONS_PATH = RAW_DATA_DIR / "new_transactions.csv"

CLUSTERED_DATA_PATH = PROCESSED_DATA_DIR / "creditcard_clustered.csv"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "fraud_predictions.csv"
SUPERVISED_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "supervised_predictions.csv"
MODEL_COMPARISON_PATH = PROCESSED_DATA_DIR / "model_comparison.csv"
FRAUD_DISTRIBUTION_CHART_PATH = FIGURES_DIR / "fraud_distribution.png"
MODEL_METRICS_CHART_PATH = FIGURES_DIR / "model_metrics_comparison.png"
CONFUSION_MATRICES_CHART_PATH = FIGURES_DIR / "model_confusion_matrices.png"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
CLUSTER_PROFILES_PATH = MODELS_DIR / "cluster_profiles.pkl"
SUPERVISED_SCALER_PATH = MODELS_DIR / "supervised_scaler.pkl"
SUPERVISED_FEATURES_PATH = MODELS_DIR / "supervised_feature_columns.pkl"
LOGISTIC_REGRESSION_MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"

DBSCAN_EPS = 2
DBSCAN_MIN_SAMPLES = 5
FRAUD_CLUSTER_MIN_COUNT = 5
FRAUD_CLUSTER_MIN_RATE = 0.50
MATCH_DISTANCE_MULTIPLIER = 1.25
NORMAL_SAMPLE_SIZE = 5000
RANDOM_STATE = 42
RANDOM_FOREST_ESTIMATORS = 100
