import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from src.config import NEW_TRANSACTIONS_PATH, PREDICTIONS_PATH


def main() -> None:
    actual_data = pd.read_csv(NEW_TRANSACTIONS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    if "Class" not in actual_data.columns:
        raise ValueError("new_transactions.csv must contain the Class column for evaluation.")

    y_true = actual_data["Class"]
    y_pred = predictions["Fraud_Prediction"].map({"Normal": 0, "Fraud": 1})

    print("Recall Score:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
