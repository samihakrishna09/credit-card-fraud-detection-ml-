import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FRAUD_DISTRIBUTION_CHART_PATH, TRAINING_DATA_PATH


def plot_fraud_distribution() -> None:
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")

    df = pd.read_csv(TRAINING_DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("creditcard.csv must contain the Class column.")

    class_counts = df["Class"].value_counts().sort_index()
    print("Fraud Distribution:")
    print(class_counts.rename(index={0: "Normal", 1: "Fraud"}))

    labels = ["Normal", "Fraud"]
    colors = ["#2f80ed", "#eb5757"]

    FRAUD_DISTRIBUTION_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, class_counts.values, color=colors)
    plt.title("Fraud vs Normal Transaction Distribution")
    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions")

    for bar, count in zip(bars, class_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(FRAUD_DISTRIBUTION_CHART_PATH, dpi=150)
    plt.close()

    print(f"Saved fraud distribution chart: {FRAUD_DISTRIBUTION_CHART_PATH}")


def main() -> None:
    plot_fraud_distribution()


if __name__ == "__main__":
    main()
