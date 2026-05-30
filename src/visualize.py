import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    CONFUSION_MATRICES_CHART_PATH,
    FRAUD_DISTRIBUTION_CHART_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METRICS_CHART_PATH,
    TRAINING_DATA_PATH,
)


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


def plot_model_metrics_comparison() -> None:
    if not MODEL_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            f"Model comparison file not found: {MODEL_COMPARISON_PATH}. "
            "Run 'python -m src.compare_models' first."
        )

    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
    metrics = ["Precision", "Recall", "F1 Score"]
    chart_df = comparison_df.set_index("Model")[metrics]

    MODEL_METRICS_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)

    ax = chart_df.plot(
        kind="bar",
        figsize=(10, 6),
        color=["#2f80ed", "#27ae60", "#f2994a"],
        width=0.75,
    )
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Metric")
    plt.xticks(rotation=20, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    plt.tight_layout()
    plt.savefig(MODEL_METRICS_CHART_PATH, dpi=150)
    plt.close()

    print(f"Saved model metrics chart: {MODEL_METRICS_CHART_PATH}")


def plot_confusion_matrices() -> None:
    if not MODEL_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            f"Model comparison file not found: {MODEL_COMPARISON_PATH}. "
            "Run 'python -m src.compare_models' first."
        )

    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
    CONFUSION_MATRICES_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(comparison_df), figsize=(14, 4))

    if len(comparison_df) == 1:
        axes = [axes]

    for ax, row in zip(axes, comparison_df.to_dict(orient="records")):
        matrix = [
            [row["True Negatives"], row["False Positives"]],
            [row["False Negatives"], row["True Positives"]],
        ]

        ax.imshow(matrix, cmap="Blues")
        ax.set_title(row["Model"])
        ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Fraud"])
        ax.set_yticks([0, 1], labels=["Actual Normal", "Actual Fraud"])

        for y in range(2):
            for x in range(2):
                ax.text(x, y, int(matrix[y][x]), ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRICES_CHART_PATH, dpi=150)
    plt.close()

    print(f"Saved confusion matrices chart: {CONFUSION_MATRICES_CHART_PATH}")


def main() -> None:
    plot_fraud_distribution()
    plot_model_metrics_comparison()
    plot_confusion_matrices()


if __name__ == "__main__":
    main()
