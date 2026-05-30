import pandas as pd


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the credit-card feature columns expected by the DBSCAN pipeline."""
    v_columns = sorted(
        [column for column in df.columns if column.startswith("V")],
        key=lambda column: int(column[1:]) if column[1:].isdigit() else column,
    )

    if "Amount" not in df.columns:
        raise ValueError("Input data must contain an 'Amount' column.")

    if not v_columns:
        raise ValueError("Input data must contain PCA feature columns such as V1, V2, ...")

    return v_columns + ["Amount"]


def validate_feature_order(df: pd.DataFrame, expected_features: list[str]) -> None:
    missing_columns = [column for column in expected_features if column not in df.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input data is missing feature columns used during training: {missing}")


def validate_training_columns(df: pd.DataFrame) -> None:
    required_columns = {"Class", "Amount"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training data is missing required columns: {missing}")

    get_feature_columns(df)


def validate_prediction_columns(df: pd.DataFrame) -> None:
    get_feature_columns(df)
