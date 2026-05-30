# Fraud Detection Using DBSCAN and Cluster Matching

This project detects potentially fraudulent credit-card transactions using DBSCAN clustering and historical fraud-cluster matching.

The project uses the Kaggle Credit Card Fraud Detection dataset as historical transaction data. It trains on known transactions, identifies fraud-dominant DBSCAN clusters, and predicts whether new unseen transactions are likely to be `Fraud` or `Normal`.

## Features

- DBSCAN-based transaction clustering
- Logistic Regression supervised classification
- Random Forest supervised classification
- Reusable preprocessing with `StandardScaler`
- Fraud-dominant historical cluster detection
- Cluster-profile matching for new transactions
- Model comparison with precision, recall, F1-score, and confusion matrix values
- CSV prediction output for analysis and reporting

## Dataset

Download the dataset from Kaggle:

[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

After downloading and extracting the ZIP file, place `creditcard.csv` here:

```text
data/raw/creditcard.csv
```

The Kaggle dataset contains:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

`Class` means:

- `0` = normal transaction
- `1` = fraud transaction

## Project Structure

```text
fraud detection ml/
|-- data/
|   |-- raw/
|   |   |-- creditcard.csv
|   |   `-- new_transactions.csv
|   `-- processed/
|       |-- creditcard_clustered.csv
|       `-- fraud_predictions.csv
|-- models/
|   |-- scaler.pkl
|   |-- cluster_profiles.pkl
|   |-- supervised_scaler.pkl
|   |-- logistic_regression_model.pkl
|   `-- random_forest_model.pkl
|-- reports/
|   `-- figures/
|       |-- fraud_distribution.png
|       |-- model_metrics_comparison.png
|       `-- model_confusion_matrices.png
|-- src/
|   |-- __init__.py
|   |-- compare_models.py
|   |-- config.py
|   |-- data_utils.py
|   |-- predict.py
|   |-- predict_supervised.py
|   |-- train.py
|   |-- train_supervised.py
|   `-- visualize.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Create New Transactions File

Kaggle provides only one main file: `creditcard.csv`.

For prediction and model comparison, this project also uses:

```text
data/raw/new_transactions.csv
```

For model comparison, create this file by taking a small sample of rows from `creditcard.csv` and keeping the `Class` column. The `Class` column is needed so `compare_models.py` can compare predicted labels with true labels.

Required columns:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

## Run the Project

Train DBSCAN and save cluster profiles:

```bash
python -m src.train
```

Predict fraud for new transactions:

```bash
python -m src.predict
```

Train Logistic Regression and Random Forest:

```bash
python -m src.train_supervised
```

Predict with Logistic Regression and Random Forest:

```bash
python -m src.predict_supervised
```

Train Logistic Regression and Random Forest, then compare all three methods:

```bash
python -m src.compare_models
```

Generate all charts:

```bash
python -m src.visualize
```

This command prints the class distribution in the terminal and saves three chart images.
<img width="979" height="689" alt="image" src="https://github.com/user-attachments/assets/ad7ddabb-8ad9-4bb7-a325-53e1f60c420f" />


Terminal output:

```text
Fraud Distribution:
Class
Normal    284315
Fraud        492
Name: count, dtype: int64
Saved fraud distribution chart: reports/figures/fraud_distribution.png
Saved model metrics chart: reports/figures/model_metrics_comparison.png
Saved confusion matrices chart: reports/figures/model_confusion_matrices.png
```

Charts are saved to:

```text
reports/figures/fraud_distribution.png
reports/figures/model_metrics_comparison.png
reports/figures/model_confusion_matrices.png
```

## Model Comparison

The file `src/compare_models.py` trains two supervised models and compares them with the DBSCAN cluster-matching result:

- DBSCAN Cluster Matching
- Logistic Regression
- Random Forest

It uses the labeled `Class` column in `new_transactions.csv` to calculate precision, recall, F1-score, true negatives, false positives, false negatives, and true positives.

Run:

```bash
python -m src.compare_models
```

Current comparison output:

```text
Model Comparison:
                  Model  Precision  Recall  F1 Score  True Negatives  False Positives  False Negatives  True Positives
          Random Forest   1.000000     1.0  1.000000              30                0                0              10
    Logistic Regression   0.909091     1.0  0.952381              29                1                0              10
DBSCAN Cluster Matching   1.000000     0.3  0.461538              30                0                7               3
```

The comparison shows that Random Forest and Logistic Regression perform better than DBSCAN on this labeled evaluation sample. DBSCAN is unsupervised, while Logistic Regression and Random Forest directly learn from the fraud labels.

## How It Works

1. `train.py` loads `data/raw/creditcard.csv`.
2. It uses `V1` to `V28` and `Amount` as model features.
3. The features are scaled using `StandardScaler`.
4. DBSCAN groups historical transactions into clusters.
5. Clusters with enough fraud transactions and high fraud rate are marked fraud-dominant.
6. The scaler and historical cluster profiles are saved in `models/`.
7. `predict.py` loads `data/raw/new_transactions.csv`.
8. New transactions are scaled using the saved scaler.
9. Each new transaction is matched to the nearest historical cluster profile.
10. Transactions close to fraud-dominant clusters are labeled `Fraud`; others are labeled `Normal`.
11. `compare_models.py` trains Logistic Regression and Random Forest.
12. `compare_models.py` compares DBSCAN, Logistic Regression, and Random Forest.
13. `train_supervised.py` trains and saves Logistic Regression and Random Forest models.
14. `predict_supervised.py` loads the saved supervised models and predicts new transactions.

## Model Settings

Main settings are stored in `src/config.py`:

```python
DBSCAN_EPS = 2
DBSCAN_MIN_SAMPLES = 5
FRAUD_CLUSTER_MIN_COUNT = 5
FRAUD_CLUSTER_MIN_RATE = 0.50
MATCH_DISTANCE_MULTIPLIER = 1.25
NORMAL_SAMPLE_SIZE = 5000
RANDOM_STATE = 42
RANDOM_FOREST_ESTIMATORS = 100
```

## Output

The final prediction file is:

```text
data/processed/fraud_predictions.csv
```

Fraud distribution chart:

```text
reports/figures/fraud_distribution.png
```

Model metrics comparison chart:

```text
reports/figures/model_metrics_comparison.png
```

This chart compares all three models using:

- Precision
- Recall
- F1-score

Model confusion matrices chart:

```text
reports/figures/model_confusion_matrices.png
```

This chart shows the confusion matrix for:

- DBSCAN Cluster Matching
- Logistic Regression
- Random Forest

Fraud distribution output:

```text
Fraud Distribution:
Class
Normal    284315
Fraud        492
Name: count, dtype: int64
```

Training output from one Kaggle run:

```text
Cluster summary:
 Cluster  total_transactions  fraud_transactions
      -1                2154                 416
       2                2624                  30
       1                  20                  20
       0                   8                   8
       3                   7                   6
       4                   6                   6
       5                   7                   4
       9                  22                   1
      19                   5                   1
       7                 211                   0
       8                 162                   0
      12                  85                   0
       6                  55                   0
      11                  48                   0
      15                  18                   0
      16                  12                   0
      13                   9                   0
      14                   9                   0
      10                   6                   0
      22                   6                   0
      18                   5                   0
      21                   5                   0
      17                   4                   0
      20                   4                   0
```

Prediction output:

```text
Fraud-dominant historical clusters: [0, 1, 3, 4]

Prediction counts:
Fraud_Prediction
Normal    37
Fraud      3
```

Supervised training output:

```text
Saved supervised scaler: models/supervised_scaler.pkl
Saved supervised feature columns: models/supervised_feature_columns.pkl
Saved Logistic Regression model: models/logistic_regression_model.pkl
Saved Random Forest model: models/random_forest_model.pkl
```

Supervised prediction output:

```text
Saved supervised predictions: data/processed/supervised_predictions.csv

Logistic Regression prediction counts:
Logistic_Regression_Prediction
Normal    29
Fraud     11

Random Forest prediction counts:
Random_Forest_Prediction
Normal    30
Fraud     10
```

Model comparison output:

```text
Model Comparison:
                  Model  Precision  Recall  F1 Score  True Negatives  False Positives  False Negatives  True Positives
          Random Forest   1.000000     1.0  1.000000              30                0                0              10
    Logistic Regression   0.909091     1.0  0.952381              29                1                0              10
DBSCAN Cluster Matching   1.000000     0.3  0.461538              30                0                7               3
```

Model comparison file:

```text
data/processed/model_comparison.csv
```

Supervised prediction file:

```text
data/processed/supervised_predictions.csv
```

## Push to GitHub

Create a new empty repository on GitHub. Then run these commands from the project folder:

```bash
git add .
git commit -m "Initial fraud detection project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git push -u origin main
```

## Important Notes

The Kaggle dataset and generated model files are ignored by Git using `.gitignore`, because CSV and `.pkl` files can be large. The GitHub repository should contain the code and instructions, not the full dataset.

DBSCAN does not have a normal `predict` method. This project solves that by saving historical cluster profiles during training and matching new transactions to those profiles during prediction.

This is a learning and portfolio project. For production fraud detection, compare this approach with Isolation Forest, Local Outlier Factor, One-Class SVM, or supervised models if reliable labels are available.

## References

- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [DBSCAN in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html)
