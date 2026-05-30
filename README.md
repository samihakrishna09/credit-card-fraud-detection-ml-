# Fraud Detection Using DBSCAN and Cluster Matching

This project detects potentially fraudulent credit-card transactions using DBSCAN clustering and historical fraud-cluster matching.

The project uses the Kaggle Credit Card Fraud Detection dataset as historical transaction data. It trains on known transactions, identifies fraud-dominant DBSCAN clusters, and predicts whether new unseen transactions are likely to be `Fraud` or `Normal`.

## Features

- DBSCAN-based transaction clustering
- Reusable preprocessing with `StandardScaler`
- Fraud-dominant historical cluster detection
- Cluster-profile matching for new transactions
- Recall, F1-score, and confusion matrix evaluation
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
|   `-- cluster_profiles.pkl
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_utils.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- train.py
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

For prediction and evaluation, this project also uses:

```text
data/raw/new_transactions.csv
```

For evaluation, create this file by taking a small sample of rows from `creditcard.csv` and keeping the `Class` column. The `Class` column is needed so `evaluate.py` can compare predicted labels with true labels.

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

Evaluate recall, F1-score, and confusion matrix:

```bash
python -m src.evaluate
```

## Evaluation

The file `src/evaluate.py` checks how well the fraud prediction worked on `new_transactions.csv`.

It compares:

- the actual label from the `Class` column
- the predicted label from `data/processed/fraud_predictions.csv`

It calculates:

- `Recall Score`: how many real fraud transactions were correctly detected
- `F1 Score`: balance between fraud precision and fraud recall
- `Confusion Matrix`: count of correct and incorrect predictions

Run evaluation after prediction:

```bash
python -m src.evaluate
```

Current evaluation output:

```text
Recall Score: 0.3
F1 Score: 0.46153846153846156

Confusion Matrix:
[[30  0]
 [ 7  3]]
```

Confusion matrix meaning:

```text
[[TN FP]
 [FN TP]]
```

For this project output:

- `30` normal transactions were correctly predicted as normal
- `0` normal transactions were wrongly predicted as fraud
- `7` fraud transactions were missed
- `3` fraud transactions were correctly predicted as fraud

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
11. `evaluate.py` compares predictions with the true `Class` labels.

## Model Settings

Main settings are stored in `src/config.py`:

```python
DBSCAN_EPS = 2
DBSCAN_MIN_SAMPLES = 5
FRAUD_CLUSTER_MIN_COUNT = 5
FRAUD_CLUSTER_MIN_RATE = 0.50
MATCH_DISTANCE_MULTIPLIER = 1.25
```

## Output

The final prediction file is:

```text
data/processed/fraud_predictions.csv
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

Evaluation output:

```text
Recall Score: 0.3
F1 Score: 0.46153846153846156

Confusion Matrix:
[[30  0]
 [ 7  3]]
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
