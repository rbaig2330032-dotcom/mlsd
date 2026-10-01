# RetailSmart: Customer Spending Prediction (DVC + Feast)

Course project for **DS-4491 Machine Learning Systems Design**.
A reproducible ML pipeline managed with **DVC**, with **Feast** (feature store) as the bonus component.

## 1. Problem and Dataset

**Problem:** predict how much a customer spends in total (`Total_spent`) from their shopping behaviour and profile.

**Dataset:** `RetailSmart-CustomerBehavior.csv` (100,000 customers, 11 columns, no missing values).

| Group | Columns |
|-------|---------|
| ID | `Customer_ID` |
| Behaviour | `Visits_per_month`, `Avg_spend_per_visit`, `Membership_yrs`, `Complaints_count`, `Satisfaction_Score` |
| Profile | `Age`, `Gender`, `City`, `Is_Loyalty_Member` |
| **Target** | `Total_spent` |

> **Note:** in this dataset `Total_spent = Visits_per_month × Avg_spend_per_visit` exactly, and the other columns are unrelated to it. The model therefore scores very high. The project is about the ML *workflow* (DVC + Feast), not about the difficulty of the prediction.

The raw CSV is tracked by DVC (not stored in Git). Place it at `data/raw/RetailSmart-CustomerBehavior.csv`.

## 2. ML Model

**Random Forest Regressor** (scikit-learn): 100 trees, max depth 10. Trains in under a minute.

- `City` and `Gender` are converted to numbers with fixed mappings (`src/common.py`).
- Train/test split: 80% / 20%, `random_state=42`.
- Hyperparameters live in `params.yaml`.

## 3. Project Structure

```
retailsmart-dvc/
├── data/
│   ├── raw/                 # raw CSV (DVC-tracked via .dvc file)
│   ├── prepared/            # train/test label tables
│   └── training/            # train.csv / test.csv produced by Feast
├── feature_repo/            # Feast feature store
│   ├── feature_store.yaml   # Feast config (registry, offline + online store)
│   ├── features.py          # entity, feature views, feature service
│   └── data/                # feature parquet files, registry.db, online_store.db
├── src/
│   ├── common.py            # shared encoding helper
│   ├── prepare.py           # stage 1
│   ├── feast_apply.py       # stage 2
│   ├── get_features.py      # stage 3
│   ├── train.py             # stage 4
│   ├── evaluate.py          # stage 5
│   └── predict_online.py    # bonus: prediction from the online store
├── models/                  # model.joblib (DVC-tracked)
├── metrics/                 # metrics.json
├── dvc.yaml                 # pipeline definition
├── params.yaml              # parameters
├── requirements.txt
└── README.md
```

## 4. DVC Pipeline

```mermaid
flowchart LR
    A[prepare] --> B[feast_apply] --> C[get_features] --> D[train] --> E[evaluate]
```

| Stage | What it does | Output |
|-------|--------------|--------|
| `prepare` | Cleans/validates the CSV, writes feature files for Feast, splits labels into train/test | feature parquets, label parquets |
| `feast_apply` | Registers feature definitions and loads the **online store** | `registry.db`, `online_store.db` |
| `get_features` | Retrieves training features from the **offline store** | `train.csv`, `test.csv` |
| `train` | Trains the Random Forest | `models/model.joblib` |
| `evaluate` | Computes MAE, RMSE, R² on the test set | `metrics/metrics.json` |

The whole pipeline is reproduced with one command: `dvc repro`.

## 5. Feast (Bonus)

- **Entity:** `customer` (key: `Customer_ID`).
- **Feature views:** `customer_stats` (behaviour) and `customer_profile` (profile).
- **Feature service:** `spend_model`, the exact feature group the model uses.
- **Offline store** (parquet files): used for *training*. `get_historical_features` joins features onto the labels.
- **Online store** (SQLite): holds the latest value of each feature for fast lookup. `feast materialize` copies data from offline to online. `src/predict_online.py` reads from it to make a prediction.

Training and prediction use the same feature definitions, so they stay consistent.

## 6. How to Run

### Setup

```bash
git clone <your-repo-url>
cd retailsmart-dvc
pip install -r requirements.txt
```

### First-time setup (new repo)

```bash
git init
dvc init
dvc remote add -d storage /path/to/dvc-remote      # a folder, Google Drive, S3, ...
cp /path/to/RetailSmart-CustomerBehavior.csv data/raw/
dvc add data/raw/RetailSmart-CustomerBehavior.csv
git add .
git commit -m "Initial project"
```

### Run the pipeline

```bash
dvc repro            # run all stages
dvc dag              # show the pipeline graph
dvc status           # check whether anything is out of date
dvc metrics show     # show the results
```

### Online prediction (Feast online store)

```bash
python src/predict_online.py CUST000001 CUST000002 CUST000003
```

### Share data and model

```bash
dvc push             # upload data/model to the DVC remote
dvc pull             # download them (on a new machine after git clone)
```

### Reproduce on another machine

```bash
git clone <your-repo-url> && cd retailsmart-dvc
pip install -r requirements.txt
dvc pull             # get the raw data from the remote
dvc repro            # rebuild everything
```

### Try changing a parameter

```bash
# edit params.yaml, e.g. train.n_estimators: 200
dvc status           # shows 'train' and 'evaluate' are stale
dvc repro            # re-runs only train + evaluate
```

## 7. Results

Test set (20,000 customers):

| Metric | Value |
|--------|-------|
| MAE | ≈ 0.23 |
| RMSE | ≈ 1.58 |
| R² | ≈ 0.9998 |

Exact values are written to `metrics/metrics.json` after `dvc repro`. The very high score is expected because the target is a product of two input columns (see the note in section 1).
