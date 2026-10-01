"""Small helpers shared by train.py, evaluate.py and predict_online.py."""

TARGET = "Total_spent"
CITIES = ["Dhaka", "Chittagong", "Khulna", "Rajshahi", "Sylhet"]
GENDERS = ["Male", "Female", "Other"]


def make_xy(df):
    """Turn a feature table into model inputs (X) and, if present, the target (y)."""
    X = df.drop(columns=["Customer_ID", "event_timestamp", TARGET], errors="ignore").copy()
    X["City"] = X["City"].map({c: i for i, c in enumerate(CITIES)})
    X["Gender"] = X["Gender"].map({g: i for i, g in enumerate(GENDERS)})
    X = X.sort_index(axis=1)          # same column order every time
    y = df[TARGET] if TARGET in df.columns else None
    return X, y
