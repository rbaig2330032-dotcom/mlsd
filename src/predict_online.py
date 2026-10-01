"""Bonus (not a pipeline stage): predict using the ONLINE store.

Usage: python src/predict_online.py CUST000001 CUST000002
"""
import sys
import joblib
import pandas as pd
from feast import FeatureStore
from common import make_xy

ids = sys.argv[1:] or ["CUST000001", "CUST000002", "CUST000003"]

store = FeatureStore(repo_path="feature_repo")
model = joblib.load("models/model.joblib")

# Fetch the latest features for these customers from the online store
online = store.get_online_features(
    features=store.get_feature_service("spend_model"),
    entity_rows=[{"Customer_ID": i} for i in ids],
).to_df()

X, _ = make_xy(online)
online["predicted_Total_spent"] = model.predict(X).round(2)
print(online[["Customer_ID", "predicted_Total_spent"]])
