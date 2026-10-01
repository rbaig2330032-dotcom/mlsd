"""Stage 1 - prepare: raw CSV -> Feast source files + train/test label tables."""
import sys
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["prepare"]
df = pd.read_csv(sys.argv[1])

# basic checks
assert df["Customer_ID"].is_unique
assert df.isna().sum().sum() == 0

feature_time = pd.Timestamp(params["feature_timestamp"], tz="UTC")
label_time = pd.Timestamp(params["label_timestamp"], tz="UTC")

# 1) Feature files for Feast (one file per feature view)
stats_cols = ["Visits_per_month", "Avg_spend_per_visit", "Membership_yrs",
              "Complaints_count", "Satisfaction_Score"]
profile_cols = ["Age", "Gender", "City", "Is_Loyalty_Member"]

stats = df[["Customer_ID"] + stats_cols].copy()
stats["event_timestamp"] = feature_time
stats.to_parquet("feature_repo/data/customer_stats.parquet", index=False)

profile = df[["Customer_ID"] + profile_cols].copy()
profile["event_timestamp"] = feature_time
profile.to_parquet("feature_repo/data/customer_profile.parquet", index=False)

# 2) Labels (the target) are kept outside the feature store, split into train/test
labels = df[["Customer_ID", "Total_spent"]].copy()
labels["event_timestamp"] = label_time
train_labels, test_labels = train_test_split(
    labels, test_size=params["test_size"], random_state=params["random_state"]
)
train_labels.to_parquet("data/prepared/train_labels.parquet", index=False)
test_labels.to_parquet("data/prepared/test_labels.parquet", index=False)

print("train:", len(train_labels), "test:", len(test_labels))
