"""Stage 3 - get_features: fetch training features from the OFFLINE store."""
import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
service = store.get_feature_service("spend_model")

for name in ["train", "test"]:
    labels = pd.read_parquet(f"data/prepared/{name}_labels.parquet")   # Customer_ID, event_timestamp, Total_spent
    # Feast joins the features onto each customer (as of event_timestamp)
    data = store.get_historical_features(entity_df=labels, features=service).to_df()
    data.to_csv(f"data/training/{name}.csv", index=False)
    print(name, data.shape)
