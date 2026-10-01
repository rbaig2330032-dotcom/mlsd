"""Stage 2 - feast_apply: register feature definitions and fill the online store."""
import sys
from datetime import datetime, timezone

sys.path.insert(0, "feature_repo")
from feast import FeatureStore
from features import customer, customer_stats, customer_profile, spend_model

store = FeatureStore(repo_path="feature_repo")

# "feast apply": save the definitions in the registry
store.apply([customer, customer_stats, customer_profile, spend_model])

# "feast materialize": copy the latest feature values into the ONLINE store
store.materialize(
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    end_date=datetime.now(timezone.utc),
)
print("Feast: applied and materialized")
