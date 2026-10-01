"""Feast feature definitions: 1 entity, 2 feature views, 1 feature service."""
from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# Entity = the "thing" we describe. Customer_ID is the key used to look features up.
customer = Entity(name="customer", join_keys=["Customer_ID"])

# --- Feature view 1: shopping behaviour ---
stats_source = FileSource(
    path="data/customer_stats.parquet",      # relative to feature_repo/
    timestamp_field="event_timestamp",
)
customer_stats = FeatureView(
    name="customer_stats",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="Visits_per_month", dtype=Int64),
        Field(name="Avg_spend_per_visit", dtype=Float64),
        Field(name="Membership_yrs", dtype=Float64),
        Field(name="Complaints_count", dtype=Int64),
        Field(name="Satisfaction_Score", dtype=Float64),
    ],
    source=stats_source,
)

# --- Feature view 2: customer profile ---
profile_source = FileSource(
    path="data/customer_profile.parquet",
    timestamp_field="event_timestamp",
)
customer_profile = FeatureView(
    name="customer_profile",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="Age", dtype=Int64),
        Field(name="Gender", dtype=String),
        Field(name="City", dtype=String),
        Field(name="Is_Loyalty_Member", dtype=Int64),
    ],
    source=profile_source,
)

# The model always uses this same group of features (training and online).
spend_model = FeatureService(name="spend_model", features=[customer_stats, customer_profile])
