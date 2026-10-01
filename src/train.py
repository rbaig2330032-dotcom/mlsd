"""Stage 4 - train: fit a Random Forest on the training features."""
import sys
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from common import make_xy

params = yaml.safe_load(open("params.yaml"))["train"]

df = pd.read_csv(sys.argv[1])
X, y = make_xy(df)

model = RandomForestRegressor(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"],
    n_jobs=-1,
)
model.fit(X, y)

joblib.dump(model, sys.argv[2])
print("model saved to", sys.argv[2])
