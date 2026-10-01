"""Stage 5 - evaluate: score the model on the test set and write metrics.json."""
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import make_xy

model = joblib.load(sys.argv[1])
X, y = make_xy(pd.read_csv(sys.argv[2]))
pred = model.predict(X)

metrics = {
    "mae": float(mean_absolute_error(y, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    "r2": float(r2_score(y, pred)),
}
with open(sys.argv[3], "w") as f:
    json.dump(metrics, f, indent=2)
print(metrics)
