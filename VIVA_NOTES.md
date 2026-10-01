# Viva Notes (not required in the repo, for your own revision)

**What does `dvc repro` do?** Reads `dvc.yaml`, checks which stages have changed inputs (code, data, params), and re-runs only those, in dependency order.

**Why DVC?** Git can't handle large data/models. DVC versions them with tiny `.dvc` / `dvc.lock` files and makes the pipeline reproducible.

**What is `dvc.yaml`?** The pipeline definition: stages with `cmd`, `deps`, `params`, `outs`, `metrics`.

**What is `params.yaml`?** All tunable settings in one file. A stage only depends on the section listed under its `params:`.

**How does DVC know what to rerun?** It stores MD5 hashes of every dep/param/out in `dvc.lock`. If a current hash differs from the lock, the stage is stale.

**Dataset changes?** Hash of the CSV changes, so `prepare` and everything after it re-run (`dvc status` shows it first).

**Parameter changes?** E.g. `train.n_estimators`: only `train` and `evaluate` re-run. `prepare`, `feast_apply`, `get_features` are skipped.

**Git vs DVC?** Git tracks code and small text files. DVC tracks data, models and outputs (content stored in the DVC cache/remote; Git stores only pointers).

**`dvc push` / `dvc pull`?** Push uploads cached files to the DVC remote. Pull downloads them into the workspace.

**Reproduce on another machine?** `git clone`, `pip install -r requirements.txt`, `dvc pull`, `dvc repro`.

**Other commands:** `dvc init` sets up DVC in a Git repo. `dvc add` starts tracking a file. `dvc status` shows what's changed. `dvc dag` draws the stage graph.

## Feast
- **Feature store:** one place that defines features and serves them for training and prediction.
- **Offline store:** historical data (parquet here), used to build training sets (`get_historical_features`).
- **Online store:** latest value per customer (SQLite here), used for fast lookups at prediction time (`get_online_features`).
- **`feast apply`:** registers definitions. **`materialize`:** copies offline data into the online store.
- **Why:** same feature definitions in training and serving, so no mismatch.
