# Example run wrappers

Copy any `.example.sh` to a new name (e.g. `my_run.sh`), **`chmod +x`**, edit `CKPT` and optional limits, then run.

They only **construct** descriptive names + (where needed) `TS` + paths — no edits to production Python scripts.

- **Cross-slice:** put human intent only in `--run-name`; the Python runner appends UTC `_YYYYMMDD_HHMMSS` (do not duplicate a bash timestamp inside `--run-name`).
- **Baseline pooling:** bash builds `baseline_<LABEL>_<TS>` for `--artifacts-dir`.
- **Linear probe:** `--run-name` is descriptive; `train_linear_probe_dbt.py` appends a **local** timestamp.
