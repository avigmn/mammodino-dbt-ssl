# Cross-slice Transformer (sandbox)

Isolated pipeline: **frozen MammoDINO CLS** → **slice sequences ordered by manifest** → **`TransformerEncoder` + volume token** → patient-level metrics compatible with `baseline_patient_pool_embeddings_dbt.py`.

**Does not modify** production scripts under `mammodino_ssl_project/scripts/`.

## Dependencies

- **Manifest + slice ordering:** `pyarrow` only (no pandas on this code path).
- **Slice loading:** sandbox dataset [`cross_slice/dataset_pyarrow.py`](cross_slice/dataset_pyarrow.py) mirrors supervised-slice IO without importing `dbt_ssl.data.supervised_slice_dataset` (which pulls pandas).
- **Backbone:** `mammodino_ssl.models.dino_ssl` (torch).
- **Metrics:** `sklearn`.
- **Plots:** `matplotlib` (optional; install if you want PNGs; otherwise use `--no-plots`).

## Slice ordering

Verified on `dbt_simclr_project/artifacts/manifests/master_manifest.parquet`:

- **`slice_index`** (`int64`) — default sort key within each patient  
- Alternatives (auto if `slice_index` missing): `instance_number`, `InstanceNumber`, `z_index`, `z_rank`  
- Fallback: basename digit parsing + lexicographic tie-break (see `cross_slice/manifest_io.py`)

Override column: `--slice-order-column <name>`.

## Naming runs (descriptive label + timestamp)

**עברית (קצר):** לכל ריצה תן שם שמתאר את הסוג (**smoke** / **full**), משאבים (**3gpu**), מגבלות (**extractMax10b**, **10ep**) והערכת סט (**evalVal** / **evalValTest**). חותמת זמן חייבת להופיע בשם התיקייה — אצל cross-slice וה-linear-probe היא נוספת אוטומטית; אצל baseline pooling תוסיף ידנית `${TS}` לנתיב `--artifacts-dir` (ראה דוגמאות ב-[`scripts/templates/`](scripts/templates/)).

Goal: every artifact directory encodes **what** you ran and **when**, so smoke vs full runs stay obvious later.

### Cross-slice sandbox (`run_cross_slice_transformer.py`)

Output folder (fails if it already exists — nothing is overwritten):

`runs/cross_slice_transformer_<run_name>_<UTC_YYYYMMDD_HHMMSS>/`

Set **`--run-name`** to a short slug (ASCII). Typical tokens:

- **`smoke`** vs **`full`**
- **`3gpu`** when `--cuda-devices 0,1,2` is used for embedding extraction
- limits: **`extractMax20b`**, **`tfMax5ep`**, **`patience15`**
- eval scope: **`evalVal`** vs **`evalValTest`** (when `--eval-splits` includes `test`)

| Intent | Example `--run-name` |
|--------|----------------------|
| Smoke, 3 GPUs, few embedding batches + few transformer epochs | `smoke_3gpu_extract10b_tf3ep` |
| Longer budget (~10 transformer epochs) | `full_3gpu_tf10ep_patience10` |

The **UTC timestamp is appended automatically** — do not bake manual clock times into `--run-name`.

### Patient-level pooling (`baseline_patient_pool_embeddings_dbt.py`)

Pass **`--artifacts-dir`** explicitly; combine a label and timestamp:

```bash
TS="$(date -u +%Y%m%d_%H%M%S)"
LABEL="smoke_patient_attn_3gpu_extract15b_attnEp5"
python scripts/baseline_patient_pool_embeddings_dbt.py \
  ... \
  --artifacts-dir experiments/dbt_cross_slice_transformer/runs/baseline_${LABEL}_${TS}
```

Use **`date -u`** for UTC (aligned with cross-slice); drop `-u` for local wall-clock.

### Slice linear probe (`train_linear_probe_dbt.py`, multi-GPU)

Use **`--run-name`** — the script **appends a local timestamp** itself:

`experiments/dbt_dino_runs/probe/[run_group/]<run_name>_<stamp>/`

```bash
torchrun --standalone --nproc_per_node=3 scripts/train_linear_probe_dbt.py \
  --num-gpus 3 \
  --run-group smoke_20260510 \
  --run-name probe_linear_3gpu_10ep_bs128 \
  ...
```

Worked examples you can copy: [`scripts/templates/`](scripts/templates/).

## Layout

```text
cross_slice/           # Python package (models, extraction, training, metrics)
scripts/run_cross_slice_transformer.py
scripts/templates/     # example shells: RUN_LABEL + timestamp patterns
runs/                  # outputs; each run is a new timestamped folder (never overwrites)
```

Per run directory:

- `config.json` — args + resolved paths + slice-order provenance  
- `metrics.json` — schema aligned with patient-attention runs (threshold tuning, confusion matrices); includes a `plots` section  
- `logs/training.log` — human-readable epoch lines  
- `logs/metrics_epoch.jsonl` — one JSON object per epoch (for reproducible learning curves). Use **`--verbose-epoch-metrics`** on short/smoke runs for richer train/val fields (F1, CE eval, confusion counts); full runs stay lightweight by default.  
- `plots/` — PNG figures (skipped if `--no-plots` or matplotlib missing):  
  - `training_curves.png` (loss + val AUROC + val balanced acc @0.5)  
  - `roc_patient_val.png`, `score_hist_patient_val.png` (+ test counterparts when `--eval-splits` includes test)  
  - `confusion_val_at_0.5.png`, `confusion_val_at_threshold.png`, and matching test confusion plots  
  - `index.json` — list of generated plot paths  
- `checkpoints/best.pt` — downstream Transformer weights only  
- `RESULTS_SUMMARY.md`

## Run (example)

From repo root (`mammodino_ssl_project/`):

```bash
python experiments/dbt_cross_slice_transformer/scripts/run_cross_slice_transformer.py \
  --checkpoint /path/to/best.pt \
  --eval-splits val,test \
  --cuda-devices 0,1,2 \
  --run-name full_3gpu_tf10ep_patience10_evalValTest \
  --runs-root experiments/dbt_cross_slice_transformer/runs
```

Use **`--help`** for all flags (`tf-layers`, `tf-dropout`, early stopping, etc.).

## Three-way comparison (same checkpoint / splits / protocol)

**Mean pooling + sklearn** (unchanged baseline script):

```bash
python scripts/baseline_patient_pool_embeddings_dbt.py \
  --checkpoint <CKPT> --mode patient_mean_embed --eval-splits val,test \
  --artifacts-dir experiments/dbt_cross_slice_transformer/runs/baseline_mean_<stamp>/
```

**Patient attention MIL**:

```bash
python scripts/baseline_patient_pool_embeddings_dbt.py \
  --checkpoint <CKPT> --mode patient_attention --eval-splits val,test \
  --artifacts-dir experiments/dbt_cross_slice_transformer/runs/baseline_attn_<stamp>/
```

**Cross-slice Transformer**: command above with matching `--checkpoint`, `--cuda-devices`, `--seed`, `--eval-splits`.

## Defaults (first experiment)

- Transformer: **2 layers**, dropout **0.15**, **8 heads** (requires `embed_dim % nhead == 0`)  
- Early stopping: **`auroc`**, patience **10** (switch with `--early-stop-metric balanced_accuracy`)
