# Liron's DINO-Full Redesign (June 2026)

> Reference note documenting the redesigned, "faithful DINO" reimplementation
> Liron created on the server. **This is the model we will deepen and fine-tune.**
> Source of truth: `/mnt/md0/Liron/mammodino_ssl_project_dino_full/` (server,
> not in this repo). Summarized here for our tracking and the final report.

## Location & status

- **New project (server):** `/mnt/md0/Liron/mammodino_ssl_project_dino_full/`
  (created 2026-06-08; the old `mammodino_ssl_project/` is left untouched).
- Also on server: `mammodino_dbt_ssl_partner_repo/` — a copy of *our* GitHub repo
  she pulled in for the model-comparison report.
- **Already trained and evaluated.** Primary checkpoint:
  `experiments/dino_full_runs/dino_full_v1_20260606_131923_388774/checkpoints/best.pt`
  (trained 2026-06-07; checkpoint stores both `student` and `teacher` weights + `center`).
- Her changelog: `docs/DINO_FULL_IMPLEMENTATION_CHANGES.md`; audit that motivated
  it: `docs/DINO_IMPLEMENTATION_AUDIT.md` (both in the new project on the server).

## Why she redesigned it

The original `mammodino_ssl_project` was a simplified DINO. An audit identified
several gaps vs the canonical Facebook DINO recipe. The `_dino_full` fork is an
additive, isolated reimplementation that closes those gaps while keeping the
MammoDINO medical adaptations. The legacy pipeline is preserved untouched for
comparison.

## What changed — old vs DINO-full

| Aspect | `mammodino_ssl_project` (old, what *we* ran) | `_dino_full` (new) |
| --- | --- | --- |
| Views | 1 weak teacher + 1 strong student (2 total) | **multi-crop**: 2 global + N local (96px) |
| Loss | single teacher/student pair CE (+ optional iBOT) | CE over all teacher-global × student pairs (canonical DINO) |
| Head | plain MLP, K=512, no L2-norm/weight-norm | MLP + L2-norm bottleneck + weight-normed prototypes, **K=4096** |
| Backbone | `TinyViT` (fixed) | configurable `VisionTransformer` (`tiny` default, **`vit_small` option**) + attention hook |
| Base image | 224 | 256 (crops taken from it; global crop fed at 224) |
| LR / WD | fixed | **cosine LR + 10-epoch warmup; cosine WD 0.04→0.4** |
| Teacher momentum | fixed 0.996 | cosine 0.996 → 1.0 |
| Teacher temp | fixed 0.04 | warmup 0.04 → target (30 ep) |
| Stability | none | **last-layer freeze (1 ep) + gradient clipping 3.0** |
| Diagnostics | none | teacher entropy / center norm / effective prototypes / KL, per epoch |
| Eval backbone | student only | configurable, **default teacher** (DINO-standard) |
| k-NN eval | none | yes (student + teacher) |
| Attention maps | not exposed | `get_last_selfattention` + saver |
| Eval suite | linear probe, attn pool | + k-NN, representation analysis (PCA/UMAP/t-SNE + silhouette/distance), layer-wise analysis, attribution analysis (patient/slice-depth/laterality) |
| Run tracking | best/last by val DINO loss | timestamped run dirs, full config+logs+metrics+diagnostics, no overwrite |
| iBOT | optional, wired in | **not** wired into faithful trainer (available via legacy pipeline; ViT exposes `forward_masked` for later) |
| Multi-GPU | DDP (legacy) | DDP via torchrun, rank-0-gated IO, identical checkpoint key format |

## Key config (`configs/dino_dbt_full.yaml`)

- `model.arch: tiny` (depth 4, embed 192, 3 heads) — or `vit_small` (ViT-S/16)
- `num_prototypes: 4096`, head 3-layer, hidden 2048, bottleneck 256, norm-last-layer
- `dataset`: base 256, global crops scale 0.4–1.0, local 0.05–0.4, `local_crops_number: 6`
- `train`: lr 5e-4 (peak), min_lr 1e-6, warmup 10 ep, wd 0.04→0.4, momentum_teacher 0.996,
  teacher_temp 0.04 (warmup 30 ep), clip_grad 3.0, freeze_last_layer 1,
  early stopping patience 5, effective batch = 16×3×4 = 192 (3-GPU)
- This matches the parameter table in the final report (§3.12).

## Her model-comparison results (test set, 2026-06-08)

Generated under `report_outputs/model_comparison/` in the new project.

| Model | Head | Test AUROC (patient) |
| --- | --- | --- |
| **DINO-full** | **Attention MIL** | **0.698** ← best |
| DINO-full | patient mean-prob (from probe) | 0.681 |
| DINO-full | patient mean-embed + logreg | 0.669 |
| DINO-full | linear probe (slice-level) | 0.636 |
| Legacy DINO-only (100ep, May) | Attention MIL | 0.643 |
| Legacy DINO+iBOT (10ep) | Cross-slice transformer | 0.662 |

- Best slice-level test AUROC: DINO-full / linear probe = 0.636.
- Best patient-level test AUROC: DINO-full / attention MIL = 0.698.
- k-NN (val only): ~0.54 (student/teacher) — weak, as expected for raw frozen features.

**Caveats she documented:** legacy DINO-only and DINO-full use different SSL
checkpoints; legacy probe early-stops on `val_acc` while DINO-full uses
`val_roc_auc`; ImageNet baseline uses ImageNet normalization (not feature-
comparable); k-NN(val) not comparable to probe(test).

## Relationship to our session's experiments

Our runs (`/mnt/data/avi/dino_only_runs`, `/mnt/data/avi/dino_ibot_runs`) used the
**old** design (K=512, no multi-crop). They are not part of her comparison table.
Notably our DINO-only + attention pool (0.698) lands at the same value as her new
DINO-full + attention MIL (0.698) — coincidental convergence across different
recipes.

## Implication for next steps

The redesigned **DINO-full** is the canonical model going forward. Any
fine-tuning / "going deep" should build on
`dino_full_v1_20260606_131923_388774/checkpoints/best.pt` (or a fresh DINO-full
run), **not** the old-design checkpoints we trained this session.

## How to run (from her changelog)

```bash
cd /mnt/md0/Liron/mammodino_ssl_project_dino_full
PY=/mnt/md0/Liron/mammodino_ssl_project/.venv/bin/python
export PYTHONPATH="$PWD/src:/mnt/md0/Liron/dbt_simclr_project/src"

# Full 3-GPU training (effective batch 192; lr not auto-scaled):
CUDA_VISIBLE_DEVICES=0,1,2 "$PY" -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/train_dino_full.py --config configs/dino_dbt_full.yaml --device cuda --num-gpus 3

# Eval bundle on best checkpoint (k-NN, probe, confusion/ROC/PR, attention, patient pool):
RUN=experiments/dino_full_runs/<run>
CUDA_VISIBLE_DEVICES=0 PY="$PY" scripts/run_dino_full_eval_bundle.sh \
  "$RUN" "$RUN/checkpoints/best.pt" configs/dino_dbt_full.yaml teacher
```
