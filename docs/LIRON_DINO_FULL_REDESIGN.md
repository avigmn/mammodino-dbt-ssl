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

---

# Staged head-evaluation pipeline + end-to-end fine-tuning (June 13–14, 2026)

> Liron added a large **staged downstream pipeline** (stages 0→10) on top of the
> DINO-full backbone, culminating in **end-to-end fine-tuning** that broke the
> ~0.75 frozen-feature ceiling. **Best validation AUROC ≈ 0.77.**
> Source (server): `report_outputs/head_evaluation/` + `scripts/head_evaluation/`.

## The headline result

| | |
|---|---|
| Stage | `stage_10_imbalance_aware_pathology_dino_mil` |
| Model | `stage10_e2e_focal_slice_mil_hard` |
| **Val AUROC** | **0.7702** ← "the 0.77" |
| Test AUROC (same run) | 0.750 |
| Test AUROC (sibling config) | 0.759 (val 0.764) |
| Best **test** AUROC across all stages | **0.7615** (Stage 7 hierarchical attention-MIL) |
| Date | 2026-06-13 |
| Init checkpoint | `experiments/dino_full_runs/dino_full_v1_20260606_131923_388774/checkpoints/best.pt` |
| Selected ckpt | `report_outputs/head_evaluation/stage_10_.../checkpoints/stage10_e2e_focal_slice_mil_hard__auroc.pt` |

⚠️ 0.7702 is **validation**; on held-out **test** the same model is ~0.75. Test was excluded from mining/thresholds; cross-split leakage check passed (0 overlaps).

## What changed — end-to-end fine-tuning

Stages 1–9 kept the backbone **frozen** and only trained heads (ceiling ~0.75).
**Stage 10 unfreezes and fine-tunes the DINO backbone end-to-end** — the first time
the encoder weights are updated by the downstream task. This is the "fine-tune the
backbone" direction; it is what pushed performance past the frozen-feature plateau.

Recipe (Stage 10, `stage10_imbalance_aware_pathology_dino_mil.py`):
- DINO-full backbone initialized from `dino_full_v1` best.pt, fine-tuned **end-to-end** (very low lr ≈ 1e-6)
- **Focal BCE** at patient level + **hard-example weighting** (hard pos/neg mined from train+val only)
- **Top-k slice MIL** (not pseudo-labeling)
- **Hierarchical view/side attention** head (carried from Stage 7)
- Optional slice **continuity** term (≈0.01)
- ~80 epochs max, early stopping patience 12; params: backbone 1.96M + hierarchy 0.65M = 2.6M

## Stage-by-stage pipeline (what each stage does + best AUROC)

All stages consume the **frozen Stage-0 DINO-full embeddings** unless marked "e2e
fine-tuned" (Stages 5/5b/10, which unfreeze the backbone). Threshold tuned on val,
applied to test. Scripts: `scripts/head_evaluation/stage{N}_*.py`; outputs:
`report_outputs/head_evaluation/stage_{N}_*/`.

| Stage | What it does | Backbone | Val | Test |
|---|---|---|---|---|
| 0 | Extract frozen teacher CLS embeddings (192-d) + metadata per split — single source of truth for all later stages | frozen | — | — |
| 1 | Slice-level heads (logreg, SVM, RF, MLP, XGB) on slice embeddings; weak (inherited) slice labels — baseline | frozen | 0.623 | 0.622 |
| 2 | Patient aggregation: prob pooling (mean/max/median/top-k/percentile/noisy-or) + embedding pooling (mean/max/concat → classifier) | frozen | 0.739 | 0.720 |
| 3 | MIL heads (attention, gated-attention, top-k, GRU, transformer) over slice-embedding bags | frozen | 0.713 | 0.714 |
| 3b | Imbalance-aware retraining of patient heads (class-weight, oversample, cost-sensitive, focal) | frozen | — | ~0.72 |
| 4 | Interpretability for selected heads (attention/instance scores) — no new training | frozen | — | — |
| 5 | End-to-end MIL fine-tuning: DINO ViT-Tiny + gated-attention MIL, conservative backbone unfreeze | **e2e** | 0.716 | 0.721 |
| 5b | End-to-end MIL with guided sampling | **e2e** | 0.716 | 0.721 |
| 6 | Teacher-guided pseudo-slice supervision | frozen | — | ~0.71 |
| 7 | **Hierarchical side/view attention MIL** (slice→view→side→patient) | frozen | 0.752 | **0.7615** ← best test |
| 8 | Patient-level supervised contrastive (SupCon) representation learning | frozen | 0.755 | 0.755 |
| 9 | SelectiveKD-inspired pathology-aware slice adapter (residual adapter + KD) | frozen | 0.748 | 0.746 |
| **10** | **Imbalance-aware pathology DINO-MIL, end-to-end**: focal BCE + hard-example mining + top-k slice-MIL + Stage-7 hierarchy | **e2e** | **0.7702** | **0.7586** |
| 10c | Stage-10 variant with view-OR aggregation | **e2e** | 0.756 | 0.754 |

**Reading the arc:** frozen heads plateau ~0.72–0.75 (Stages 1–9); the two levers
that helped most were (a) the **Stage-7 hierarchy** (slice→view→side→patient
attention, best *frozen* test 0.7615) and (b) **end-to-end fine-tuning** in Stage
10 (best val 0.7702). Stage 10 combines both + imbalance-aware training.

## Artifacts (server)

- Scripts: `scripts/head_evaluation/stage{0..10}_*.py` (+ `stage_*_commands.sh`)
- Per-stage outputs: `report_outputs/head_evaluation/stage_*/` — `stage_N_metrics.{json,csv,md}`, predictions, training curves, figures, error-space diagnostics, Hebrew summaries
- Stage 10 checkpoints: `.../stage_10_.../checkpoints/stage10_e2e_focal_slice_mil_hard__{auroc,ap,balacc,sens80_spec,final}.pt`

## Implication

The project headline shifts from **frozen-backbone DINO ≈ 0.70** to
**end-to-end fine-tuned DINO-MIL ≈ 0.77 (val) / ~0.76 (test)**. This is the model
to deepen/report. Note the val–test gap (~0.77 → ~0.75) — report test numbers as
the honest generalization estimate.
