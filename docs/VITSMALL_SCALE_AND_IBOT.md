# Scale experiment: ViT-small vs TinyViT, and does iBOT help at scale?

Follow-up to the TinyViT iBOT ablation. We trained the same controlled pair at a
**larger backbone (ViT-S/16, 12-layer/384-dim, ~3× TinyViT)** — DINO-only and
DINO+iBOT, identical settings except `--w-ibot` — and evaluated both through
Liron's staged framework. Two questions: (1) does a bigger backbone close the gap
toward supervised? (2) does iBOT finally help at scale?

## Setup
- Same harness as the tiny ablation (`experiments/dino_full_ibot/`), config
  `dino_dbt_full_vitsmall.yaml` (`arch: vit_small`). Single-GPU, identical seed/
  batch/epochs; only `w_ibot` differs (0.0 vs 0.1).
- Training (early-stopped): DINO-only best val 3.099 @ep33; DINO+iBOT best val 3.269 @ep32.
- Eval: teacher backbone → Stage 0 embeddings → Stages 1/2/3/7.

## Results — test AUROC (best per stage)

| Stage | Head | tiny DINO-only | tiny DINO+iBOT | **ViT-S DINO-only** | **ViT-S DINO+iBOT** |
|---|---|---|---|---|---|
| 1 | slice | 0.601 | 0.602 | 0.632 | **0.668** |
| 2 | patient pooling | 0.713 | 0.687 | **0.720** | 0.719 |
| 3 | MIL | 0.672 | 0.656 | **0.690** | 0.671 |
| 7 | hierarchical attention | 0.731 | 0.723 | **0.746** | 0.735 |

Reference points: Liron DINO-full (tiny, 3-GPU) Stage-7 test **0.762**; Itai supervised **0.9584**.

## Findings

1. **A bigger backbone helps.** DINO-only Stage-7 test rises **0.731 → 0.746**
   going tiny → ViT-small, and gains appear at every stage (slice 0.601→0.632,
   MIL 0.672→0.690). ViT-small DINO-only (0.746) approaches Liron's 3-GPU tiny
   DINO-full (0.762) on a single GPU — scale is a real lever.

2. **iBOT still does not beat DINO-only at the patient decision (Stage 7),**
   even at ViT-small (0.735 vs 0.746). So the headline answer is unchanged:
   plain DINO is the better choice for the final cancer score.

3. **But iBOT becomes competitive — even helpful — at scale.** At ViT-small it
   *wins* the slice-level head (Stage 1: 0.668 vs 0.632) and ties patient pooling
   (Stage 2). At tiny it lost or tied everywhere. This matches intuition: the
   patch-level objective needs encoder capacity to coexist with the CLS objective;
   the 4-layer tiny ViT couldn't afford it, the 12-layer ViT-small partly can.

**Takeaway:** for this DBT pipeline, **scale up the backbone, keep DINO-only**.
iBOT's benefit is confined to slice-level features and doesn't propagate to the
patient-level hierarchy that drives the final AUROC.

## Caveats
- Single-GPU (~32–38 epochs); absolute numbers a touch below the 3-GPU reference.
- The iBOT-vs-DINO and tiny-vs-small deltas are the controlled results; all four
  runs share the same harness/splits/eval.

## Artifacts
- Per-stage metrics + Stage-7 plots: `plots/vitsmall_scale/`
- Config: `experiments/dino_full_ibot/dino_dbt_full_vitsmall.yaml`
- Server: `/mnt/data/avi/dino_full_ibot_runs/` (checkpoints), `/mnt/data/avi/ibot_staged/{dino_only_vitsmall,dino_ibot_vitsmall}/` (eval).
