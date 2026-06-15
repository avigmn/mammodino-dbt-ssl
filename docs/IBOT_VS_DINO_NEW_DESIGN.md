# Does iBOT help in the new (faithful DINO-full) design? — controlled ablation

Liron's redesigned DINO-full is **DINO-only** (iBOT was dropped from the faithful
trainer). We implemented iBOT *into* that design and ran a **controlled ablation**:
two runs from the same script, identical in every way except `--w-ibot`.

## Setup
- Implementation (additive fork, her code untouched): `experiments/dino_full_ibot/`
  — `dino_full_ibot_module.py` (CLS head + patch head), `ibot_patch_loss.py`
  (masked-patch CE + tissue-aware mask sampling), `train_dino_full_ibot.py`.
- Both runs: faithful DINO-full recipe (multi-crop 2 global + 6 local, cosine
  LR/WD/momentum, teacher-temp warmup, last-layer freeze, grad clip), arch=tiny,
  **single GPU**, identical seed/batch/epochs. Only difference: `w_ibot` = 0.0 vs 0.1.
- iBOT settings: patch head K=4096, mask ratio 0.3, tissue-biased masking 0.6,
  patch teacher-temp warmup (same schedule as CLS).
- Training (early-stopped): DINO-only best val 3.208 @ep28 (33 ep); DINO+iBOT best
  val 3.391 @ep26 (31 ep). (val losses not comparable — iBOT adds a loss term.)
- Evaluation: both checkpoints (teacher backbone) → Liron's staged framework
  (Stage 0 embeddings → Stage 1/2/3/7), same splits/metrics/code.

## Results (best AUROC per stage, same splits)

| Stage | Head | DINO-only val/test | DINO+iBOT val/test | Winner (test) |
|---|---|---|---|---|
| 1 | slice heads | 0.676 / 0.601 | 0.645 / 0.602 | tie |
| 2 | patient pooling | 0.751 / **0.713** | 0.757 / 0.687 | DINO-only |
| 3 | MIL | 0.725 / **0.672** | 0.715 / 0.656 | DINO-only |
| 7 | hierarchical attention | 0.774 / **0.731** | 0.755 / 0.723 | DINO-only |

## Conclusion

**iBOT does not help in the new faithful design.** Under identical conditions,
DINO-only matches or beats DINO+iBOT at every stage on the test set — the gap is
clearest at the structured heads (Stage 2/3/7). This is a clean controlled
confirmation of the earlier old-design observation (DINO-only ≈/> DINO+iBOT), and
it independently validates Liron's decision to drop iBOT from the faithful trainer.

Likely reason (consistent with the representation analysis): at TinyViT scale the
patch-level masked objective competes with the CLS objective rather than
complementing it; the encoder lacks capacity to serve both well.

## Caveats
- Single-GPU runs (~31–33 epochs) — absolute numbers sit a little below Liron's
  3-GPU DINO-full (e.g. our DINO-only Stage-7 test 0.731 vs her 0.762). The
  **iBOT-vs-DINO delta** is the controlled result; absolute level is not the point.
- Conclusion is specific to the TinyViT (4-layer) backbone. iBOT could still help
  at larger scale (ViT-small) — an open follow-up.

## Artifacts
- Per-stage metrics + Stage-7 variant plots: `plots/ibot_new_design/`
- Server: `/mnt/data/avi/dino_full_ibot_runs/` (checkpoints) and
  `/mnt/data/avi/ibot_staged/{dino_only_new,dino_ibot_new}/` (staged eval).
