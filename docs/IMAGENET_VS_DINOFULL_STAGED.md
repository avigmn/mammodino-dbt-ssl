# ImageNet vs DINO-full — apples-to-apples through Liron's staged framework

We re-ran a **frozen ImageNet ViT-tiny** backbone through Liron's *exact* staged
head-evaluation pipeline (Stage 0 embeddings → Stage 1/2/3/7 heads), using the
same splits, metrics, and code as her DINO-full evaluation. This isolates the
contribution of **domain-specific SSL** vs **generic ImageNet features** with the
heads held identical.

## How it was produced
- Backbone: `timm vit_tiny_patch16_224` (ImageNet weights, **frozen**), 192-d CLS, ImageNet mean/std normalization.
- Stage-0 embeddings produced by `experiments/imagenet_staged/extract_imagenet_stage0.py` — reuses Liron's `stage0_extract_embeddings` metadata/assembly so the parquet schema is byte-identical (12 meta cols + `emb_cls_final`).
- Stages run with Liron's own scripts: `stage1_slice_heads.py`, `stage2_patient_pooling.py`, `stage3_mil.py`, `stage7_hierarchical_view_side_attention_mil.py` (`--stage0-dir`/`--embeddings-dir` → our ImageNet embeddings; `--output-dir` → our writable area).
- Server outputs: `/mnt/data/avi/imagenet_staged/`. Metric JSONs copied to `plots/imagenet_staged/`.

## Results (best AUROC, same splits)

| Stage | Head | ImageNet val / test | DINO-full val / test | Δ test |
|---|---|---|---|---|
| 1 | slice heads (frozen) | 0.623 / 0.621 | 0.623 / 0.622 | ~0 |
| 2 | patient pooling | 0.771 / 0.711 | 0.739 / 0.720 | +0.009 DINO |
| 3 | MIL | 0.673 / 0.633 | 0.713 / 0.714 | **+0.081 DINO** |
| 7 | hierarchical view/side attention | 0.749 / 0.725 | 0.752 / **0.762** | **+0.037 DINO** |
| 10 | end-to-end fine-tune (DINO-only) | — (not run) | 0.770 / 0.759 | — |

DINO-full numbers are from Liron's `report_outputs/head_evaluation/` (her run).

## Reading

- **At the slice level the two are tied (~0.62)** — raw frozen features alone carry
  little malignancy signal, regardless of pretraining source. (Consistent with the
  representation analysis: the SSL space encodes patient/depth, not cancer label.)
- **As the head gets more structured, DINO-full pulls ahead** — biggest gap at MIL
  (Stage 3: 0.714 vs 0.633) and a clear margin at the hierarchy (Stage 7: 0.762 vs
  0.725). Domain-specific SSL features are more *aggregatable* into a patient decision.
- **ImageNet is a surprisingly strong baseline** through the Stage-7 hierarchy
  (test 0.725) — i.e. much of Liron's frozen-feature performance comes from the
  hierarchical/anatomical head, but DINO pretraining still adds ~0.04 test AUROC.
- **Stage 2 caveat:** ImageNet's val (0.771) is inflated vs its test (0.711) — the
  pooled logistic classifier overfits val; test is the honest number and is below DINO-full.
- **The real separation is end-to-end fine-tuning (Stage 10, DINO-only, 0.759 test)**,
  which no frozen ImageNet head reaches.

## Caveat
ImageNet ViT-tiny is 12-layer (deeper than DINO-full's 4-layer tiny ViT), so the
comparison if anything *favors* the ImageNet baseline on raw capacity — making
DINO-full's win at the structured stages more meaningful.
