# Comparison-model summary — all baselines through Liron's staged framework

Consolidated view of every comparison/ablation model we ran, all evaluated with
**Liron's staged head-evaluation** (Stage 0 frozen embeddings → Stage 1/2/3/7,
identical splits/metrics). These models exist to **contextualize and validate
Liron's DINO-full** — not to outperform it.

## Test AUROC by stage

| Model | Backbone | S1 slice | S2 pool | S3 MIL | S7 hierarchy |
|---|---|---|---|---|---|
| **Random-init** (lower bound) | ViT-tiny (12L) frozen | 0.525 | 0.669 | 0.576 | 0.721 |
| ImageNet | ViT-tiny (12L) frozen | 0.621 | 0.711 | 0.633 | 0.725 |
| DINO-only | TinyViT (4L) | 0.601 | 0.713 | 0.672 | 0.731 |
| DINO+iBOT | TinyViT (4L) | 0.602 | 0.687 | 0.656 | 0.723 |
| DINO-only | ViT-small (12L) | 0.632 | 0.720 | 0.690 | **0.746** |
| DINO+iBOT | ViT-small (12L) | 0.668 | 0.719 | 0.671 | 0.735 |
| **Liron DINO-full** (3-GPU) | TinyViT (4L) | — | 0.720 | 0.714 | **0.762** |
| **Liron Stage-10 e2e** (fine-tuned) | TinyViT (4L) | — | — | — | **0.759** (val 0.77) |
| *Itai supervised (reference)* | SwinV2-tiny | — | — | — | *0.9584* |

## What the comparison establishes

1. **SSL adds real value at the slice level.** Random-init is ≈chance (S1 0.525);
   ImageNet and DINO reach 0.60–0.67. So learned features matter where the head
   can't compensate.
2. **The anatomical hierarchy head does most of the patient-level work.** S7 reaches
   **0.721 from random frozen features** — the view/side attention + label-trained
   head is powerful on its own. SSL adds only ~0.01–0.04 on top at S7. (Important
   caveat for reading any Stage-7 number.)
3. **Domain SSL > generic ImageNet** at the structured heads (DINO-only tiny S7
   0.731 > ImageNet 0.725; gap widens with the hierarchy and at ViT-small).
4. **iBOT does not help** at the patient decision (S7) at either scale — validates
   Liron dropping it. It only becomes competitive at slice level at ViT-small.
5. **Scale helps** (DINO-only S7 0.731 → 0.746 tiny → ViT-small).
6. **Liron's model is best overall** — her well-trained 3-GPU DINO-full (0.762) and
   her Stage-10 end-to-end fine-tuning (0.759 / val 0.77) lead all SSL variants.
7. **Supervised remains far ahead** (0.9584) — the SSL↔supervised gap is the
   standing open problem, not addressed by these frozen-feature comparisons.

## Caveats
- Our DINO/iBOT runs are single-GPU (~32–38 ep); Liron's reference is 3-GPU/~69 ep,
  so her absolute lead partly reflects training budget, not just method.
- The controlled deltas (random vs ImageNet vs DINO; DINO vs iBOT; tiny vs ViT-small)
  are the trustworthy results — all share the same eval harness.

## Artifacts
- Per-model metrics + Stage-7 plots: `plots/{randinit_baseline,imagenet_staged,ibot_new_design,vitsmall_scale}/`
- Docs: `IMAGENET_VS_DINOFULL_STAGED.md`, `IBOT_VS_DINO_NEW_DESIGN.md`, `VITSMALL_SCALE_AND_IBOT.md`, `LIRON_DINO_FULL_REDESIGN.md`.
