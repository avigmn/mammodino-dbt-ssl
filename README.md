# mammodino-ssl

Self-distillation experiments (DINO → iBOT → cross-view), separate from [`dbt_simclr_project`](../dbt_simclr_project).

## Comparison experiments & results (avi)

Controlled comparison/ablation models built to contextualize and validate Liron's
DINO-full model, all evaluated through her staged head-evaluation framework.

- **`docs/COMPARISON_SUMMARY.md`** — consolidated table across all baselines (start here).
- `docs/LIRON_DINO_FULL_REDESIGN.md` — her faithful DINO-full redesign + staged pipeline (0→10) + Stage-10 end-to-end (val 0.77).
- `docs/IMAGENET_VS_DINOFULL_STAGED.md` — ImageNet ViT-tiny vs DINO-full (domain SSL value).
- `docs/IBOT_VS_DINO_NEW_DESIGN.md` — controlled iBOT ablation (iBOT doesn't help at TinyViT scale).
- `docs/VITSMALL_SCALE_AND_IBOT.md` — backbone scale (ViT-small) + iBOT-at-scale.
- `experiments/` — our configs/scripts: `dino_only/`, `imagenet_baseline/`, `imagenet_staged/`, `dino_full_ibot/`.
- `plots/` — per-experiment metrics + figures.

Headline (test AUROC, Stage-7 hierarchy): random-init 0.721 · ImageNet 0.725 ·
DINO-only tiny 0.731 · DINO-only ViT-small 0.746 · **Liron DINO-full 0.762** ·
Liron Stage-10 e2e 0.759 (val 0.77) · Itai supervised (ref) 0.9584.

## Layout

- `src/mammodino_ssl/` — models, losses, trainers, toy datasets.
- `scripts/` — runnable entrypoints.
- Reuse: CIFAR helpers and future DBT manifest loaders live in `dbt_ssl` (sibling repo). Scripts add both `src` roots to `PYTHONPATH`.

## Install

From this directory:

```bash
pip install -e ../dbt_simclr_project[train]
pip install -e ".[train]"
```

## Toy (CIFAR-10, DINO phase-1)

Run the **script** directly (not via `pytest`):

```bash
python scripts/run_toy_dino_cifar10.py --epochs 30 --num-prototypes 512
```

If the progress bar is slow to appear, try `--num-workers 0` or `PYTHONUNBUFFERED=1`.  
If CUDA / driver errors occur: `--device cpu --no-amp`.

See `python scripts/run_toy_dino_cifar10.py --help`.
