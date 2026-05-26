# mammodino-ssl

Self-distillation experiments (DINO → iBOT → cross-view), separate from [`dbt_simclr_project`](../dbt_simclr_project).

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
