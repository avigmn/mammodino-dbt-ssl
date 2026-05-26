#!/usr/bin/env python3
"""Sandbox runner: frozen CLS -> ordered slices -> TransformerEncoder + volume token -> patient metrics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_SANDBOX_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _SANDBOX_ROOT.parents[1]
_DBT_SRC = _REPO_ROOT.parent / "dbt_simclr_project" / "src"

sys.path.insert(0, str(_SANDBOX_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_DBT_SRC) not in sys.path:
    sys.path.insert(0, str(_DBT_SRC))

from cross_slice.embeddings import (  # noqa: E402
    collate_supervised_with_sort,
    extract_cls_embeddings,
    group_slices_per_patient_ordered,
)
from cross_slice.dataset_pyarrow import DBTSupervisedSliceDatasetPyArrow, SandboxSliceConfig  # noqa: E402
from cross_slice.manifest_io import build_slice_sort_lookup  # noqa: E402
from cross_slice.metrics_utils import finalize_threshold_outputs  # noqa: E402
from cross_slice.plots import save_cross_slice_run_plots  # noqa: E402
from cross_slice.train_loop import eval_patient_probs, train_transformer  # noqa: E402
from mammodino_ssl.models.dino_ssl import create_dino_ssl  # noqa: E402


def _parse_cuda_devices(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _normalize_ddp_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        nk = nk.replace("student_backbone.module.", "student_backbone.")
        nk = nk.replace("student_head.module.", "student_head.")
        nk = nk.replace("student_patch_head.module.", "student_patch_head.")
        out[nk] = v
    return out


def _load_dino_model_section(config_path: Path) -> dict:
    if not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return dict(raw.get("model") or {})


def parse_eval_splits(s: str) -> list[str]:
    raw = [x.strip().lower() for x in (s or "").split(",") if x.strip()]
    allowed = {"val", "test"}
    for x in raw:
        if x not in allowed:
            raise ValueError(f"eval split must be one of {allowed}, got {x!r}")
    seen: set[str] = set()
    out: list[str] = []
    for x in raw:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out if out else ["val"]


def try_git_rev(root: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL, timeout=3
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def json_safe(o: object) -> object:
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [json_safe(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, np.generic):
        return o.item()
    return o


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-slice Transformer on frozen CLS (sandbox).")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dino-config", type=Path, default=_REPO_ROOT / "configs/dino_dbt.yaml")
    parser.add_argument("--data-repo-root", type=Path, default=_REPO_ROOT.parent / "dbt_simclr_project")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128, help="Slice DataLoader batch for extraction.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default="",
        help="Comma-separated GPU ids for DataParallel during embedding extraction only.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--eval-splits", type=str, default="val")
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--slice-order-column", type=str, default=None)

    parser.add_argument("--batch-patients", type=int, default=16)
    parser.add_argument("--epochs-max", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--tf-layers", type=int, default=2)
    parser.add_argument("--tf-dropout", type=float, default=0.15)
    parser.add_argument("--tf-nhead", type=int, default=8)
    parser.add_argument(
        "--early-stop-metric",
        choices=("auroc", "balanced_accuracy"),
        default="auroc",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)

    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--runs-root", type=Path, default=_SANDBOX_ROOT / "runs")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib PNG generation.")
    parser.add_argument(
        "--verbose-epoch-metrics",
        action="store_true",
        help="Per epoch: extra train/val diagnostics (AUROC, F1, CE eval, confusion counts). "
        "Heavier; intended for short/smoke runs.",
    )

    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    runs_root = args.runs_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / f"cross_slice_transformer_{args.run_name}_{ts}"
    run_dir.mkdir(parents=False, exist_ok=False)

    eval_splits = parse_eval_splits(args.eval_splits)
    run_test = "test" in eval_splits

    cuda_dev_ids = _parse_cuda_devices(args.cuda_devices)
    if cuda_dev_ids:
        if not torch.cuda.is_available():
            raise ValueError("--cuda-devices requires CUDA")
        n_cuda = torch.cuda.device_count()
        for gid in cuda_dev_ids:
            if gid < 0 or gid >= n_cuda:
                raise ValueError(f"Invalid --cuda-devices index {gid} (available: 0..{n_cuda - 1})")
        device = torch.device(f"cuda:{cuda_dev_ids[0]}")
    else:
        dn = args.device
        if dn.startswith("cuda") and not torch.cuda.is_available():
            dn = "cpu"
        device = torch.device(dn if ":" in dn or dn == "cpu" else "cuda:0")

    amp = (not args.no_amp) and device.type == "cuda"
    show_prog = not bool(args.no_progress)

    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else (_REPO_ROOT / args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    dino_cfg_path = args.dino_config if args.dino_config.is_absolute() else (_REPO_ROOT / args.dino_config).resolve()
    model_yaml = _load_dino_model_section(dino_cfg_path)
    image_size = int(model_yaml.get("image_size", 224))
    num_prototypes = int(model_yaml.get("num_prototypes", 512))
    embed_dim = int(model_yaml.get("embed_dim", 192))

    data_root = args.data_repo_root if args.data_repo_root.is_absolute() else (_REPO_ROOT / args.data_repo_root).resolve()
    data_cfg = yaml.safe_load((data_root / "configs" / "data.yaml").read_text(encoding="utf-8"))
    artifacts = data_root / data_cfg.get("artifacts_dir", "artifacts")
    manifest_path = artifacts / data_cfg.get("manifest_rel_path", "manifests/master_manifest.parquet")
    split_path = artifacts / data_cfg.get("split_rel_path", "splits/patient_split_v1.json")

    sort_lookup, sort_prov = build_slice_sort_lookup(
        manifest_path, order_column_override=args.slice_order_column
    )

    cfg_dump = {
        "argv": sys.argv,
        "checkpoint": str(ckpt_path),
        "dino_config": str(dino_cfg_path),
        "data_repo_root": str(data_root),
        "manifest_path": str(manifest_path),
        "split_path": str(split_path),
        "slice_ordering": sort_prov,
        "git_rev": try_git_rev(_REPO_ROOT),
        "run_dir": str(run_dir),
        "verbose_epoch_metrics": bool(args.verbose_epoch_metrics),
    }
    (run_dir / "config.json").write_text(json.dumps(json_safe(cfg_dump), indent=2), encoding="utf-8")

    use_cache = bool(data_cfg.get("use_processed_cache", False))
    cache_dir = Path(data_cfg.get("processed_cache_rel_path", "")) if use_cache else None
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = data_root / cache_dir

    ds_cfg = SandboxSliceConfig(
        resize_height=image_size,
        resize_width=image_size,
        normalize=bool(data_cfg.get("normalize", True)),
        split_seed=int(args.seed),
        use_processed_cache=use_cache,
        processed_cache_dir=cache_dir,
        processed_cache_token=str(data_cfg.get("processed_cache_token", "")),
    )

    train_ds = DBTSupervisedSliceDatasetPyArrow(
        manifest_path=manifest_path, split_path=split_path, split="train", config=ds_cfg
    )
    val_ds = DBTSupervisedSliceDatasetPyArrow(
        manifest_path=manifest_path, split_path=split_path, split="val", config=ds_cfg
    )
    test_ds = (
        DBTSupervisedSliceDatasetPyArrow(
            manifest_path=manifest_path, split_path=split_path, split="test", config=ds_cfg
        )
        if run_test
        else None
    )

    g = torch.Generator()
    g.manual_seed(int(args.seed))

    def make_loader(ds: DBTSupervisedSliceDatasetPyArrow, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=shuffle,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            collate_fn=lambda b: collate_supervised_with_sort(b, sort_lookup),
            generator=g if shuffle else None,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False) if test_ds is not None else None

    ssl = create_dino_ssl(
        image_size=image_size,
        num_prototypes=num_prototypes,
        embed_dim=embed_dim,
        depth=int(model_yaml.get("depth", 4)),
        num_heads=int(model_yaml.get("num_heads", 3)),
        head_hidden_dim=int(model_yaml.get("head_hidden_dim", 512)),
        head_bottleneck_dim=int(model_yaml.get("head_bottleneck_dim", 256)),
    ).to(device)
    if show_prog:
        tqdm.write(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        ssl.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError:
        ssl.load_state_dict(_normalize_ddp_state_dict_keys(ckpt["model_state"]), strict=True)

    backbone: nn.Module = ssl.student_backbone
    use_dp = bool(cuda_dev_ids) and len(cuda_dev_ids) > 1
    if use_dp:
        backbone = nn.DataParallel(ssl.student_backbone, device_ids=cuda_dev_ids, output_device=cuda_dev_ids[0])

    max_train_batches = int(args.max_train_batches) if int(args.max_train_batches) > 0 else None
    max_val_batches = int(args.max_val_batches) if int(args.max_val_batches) > 0 else None
    max_test_batches = int(args.max_test_batches) if int(args.max_test_batches) > 0 else None

    Xtr, ytr, pid_tr, sk_tr = extract_cls_embeddings(
        backbone,
        train_loader,
        device,
        amp=amp,
        max_batches=max_train_batches,
        desc="CLS embeddings (train)",
        progress=show_prog,
    )
    Xva, yva, pid_va, sk_va = extract_cls_embeddings(
        backbone,
        val_loader,
        device,
        amp=amp,
        max_batches=max_val_batches,
        desc="CLS embeddings (val)",
        progress=show_prog,
    )
    Xte = yte = pid_te = sk_te = None
    if run_test and test_loader is not None:
        Xte_t, yte_t, pid_te_t, sk_te_t = extract_cls_embeddings(
            backbone,
            test_loader,
            device,
            amp=amp,
            max_batches=max_test_batches,
            desc="CLS embeddings (test)",
            progress=show_prog,
        )
        Xte, yte, pid_te, sk_te = Xte_t, yte_t, pid_te_t, sk_te_t

    if args.shuffle_labels and len(ytr):
        rng = np.random.default_rng(int(args.seed))
        ytr = rng.permutation(ytr)

    _, tr_mats, tr_y, tr_diag = group_slices_per_patient_ordered(Xtr, ytr, pid_tr, sk_tr)
    _, va_mats, va_y, va_diag = group_slices_per_patient_ordered(Xva, yva, pid_va, sk_va)
    te_mats = te_y_np = None
    te_diag: dict | None = None
    if Xte is not None and yte is not None and pid_te is not None and sk_te is not None:
        _, te_mats, te_y_np, te_diag = group_slices_per_patient_ordered(Xte, yte, pid_te, sk_te)

    cw = None
    if args.class_weight and len(tr_y):
        n0 = int((tr_y == 0).sum())
        n1 = int((tr_y == 1).sum())
        w0 = len(tr_y) / (2 * max(1, n0))
        w1 = len(tr_y) / (2 * max(1, n1))
        cw = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    dim_use = int(tr_mats[0].shape[1]) if tr_mats else embed_dim
    model = train_transformer(
        train_mats=tr_mats,
        train_y=tr_y,
        val_mats=va_mats,
        val_y=va_y,
        device=device,
        dim=dim_use,
        epochs_max=int(args.epochs_max),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_patients=int(args.batch_patients),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        class_weights=cw,
        tf_dropout=float(args.tf_dropout),
        tf_nhead=int(args.tf_nhead),
        tf_layers=int(args.tf_layers),
        early_stop_metric=str(args.early_stop_metric),
        early_stop_patience=int(args.early_stop_patience),
        progress=show_prog,
        ckpt_path=run_dir / "checkpoints" / "best.pt",
        log_file=run_dir / "logs" / "training.log",
        metrics_jsonl_path=run_dir / "logs" / "metrics_epoch.jsonl",
        verbose_epoch_metrics=bool(args.verbose_epoch_metrics),
    )

    p_val = eval_patient_probs(
        model,
        va_mats,
        device,
        int(args.batch_patients),
        int(args.num_workers),
        progress=show_prog,
        desc="val infer (final)",
    )
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    pred_val_05 = (p_val >= 0.5).astype(np.int64) if len(p_val) else np.zeros((0,), dtype=np.int64)
    pat_auroc = float(roc_auc_score(va_y, p_val)) if len(np.unique(va_y)) >= 2 else float("nan")
    pat_bal = float(balanced_accuracy_score(va_y, pred_val_05)) if len(va_y) else float("nan")

    ex_devs = (
        list(cuda_dev_ids)
        if cuda_dev_ids
        else ([int(device.index)] if device.type == "cuda" and device.index is not None else [])
    )

    out: dict = {
        "checkpoint": str(ckpt_path),
        "mode": "cross_slice_transformer",
        "shuffle_labels": bool(args.shuffle_labels),
        "eval_splits": eval_splits,
        "embedding_extraction": {"data_parallel": use_dp, "cuda_devices": ex_devs},
        "slice_ordering": sort_prov,
        "ordering_diagnostics_train": tr_diag,
        "ordering_diagnostics_val": va_diag,
        "ordering_diagnostics_test": te_diag,
        "n_train_slices": int(len(ytr)),
        "embed_dim": dim_use,
        "slice_level": None,
        "patient_level": {
            "primary_eval": True,
            "embedding_pool": "cross_slice_transformer_volume_token",
            "tf_layers": int(args.tf_layers),
            "tf_dropout": float(args.tf_dropout),
            "tf_nhead": int(args.tf_nhead),
            "epochs_max": int(args.epochs_max),
            "early_stop_metric": str(args.early_stop_metric),
            "early_stop_patience": int(args.early_stop_patience),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_patients": int(args.batch_patients),
            "class_weight": bool(args.class_weight),
            "auroc": pat_auroc,
            "balanced_accuracy@0.5": pat_bal,
            "n_train_patients": int(len(tr_y)),
            "n_val_patients": int(len(va_y)),
            "n_val_slices": int(len(yva)),
        },
    }

    ytp = ptp = None
    n_te_sl = 0
    if run_test and te_mats is not None and te_y_np is not None:
        p_tp = eval_patient_probs(
            model,
            te_mats,
            device,
            int(args.batch_patients),
            int(args.num_workers),
            progress=show_prog,
            desc="test infer",
        )
        ytp, ptp = te_y_np, p_tp
        n_te_sl = int(len(yte)) if yte is not None else 0

    finalize_threshold_outputs(
        out,
        patient_block=out["patient_level"],
        score_definition="cross_slice_transformer_volume_token",
        y_val_pat=va_y,
        p_val_pat=p_val,
        y_test_pat=ytp,
        p_test_pat=ptp,
        run_test=run_test,
        n_test_slices=n_te_sl,
    )

    warn_fn = tqdm.write
    if args.no_plots:
        out["plots"] = {"skipped": True, "reason": "--no-plots"}
    else:
        out["plots"] = save_cross_slice_run_plots(
            run_dir=run_dir,
            metrics_dict=out,
            run_name=str(args.run_name),
            val_y=va_y,
            val_scores=p_val,
            test_y=ytp,
            test_scores=ptp,
            ran_test=bool(run_test),
            metrics_epoch_jsonl=run_dir / "logs" / "metrics_epoch.jsonl",
            warn_print=warn_fn,
        )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(json_safe(out), indent=2), encoding="utf-8")

    pte_block = out.get("patient_level_test")
    summary_lines = [
        f"# cross_slice_transformer `{args.run_name}`",
        "",
        f"- run_dir: `{run_dir}`",
        f"- val AUROC: {out['patient_level'].get('auroc')}",
        f"- val balanced_accuracy@0.5: {out['patient_level'].get('balanced_accuracy@0.5')}",
        f"- threshold (val balanced_acc): {out.get('threshold_tuning', {}).get('threshold_value')}",
    ]
    if run_test and pte_block:
        summary_lines += [
            f"- test AUROC: {pte_block.get('auroc')}",
            f"- test balanced_accuracy@0.5: {pte_block.get('balanced_accuracy@0.5')}",
            f"- test balanced_accuracy@threshold_from_val: {pte_block.get('balanced_accuracy@threshold_from_val')}",
        ]
    if isinstance(out.get("plots"), dict) and out["plots"].get("files"):
        summary_lines += ["", "- plots: see `plots/` (training curves, ROC, confusion matrices, `plots/index.json`)."]
    (run_dir / "RESULTS_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(metrics_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
