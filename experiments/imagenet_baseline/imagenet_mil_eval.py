#!/usr/bin/env python3
"""ImageNet ViT-tiny (frozen) + MIL attention pool on DBT.

Baseline: generic ImageNet features, no domain-specific SSL.
Compare to DINO-only and DINO+iBOT to show value of DBT pre-training.

Run with Liron's python3.12 (has timm via --user):
    /mnt/md0/Liron/tools/python312/python/bin/python3.12 imagenet_mil_eval.py \
        --data-repo-root /mnt/md0/Liron/dbt_simclr_project \
        --artifacts-dir /mnt/data/avi/imagenet_mil_runs \
        --save-json /mnt/data/avi/imagenet_mil_runs/metrics.json \
        --eval-splits val,test
"""

from __future__ import annotations

import sys
from pathlib import Path

# Inject venv site-packages so we get torch, numpy, sklearn, etc.
_VENV_SP = "/mnt/md0/Liron/mammodino_ssl_project/.venv_py312_clean/lib/python3.12/site-packages"
if _VENV_SP not in sys.path:
    sys.path.insert(0, _VENV_SP)

# Inject project src dirs
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DBT_SRC = Path("/mnt/md0/Liron/dbt_simclr_project/src")
for _p in (_REPO_ROOT / "src", _DBT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import json
from collections import defaultdict

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dbt_ssl.data.supervised_slice_dataset import DBTSupervisedSliceDataset, SupervisedSliceConfig

# ImageNet normalization constants
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------

class ImageNetViTBackbone(nn.Module):
    """Frozen timm ViT-tiny pretrained on ImageNet. Returns (cls_emb, None)."""

    def __init__(self, model_name: str = "vit_tiny_patch16_224") -> None:
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim: int = self.model.embed_dim  # 192 for vit_tiny

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # x: [B, 3, H, W] — already ImageNet-normalized
        cls = self.model(x)  # [B, embed_dim]
        return cls, None


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _collate_imagenet(batch: list[dict]) -> dict:
    """Convert raw uint8 images to ImageNet-normalized float tensors."""
    imgs = []
    for item in batch:
        arr = item["image"]  # shape (1, H, W), uint8 or float
        t = torch.from_numpy(arr.astype(np.float32))
        if t.max() > 1.0:
            t = t / 255.0
        t = t.repeat(3, 1, 1)  # grayscale -> 3-channel
        t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
        imgs.append(t)
    x = torch.stack(imgs, dim=0)
    y = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    meta = [{"canonical_patient_id": item.get("canonical_patient_id")} for item in batch]
    return {"image": x, "label": y, "meta": meta}


@torch.no_grad()
def _extract_embeddings(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    backbone.eval()
    xs, ys, pids = [], [], []
    for batch in tqdm(loader, desc=desc, dynamic_ncols=True):
        x = batch["image"].to(device)
        y = batch["label"]
        cls, _ = backbone(x)
        xs.append(cls.float().cpu().numpy())
        ys.append(y.numpy())
        for m in batch["meta"]:
            pids.append(str(m["canonical_patient_id"]))
    X = np.concatenate(xs, axis=0) if xs else np.zeros((0, 0), dtype=np.float32)
    Y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64)
    return X, Y, pids


def _group_by_patient(
    X: np.ndarray, y: np.ndarray, pids: list[str]
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    by_pid: dict[str, dict] = defaultdict(lambda: {"rows": [], "label": None})
    for i in range(len(y)):
        pid = pids[i]
        by_pid[pid]["rows"].append(X[i])
        lab = int(y[i])
        if by_pid[pid]["label"] is None:
            by_pid[pid]["label"] = lab
    p_sorted = sorted(by_pid.keys())
    mats, labels = [], np.empty(len(p_sorted), dtype=np.int64)
    for k, pid in enumerate(p_sorted):
        mats.append(np.stack(by_pid[pid]["rows"], axis=0))
        labels[k] = int(by_pid[pid]["label"])
    return p_sorted, mats, labels


# ---------------------------------------------------------------------------
# MIL attention pool (identical to Liron's baseline_patient_pool_embeddings_dbt.py)
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        return (w * x).sum(dim=1)


class AttentionPatientClassifier(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pool = AttentionPool(dim)
        self.fc = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x, mask))


class PatientEmbeddingDataset(Dataset):
    def __init__(self, mats: list[np.ndarray], labels: np.ndarray) -> None:
        self.mats = mats
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.mats)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return self.mats[idx], int(self.labels[idx])


def _collate_patients(batch):
    embs = [torch.from_numpy(b[0]).float() for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    t_max = max(e.shape[0] for e in embs)
    d = embs[0].shape[1]
    padded = torch.zeros(len(batch), t_max, d)
    mask = torch.zeros(len(batch), t_max, dtype=torch.bool)
    for i, e in enumerate(embs):
        padded[i, :e.shape[0]] = e
        mask[i, :e.shape[0]] = True
    return padded, mask, labels


def _train_mil(
    mats: list[np.ndarray],
    labels: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[AttentionPatientClassifier, list[float]]:
    torch.manual_seed(seed)
    dim = mats[0].shape[1]
    model = AttentionPatientClassifier(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    ds = PatientEmbeddingDataset(mats, labels)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate_patients, generator=g)
    history: list[float] = []
    model.train()
    for _ in tqdm(range(epochs), desc="MIL training", dynamic_ncols=True):
        running, n = 0.0, 0
        for padded, mask, y in loader:
            padded, mask, y = padded.to(device), mask.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(padded, mask), y)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            n += 1
        history.append(running / max(1, n))
    return model, history


@torch.no_grad()
def _eval_mil(
    model: AttentionPatientClassifier,
    mats: list[np.ndarray],
    device: torch.device,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    model.eval()
    ds = PatientEmbeddingDataset(mats, np.zeros(len(mats), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: _collate_patients([(x, 0) for x, _ in b]))
    probs = []
    for padded, mask, _ in tqdm(loader, desc=desc, dynamic_ncols=True):
        padded, mask = padded.to(device), mask.to(device)
        p = torch.softmax(model(padded, mask).float(), dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    auroc = float(roc_auc_score(y, probs)) if len(np.unique(y)) >= 2 else float("nan")
    bal = float(balanced_accuracy_score(y, (probs >= threshold).astype(int)))
    return {"auroc": auroc, "balanced_accuracy": bal, "n_patients": int(len(y))}


def _best_threshold(y: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import balanced_accuracy_score
    candidates = set([0.0, 1.0])
    uniq = np.unique(probs)
    candidates.update(uniq.tolist())
    if len(uniq) > 1:
        candidates.update(((uniq[:-1] + uniq[1:]) / 2).tolist())
    best_t, best_b = 0.5, -1.0
    for t in sorted(candidates):
        b = balanced_accuracy_score(y, (probs >= t).astype(int))
        if b > best_b:
            best_b, best_t = b, float(t)
    return best_t, float(best_b)


def _confusion_matrix(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y.astype(int), pred.astype(int), labels=[0, 1])
    return {"labels_row_true_col_pred": [0, 1], "matrix": cm.tolist()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ImageNet ViT-tiny (frozen) + MIL attention pool baseline for DBT."
    )
    parser.add_argument("--data-repo-root", type=Path,
                        default=Path("/mnt/md0/Liron/dbt_simclr_project"))
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--eval-splits", type=str, default="val,test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--attn-epochs", type=int, default=30)
    parser.add_argument("--attn-lr", type=float, default=3e-4)
    parser.add_argument("--attn-batch-patients", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-name", type=str, default="vit_tiny_patch16_224")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_test = "test" in args.eval_splits.split(",")

    # --- data ---
    data_root = args.data_repo_root
    data_cfg = yaml.safe_load((data_root / "configs/data.yaml").read_text())
    artifacts = data_root / data_cfg.get("artifacts_dir", "artifacts")
    manifest_path = artifacts / data_cfg.get("manifest_rel_path", "manifests/master_manifest.parquet")
    split_path = artifacts / data_cfg.get("split_rel_path", "splits/patient_split_v1.json")

    # normalize=False: we get raw values and apply ImageNet normalization ourselves
    ds_cfg = SupervisedSliceConfig(
        resize_height=224, resize_width=224,
        normalize=False,
        split_seed=args.seed,
        use_processed_cache=False,
    )

    def _make_loader(split: str, shuffle: bool) -> DataLoader:
        ds = DBTSupervisedSliceDataset(
            manifest_path=manifest_path, split_path=split_path,
            split=split, config=ds_cfg,
        )
        g = torch.Generator()
        g.manual_seed(args.seed)
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            collate_fn=_collate_imagenet, generator=g,
        )

    train_loader = _make_loader("train", shuffle=True)
    val_loader = _make_loader("val", shuffle=False)
    test_loader = _make_loader("test", shuffle=False) if run_test else None

    # --- backbone ---
    tqdm.write(f"Loading {args.model_name} with ImageNet weights ...")
    backbone = ImageNetViTBackbone(args.model_name).to(device)
    tqdm.write(f"Backbone embed_dim: {backbone.embed_dim}")

    # --- extract embeddings ---
    Xtr, ytr, pid_tr = _extract_embeddings(backbone, train_loader, device, "Embeddings (train)")
    Xva, yva, pid_va = _extract_embeddings(backbone, val_loader, device, "Embeddings (val)")
    Xte, yte, pid_te = None, None, None
    if run_test and test_loader is not None:
        Xte, yte, pid_te = _extract_embeddings(backbone, test_loader, device, "Embeddings (test)")

    tqdm.write(f"Train: {Xtr.shape}, Val: {Xva.shape}" +
               (f", Test: {Xte.shape}" if Xte is not None else ""))

    # --- group by patient ---
    _, tr_mats, tr_y = _group_by_patient(Xtr, ytr, pid_tr)
    _, va_mats, va_y = _group_by_patient(Xva, yva, pid_va)
    te_mats, te_y = None, None
    if Xte is not None and yte is not None and pid_te is not None:
        _, te_mats, te_y = _group_by_patient(Xte, yte, pid_te)

    # --- train MIL head ---
    mil, mil_history = _train_mil(tr_mats, tr_y, device,
                                  epochs=args.attn_epochs, lr=args.attn_lr,
                                  batch_size=args.attn_batch_patients, seed=args.seed)

    # --- evaluate ---
    p_val = _eval_mil(mil, va_mats, device, args.attn_batch_patients, "MIL infer (val)")
    tau, bal_val_tau = _best_threshold(va_y, p_val)
    val_metrics = _compute_metrics(va_y, p_val, 0.5)
    val_metrics["balanced_accuracy@threshold"] = bal_val_tau
    val_metrics["threshold"] = tau

    out: dict = {
        "model": args.model_name,
        "imagenet_pretrained": True,
        "embed_dim": backbone.embed_dim,
        "n_train_slices": int(len(ytr)),
        "n_train_patients": int(len(tr_y)),
        "n_val_patients": int(len(va_y)),
        "attn_epochs": args.attn_epochs,
        "attn_lr": args.attn_lr,
        "mil_train_loss_history": mil_history,
        "val": val_metrics,
    }

    if run_test and te_mats is not None and te_y is not None:
        p_test = _eval_mil(mil, te_mats, device, args.attn_batch_patients, "MIL infer (test)")
        test_metrics = _compute_metrics(te_y, p_test, 0.5)
        test_metrics["balanced_accuracy@threshold_from_val"] = float(
            __import__("sklearn.metrics", fromlist=["balanced_accuracy_score"])
            .balanced_accuracy_score(te_y, (p_test >= tau).astype(int))
        )
        test_metrics["threshold_from_val"] = tau
        test_metrics["n_patients"] = int(len(te_y))
        test_metrics["confusion_matrix_at_0.5"] = _confusion_matrix(
            te_y, (p_test >= 0.5).astype(int))
        test_metrics["confusion_matrix_at_threshold"] = _confusion_matrix(
            te_y, (p_test >= tau).astype(int))
        out["test"] = test_metrics

        # save fpr/tpr for ROC plot
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(te_y, p_test)
        out["test"]["fpr"] = fpr.tolist()
        out["test"]["tpr"] = tpr.tolist()

    def _safe(o):
        import math
        if isinstance(o, dict):
            return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_safe(v) for v in o]
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, np.generic):
            return o.item()
        return o

    out_txt = json.dumps(_safe(out), indent=2)
    print(out_txt)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(out_txt, encoding="utf-8")
        tqdm.write(f"Metrics saved to {args.save_json}")

    if args.artifacts_dir is not None:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (args.artifacts_dir / "metrics.json").write_text(out_txt, encoding="utf-8")
        tqdm.write(f"Artifacts saved to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
