#!/usr/bin/env python3
"""Patient attention pool evaluation for DINO+iBOT checkpoint.

Extends baseline_patient_pool_embeddings_dbt.py to save patient-level
probabilities and generate a ROC curve plot. AUC on the plot matches
the reported metric (same method, same probabilities).

Run with Liron's venv:
    source /mnt/md0/Liron/mammodino_ssl_project/.venv_py312_clean/bin/activate
    python eval_attention_pool_with_roc.py \
        --checkpoint /mnt/data/avi/dino_ibot_runs/.../checkpoints/best.pt \
        --artifacts-dir /mnt/data/avi/dino_ibot_runs/.../probe_patient_attention_roc \
        --eval-splits val,test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DBT_SRC = Path("/mnt/md0/Liron/dbt_simclr_project/src")
for _p in (_REPO_ROOT / "src", _DBT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dbt_ssl.data.supervised_slice_dataset import DBTSupervisedSliceDataset, SupervisedSliceConfig
from mammodino_ssl.models.dino_ssl import create_dino_ssl


# ---------------------------------------------------------------------------
# Utilities (identical to baseline_patient_pool_embeddings_dbt.py)
# ---------------------------------------------------------------------------

def _to_tensor3(image_nchw) -> torch.Tensor:
    t = torch.from_numpy(image_nchw).float()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return t


def _collate(batch):
    x = torch.stack([_to_tensor3(item["image"]) for item in batch])
    y = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    meta = [{"canonical_patient_id": item.get("canonical_patient_id")} for item in batch]
    return {"image": x, "label": y, "meta": meta}


@torch.no_grad()
def _extract_embeddings(backbone, loader, device, desc):
    backbone.eval()
    xs, ys, pids = [], [], []
    for batch in tqdm(loader, desc=desc, dynamic_ncols=True):
        x = batch["image"].to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            cls, _ = backbone(x)
        xs.append(cls.float().cpu().numpy())
        ys.append(batch["label"].numpy())
        for m in batch["meta"]:
            pids.append(str(m["canonical_patient_id"]))
    X = np.concatenate(xs) if xs else np.zeros((0, 0), dtype=np.float32)
    Y = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
    return X, Y, pids


def _group_by_patient(X, y, pids):
    by_pid = defaultdict(lambda: {"rows": [], "label": None})
    for i in range(len(y)):
        by_pid[pids[i]]["rows"].append(X[i])
        if by_pid[pids[i]]["label"] is None:
            by_pid[pids[i]]["label"] = int(y[i])
    p_sorted = sorted(by_pid.keys())
    mats, labels = [], np.empty(len(p_sorted), dtype=np.int64)
    for k, pid in enumerate(p_sorted):
        mats.append(np.stack(by_pid[pid]["rows"]))
        labels[k] = by_pid[pid]["label"]
    return p_sorted, mats, labels


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        return (torch.softmax(s, dim=1).unsqueeze(-1) * x).sum(dim=1)


class AttentionPatientClassifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = AttentionPool(dim)
        self.fc = nn.Linear(dim, 2)

    def forward(self, x, mask):
        return self.fc(self.pool(x, mask))


class PatientDataset(Dataset):
    def __init__(self, mats, labels):
        self.mats = mats
        self.labels = labels.astype(np.int64)

    def __len__(self): return len(self.mats)
    def __getitem__(self, i): return self.mats[i], int(self.labels[i])


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


def _train_mil(mats, labels, device, epochs, lr, batch_size, seed):
    torch.manual_seed(seed)
    dim = mats[0].shape[1]
    model = AttentionPatientClassifier(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(PatientDataset(mats, labels), batch_size=batch_size,
                        shuffle=True, collate_fn=_collate_patients, generator=g)
    model.train()
    for _ in tqdm(range(epochs), desc="MIL training", dynamic_ncols=True):
        for padded, mask, y in loader:
            padded, mask, y = padded.to(device), mask.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            crit(model(padded, mask), y).backward()
            opt.step()
    return model


@torch.no_grad()
def _get_probs(model, mats, device, batch_size, desc):
    model.eval()
    ds = PatientDataset(mats, np.zeros(len(mats), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: _collate_patients([(x, 0) for x, _ in b]))
    probs = []
    for padded, mask, _ in tqdm(loader, desc=desc, dynamic_ncols=True):
        padded, mask = padded.to(device), mask.to(device)
        p = torch.softmax(model(padded, mask).float(), dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dino-config", type=Path,
                        default=Path("/mnt/md0/Liron/mammodino_ssl_project/configs/dino_dbt.yaml"))
    parser.add_argument("--data-repo-root", type=Path,
                        default=Path("/mnt/md0/Liron/dbt_simclr_project"))
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--eval-splits", type=str, default="val,test")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--attn-epochs", type=int, default=30)
    parser.add_argument("--attn-lr", type=float, default=3e-4)
    parser.add_argument("--attn-batch-patients", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_test = "test" in args.eval_splits
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # data
    data_root = args.data_repo_root
    data_cfg = yaml.safe_load((data_root / "configs/data.yaml").read_text())
    artifacts = data_root / data_cfg.get("artifacts_dir", "artifacts")
    manifest_path = artifacts / data_cfg.get("manifest_rel_path", "manifests/master_manifest.parquet")
    split_path = artifacts / data_cfg.get("split_rel_path", "splits/patient_split_v1.json")

    model_cfg = yaml.safe_load(args.dino_config.read_text()).get("model", {})
    image_size = int(model_cfg.get("image_size", 224))

    ds_cfg = SupervisedSliceConfig(resize_height=image_size, resize_width=image_size,
                                   normalize=True, split_seed=args.seed,
                                   use_processed_cache=False)

    def _loader(split, shuffle):
        ds = DBTSupervisedSliceDataset(manifest_path=manifest_path, split_path=split_path,
                                       split=split, config=ds_cfg)
        g = torch.Generator(); g.manual_seed(args.seed)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                          collate_fn=_collate, generator=g)

    # backbone
    ssl = create_dino_ssl(
        image_size=image_size,
        num_prototypes=int(model_cfg.get("num_prototypes", 512)),
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        depth=int(model_cfg.get("depth", 4)),
        num_heads=int(model_cfg.get("num_heads", 3)),
        head_hidden_dim=int(model_cfg.get("head_hidden_dim", 512)),
        head_bottleneck_dim=int(model_cfg.get("head_bottleneck_dim", 256)),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    state = {k.replace("student_backbone.module.", "student_backbone."): v for k, v in state.items()}
    ssl.load_state_dict(state, strict=True)
    backbone = ssl.student_backbone

    # extract embeddings
    Xtr, ytr, pid_tr = _extract_embeddings(backbone, _loader("train", True), device, "Embeddings (train)")
    Xva, yva, pid_va = _extract_embeddings(backbone, _loader("val", False), device, "Embeddings (val)")
    Xte, yte, pid_te = None, None, None
    if run_test:
        Xte, yte, pid_te = _extract_embeddings(backbone, _loader("test", False), device, "Embeddings (test)")

    _, tr_mats, tr_y = _group_by_patient(Xtr, ytr, pid_tr)
    _, va_mats, va_y = _group_by_patient(Xva, yva, pid_va)
    te_mats, te_y = None, None
    if Xte is not None:
        _, te_mats, te_y = _group_by_patient(Xte, yte, pid_te)

    # train MIL
    mil = _train_mil(tr_mats, tr_y, device, args.attn_epochs, args.attn_lr,
                     args.attn_batch_patients, args.seed)

    # get probabilities
    p_val = _get_probs(mil, va_mats, device, args.attn_batch_patients, "Probs (val)")
    p_test = _get_probs(mil, te_mats, device, args.attn_batch_patients, "Probs (test)") \
             if te_mats is not None else None

    # metrics
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve

    def _best_thresh(y, p):
        best_t, best_b = 0.5, -1.0
        for t in sorted(set(p.tolist()) | {0.5}):
            b = balanced_accuracy_score(y, (p >= t).astype(int))
            if b > best_b: best_b, best_t = b, t
        return float(best_t), float(best_b)

    val_auc = float(roc_auc_score(va_y, p_val))
    tau, _ = _best_thresh(va_y, p_val)
    tqdm.write(f"Val AUROC: {val_auc:.4f} | threshold from val: {tau:.4f}")

    out = {"val_auroc": val_auc, "threshold_from_val": tau,
           "val_probs": p_val.tolist(), "val_labels": va_y.tolist()}

    if p_test is not None and te_y is not None:
        test_auc = float(roc_auc_score(te_y, p_test))
        test_bal = float(balanced_accuracy_score(te_y, (p_test >= tau).astype(int)))
        fpr, tpr, _ = roc_curve(te_y, p_test)
        tqdm.write(f"Test AUROC: {test_auc:.4f} | Balanced Acc @val_thresh: {test_bal:.4f}")
        out.update({"test_auroc": test_auc, "test_balanced_accuracy": test_bal,
                    "test_probs": p_test.tolist(), "test_labels": te_y.tolist(),
                    "fpr": fpr.tolist(), "tpr": tpr.tolist()})

    # save metrics
    (args.artifacts_dir / "metrics_with_probs.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    tqdm.write(f"Saved to {args.artifacts_dir}/metrics_with_probs.json")

    # ROC curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if "fpr" in out:
            plt.figure(figsize=(7, 6))
            plt.plot(out["fpr"], out["tpr"],
                     label=f"DINO+iBOT (AUC={out['test_auroc']:.3f})", color="steelblue")
            plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curve — DINO+iBOT (Test Set)")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.artifacts_dir / "roc_curve_test.png", dpi=150)
            plt.close()
            tqdm.write("roc_curve_test.png saved")
    except ImportError:
        tqdm.write("matplotlib not available — skipping ROC plot")


if __name__ == "__main__":
    main()
