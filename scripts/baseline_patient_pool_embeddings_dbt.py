#!/usr/bin/env python3
"""Frozen DINO CLS embeddings: slice-level vs patient-level pooling (mean/max/attention) + metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DBT_SRC = _REPO_ROOT.parent / "dbt_simclr_project" / "src"
for p in (_REPO_ROOT / "src", _DBT_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from dbt_ssl.data.supervised_slice_dataset import DBTSupervisedSliceDataset, SupervisedSliceConfig
from mammodino_ssl.models.dino_ssl import create_dino_ssl


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


def _load_dino_model_section(config_path: Path) -> dict:
    if not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return dict(raw.get("model") or {})


def _normalize_ddp_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        nk = nk.replace("student_backbone.module.", "student_backbone.")
        nk = nk.replace("student_head.module.", "student_head.")
        nk = nk.replace("student_patch_head.module.", "student_patch_head.")
        out[nk] = v
    return out


def _to_tensor3(image_nchw: object) -> torch.Tensor:
    t = torch.from_numpy(image_nchw).float()  # type: ignore[arg-type]
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return t


def collate_supervised(batch: list[dict]) -> dict[str, object]:
    x = torch.stack([_to_tensor3(item["image"]) for item in batch], dim=0)
    y = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    meta = [
        {
            "canonical_patient_id": item.get("canonical_patient_id"),
            "slice_abspath": item.get("slice_abspath"),
        }
        for item in batch
    ]
    return {"image": x, "label": y, "meta": meta}


@torch.no_grad()
def _extract_cls_embeddings_with_patients(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None,
    desc: str,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    backbone.eval()
    amp_on = amp and device.type == "cuda"
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pids: list[str] = []
    try:
        n_loader = len(loader)
    except TypeError:
        n_loader = None
    total_steps = None if n_loader is None else (n_loader if max_batches is None else min(max_batches, n_loader))
    stream = tqdm(loader, desc=desc, total=total_steps, leave=True, dynamic_ncols=True) if progress else loader
    for step_idx, batch in enumerate(stream):
        if max_batches is not None and step_idx >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        meta = batch["meta"]
        with torch.amp.autocast("cuda", enabled=amp_on):
            cls, _ = backbone(x)
        xs.append(cls.detach().float().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        for m in meta:
            pids.append(str(m["canonical_patient_id"]))
    X = np.concatenate(xs, axis=0) if xs else np.zeros((0, 0), dtype=np.float32)
    Y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64)
    return X, Y, pids


def _group_slices_per_patient(
    X: np.ndarray, y: np.ndarray, patient_ids: list[str]
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    by_pid: dict[str, dict] = defaultdict(lambda: {"rows": [], "label": None})
    for i in range(len(y)):
        pid = patient_ids[i]
        block = by_pid[pid]
        block["rows"].append(X[i])
        lab = int(y[i])
        if block["label"] is None:
            block["label"] = lab
        elif block["label"] != lab:
            raise ValueError(f"Inconsistent labels for patient {pid}: {block['label']} vs {lab}")

    p_sorted = sorted(by_pid.keys())
    mats: list[np.ndarray] = []
    labels = np.empty(len(p_sorted), dtype=np.int64)
    for k, pid in enumerate(p_sorted):
        rows = by_pid[pid]["rows"]
        mats.append(np.stack(rows, axis=0))
        labels[k] = int(by_pid[pid]["label"])
    return p_sorted, mats, labels


def _aggregate_embeddings(mats: list[np.ndarray], how: str) -> np.ndarray:
    """mats[i] has shape (n_slices_i, D) -> stacked (N_patients, D)."""
    out = []
    for m in mats:
        if how == "mean":
            out.append(m.mean(axis=0))
        elif how == "max":
            out.append(m.max(axis=0))
        else:
            raise ValueError(how)
    return np.stack(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def _aggregate_slice_probs_to_patients(
    slice_probs: np.ndarray, y_slices: np.ndarray, patient_ids: list[str], agg: str
) -> tuple[np.ndarray, np.ndarray]:
    by_pid: dict[str, dict] = defaultdict(lambda: {"scores": [], "y": None})
    for i in range(len(slice_probs)):
        pid = patient_ids[i]
        by_pid[pid]["scores"].append(float(slice_probs[i]))
        lab = int(y_slices[i])
        if by_pid[pid]["y"] is None:
            by_pid[pid]["y"] = lab
        elif by_pid[pid]["y"] != lab:
            raise ValueError(f"label mismatch for patient {pid}")

    scores: list[float] = []
    y_pat: list[int] = []
    for pid in sorted(by_pid.keys()):
        sc = np.array(by_pid[pid]["scores"], dtype=np.float64)
        if agg == "mean":
            scores.append(float(sc.mean()))
        elif agg == "max":
            scores.append(float(sc.max()))
        else:
            raise ValueError(agg)
        y_pat.append(by_pid[pid]["y"])

    return np.asarray(y_pat, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def _patient_metrics_from_slice_probs(
    slice_probs: np.ndarray, y_slices: np.ndarray, patient_ids: list[str], agg: str
) -> tuple[float, float, int]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y_pat_arr, s_arr = _aggregate_slice_probs_to_patients(slice_probs, y_slices, patient_ids, agg)
    if len(y_pat_arr) == 0:
        return float("nan"), float("nan"), 0
    pred = (s_arr >= 0.5).astype(np.int64)
    auroc = float(roc_auc_score(y_pat_arr, s_arr)) if len(np.unique(y_pat_arr)) >= 2 else float("nan")
    bal = float(balanced_accuracy_score(y_pat_arr, pred))
    return auroc, bal, int(len(y_pat_arr))


def _best_threshold_balanced_accuracy(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing balanced accuracy (binary). Returns (threshold, balanced_accuracy_at_threshold)."""
    from sklearn.metrics import balanced_accuracy_score

    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(y_true) == 0:
        return 0.5, float("nan")
    candidates: set[float] = {0.0, 1.0}
    uniq = np.unique(scores)
    for u in uniq:
        candidates.add(float(u))
    if len(uniq) > 1:
        for mid in (uniq[:-1] + uniq[1:]) / 2.0:
            candidates.add(float(mid))
    best_t = 0.5
    best_bal = -1.0
    for t in sorted(candidates):
        pred = (scores >= t).astype(np.int64)
        bal = balanced_accuracy_score(y_true, pred)
        if bal > best_bal:
            best_bal = bal
            best_t = float(t)
    return best_t, float(best_bal)


def _confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "labels_row_true_col_pred": [0, 1],
        "matrix": cm.tolist(),
        "note": "sklearn confusion_matrix: rows=true class, cols=predicted class",
    }


def _patient_prob_metrics_dict(y_pat: np.ndarray, p_pat: np.ndarray, *, threshold: float) -> dict[str, float | int]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y_pat = np.asarray(y_pat, dtype=np.int64)
    p_pat = np.asarray(p_pat, dtype=np.float64)
    if len(y_pat) == 0:
        return {"auroc": float("nan"), "balanced_accuracy": float("nan"), "n_patients": 0}
    pred = (p_pat >= float(threshold)).astype(np.int64)
    auroc = float(roc_auc_score(y_pat, p_pat)) if len(np.unique(y_pat)) >= 2 else float("nan")
    bal = float(balanced_accuracy_score(y_pat, pred))
    return {"auroc": auroc, "balanced_accuracy": bal, "n_patients": int(len(y_pat))}


def _parse_eval_splits(s: str) -> list[str]:
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


class AttentionPool(nn.Module):
    """Single-query attention over slice tokens (batch, T, D)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] bool True = valid
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        return (w * x).sum(dim=1)


class AttentionPatientClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.pool = AttentionPool(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.pool(x, mask)
        return self.fc(z)


class PatientEmbeddingDataset(Dataset):
    def __init__(self, mats: list[np.ndarray], labels: np.ndarray) -> None:
        self.mats = mats
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.mats)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return self.mats[idx], int(self.labels[idx])


def collate_patient_batch(batch: list[tuple[np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embs = [torch.from_numpy(b[0]).float() for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    t_max = max(e.shape[0] for e in embs)
    d = embs[0].shape[1]
    bsz = len(batch)
    padded = torch.zeros(bsz, t_max, d)
    mask = torch.zeros(bsz, t_max, dtype=torch.bool)
    for i, e in enumerate(embs):
        t = e.shape[0]
        padded[i, :t] = e
        mask[i, :t] = True
    return padded, mask, labels


def _train_attention_classifier(
    train_mats: list[np.ndarray],
    train_y: np.ndarray,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    batch_patients: int,
    seed: int,
    num_workers: int,
    class_weights: torch.Tensor | None,
    progress: bool,
) -> AttentionPatientClassifier:
    torch.manual_seed(seed)
    dim = train_mats[0].shape[1]
    model = AttentionPatientClassifier(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    ds = PatientEmbeddingDataset(train_mats, train_y)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_patients,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_patient_batch,
        generator=g,
        drop_last=False,
    )

    model.train()
    epoch_iter = range(epochs)
    if progress:
        epoch_iter = tqdm(epoch_iter, desc="attention train", leave=True, dynamic_ncols=True)
    for _ in epoch_iter:
        running = 0.0
        n_batches = 0
        for padded, mask, labels in loader:
            padded = padded.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(padded, mask)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            n_batches += 1
        if progress and hasattr(epoch_iter, "set_postfix"):
            avg = running / max(1, n_batches)
            epoch_iter.set_postfix(loss=f"{avg:.4f}")
    return model


@torch.no_grad()
def _eval_attention_patient_probs(
    model: AttentionPatientClassifier,
    mats: list[np.ndarray],
    device: torch.device,
    batch_patients: int,
    num_workers: int,
    *,
    progress: bool,
    desc: str = "attention infer",
) -> np.ndarray:
    model.eval()
    ds = PatientEmbeddingDataset(mats, np.zeros(len(mats), dtype=np.int64))
    loader = DataLoader(
        ds,
        batch_size=batch_patients,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_patient_batch([(b[0], 0) for b in batch]),
    )
    probs: list[float] = []
    batches = loader
    if progress:
        batches = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True)
    for padded, mask, _ in batches:
        padded = padded.to(device)
        mask = mask.to(device)
        logits = model(padded, mask)
        p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy().tolist()
        probs.extend([float(x) for x in p])
    return np.asarray(probs, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient-level pooling on frozen DINO CLS embeddings (DBT): mean/max/attention vs slice LogReg."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--dino-config",
        type=Path,
        default=_REPO_ROOT / "configs/dino_dbt.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=("slice_logreg", "patient_mean_embed", "patient_max_embed", "patient_attention"),
        default="patient_attention",
        help="slice_logreg: sklearn on slices (+ patient metrics via prob aggregation). "
        "patient_*: pool embeddings per patient then classify.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128, help="Slice DataLoader batch (embedding extraction).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default="",
        help="Comma-separated GPU indices for nn.DataParallel during embedding extraction only, e.g. 0,1,2. "
        "First id is the output device. Empty = single GPU from --device.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm bars (quiet logs).")
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 = full train split")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 = full val split")
    parser.add_argument("--max-test-batches", type=int, default=0, help="0 = full test split (only if test in --eval-splits)")
    parser.add_argument(
        "--eval-splits",
        type=str,
        default="val",
        help="Comma-separated splits to evaluate after training on train: val, test (e.g. val,test). Test uses threshold tuned on val.",
    )
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument(
        "--data-repo-root",
        type=Path,
        default=Path("../dbt_simclr_project"),
    )
    # Attention-only
    parser.add_argument("--attn-epochs", type=int, default=30)
    parser.add_argument("--attn-lr", type=float, default=3e-4)
    parser.add_argument("--attn-batch-patients", type=int, default=32)
    parser.add_argument("--attn-class-weight", action="store_true", help="Inverse-frequency CE weights on train patients.")
    parser.add_argument("--save-json", type=Path, default=None, help="Write full metrics dict as JSON (safe for NaN).")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Create directory and write metrics.json (same as stdout payload).",
    )
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else (_REPO_ROOT / args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

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
        device_name = args.device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        if device_name.startswith("cuda"):
            device = torch.device(device_name if ":" in device_name else "cuda:0")
        else:
            device = torch.device(device_name)

    amp = (not args.no_amp) and device.type == "cuda"

    dino_cfg_path = args.dino_config if args.dino_config.is_absolute() else (_REPO_ROOT / args.dino_config).resolve()
    model_yaml = _load_dino_model_section(dino_cfg_path)
    image_size = int(model_yaml.get("image_size", 224))
    num_prototypes = int(model_yaml.get("num_prototypes", 512))

    data_root = args.data_repo_root if args.data_repo_root.is_absolute() else (_REPO_ROOT / args.data_repo_root).resolve()
    data_cfg = yaml.safe_load((data_root / "configs" / "data.yaml").read_text(encoding="utf-8"))
    artifacts = data_root / data_cfg.get("artifacts_dir", "artifacts")
    manifest_path = artifacts / data_cfg.get("manifest_rel_path", "manifests/master_manifest.parquet")
    split_path = artifacts / data_cfg.get("split_rel_path", "splits/patient_split_v1.json")
    use_cache = bool(data_cfg.get("use_processed_cache", False))
    cache_dir = Path(data_cfg.get("processed_cache_rel_path", "")) if use_cache else None
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = data_root / cache_dir

    ds_cfg = SupervisedSliceConfig(
        resize_height=image_size,
        resize_width=image_size,
        normalize=bool(data_cfg.get("normalize", True)),
        split_seed=int(args.seed),
        use_processed_cache=use_cache,
        processed_cache_dir=cache_dir,
        processed_cache_token=str(data_cfg.get("processed_cache_token", "")),
    )
    eval_splits = _parse_eval_splits(args.eval_splits)
    run_test = "test" in eval_splits

    train_ds = DBTSupervisedSliceDataset(manifest_path=manifest_path, split_path=split_path, split="train", config=ds_cfg)
    val_ds = DBTSupervisedSliceDataset(manifest_path=manifest_path, split_path=split_path, split="val", config=ds_cfg)
    test_ds = (
        DBTSupervisedSliceDataset(manifest_path=manifest_path, split_path=split_path, split="test", config=ds_cfg)
        if run_test
        else None
    )
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_supervised,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_supervised,
        generator=g,
    )
    test_loader = None
    if run_test and test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            collate_fn=collate_supervised,
            generator=g,
        )

    ssl = create_dino_ssl(
        image_size=image_size,
        num_prototypes=num_prototypes,
        embed_dim=int(model_yaml.get("embed_dim", 192)),
        depth=int(model_yaml.get("depth", 4)),
        num_heads=int(model_yaml.get("num_heads", 3)),
        head_hidden_dim=int(model_yaml.get("head_hidden_dim", 512)),
        head_bottleneck_dim=int(model_yaml.get("head_bottleneck_dim", 256)),
    ).to(device)
    if not args.no_progress:
        tqdm.write(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        ssl.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError:
        ssl.load_state_dict(_normalize_ddp_state_dict_keys(ckpt["model_state"]), strict=True)

    backbone_for_extract: nn.Module = ssl.student_backbone
    use_dp = bool(cuda_dev_ids) and len(cuda_dev_ids) > 1
    if use_dp:
        backbone_for_extract = nn.DataParallel(
            ssl.student_backbone, device_ids=cuda_dev_ids, output_device=cuda_dev_ids[0]
        )

    max_train_batches = int(args.max_train_batches) if int(args.max_train_batches) > 0 else None
    max_val_batches = int(args.max_val_batches) if int(args.max_val_batches) > 0 else None
    max_test_batches = int(args.max_test_batches) if int(args.max_test_batches) > 0 else None
    show_prog = not bool(args.no_progress)
    Xtr, ytr, pid_tr = _extract_cls_embeddings_with_patients(
        backbone_for_extract,
        train_loader,
        device,
        amp=amp,
        max_batches=max_train_batches,
        desc="CLS embeddings (train)",
        progress=show_prog,
    )
    Xva, yva, pid_va = _extract_cls_embeddings_with_patients(
        backbone_for_extract,
        val_loader,
        device,
        amp=amp,
        max_batches=max_val_batches,
        desc="CLS embeddings (val)",
        progress=show_prog,
    )
    Xte: np.ndarray | None = None
    yte: np.ndarray | None = None
    pid_te: list[str] | None = None
    if run_test and test_loader is not None:
        Xte_t, yte_t, pid_te_t = _extract_cls_embeddings_with_patients(
            backbone_for_extract,
            test_loader,
            device,
            amp=amp,
            max_batches=max_test_batches,
            desc="CLS embeddings (test)",
            progress=show_prog,
        )
        Xte, yte, pid_te = Xte_t, yte_t, pid_te_t

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    ex_devs: list[int] = (
        list(cuda_dev_ids)
        if cuda_dev_ids
        else ([int(device.index)] if device.type == "cuda" and device.index is not None else [])
    )
    out: dict = {
        "checkpoint": str(ckpt_path),
        "mode": str(args.mode),
        "shuffle_labels": bool(args.shuffle_labels),
        "eval_splits": eval_splits,
        "embedding_extraction": {
            "data_parallel": use_dp,
            "cuda_devices": ex_devs,
        },
    }

    def _finalize_threshold_outputs(
        patient_block: dict,
        *,
        score_definition: str,
        y_val_pat: np.ndarray,
        p_val_pat: np.ndarray,
        y_test_pat: np.ndarray | None,
        p_test_pat: np.ndarray | None,
        n_test_slices: int,
    ) -> None:
        tau, bal_at_tau_val = _best_threshold_balanced_accuracy(y_val_pat, p_val_pat)
        patient_block["threshold_tuned_on_val"] = float(tau)
        patient_block["balanced_accuracy@threshold_tuned_on_val"] = float(bal_at_tau_val)
        out["threshold_tuning"] = {
            "score_definition": score_definition,
            "threshold_fit_on": "val",
            "objective": "balanced_accuracy",
            "threshold_value": float(tau),
            "val_balanced_accuracy_at_threshold": float(bal_at_tau_val),
        }
        pred_val_05 = (p_val_pat >= 0.5).astype(np.int64)
        pred_val_tau = (p_val_pat >= tau).astype(np.int64)
        patient_block["confusion_matrix_patient_val_at_0.5"] = _confusion_matrix_binary(y_val_pat, pred_val_05)
        patient_block["confusion_matrix_patient_val_at_threshold"] = _confusion_matrix_binary(y_val_pat, pred_val_tau)
        if run_test and y_test_pat is not None and p_test_pat is not None and len(y_test_pat):
            m05 = _patient_prob_metrics_dict(y_test_pat, p_test_pat, threshold=0.5)
            mtau = _patient_prob_metrics_dict(y_test_pat, p_test_pat, threshold=tau)
            pred_te_05 = (p_test_pat >= 0.5).astype(np.int64)
            pred_te_tau = (p_test_pat >= tau).astype(np.int64)
            out["patient_level_test"] = {
                "score_definition": score_definition,
                "auroc": m05["auroc"],
                "balanced_accuracy@0.5": m05["balanced_accuracy"],
                "balanced_accuracy@threshold_from_val": mtau["balanced_accuracy"],
                "threshold_from_val": float(tau),
                "n_patients": m05["n_patients"],
                "n_slices": int(n_test_slices),
                "confusion_matrix_patient_test_at_0.5": _confusion_matrix_binary(y_test_pat, pred_te_05),
                "confusion_matrix_patient_test_at_threshold_from_val": _confusion_matrix_binary(y_test_pat, pred_te_tau),
            }

    if args.shuffle_labels and len(ytr):
        rng = np.random.default_rng(int(args.seed))
        ytr = rng.permutation(ytr)

    slice_block: dict | None = None
    patient_block: dict

    if args.mode == "slice_logreg":
        if show_prog:
            tqdm.write("fitting slice-level LogisticRegression …")
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=None)
        clf.fit(Xtr, ytr)
        p_va = clf.predict_proba(Xva)[:, 1] if len(Xva) else np.zeros((0,), dtype=np.float64)
        y_hat = (p_va >= 0.5).astype(np.int64) if len(p_va) else np.zeros((0,), dtype=np.int64)
        slice_auroc = float(roc_auc_score(yva, p_va)) if len(np.unique(yva)) >= 2 else float("nan")
        slice_bal = float(balanced_accuracy_score(yva, y_hat)) if len(yva) else float("nan")
        pm_auroc, pm_bal, npm = _patient_metrics_from_slice_probs(p_va, yva, pid_va, "mean")
        px_auroc, px_bal, _ = _patient_metrics_from_slice_probs(p_va, yva, pid_va, "max")
        slice_block = {
            "auroc": slice_auroc,
            "balanced_accuracy@0.5": slice_bal,
            "n_val_slices": int(len(yva)),
            "confusion_matrix_slice_val_at_0.5": _confusion_matrix_binary(yva, y_hat),
        }
        patient_block = {
            "primary_eval": True,
            "from_slice_classifier_prob_agg_mean": {"auroc": pm_auroc, "balanced_accuracy@0.5": pm_bal, "n_val_patients": npm},
            "from_slice_classifier_prob_agg_max": {"auroc": px_auroc, "balanced_accuracy@0.5": px_bal, "n_val_patients": npm},
        }
        out["n_train_slices"] = int(len(ytr))
        out["embed_dim"] = int(Xtr.shape[1]) if Xtr.ndim == 2 else None
        yvp, pvp = _aggregate_slice_probs_to_patients(p_va, yva, pid_va, "mean")
        ytp, ptp = None, None
        n_te_sl = 0
        if run_test and Xte is not None and yte is not None and pid_te is not None and len(Xte):
            p_te = clf.predict_proba(Xte)[:, 1]
            ytp, ptp = _aggregate_slice_probs_to_patients(p_te, yte, pid_te, "mean")
            n_te_sl = int(len(yte))
        patient_block["n_val_slices"] = int(len(yva))
        _finalize_threshold_outputs(
            patient_block["from_slice_classifier_prob_agg_mean"],
            score_definition="slice_classifier_prob_agg_mean",
            y_val_pat=yvp,
            p_val_pat=pvp,
            y_test_pat=ytp,
            p_test_pat=ptp,
            n_test_slices=n_te_sl,
        )

    elif args.mode in ("patient_mean_embed", "patient_max_embed"):
        how = "mean" if args.mode == "patient_mean_embed" else "max"
        _, tr_mats, tr_y = _group_slices_per_patient(Xtr, ytr, pid_tr)
        _, va_mats, va_y = _group_slices_per_patient(Xva, yva, pid_va)
        Xptr = _aggregate_embeddings(tr_mats, how)
        Xpva = _aggregate_embeddings(va_mats, how)
        if show_prog:
            tqdm.write("fitting patient-level LogisticRegression (on pooled embeddings) …")
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=None)
        clf.fit(Xptr, tr_y)
        p_pat = clf.predict_proba(Xpva)[:, 1] if len(Xpva) else np.zeros((0,), dtype=np.float64)
        y_hat_p = (p_pat >= 0.5).astype(np.int64) if len(p_pat) else np.zeros((0,), dtype=np.int64)
        pat_auroc = float(roc_auc_score(va_y, p_pat)) if len(np.unique(va_y)) >= 2 else float("nan")
        pat_bal = float(balanced_accuracy_score(va_y, y_hat_p)) if len(va_y) else float("nan")
        slice_block = None
        patient_block = {
            "primary_eval": True,
            "embedding_pool": how,
            "auroc": pat_auroc,
            "balanced_accuracy@0.5": pat_bal,
            "n_train_patients": int(len(tr_y)),
            "n_val_patients": int(len(va_y)),
        }
        out["n_train_slices"] = int(len(ytr))
        out["embed_dim"] = int(Xptr.shape[1]) if Xptr.ndim == 2 else None
        ytp, ptp = None, None
        n_te_sl = 0
        if run_test and Xte is not None and yte is not None and pid_te is not None and len(Xte):
            _, te_mats, te_y = _group_slices_per_patient(Xte, yte, pid_te)
            Xpte = _aggregate_embeddings(te_mats, how)
            ptp = clf.predict_proba(Xpte)[:, 1] if len(Xpte) else np.zeros((0,), dtype=np.float64)
            ytp = te_y
            n_te_sl = int(len(yte))
        patient_block["n_val_slices"] = int(len(yva))
        _finalize_threshold_outputs(
            patient_block,
            score_definition=f"patient_{how}_embed_then_logreg",
            y_val_pat=va_y,
            p_val_pat=p_pat,
            y_test_pat=ytp,
            p_test_pat=ptp,
            n_test_slices=n_te_sl,
        )

    else:
        assert args.mode == "patient_attention"
        _, tr_mats, tr_y = _group_slices_per_patient(Xtr, ytr, pid_tr)
        _, va_mats, va_y = _group_slices_per_patient(Xva, yva, pid_va)

        cw = None
        if args.attn_class_weight and len(tr_y):
            n0 = int((tr_y == 0).sum())
            n1 = int((tr_y == 1).sum())
            # sklearn-style: n_samples / (n_classes * count)
            w0 = len(tr_y) / (2 * max(1, n0))
            w1 = len(tr_y) / (2 * max(1, n1))
            cw = torch.tensor([w0, w1], dtype=torch.float32, device=device)

        attn = _train_attention_classifier(
            tr_mats,
            tr_y,
            device,
            epochs=int(args.attn_epochs),
            lr=float(args.attn_lr),
            batch_patients=int(args.attn_batch_patients),
            seed=int(args.seed),
            num_workers=int(args.num_workers),
            class_weights=cw,
            progress=show_prog,
        )
        p_pat = _eval_attention_patient_probs(
            attn,
            va_mats,
            device,
            int(args.attn_batch_patients),
            int(args.num_workers),
            progress=show_prog,
            desc="attention infer (val)",
        )
        y_hat_p = (p_pat >= 0.5).astype(np.int64) if len(p_pat) else np.zeros((0,), dtype=np.int64)
        pat_auroc = float(roc_auc_score(va_y, p_pat)) if len(np.unique(va_y)) >= 2 else float("nan")
        pat_bal = float(balanced_accuracy_score(va_y, y_hat_p)) if len(va_y) else float("nan")
        slice_block = None
        patient_block = {
            "primary_eval": True,
            "embedding_pool": "attention",
            "attn_epochs": int(args.attn_epochs),
            "attn_lr": float(args.attn_lr),
            "attn_batch_patients": int(args.attn_batch_patients),
            "attn_class_weight": bool(args.attn_class_weight),
            "auroc": pat_auroc,
            "balanced_accuracy@0.5": pat_bal,
            "n_train_patients": int(len(tr_y)),
            "n_val_patients": int(len(va_y)),
        }
        out["n_train_slices"] = int(len(ytr))
        out["embed_dim"] = int(tr_mats[0].shape[1]) if tr_mats else None
        ytp, ptp = None, None
        n_te_sl = 0
        if run_test and Xte is not None and yte is not None and pid_te is not None and len(Xte):
            _, te_mats, te_y = _group_slices_per_patient(Xte, yte, pid_te)
            p_tp = _eval_attention_patient_probs(
                attn,
                te_mats,
                device,
                int(args.attn_batch_patients),
                int(args.num_workers),
                progress=show_prog,
                desc="attention infer (test)",
            )
            ytp, ptp = te_y, p_tp
            n_te_sl = int(len(yte))
        patient_block["n_val_slices"] = int(len(yva))
        _finalize_threshold_outputs(
            patient_block,
            score_definition="patient_attention_pool",
            y_val_pat=va_y,
            p_val_pat=p_pat,
            y_test_pat=ytp,
            p_test_pat=ptp,
            n_test_slices=n_te_sl,
        )

    out["slice_level"] = slice_block
    out["patient_level"] = patient_block

    def _json_safe(o: object) -> object:
        import math

        if isinstance(o, dict):
            return {str(k): _json_safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_json_safe(v) for v in o]
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, np.generic):
            return o.item()
        return o

    out_txt = json.dumps(_json_safe(out), indent=2)
    print(out_txt)
    if args.save_json is not None:
        sp = args.save_json if args.save_json.is_absolute() else (_REPO_ROOT / args.save_json).resolve()
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(out_txt, encoding="utf-8")
    if args.artifacts_dir is not None:
        ad = args.artifacts_dir if args.artifacts_dir.is_absolute() else (_REPO_ROOT / args.artifacts_dir).resolve()
        ad.mkdir(parents=True, exist_ok=True)
        (ad / "metrics.json").write_text(out_txt, encoding="utf-8")


if __name__ == "__main__":
    main()
