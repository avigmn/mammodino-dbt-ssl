"""Train / eval patient-level Transformer on padded slice sequences."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cross_slice.model import VolumeTokenTransformer


class PatientTensorDataset(Dataset):
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


def _binary_patient_metrics(y_true: np.ndarray, scores: np.ndarray, *, threshold: float = 0.5) -> dict[str, float | int]:
    """Classification diagnostics at a fixed score threshold (patient-level)."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    empty: dict[str, float | int] = {
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "auroc": float("nan"),
        "f1_pos": float("nan"),
        "precision_pos": float("nan"),
        "recall_pos": float("nan"),
        "pct_pred_positive": float("nan"),
        "mean_score": float("nan"),
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
    }
    if len(y_true) == 0:
        return empty

    pred = (scores >= threshold).astype(np.int64)
    out: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auroc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) >= 2 else float("nan"),
        "f1_pos": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision_pos": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "pct_pred_positive": float(pred.mean()),
        "mean_score": float(scores.mean()),
    }
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["tn"], out["fp"], out["fn"], out["tp"] = int(tn), int(fp), int(fn), int(tp)
    return out


@torch.no_grad()
def eval_patient_avg_ce(
    model: VolumeTokenTransformer,
    mats: list[np.ndarray],
    labels_np: np.ndarray,
    device: torch.device,
    batch_patients: int,
    num_workers: int,
    crit: nn.Module,
    *,
    progress: bool,
    desc: str,
) -> float:
    """Mean cross-entropy over patients (eval mode)."""
    model.eval()
    labels_np = np.asarray(labels_np, dtype=np.int64)
    if len(mats) == 0:
        return float("nan")
    ds = PatientTensorDataset(mats, labels_np)
    loader = DataLoader(
        ds,
        batch_size=batch_patients,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patient_batch,
    )
    total = 0.0
    n = 0
    stream = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if progress else loader
    for padded, mask, labels in stream:
        padded = padded.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        logits = model(padded, mask)
        bs = int(labels.size(0))
        batch_loss = crit(logits, labels)
        total += float(batch_loss.detach().cpu()) * bs
        n += bs
    return total / max(1, n)


@torch.no_grad()
def eval_patient_probs(
    model: VolumeTokenTransformer,
    mats: list[np.ndarray],
    device: torch.device,
    batch_patients: int,
    num_workers: int,
    progress: bool,
    desc: str,
) -> np.ndarray:
    model.eval()
    ds = PatientTensorDataset(mats, np.zeros(len(mats), dtype=np.int64))
    loader = DataLoader(
        ds,
        batch_size=batch_patients,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_patient_batch([(b[0], 0) for b in batch]),
    )
    probs: list[float] = []
    stream = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True) if progress else loader
    for padded, mask, _ in stream:
        padded = padded.to(device)
        mask = mask.to(device)
        logits = model(padded, mask)
        p = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy().tolist()
        probs.extend(float(x) for x in p)
    return np.asarray(probs, dtype=np.float64)


def train_transformer(
    *,
    train_mats: list[np.ndarray],
    train_y: np.ndarray,
    val_mats: list[np.ndarray],
    val_y: np.ndarray,
    device: torch.device,
    dim: int,
    epochs_max: int,
    lr: float,
    weight_decay: float,
    batch_patients: int,
    num_workers: int,
    seed: int,
    class_weights: torch.Tensor | None,
    tf_dropout: float,
    tf_nhead: int,
    tf_layers: int,
    early_stop_metric: str,
    early_stop_patience: int,
    progress: bool,
    ckpt_path: Path,
    log_file: Path,
    metrics_jsonl_path: Path | None = None,
    verbose_epoch_metrics: bool = False,
) -> VolumeTokenTransformer:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = VolumeTokenTransformer(
        dim,
        n_layers=tf_layers,
        nhead=tf_nhead,
        dropout=tf_dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    ds = PatientTensorDataset(train_mats, train_y)
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

    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    patience_left = int(early_stop_patience)

    def score_val(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score

        if len(y) == 0:
            return float("nan"), float("nan")
        pred05 = (p >= 0.5).astype(np.int64)
        bal = float(balanced_accuracy_score(y, pred05))
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan")
        return auc, bal

    log_header = (
        f"# train log early_stop_metric={early_stop_metric} "
        f"verbose_epoch_metrics={verbose_epoch_metrics}\n"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(log_header, encoding="utf-8")

    if metrics_jsonl_path is not None:
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path.write_text("", encoding="utf-8")

    def _append_jsonl(record: dict) -> None:
        if metrics_jsonl_path is None:
            return
        safe = {}
        for k, v in record.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                safe[k] = None
            else:
                safe[k] = v
        with metrics_jsonl_path.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(safe) + "\n")

    for epoch in range(int(epochs_max)):
        model.train()
        running = 0.0
        n_batches = 0
        epoch_iter = (
            tqdm(loader, desc=f"epoch {epoch + 1}/{epochs_max}", leave=False, dynamic_ncols=True)
            if progress
            else loader
        )
        for padded, mask, labels in epoch_iter:
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
        train_loss = running / max(1, n_batches)

        p_val = eval_patient_probs(
            model,
            val_mats,
            device,
            batch_patients,
            num_workers,
            progress=False,
            desc=f"epoch {epoch + 1} val infer",
        )

        if verbose_epoch_metrics:
            p_train = eval_patient_probs(
                model,
                train_mats,
                device,
                batch_patients,
                num_workers,
                progress=False,
                desc=f"epoch {epoch + 1} train infer",
            )
            train_ce_eval = eval_patient_avg_ce(
                model,
                train_mats,
                train_y,
                device,
                batch_patients,
                num_workers,
                crit,
                progress=False,
                desc=f"epoch {epoch + 1} train CE",
            )
            val_ce_eval = eval_patient_avg_ce(
                model,
                val_mats,
                val_y,
                device,
                batch_patients,
                num_workers,
                crit,
                progress=False,
                desc=f"epoch {epoch + 1} val CE",
            )
            m_tr = _binary_patient_metrics(train_y, p_train)
            m_va = _binary_patient_metrics(val_y, p_val)
            auc = float(m_va["auroc"])
            bal = float(m_va["balanced_accuracy"])
        else:
            auc, bal = score_val(val_y, p_val)

        if early_stop_metric == "auroc":
            metric = auc if not math.isnan(auc) else bal
        elif early_stop_metric == "balanced_accuracy":
            metric = bal if not math.isnan(bal) else auc
        else:
            raise ValueError(f"unknown early_stop_metric {early_stop_metric!r}")

        if math.isnan(metric):
            metric = float("-inf")

        improved = metric > best_score + 1e-6
        if improved:
            best_score = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(early_stop_patience)
        else:
            patience_left -= 1

        ep = epoch + 1

        if verbose_epoch_metrics:
            block = (
                f"\n=== epoch {ep}/{epochs_max} ===\n"
                f"loss_train_batches_mean={train_loss:.6f}  ce_eval_train={train_ce_eval:.6f}  "
                f"ce_eval_val={val_ce_eval:.6f}\n"
                f"train  AUROC={m_tr['auroc']}  bal_acc@0.5={m_tr['balanced_accuracy']}  acc={m_tr['accuracy']}  "
                f"f1_pos={m_tr['f1_pos']}  prec_pos={m_tr['precision_pos']}  rec_pos={m_tr['recall_pos']}  "
                f"pct_pred_pos={m_tr['pct_pred_positive']:.4f}  mean_score={m_tr['mean_score']:.4f}  "
                f"cm[tn,fp,fn,tp]={m_tr['tn']},{m_tr['fp']},{m_tr['fn']},{m_tr['tp']}\n"
                f"val    AUROC={m_va['auroc']}  bal_acc@0.5={m_va['balanced_accuracy']}  acc={m_va['accuracy']}  "
                f"f1_pos={m_va['f1_pos']}  prec_pos={m_va['precision_pos']}  rec_pos={m_va['recall_pos']}  "
                f"pct_pred_pos={m_va['pct_pred_positive']:.4f}  mean_score={m_va['mean_score']:.4f}  "
                f"cm[tn,fp,fn,tp]={m_va['tn']},{m_va['fp']},{m_va['fn']},{m_va['tp']}\n"
                f"early_stop: metric={early_stop_metric} current={metric} best={best_score} "
                f"patience_left={patience_left} improved={improved}\n"
            )
            with log_file.open("a", encoding="utf-8") as lf:
                lf.write(block)
            _append_jsonl(
                {
                    "epoch": ep,
                    "verbose_epoch_metrics": True,
                    "train_loss_batches_mean": train_loss,
                    "train_ce_eval_mean": train_ce_eval,
                    "val_ce_eval_mean": val_ce_eval,
                    "train_auroc": m_tr["auroc"],
                    "train_bal_acc05": m_tr["balanced_accuracy"],
                    "train_accuracy": m_tr["accuracy"],
                    "train_f1_pos": m_tr["f1_pos"],
                    "train_precision_pos": m_tr["precision_pos"],
                    "train_recall_pos": m_tr["recall_pos"],
                    "train_pct_pred_positive": m_tr["pct_pred_positive"],
                    "train_mean_score": m_tr["mean_score"],
                    "train_tn": m_tr["tn"],
                    "train_fp": m_tr["fp"],
                    "train_fn": m_tr["fn"],
                    "train_tp": m_tr["tp"],
                    "val_auroc": m_va["auroc"],
                    "val_bal_acc05": m_va["balanced_accuracy"],
                    "val_accuracy": m_va["accuracy"],
                    "val_f1_pos": m_va["f1_pos"],
                    "val_precision_pos": m_va["precision_pos"],
                    "val_recall_pos": m_va["recall_pos"],
                    "val_pct_pred_positive": m_va["pct_pred_positive"],
                    "val_mean_score": m_va["mean_score"],
                    "val_tn": m_va["tn"],
                    "val_fp": m_va["fp"],
                    "val_fn": m_va["fn"],
                    "val_tp": m_va["tp"],
                    "early_stop_metric": early_stop_metric,
                    "early_stop_current": metric,
                    "best_score": best_score,
                    "patience_left": patience_left,
                    "improved": improved,
                }
            )
        else:
            line = (
                f"epoch={ep} train_loss={train_loss:.6f} val_auroc={auc} "
                f"val_bal_acc05={bal} best_score={best_score} patience={patience_left}\n"
            )
            with log_file.open("a", encoding="utf-8") as lf:
                lf.write(line)
            _append_jsonl(
                {
                    "epoch": ep,
                    "verbose_epoch_metrics": False,
                    "train_loss": train_loss,
                    "val_auroc": auc,
                    "val_bal_acc05": bal,
                    "early_stop_metric": early_stop_metric,
                    "best_score": best_score,
                    "patience_left": patience_left,
                    "improved": improved,
                }
            )

        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    return model
