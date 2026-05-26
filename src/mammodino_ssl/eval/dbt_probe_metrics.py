"""Confusion matrix + ROC/AUC for binary DBT linear probe (shared by train + eval scripts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from torch import nn

try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        average_precision_score,
        precision_recall_fscore_support,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
except ImportError:
    roc_auc_score = None  # type: ignore[misc, assignment]
    roc_curve = None  # type: ignore[misc, assignment]
    balanced_accuracy_score = None  # type: ignore[misc, assignment]
    precision_recall_fscore_support = None  # type: ignore[misc, assignment]
    precision_recall_curve = None  # type: ignore[misc, assignment]
    average_precision_score = None  # type: ignore[misc, assignment]


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def _collect_scores(
    model: "nn.Module",
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    max_steps: int | None,
    positive_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    amp_on = amp and device.type == "cuda"
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    pc = int(positive_class)
    for step_idx, batch in enumerate(tqdm(loader, desc="eval_metrics", leave=False, dynamic_ncols=True)):
        if max_steps is not None and step_idx >= max_steps:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x)
            prob = F.softmax(logits.float(), dim=1)[:, pc]
            pred = logits.argmax(dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        y_score.extend(prob.detach().cpu().tolist())
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(y_score, dtype=np.float64),
    )


def _plot_confusion(cm: np.ndarray, *, split: str, acc: float, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion matrix ({split}, acc={acc:.4f})")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, *, split: str, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    auc_label = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "n/a"
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_label})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC ({split})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _majority_baseline(y_true: np.ndarray) -> dict[str, float | int]:
    n0 = int((y_true == 0).sum())
    n1 = int((y_true == 1).sum())
    total = max(1, n0 + n1)
    majority_class = 0 if n0 >= n1 else 1
    return {
        "n_class_0": n0,
        "n_class_1": n1,
        "majority_class": majority_class,
        "majority_baseline_accuracy": float(max(n0, n1) / total),
    }


def _plot_pr(precision: np.ndarray, recall: np.ndarray, ap: float, *, split: str, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ap_label = f"{ap:.4f}" if not np.isnan(ap) else "n/a"
    ax.plot(recall, precision, label=f"PR (AP = {ap_label})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall ({split})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_confusion_roc_metrics(
    probe: "nn.Module",
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
    out_dir: Path,
    split: str,
    positive_class: int = 1,
    max_steps: int | None = None,
    dino_checkpoint: str | None = None,
    probe_checkpoint: str | None = None,
    extra_fields: dict | None = None,
) -> dict:
    """Writes confusion_matrix_*.png, roc_curve_*.png, eval_metrics_*.json. Returns summary dict."""
    if roc_curve is None or roc_auc_score is None or balanced_accuracy_score is None:
        raise ImportError(
            "save_confusion_roc_metrics requires scikit-learn. "
            "pip install 'mammodino-ssl[train]' or pip install scikit-learn"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, y_score = _collect_scores(
        probe, loader, device, amp=amp, max_steps=max_steps, positive_class=positive_class
    )
    cm = _confusion_matrix(y_true, y_pred, num_classes=2)
    correct = int(np.trace(cm))
    total = int(cm.sum())
    acc = correct / max(1, total)
    maj_info = _majority_baseline(y_true)
    pct_pred_positive = float((y_pred == 1).mean()) if y_pred.size else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=int(positive_class),
        zero_division=0,
    )

    cm_png = out_dir / f"confusion_matrix_{split}.png"
    _plot_confusion(cm, split=split, acc=acc, out_path=cm_png)

    roc_auc = float("nan")
    roc_auc_note = ""
    fpr = np.array([0.0, 1.0])
    tpr = np.array([0.0, 1.0])
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = float(roc_auc_score(y_true, y_score))
    else:
        roc_auc_note = "single_class_in_y_true"

    roc_png = out_dir / f"roc_curve_{split}.png"
    _plot_roc(fpr, tpr, roc_auc, split=split, out_path=roc_png)

    pr_auc = float("nan")
    pr_note = ""
    pr_precision = np.array([1.0])
    pr_recall = np.array([0.0])
    if precision_recall_curve is not None and average_precision_score is not None and len(np.unique(y_true)) >= 2:
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score, pos_label=int(positive_class))
        pr_auc = float(average_precision_score(y_true, y_score))
    else:
        pr_note = "precision_recall_curve_unavailable_or_single_class"

    pr_png = out_dir / f"pr_curve_{split}.png"
    _plot_pr(pr_precision, pr_recall, pr_auc, split=split, out_path=pr_png)

    summary: dict = {
        "split": split,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "roc_auc_note": roc_auc_note,
        "average_precision": pr_auc,
        "average_precision_note": pr_note,
        "pct_pred_positive": pct_pred_positive,
        **maj_info,
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "pr_precision": pr_precision.tolist(),
        "pr_recall": pr_recall.tolist(),
        "n_samples_used": total,
        "positive_class": int(positive_class),
    }
    if dino_checkpoint:
        summary["dino_checkpoint"] = dino_checkpoint
    if probe_checkpoint:
        summary["probe_checkpoint"] = probe_checkpoint
    if extra_fields:
        summary.update(extra_fields)

    json_path = out_dir / f"eval_metrics_{split}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["_paths"] = {
        "confusion_png": str(cm_png),
        "roc_png": str(roc_png),
        "pr_png": str(pr_png),
        "metrics_json": str(json_path),
    }
    return summary


@torch.no_grad()
def collect_slice_positive_probs_with_meta(
    probe: "nn.Module",
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    max_steps: int | None,
    positive_class: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-slice positive-class probabilities with canonical_patient_id (matches supervised collate `meta`)."""
    probe.eval()
    amp_on = amp and device.type == "cuda"
    pc = int(positive_class)
    y_true: list[int] = []
    y_score: list[float] = []
    pids: list[str] = []
    for step_idx, batch in enumerate(tqdm(loader, desc="patient_agg_slices", leave=False, dynamic_ncols=True)):
        if max_steps is not None and step_idx >= max_steps:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        meta = batch.get("meta") or []
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = probe(x)
            prob = F.softmax(logits.float(), dim=1)[:, pc]
        y_true.extend(y.detach().cpu().tolist())
        y_score.extend(prob.detach().cpu().tolist())
        for i in range(y.numel()):
            pid = str(meta[i]["canonical_patient_id"]) if i < len(meta) else ""
            pids.append(pid)
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_score, dtype=np.float64),
        pids,
    )


def aggregate_mean_prob_per_patient(
    y_slice: np.ndarray, score_slice: np.ndarray, patient_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """One score per patient: mean slice probability; labels must agree per patient."""
    from collections import defaultdict

    by_pid: dict[str, dict] = defaultdict(lambda: {"scores": [], "y": None})
    for i in range(len(score_slice)):
        pid = patient_ids[i]
        by_pid[pid]["scores"].append(float(score_slice[i]))
        lab = int(y_slice[i])
        if by_pid[pid]["y"] is None:
            by_pid[pid]["y"] = lab
        elif by_pid[pid]["y"] != lab:
            raise ValueError(f"Inconsistent labels for patient {pid}")
    scores: list[float] = []
    y_pat: list[int] = []
    for pid in sorted(by_pid.keys()):
        arr = np.array(by_pid[pid]["scores"], dtype=np.float64)
        scores.append(float(arr.mean()))
        y_pat.append(int(by_pid[pid]["y"]))
    return np.asarray(y_pat, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def best_threshold_balanced_accuracy_binary(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if balanced_accuracy_score is None:
        raise ImportError("scikit-learn required")
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
        bal = float(balanced_accuracy_score(y_true, pred))
        if bal > best_bal:
            best_bal = bal
            best_t = float(t)
    return best_t, best_bal


def patient_level_auroc_and_balanced_acc(
    y_pat: np.ndarray, score_pat: np.ndarray, *, threshold: float
) -> dict[str, float | int]:
    if roc_auc_score is None or balanced_accuracy_score is None:
        raise ImportError("scikit-learn required")
    y_pat = np.asarray(y_pat, dtype=np.int64)
    score_pat = np.asarray(score_pat, dtype=np.float64)
    if len(y_pat) == 0:
        return {"auroc": float("nan"), "balanced_accuracy": float("nan"), "n_patients": 0}
    pred = (score_pat >= float(threshold)).astype(np.int64)
    auroc = float(roc_auc_score(y_pat, score_pat)) if len(np.unique(y_pat)) >= 2 else float("nan")
    bal = float(balanced_accuracy_score(y_pat, pred))
    return {"auroc": auroc, "balanced_accuracy": bal, "n_patients": int(len(y_pat))}


def summarize_patient_level_mean_probs_val_and_test(
    probe: "nn.Module",
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    *,
    device: torch.device,
    amp: bool,
    positive_class: int,
    max_steps_val: int | None,
    max_steps_test: int | None,
) -> dict[str, object]:
    """Mean aggregate slice probs → patient; tune balanced-acc threshold on val; report val + optional test."""
    ys_v, pr_v, pid_v = collect_slice_positive_probs_with_meta(
        probe, val_loader, device, amp=amp, max_steps=max_steps_val, positive_class=positive_class
    )
    yv_pat, pv_pat = aggregate_mean_prob_per_patient(ys_v, pr_v, pid_v)
    tau, bal_tau_val = best_threshold_balanced_accuracy_binary(yv_pat, pv_pat)
    val05 = patient_level_auroc_and_balanced_acc(yv_pat, pv_pat, threshold=0.5)
    val_tau = patient_level_auroc_and_balanced_acc(yv_pat, pv_pat, threshold=tau)
    out: dict[str, object] = {
        "aggregation": "mean_slice_probability_per_patient",
        "val": {
            **val05,
            "balanced_accuracy@threshold_tuned_on_val": val_tau["balanced_accuracy"],
            "threshold_tuned_on_val": tau,
            "note_threshold_fit_on_val": "threshold maximizes val balanced accuracy (same split)",
        },
        "threshold_tuning": {
            "threshold_fit_on": "val",
            "objective": "balanced_accuracy",
            "threshold_value": tau,
            "val_balanced_accuracy_at_threshold": bal_tau_val,
        },
    }
    if test_loader is not None:
        ys_t, pr_t, pid_t = collect_slice_positive_probs_with_meta(
            probe, test_loader, device, amp=amp, max_steps=max_steps_test, positive_class=positive_class
        )
        yt_pat, pt_pat = aggregate_mean_prob_per_patient(ys_t, pr_t, pid_t)
        te05 = patient_level_auroc_and_balanced_acc(yt_pat, pt_pat, threshold=0.5)
        te_tau = patient_level_auroc_and_balanced_acc(yt_pat, pt_pat, threshold=tau)
        out["test"] = {
            **te05,
            "balanced_accuracy@threshold_from_val": te_tau["balanced_accuracy"],
            "threshold_from_val": tau,
        }
    return out
