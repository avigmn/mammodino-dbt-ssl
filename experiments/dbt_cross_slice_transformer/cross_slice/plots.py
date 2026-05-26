"""Save PNG plots under each run directory (matplotlib Agg backend)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _safe_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves_from_jsonl(jsonl_path: Path, out_path: Path, *, title: str | None = None) -> bool:
    """Plot epoch curves from ``metrics_epoch.jsonl``. Returns False if file missing or empty."""
    if not jsonl_path.is_file():
        return False
    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return False

    def _flt(row: dict[str, Any], key: str, alt: str | None = None) -> float:
        v = row.get(key)
        if v is None and alt:
            v = row.get(alt)
        if v is None:
            return float("nan")
        return float(v)

    epochs = [int(r["epoch"]) for r in rows]
    verbose = bool(rows[0].get("verbose_epoch_metrics"))

    plt = _safe_import_matplotlib()

    if not verbose:
        train_loss = [_flt(r, "train_loss") for r in rows]
        val_auroc = [_flt(r, "val_auroc") for r in rows]
        val_bal = [_flt(r, "val_bal_acc05") for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        if title:
            fig.suptitle(title)
        axes[0].plot(epochs, train_loss, marker="o", markersize=2)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("train CE loss (batch mean)")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, val_auroc, color="C1", marker="o", markersize=2)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("val AUROC")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(epochs, val_bal, color="C2", marker="o", markersize=2)
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("val balanced acc @0.5")
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].grid(True, alpha=0.3)
    else:
        train_loss = [_flt(r, "train_loss_batches_mean") for r in rows]
        train_ce_e = [_flt(r, "train_ce_eval_mean") for r in rows]
        val_ce_e = [_flt(r, "val_ce_eval_mean") for r in rows]
        train_auroc = [_flt(r, "train_auroc") for r in rows]
        val_auroc = [_flt(r, "val_auroc") for r in rows]
        train_bal = [_flt(r, "train_bal_acc05") for r in rows]
        val_bal = [_flt(r, "val_bal_acc05") for r in rows]
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        if title:
            fig.suptitle(title)
        axes[0, 0].plot(epochs, train_loss, marker="o", markersize=2, label="train (batch mean)")
        axes[0, 0].set_xlabel("epoch")
        axes[0, 0].set_ylabel("CE loss")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(epochs, train_ce_e, marker="o", markersize=2, label="train CE eval")
        axes[0, 1].plot(epochs, val_ce_e, marker="o", markersize=2, label="val CE eval")
        axes[0, 1].set_xlabel("epoch")
        axes[0, 1].set_ylabel("mean CE (eval)")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(epochs, train_auroc, marker="o", markersize=2, label="train")
        axes[1, 0].plot(epochs, val_auroc, marker="o", markersize=2, label="val")
        axes[1, 0].set_xlabel("epoch")
        axes[1, 0].set_ylabel("AUROC")
        axes[1, 0].set_ylim(-0.02, 1.02)
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(epochs, train_bal, marker="o", markersize=2, label="train bal@0.5")
        axes[1, 1].plot(epochs, val_bal, marker="o", markersize=2, label="val bal@0.5")
        axes[1, 1].set_xlabel("epoch")
        axes[1, 1].set_ylabel("balanced accuracy @0.5")
        axes[1, 1].set_ylim(-0.02, 1.02)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        train_f1 = [_flt(r, "train_f1_pos") for r in rows]
        val_f1 = [_flt(r, "val_f1_pos") for r in rows]
        axes[2, 0].plot(epochs, train_f1, marker="o", markersize=2, label="train F1 pos")
        axes[2, 0].plot(epochs, val_f1, marker="o", markersize=2, label="val F1 pos")
        axes[2, 0].set_xlabel("epoch")
        axes[2, 0].set_ylabel("F1 (class 1)")
        axes[2, 0].set_ylim(-0.02, 1.02)
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(True, alpha=0.3)
        train_pp = [_flt(r, "train_pct_pred_positive") for r in rows]
        val_pp = [_flt(r, "val_pct_pred_positive") for r in rows]
        axes[2, 1].plot(epochs, train_pp, marker="o", markersize=2, label="train %pred pos")
        axes[2, 1].plot(epochs, val_pp, marker="o", markersize=2, label="val %pred pos")
        axes[2, 1].set_xlabel("epoch")
        axes[2, 1].set_ylabel("fraction pred positive @0.5")
        axes[2, 1].set_ylim(-0.02, 1.02)
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_binary_confusion(cm_dict: dict[str, Any], out_path: Path, *, title: str) -> None:
    """``cm_dict`` has sklearn-style matrix list under ``matrix``."""
    plt = _safe_import_matplotlib()
    mat = np.asarray(cm_dict["matrix"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(mat.shape[1]),
        yticks=np.arange(mat.shape[0]),
        xticklabels=["pred 0", "pred 1"],
        yticklabels=["true 0", "true 1"],
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    thresh = mat.max() / 2.0 if mat.size else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(int(mat[i, j]), "d"), ha="center", va="center", color="w" if mat[i, j] > thresh else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_binary(y_true: np.ndarray, scores: np.ndarray, out_path: Path, *, title: str) -> bool:
    from sklearn.metrics import auc, roc_curve

    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return False
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt = _safe_import_matplotlib()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_all_from_metrics_json(metrics: dict[str, Any], plots_dir: Path, *, prefix: str = "") -> list[str]:
    """Generate ROC + confusion PNGs from final metrics dict (after threshold tuning)."""
    written: list[str] = []
    pf = f"{prefix}_" if prefix else ""

    patient = metrics.get("patient_level") or {}
    pte = metrics.get("patient_level_test")

    # Val confusion matrices stored on patient_level
    if isinstance(patient, dict):
        if cm := patient.get("confusion_matrix_patient_val_at_0.5"):
            png_path = plots_dir / f"{pf}confusion_val_at_0.5.png"
            plot_binary_confusion(cm, png_path, title="Val confusion @0.5")
            written.append(str(png_path))
        if cm := patient.get("confusion_matrix_patient_val_at_threshold"):
            png_path = plots_dir / f"{pf}confusion_val_at_threshold.png"
            plot_binary_confusion(cm, png_path, title="Val confusion @val-tuned threshold")
            written.append(str(png_path))

    if isinstance(pte, dict):
        if cm := pte.get("confusion_matrix_patient_test_at_0.5"):
            png_path = plots_dir / f"{pf}confusion_test_at_0.5.png"
            plot_binary_confusion(cm, png_path, title="Test confusion @0.5")
            written.append(str(png_path))
        if cm := pte.get("confusion_matrix_patient_test_at_threshold_from_val"):
            png_path = plots_dir / f"{pf}confusion_test_at_threshold.png"
            plot_binary_confusion(cm, png_path, title="Test confusion @threshold val→test")
            written.append(str(png_path))

    return written


def plot_calibration_histogram(
    y_true: np.ndarray, scores: np.ndarray, out_path: Path, *, title: str, bins: int = 20
) -> bool:
    """Score distributions stratified by class (informal calibration view)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(y_true) == 0:
        return False
    plt = _safe_import_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[y_true == 0], bins=bins, alpha=0.6, label="class 0", density=True)
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.6, label="class 1", density=True)
    ax.set_xlabel("predicted p(class 1)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def save_cross_slice_run_plots(
    *,
    run_dir: Path,
    metrics_dict: dict[str, Any],
    run_name: str,
    val_y: np.ndarray,
    val_scores: np.ndarray,
    test_y: np.ndarray | None,
    test_scores: np.ndarray | None,
    ran_test: bool,
    metrics_epoch_jsonl: Path,
    warn_print,
) -> dict[str, Any]:
    """Generate standard PNGs under ``run_dir/plots/``. Uses ``warn_print`` e.g. ``tqdm.write``."""
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"plots_dir": str(plots_dir), "files": []}

    run_resolved = run_dir.resolve()

    def rel_to_run(p: Path) -> str:
        return str(p.resolve().relative_to(run_resolved))

    try:
        _safe_import_matplotlib()
    except ImportError as e:
        warn_print(f"[WARN] matplotlib unavailable; skipping plots: {e}")
        return manifest

    title_prefix = f"{run_name} · cross-slice Transformer"

    try:
        tc = plots_dir / "training_curves.png"
        if plot_training_curves_from_jsonl(metrics_epoch_jsonl, tc, title=title_prefix):
            manifest["files"].append(rel_to_run(tc))

        rv = plots_dir / "roc_patient_val.png"
        if plot_roc_binary(val_y, val_scores, rv, title=f"{title_prefix} · val"):
            manifest["files"].append(rel_to_run(rv))

        hv = plots_dir / "score_hist_patient_val.png"
        if plot_calibration_histogram(val_y, val_scores, hv, title=f"{title_prefix} · val p(class=1)"):
            manifest["files"].append(rel_to_run(hv))

        if ran_test and test_y is not None and test_scores is not None and len(test_y):
            rt = plots_dir / "roc_patient_test.png"
            if plot_roc_binary(test_y, test_scores, rt, title=f"{title_prefix} · test"):
                manifest["files"].append(rel_to_run(rt))
            ht = plots_dir / "score_hist_patient_test.png"
            if plot_calibration_histogram(test_y, test_scores, ht, title=f"{title_prefix} · test p(class=1)"):
                manifest["files"].append(rel_to_run(ht))

        cms = plot_all_from_metrics_json(metrics_dict, plots_dir)
        for x in cms:
            manifest["files"].append(rel_to_run(Path(x)))

        manifest["files"] = sorted(set(manifest["files"]))
        idx_path = plots_dir / "index.json"
        idx_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["index_json"] = rel_to_run(idx_path)
    except Exception as e:
        warn_print(f"[WARN] plot generation failed: {type(e).__name__}: {e}")

    return manifest
