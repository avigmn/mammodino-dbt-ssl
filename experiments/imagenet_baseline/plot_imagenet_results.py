#!/usr/bin/env python3
"""Generate plots from ImageNet MIL baseline metrics.json."""

from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = Path("/mnt/data/avi/imagenet_mil_runs/metrics.json")
OUT_DIR = Path("/mnt/data/avi/imagenet_mil_runs")

def main() -> None:
    data = json.loads(METRICS.read_text())
    test = data.get("test", {})
    auc = test.get("auroc", float("nan"))

    # MIL training loss curve
    history = data.get("mil_train_loss_history", [])
    if history:
        epochs = list(range(1, len(history) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history, marker="o", markersize=3, color="darkorange", label="MIL Train Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("MIL Attention Pool Training — ImageNet Baseline")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "mil_training_curves.png", dpi=150)
        plt.close()
        print("mil_training_curves.png saved")

    # ROC curve
    fpr = test.get("fpr")
    tpr = test.get("tpr")
    if fpr and tpr:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"ImageNet ViT-tiny (AUC={auc:.3f})", color="darkorange")
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — ImageNet Baseline (Test Set)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "roc_curve_test.png", dpi=150)
        plt.close()
        print("roc_curve_test.png saved")

    # Confusion matrix
    cm_data = test.get("confusion_matrix_at_0.5", {})
    cm = cm_data.get("matrix")
    if cm:
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Oranges")
        plt.colorbar(im, ax=ax)
        classes = ["Negative", "Positive"]
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(classes); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — ImageNet Baseline (Test Set)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "confusion_matrix_test.png", dpi=150)
        plt.close()
        print("confusion_matrix_test.png saved")

    # Summary print
    print(f"\n=== ImageNet ViT-tiny + MIL Baseline ===")
    print(f"Val  AUROC: {data.get('val', {}).get('auroc', 'N/A'):.3f}")
    print(f"Test AUROC: {auc:.3f}")
    print(f"Test Balanced Acc @0.5:         {test.get('balanced_accuracy', 'N/A'):.3f}")
    print(f"Test Balanced Acc @val threshold: {test.get('balanced_accuracy@threshold_from_val', 'N/A'):.3f}")

if __name__ == "__main__":
    main()
