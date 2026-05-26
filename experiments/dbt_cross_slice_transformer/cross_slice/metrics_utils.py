"""Patient metrics / threshold tuning aligned with baseline_patient_pool_embeddings_dbt.py."""

from __future__ import annotations

from typing import Any

import numpy as np


def best_threshold_balanced_accuracy(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
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
            best_bal = float(bal)
            best_t = float(t)
    return best_t, float(best_bal)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "labels_row_true_col_pred": [0, 1],
        "matrix": cm.tolist(),
        "note": "sklearn confusion_matrix: rows=true class, cols=predicted class",
    }


def patient_prob_metrics_dict(y_pat: np.ndarray, p_pat: np.ndarray, *, threshold: float) -> dict[str, float | int]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y_pat = np.asarray(y_pat, dtype=np.int64)
    p_pat = np.asarray(p_pat, dtype=np.float64)
    if len(y_pat) == 0:
        return {"auroc": float("nan"), "balanced_accuracy": float("nan"), "n_patients": 0}
    pred = (p_pat >= float(threshold)).astype(np.int64)
    auroc = float(roc_auc_score(y_pat, p_pat)) if len(np.unique(y_pat)) >= 2 else float("nan")
    bal = float(balanced_accuracy_score(y_pat, pred))
    return {"auroc": auroc, "balanced_accuracy": bal, "n_patients": int(len(y_pat))}


def finalize_threshold_outputs(
    out: dict[str, Any],
    *,
    patient_block: dict[str, Any],
    score_definition: str,
    y_val_pat: np.ndarray,
    p_val_pat: np.ndarray,
    y_test_pat: np.ndarray | None,
    p_test_pat: np.ndarray | None,
    run_test: bool,
    n_test_slices: int,
) -> None:
    tau, bal_at_tau_val = best_threshold_balanced_accuracy(y_val_pat, p_val_pat)
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
    patient_block["confusion_matrix_patient_val_at_0.5"] = confusion_matrix_binary(y_val_pat, pred_val_05)
    patient_block["confusion_matrix_patient_val_at_threshold"] = confusion_matrix_binary(y_val_pat, pred_val_tau)
    if run_test and y_test_pat is not None and p_test_pat is not None and len(y_test_pat):
        m05 = patient_prob_metrics_dict(y_test_pat, p_test_pat, threshold=0.5)
        mtau = patient_prob_metrics_dict(y_test_pat, p_test_pat, threshold=tau)
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
            "confusion_matrix_patient_test_at_0.5": confusion_matrix_binary(y_test_pat, pred_te_05),
            "confusion_matrix_patient_test_at_threshold_from_val": confusion_matrix_binary(
                y_test_pat, pred_te_tau
            ),
        }
