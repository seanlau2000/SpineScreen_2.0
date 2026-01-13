from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics_hf_auc_acc(eval_pred) -> Dict[str, float]:
    """
    HF Trainer-compatible compute_metrics: AUC + accuracy.
    Mirrors your compute_metrics in the CV script.
    """
    logits = eval_pred.predictions
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    logits_t = torch.tensor(logits)
    probs = torch.softmax(logits_t, dim=1).cpu().numpy()

    labels = np.array(eval_pred.label_ids).astype(int).flatten()

    if len(np.unique(labels)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(labels, probs[:, 1]))

    preds = np.argmax(probs, axis=-1)
    accuracy = float((preds == labels).mean())

    return {"auc": auc, "accuracy": accuracy}


def optimize_thresholds_by_beta(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    betas: Iterable[float],
    threshold_step: float = 0.01,
) -> Dict[float, Dict[str, float]]:
    """
    Reproduces your per-fold threshold search for each beta over [0,1] step=0.01.

    Returns dict:
      beta -> {optimal_threshold, sensitivity, specificity, ppv, npv, accuracy, f1}
    """
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    true_labels = np.asarray(true_labels, dtype=int).reshape(-1)

    thresholds = np.arange(0.0, 1.0 + 1e-9, threshold_step)

    out: Dict[float, Dict[str, float]] = {}

    for beta in betas:
        fbeta_scores: List[float] = []
        for thr in thresholds:
            preds_thr = (probabilities >= thr).astype(int)
            prec = precision_score(true_labels, preds_thr, zero_division=0)
            rec = recall_score(true_labels, preds_thr, zero_division=0)
            denom = (beta**2 * prec + rec)
            fbeta = 0.0 if denom == 0 else (1 + beta**2) * (prec * rec) / denom
            fbeta_scores.append(float(fbeta))

        optimal_idx = int(np.argmax(fbeta_scores))
        optimal_thr = float(thresholds[optimal_idx])

        preds_opt = (probabilities >= optimal_thr).astype(int)

        # confusion matrix may fail if only one class predicted; handle robustly
        cm = confusion_matrix(true_labels, preds_opt, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        accuracy = (TP + TN) / max((TP + TN + FP + FN), 1)
        f1 = float(f1_score(true_labels, preds_opt, zero_division=0))

        out[float(beta)] = {
            "optimal_threshold": optimal_thr,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
            "accuracy": float(accuracy),
            "f1": float(f1),
        }

    return out


def roc_pr_fold_curves(true_labels: np.ndarray, probabilities: np.ndarray):
    """
    Returns per-fold:
      roc: (fpr, tpr, auc)
      pr: (precision, recall, ap)
    """
    y = np.asarray(true_labels, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)

    fpr, tpr, _ = roc_curve(y, p)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5

    pr_precision, pr_recall, _ = precision_recall_curve(y, p)
    ap = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else 0.0

    return (fpr, tpr, auc), (pr_precision, pr_recall, ap)


def aggregate_roc(fprs: List[np.ndarray], tprs: List[np.ndarray], mean_fpr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate TPRs onto mean_fpr grid, return (mean_tpr, std_tpr).
    """
    tprs_interp = []
    for fpr, tpr in zip(fprs, tprs):
        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs_interp.append(tpr_i)

    mean_tpr = np.mean(tprs_interp, axis=0)
    std_tpr = np.std(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    return mean_tpr, std_tpr


def aggregate_pr(precisions: List[np.ndarray], recalls: List[np.ndarray], mean_recall: np.ndarray) -> np.ndarray:
    """
    Interpolate precision onto mean_recall grid using monotonic precision fix.
    Returns mean precision curve.
    """
    prec_interp_all = []
    for pr_precision, pr_recall in zip(precisions, recalls):
        pr_precision_mono = np.maximum.accumulate(pr_precision[::-1])[::-1]
        prec_i = np.interp(mean_recall, pr_recall, pr_precision_mono)
        prec_interp_all.append(prec_i)
    return np.mean(prec_interp_all, axis=0)


def bootstrap_ci_mean(values: Iterable[float], B: int = 10000, seed: int = 42):
    """
    Percentile bootstrap CI for the mean.
    Returns (mean, std, ci_lower, ci_upper).
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.shape[0]
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, n, size=(B, n))
    boot_means = arr[idx].mean(axis=1)
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    return mean, std, float(ci_lower), float(ci_upper)


def bootstrap_auc_ci(aucs: Iterable[float], B: int = 10000, seed: int = 42):
    """
    Mean AUC, std, and bootstrap 95% CI over fold-level AUCs.
    """
    arr = np.asarray(list(aucs), dtype=float)
    n = arr.shape[0]
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, n, size=(B, n))
    boot_means = arr[idx].mean(axis=1)

    mean_auc = float(arr.mean())
    std_auc = float(arr.std(ddof=0))
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    return mean_auc, std_auc, float(ci_lower), float(ci_upper)

