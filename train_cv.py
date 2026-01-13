#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gc
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.nn import CrossEntropyLoss

from screening_bert.config import Config
from screening_bert.data import (
    load_excel_decrypt,
    preprocess_dataframe,
    build_tokenizer,
    tokenize_texts,
    PatientScreeningDataset,
    compute_class_weights_from_labels,
)
from screening_bert.model import build_model
from screening_bert.metrics import (
    compute_metrics_hf_auc_acc,
    optimize_thresholds_by_beta,
    roc_pr_fold_curves,
    bootstrap_ci_mean,
    bootstrap_auc_ci,
    aggregate_roc,
)
from screening_bert.utils import seed_everything, ensure_dir


class EvaluationCallback(TrainerCallback):
    """
    Reproduces your notebook callback: force evaluate every eval_steps by toggling should_evaluate on log.
    """
    def __init__(self, eval_steps: int):
        self.eval_steps = eval_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            control.should_evaluate = True


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("Labels are missing from inputs!")
        labels = labels.view(-1)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights if self.class_weights is not None else None)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def parse_args():
    p = argparse.ArgumentParser(description="10-fold CV training + threshold optimization + ROC aggregation.")
    cfg = Config()

    p.add_argument("--excel_path", type=str, default=cfg.excel_path)
    p.add_argument("--excel_password", type=str, default=cfg.excel_password)

    p.add_argument("--model_name", type=str, default=cfg.model_name)
    p.add_argument("--max_length", type=int, default=cfg.max_length)
    p.add_argument("--seed", type=int, default=cfg.seed)

    p.add_argument("--n_splits", type=int, default=cfg.n_splits)
    p.add_argument("--output_dir", type=str, default="./cv_outputs")

    # training hyperparams (default to your tuned values)
    p.add_argument("--learning_rate", type=float, default=cfg.learning_rate)
    p.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    p.add_argument("--train_batch_size", type=int, default=cfg.train_batch_size)
    p.add_argument("--eval_batch_size", type=int, default=cfg.eval_batch_size)
    p.add_argument("--num_train_epochs", type=float, default=cfg.num_train_epochs)

    # step eval
    p.add_argument("--eval_steps", type=int, default=cfg.eval_steps)
    p.add_argument("--logging_steps", type=int, default=cfg.logging_steps)
    p.add_argument("--save_steps", type=int, default=cfg.save_steps)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(seed=args.seed, model_name=args.model_name, max_length=args.max_length)

    seed_everything(cfg.seed)
    outdir = ensure_dir(args.output_dir)

    # Load and preprocess
    df_raw = load_excel_decrypt(args.excel_path, args.excel_password)
    df = preprocess_dataframe(
        df_raw,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        remove_duplicate_col=cfg.remove_duplicate_col,
        remove_duplicate_value=cfg.remove_duplicate_value,
        mri_imaging_col=cfg.mri_imaging_col,
        mri_imaging_required_value=cfg.mri_imaging_required_value,
        excluded_history_col=cfg.excluded_history_col,
        excluded_history_required_value=cfg.excluded_history_required_value,
        exclusion_followup_col=cfg.exclusion_followup_col,
        exclusion_followup_required_value=cfg.exclusion_followup_required_value,
        positive_label=cfg.positive_label,
        negative_label=cfg.negative_label,
    )

    documents = df[cfg.text_column].tolist()
    labels = df[cfg.label_column].astype(int).tolist()

    tokenizer = build_tokenizer(cfg.model_name)
    kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=cfg.seed)

    # Containers
    betas = list(cfg.betas)
    metrics_by_beta = {
        beta: {"thresholds": [], "sensitivity": [], "specificity": [], "ppv": [], "npv": [], "accuracy": [], "f1": []}
        for beta in betas
    }

    fprs_folds, tprs_folds, aucs = [], [], []

    mean_fpr = np.linspace(0, 1, 101)

    per_fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(documents, labels), start=1):
        print(f"\n===== Training fold {fold}/{args.n_splits} =====")

        train_texts = [documents[i] for i in train_idx]
        val_texts = [documents[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        # Tokenize + dataset
        train_enc = tokenize_texts(tokenizer, train_texts, cfg.max_length)
        val_enc = tokenize_texts(tokenizer, val_texts, cfg.max_length)

        train_ds = PatientScreeningDataset(train_enc, train_labels)
        val_ds = PatientScreeningDataset(val_enc, val_labels)

        # per-fold class weights from train labels
        class_weights = compute_class_weights_from_labels(train_labels)

        # Model
        model = build_model(cfg.model_name, num_labels=2)

        # HF args
        fold_dir = ensure_dir(outdir / f"fold_{fold}")
        training_args = TrainingArguments(
            output_dir=str(fold_dir / "hf_out"),
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            greater_is_better=True,
            seed=cfg.seed,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics_hf_auc_acc,
            class_weights=class_weights,
            callbacks=[EvaluationCallback(eval_steps=args.eval_steps)],
        )

        trainer.train()

        # Predict fold
        preds = trainer.predict(val_ds)
        logits = preds.predictions
        probs = softmax(logits, axis=1)[:, 1]
        y_true = np.asarray(preds.label_ids).astype(int)

        # Threshold optimization for each beta
        fold_beta_metrics = optimize_thresholds_by_beta(
            probabilities=probs,
            true_labels=y_true,
            betas=betas,
            threshold_step=cfg.threshold_grid_step,
        )

        for beta in betas:
            m = fold_beta_metrics[beta]
            metrics_by_beta[beta]["thresholds"].append(m["optimal_threshold"])
            metrics_by_beta[beta]["sensitivity"].append(m["sensitivity"])
            metrics_by_beta[beta]["specificity"].append(m["specificity"])
            metrics_by_beta[beta]["ppv"].append(m["ppv"])
            metrics_by_beta[beta]["npv"].append(m["npv"])
            metrics_by_beta[beta]["accuracy"].append(m["accuracy"])
            metrics_by_beta[beta]["f1"].append(m["f1"])

            per_fold_rows.append(
                {
                    "Beta": beta,
                    "Fold": fold,
                    "Optimal_Threshold": m["optimal_threshold"],
                    "Sensitivity": m["sensitivity"],
                    "Specificity": m["specificity"],
                    "PPV": m["ppv"],
                    "NPV": m["npv"],
                    "Accuracy": m["accuracy"],
                    "F1": m["f1"],
                }
            )
            print(f"Fold {fold} | β={beta} | Optimal Threshold={m['optimal_threshold']:.2f} | F1={m['f1']:.4f}")

        # ROC fold curve + AUC
        (fpr, tpr, roc_auc), _ = roc_pr_fold_curves(y_true, probs)
        fprs_folds.append(fpr)
        tprs_folds.append(tpr)
        aucs.append(roc_auc)

        # Clean up GPU memory like your notebook
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ---- Save per-fold table ----
    model_tag = Path(cfg.model_name).name.replace("/", "_")
    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_path = outdir / f"{model_tag}_per_fold_metrics_by_beta_F1.csv"
    per_fold_df.to_csv(per_fold_path, index=False)
    print(f"\nSaved per-fold metrics to {per_fold_path}")

    # ---- Bootstrap CI summary per beta ----
    summary_rows = []
    for beta in betas:
        m = metrics_by_beta[beta]
        B = cfg.bootstrap_B
        seed = cfg.bootstrap_seed

        thr_m, thr_s, thr_l, thr_u = bootstrap_ci_mean(m["thresholds"], B=B, seed=seed)
        sens_m, sens_s, sens_l, sens_u = bootstrap_ci_mean(m["sensitivity"], B=B, seed=seed)
        spec_m, spec_s, spec_l, spec_u = bootstrap_ci_mean(m["specificity"], B=B, seed=seed)
        ppv_m, ppv_s, ppv_l, ppv_u = bootstrap_ci_mean(m["ppv"], B=B, seed=seed)
        npv_m, npv_s, npv_l, npv_u = bootstrap_ci_mean(m["npv"], B=B, seed=seed)
        acc_m, acc_s, acc_l, acc_u = bootstrap_ci_mean(m["accuracy"], B=B, seed=seed)
        f1_m, f1_s, f1_l, f1_u = bootstrap_ci_mean(m["f1"], B=B, seed=seed)

        summary_rows.append(
            {
                "Beta": beta,
                "Optimal_Threshold_Mean": thr_m,
                "Optimal_Threshold_Std": thr_s,
                "Optimal_Threshold_CI_Lower": thr_l,
                "Optimal_Threshold_CI_Upper": thr_u,
                "Sensitivity_Mean": sens_m,
                "Sensitivity_Std": sens_s,
                "Sensitivity_CI_Lower": sens_l,
                "Sensitivity_CI_Upper": sens_u,
                "Specificity_Mean": spec_m,
                "Specificity_Std": spec_s,
                "Specificity_CI_Lower": spec_l,
                "Specificity_CI_Upper": spec_u,
                "PPV_Mean": ppv_m,
                "PPV_Std": ppv_s,
                "PPV_CI_Lower": ppv_l,
                "PPV_CI_Upper": ppv_u,
                "NPV_Mean": npv_m,
                "NPV_Std": npv_s,
                "NPV_CI_Lower": npv_l,
                "NPV_CI_Upper": npv_u,
                "Accuracy_Mean": acc_m,
                "Accuracy_Std": acc_s,
                "Accuracy_CI_Lower": acc_l,
                "Accuracy_CI_Upper": acc_u,
                "F1_Mean": f1_m,
                "F1_Std": f1_s,
                "F1_CI_Lower": f1_l,
                "F1_CI_Upper": f1_u,
            }
        )

    metrics_df = pd.DataFrame(summary_rows)
    bootstrap_path = outdir / f"{model_tag}_mean_performance_by_beta_F1_bootstrapCI.csv"
    metrics_df.to_csv(bootstrap_path, index=False)
    print(f"Saved bootstrap 95% CIs to {bootstrap_path}")

    # ---- Mean ROC curve + plot ----
    mean_tpr, std_tpr = aggregate_roc(fprs_folds, tprs_folds, mean_fpr=mean_fpr)
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    roc_df = pd.DataFrame({"fpr": mean_fpr, "tpr": mean_tpr})
    roc_csv_path = outdir / f"{model_tag}_mean_roc.csv"
    roc_df.to_csv(roc_csv_path, index=False)
    print(f"Saved mean ROC curve to {roc_csv_path} (AUC={mean_auc:.4f})")

    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, label="±1 Std. Dev.")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curve with Confidence Band")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    roc_png_path = outdir / f"{model_tag}_MEANROC_withConfidenceBand.png"
    plt.savefig(roc_png_path)
    plt.close()
    print(f"Saved ROC figure to {roc_png_path}")

    # ---- Fold-level AUC bootstrap CI ----
    mean_auc, std_auc, ci_lower, ci_upper = bootstrap_auc_ci(aucs, B=cfg.bootstrap_B, seed=42)
    auc_summary = {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "auc_ci_lower_95": ci_lower,
        "auc_ci_upper_95": ci_upper,
        "n_folds": len(aucs),
    }
    auc_json_path = outdir / f"{model_tag}_auc_bootstrap_summary.json"
    with open(auc_json_path, "w") as f:
        json.dump(auc_summary, f, indent=2)
    print(f"Saved AUC bootstrap summary to {auc_json_path}")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f} | 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]")


if __name__ == "__main__":
    main()
