#!/usr/bin/env python
from __future__ import annotations

import argparse

import optuna
import torch
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split

from screening_bert.config import Config
from screening_bert.data import (
    load_excel_decrypt,
    preprocess_dataframe,
    build_tokenizer,
    tokenize_texts,
    PatientScreeningDataset,
    compute_class_weights_from_labels,
)
from screening_bert.metrics import compute_metrics_hf_auc_acc
from screening_bert.model import build_model
from screening_bert.utils import seed_everything


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
    cfg = Config()
    p = argparse.ArgumentParser(description="Optuna hyperparameter optimization.")
    p.add_argument("--excel_path", type=str, default=cfg.excel_path)
    p.add_argument("--excel_password", type=str, default=cfg.excel_password)
    p.add_argument("--model_name", type=str, default=cfg.model_name)
    p.add_argument("--max_length", type=int, default=cfg.max_length)
    p.add_argument("--seed", type=int, default=cfg.seed)

    p.add_argument("--output_dir", type=str, default="./optuna_results")
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--valid_frac", type=float, default=0.1)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(seed=args.seed, model_name=args.model_name, max_length=args.max_length)
    seed_everything(cfg.seed)

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

    texts = df[cfg.text_column].tolist()
    labels = df[cfg.label_column].tolist()

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=args.valid_frac, random_state=cfg.seed, stratify=labels
    )

    tokenizer = build_tokenizer(cfg.model_name)
    train_enc = tokenize_texts(tokenizer, train_texts, cfg.max_length)
    valid_enc = tokenize_texts(tokenizer, valid_texts, cfg.max_length)

    train_ds = PatientScreeningDataset(train_enc, train_labels)
    valid_ds = PatientScreeningDataset(valid_enc, valid_labels)

    class_weights = compute_class_weights_from_labels(train_labels)

    def _model_init():
        return build_model(cfg.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="auc",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        seed=cfg.seed,
    )

    trainer = CustomTrainer(
        class_weights=class_weights,
        model_init=_model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics_hf_auc_acc,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    def hp_space(trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("batch_size", [4]),
        }

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        n_trials=args.n_trials,
        hp_space=hp_space,
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        compute_objective=lambda metrics: metrics["eval_auc"],
        pruner=optuna.pruners.MedianPruner(),
    )

    print("Best trial:")
    print(best_trial)


if __name__ == "__main__":
    main()

