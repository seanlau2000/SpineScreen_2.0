from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 3

    # Model/tokenizer
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 512

    # Data columns (keep exact names)
    text_column: str = "dictation"
    label_column: str = 'outcome ( Y- offered surgery within 1 year of first appt”)'

    # Excel file (publication code should allow overriding via CLI)
    excel_path: str = "../../BERT_3500patientsv4_protected1.xlsx"
    excel_password: str = "pathologic"

    # Filtering columns/values
    remove_duplicate_col: str = "Remove this Duplicate?"
    remove_duplicate_value: str = "Y"

    mri_imaging_col: str = "MRI_imaging"
    mri_imaging_required_value: int = 1

    excluded_history_col: str = "Excluded based on Neuroimaging History"
    excluded_history_required_value: int = 0

    exclusion_followup_col: str = "Exclusion over Follow- up"
    exclusion_followup_required_value: int = 0

    # Labels
    positive_label: str = "Y"
    negative_label: str = "N"

    # CV defaults
    n_splits: int = 10

    # Your tuned hyperparams (from the CV script)
    learning_rate: float = 1.2381002553887929e-05
    weight_decay: float = 2.7893000092193426e-05
    train_batch_size: int = 4
    eval_batch_size: int = 4
    num_train_epochs: float = 3.0

    # Step-eval settings (from CV)
    eval_steps: int = 150
    logging_steps: int = 150
    save_steps: int = 150

    # Threshold search / betas
    betas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    threshold_grid_step: float = 0.01  # 0..1 inclusive by 0.01

    # Bootstrap
    bootstrap_B: int = 10000
    bootstrap_seed: int = 12345

