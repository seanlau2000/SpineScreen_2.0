from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

try:
    import msoffcrypto
except Exception:  # pragma: no cover
    msoffcrypto = None


def load_excel_decrypt(path: str, password: str) -> pd.DataFrame:
    if msoffcrypto is None:
        raise ImportError("Install: pip install msoffcrypto-tool")
    file = msoffcrypto.OfficeFile(open(path, "rb"))
    decrypted = io.BytesIO()
    file.load_key(password=password)
    file.decrypt(decrypted)
    decrypted.seek(0)
    return pd.read_excel(decrypted, engine="openpyxl")


def extract_impression(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    idx = s.lower().find("impression:")
    if idx != -1:
        return s[idx + len("impression:") :].strip()
    return s


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    remove_duplicate_col: str,
    remove_duplicate_value: str,
    mri_imaging_col: str,
    mri_imaging_required_value: int,
    excluded_history_col: str,
    excluded_history_required_value: int,
    exclusion_followup_col: str,
    exclusion_followup_required_value: int,
    positive_label: str = "Y",
    negative_label: str = "N",
) -> pd.DataFrame:
    df = df.copy()

    # Filters
    if remove_duplicate_col in df.columns:
        df = df[df[remove_duplicate_col] != remove_duplicate_value]
    if mri_imaging_col in df.columns:
        df = df[df[mri_imaging_col] == mri_imaging_required_value]
    if excluded_history_col in df.columns:
        df = df[df[excluded_history_col] == excluded_history_required_value]
    if exclusion_followup_col in df.columns:
        df = df[df[exclusion_followup_col] == exclusion_followup_required_value]

    df = df[[text_column, label_column]]

    # Normalize 'n' -> 'N' (your script)
    df[label_column] = df[label_column].astype(str).str.replace("n", "N", regex=False)

    # Keep Y/N only
    df = df[df[label_column].isin([positive_label, negative_label])]

    # Drop missing text
    df = df.dropna(subset=[text_column])

    # Map labels
    mapping = {positive_label: 1, negative_label: 0}
    df[label_column] = df[label_column].map(mapping).astype(int)

    # Text cleanup
    df[text_column] = df[text_column].apply(extract_impression)
    df[text_column] = df[text_column].apply(lambda x: str(x).lower().replace("\n", " "))
    df[text_column] = df[text_column].str.replace("\xa0", " ")

    return df.reset_index(drop=True)


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(tokenizer, texts: List[str], max_length: int) -> Dict[str, List[int]]:
    texts = [str(t) for t in texts if t is not None]
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length)


class PatientScreeningDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)  # scalar label
        return item

    def __len__(self) -> int:
        return len(self.labels)


def compute_class_weights_from_labels(labels: List[int]) -> torch.Tensor:
    y = np.asarray(labels, dtype=int)
    classes = np.array([0, 1], dtype=int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float)

