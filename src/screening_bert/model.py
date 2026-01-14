from __future__ import annotations

from transformers import AutoModelForSequenceClassification


def build_model(model_name: str, num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

