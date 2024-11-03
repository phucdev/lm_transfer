from transformers import TrainingArguments
from dataclasses import dataclass


@dataclass
class CustomTrainingArguments(TrainingArguments):
    early_stopping_patience: int = -1
    early_stopping_threshold: float = 0.0
