from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SLMTrainConfig:
    base_model: str
    output_dir: str
    train_file: str
    validation_file: str
    max_seq_length: int = 768
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42

    @classmethod
    def load(cls, path: str | Path) -> "SLMTrainConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
