from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_size: list
    num_classes: int
    pretrained: bool
    checkpoint: Path


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    epochs: int
    batch_size: int
    model_save_path: Path
    dataset_path: Path
