from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    # -----------------------------
    # Directory settings
    # -----------------------------
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    result_dir: Path = Path("results")
    figure_dir: Path = Path("results/figures")

    # -----------------------------
    # Binary CIFAR-10 task
    # CIFAR-10:
    # 0 airplane, 1 automobile, 3 cat, 5 dog
    # -----------------------------
    class_ids: tuple[int, int] = (0, 1)
    class_names: tuple[str, str] = ("airplane", "automobile")

    # -----------------------------
    # Data settings
    # -----------------------------
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 0
    validation_fraction: float = 0.1

    # -----------------------------
    # Training settings
    # -----------------------------
    seed: int = 42
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "resnet18_binary_best.pt"

    def create_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)