"""Narrow adapters from the isolated benchmark to repository code."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from additive_planner import action_to_bits
from config import ExperimentConfig
from data import BinaryCIFAR10, build_transforms
from model import load_binary_resnet18_checkpoint
from quantization import apply_fake_quantization_to_module


def load_reference_model(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load the unchanged FP32 reference model through the repository loader."""

    # TODO(go-no-go): Keep this as a narrow call-through if the repository's
    # checkpoint loader signature changes; do not reimplement checkpoint I/O.
    model, _ = load_binary_resnet18_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return model


def build_binary_eval_dataset(
    source_split: str,
) -> Dataset:
    """Build a binary train or test pool with no random augmentation."""

    normalized_split = source_split.strip().lower()

    if normalized_split not in {"train", "test"}:
        raise ValueError(
            "source_split must be either 'train' or 'test'."
        )

    config = ExperimentConfig()
    _, evaluation_transform = build_transforms(
        config.image_size
    )

    # TODO(go-no-go): Continue using BinaryCIFAR10 and build_transforms rather
    # than creating benchmark-local class filtering or normalization logic.
    return BinaryCIFAR10(
        root=str(config.data_dir),
        train=normalized_split == "train",
        class_ids=config.class_ids,
        transform=evaluation_transform,
        source_indices=None,
        download=False,
    )


def apply_existing_quantization_inplace(
    module: nn.Module,
    action_name: str,
) -> None:
    """Apply the repository's current fake quantizer to one copied module."""

    # TODO(go-no-go): Keep action parsing synchronized through action_to_bits;
    # never add a benchmark-local quantization implementation here.
    bits = action_to_bits(
        action_name.strip().lower()
    )

    apply_fake_quantization_to_module(
        module=module,
        bits=bits,
    )


def get_checkpoint_default_path() -> Path:
    """Return the checkpoint path already defined by ExperimentConfig."""

    # TODO(go-no-go): Continue deriving this default from ExperimentConfig so
    # the benchmark does not establish a second checkpoint convention.
    return ExperimentConfig().checkpoint_path
