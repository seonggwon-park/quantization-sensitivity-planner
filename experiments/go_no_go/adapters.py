"""Narrow adapters from the isolated benchmark to repository code."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from additive_planner import action_to_bits
from config import ExperimentConfig
from data import build_dataloaders
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


def build_binary_eval_dataset() -> Dataset:
    """Return the test dataset built by the existing data pipeline."""

    # TODO(go-no-go): If evaluation configuration becomes injectable, thread
    # it through here while continuing to call the repository data builder.
    dataloaders = build_dataloaders(
        ExperimentConfig()
    )

    return dataloaders["test"].dataset


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

