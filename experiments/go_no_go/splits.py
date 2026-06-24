"""Deterministic, disjoint ranking and holdout evaluation splits."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class EvaluationSplitLoaders:
    """Loaders and source indices for the two benchmark partitions."""

    ranking: DataLoader
    holdout: DataLoader
    ranking_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]


def build_disjoint_eval_loaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    ranking_fraction: float = 0.5,
    max_samples: int | None = None,
) -> EvaluationSplitLoaders:
    """Partition an evaluation dataset reproducibly without overlap."""

    if not 0.0 < ranking_fraction < 1.0:
        raise ValueError(
            "ranking_fraction must be strictly between 0 and 1."
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    dataset_size = len(dataset)

    if dataset_size < 2:
        raise ValueError(
            "At least two evaluation samples are required."
        )

    if max_samples is not None and max_samples < 2:
        raise ValueError(
            "max_samples must be at least 2 when provided."
        )

    selected_size = (
        dataset_size
        if max_samples is None
        else min(dataset_size, max_samples)
    )

    generator = torch.Generator().manual_seed(seed)
    selected_indices = torch.randperm(
        dataset_size,
        generator=generator,
    )[:selected_size].tolist()

    ranking_size = int(
        selected_size * ranking_fraction
    )
    ranking_size = max(
        1,
        min(ranking_size, selected_size - 1),
    )

    ranking_indices = tuple(
        selected_indices[:ranking_size]
    )
    holdout_indices = tuple(
        selected_indices[ranking_size:]
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    ranking_loader = DataLoader(
        Subset(dataset, ranking_indices),
        shuffle=False,
        **loader_kwargs,
    )
    holdout_loader = DataLoader(
        Subset(dataset, holdout_indices),
        shuffle=False,
        **loader_kwargs,
    )

    return EvaluationSplitLoaders(
        ranking=ranking_loader,
        holdout=holdout_loader,
        ranking_indices=ranking_indices,
        holdout_indices=holdout_indices,
    )

