"""Benchmark metrics built on the repository's model comparison routine."""

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics import compare_binary_models


def evaluate_model_pair(
    reference_model,
    candidate_model,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Delegate numerical evaluation to the existing project metric code."""

    return compare_binary_models(
        fp32_model=reference_model,
        quantized_model=candidate_model,
        dataloader=dataloader,
        device=device,
    )


def spearman_rank_correlation(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Compute tie-aware Spearman correlation without a SciPy dependency."""

    if len(first) != len(second):
        raise ValueError(
            "Rank-correlation inputs must have the same length."
        )

    if len(first) < 2:
        return math.nan

    first_ranks = pd.Series(first).rank(
        method="average"
    ).to_numpy(dtype=float)
    second_ranks = pd.Series(second).rank(
        method="average"
    ).to_numpy(dtype=float)

    if (
        np.ptp(first_ranks) == 0
        or np.ptp(second_ranks) == 0
    ):
        return math.nan

    return float(
        np.corrcoef(first_ranks, second_ranks)[0, 1]
    )


def ranked_labels(
    labels: Sequence[str],
    values: Sequence[float],
) -> list[str]:
    """Return labels from largest (highest risk) to smallest value."""

    if len(labels) != len(values):
        raise ValueError(
            "Labels and ranking values must have the same length."
        )

    pairs = list(zip(labels, values))
    pairs.sort(
        key=lambda pair: pair[1],
        reverse=True,
    )

    return [label for label, _ in pairs]


def top_k_overlap_rate(
    first_ranking: Sequence[str],
    second_ranking: Sequence[str],
    top_k: int,
) -> float:
    """Return the fraction of top-k labels shared by two rankings."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    effective_k = min(
        top_k,
        len(first_ranking),
        len(second_ranking),
    )

    if effective_k == 0:
        return math.nan

    overlap = set(
        first_ranking[:effective_k]
    ) & set(second_ranking[:effective_k])

    return len(overlap) / effective_k

