"""Benchmark metrics built on the repository's model comparison routine."""

from collections.abc import Sequence
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.inference_mode()
def binary_decision_metrics(
    base_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    if base_logits.ndim != 2 or candidate_logits.ndim != 2:
        raise ValueError("Expected logits with shape [N, 2].")

    if base_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"Shape mismatch: {base_logits.shape} vs {candidate_logits.shape}"
        )

    if base_logits.shape[1] != 2:
        raise ValueError(
            "This benchmark is binary-only. Expected logits with shape [N, 2]."
        )

    base_score = base_logits[:, 1] - base_logits[:, 0]
    candidate_score = candidate_logits[:, 1] - candidate_logits[:, 0]

    base_pred = base_logits.argmax(dim=1)
    candidate_pred = candidate_logits.argmax(dim=1)

    direction = torch.where(
        base_pred == 1,
        torch.ones_like(base_score),
        -torch.ones_like(base_score),
    )

    delta_score = candidate_score - base_score
    margin_erosion = (-direction * delta_score).clamp_min(0.0)
    margin = base_score.abs()
    decision_risk = margin_erosion / (margin + eps)
    flips = candidate_pred.ne(base_pred)

    base_prob = F.softmax(base_logits, dim=1)
    candidate_log_prob = F.log_softmax(candidate_logits, dim=1)
    output_kl = (
        base_prob
        * (base_prob.clamp_min(eps).log() - candidate_log_prob)
    ).sum(dim=1)

    return {
        "base_score": base_score,
        "candidate_score": candidate_score,
        "abs_delta_score": delta_score.abs(),
        "margin": margin,
        "margin_erosion": margin_erosion,
        "decision_risk": decision_risk,
        "flip": flips.float(),
        "output_kl": output_kl,
        "base_pred": base_pred,
        "candidate_pred": candidate_pred,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def weight_relative_l2(
    base_weight: torch.Tensor,
    candidate_weight: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Return ||Wq - W||_2 / (||W||_2 + eps)."""

    return (
        (candidate_weight - base_weight).norm(p=2)
        / (base_weight.norm(p=2) + eps)
    ).item()


def _capture_tensor_output(storage: list[torch.Tensor]):
    def hook(module, inputs, output):
        del module, inputs

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Expected the modified module to return one tensor."
            )

        storage.append(output.detach())

    return hook


def _summarize_decision_metrics(
    values: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    concatenated = {
        name: torch.cat(tensors)
        for name, tensors in values.items()
    }
    decision_risk = concatenated["decision_risk"]

    return {
        "output_kl_mean": (
            concatenated["output_kl"].mean().item()
        ),
        "abs_delta_score_mean": (
            concatenated["abs_delta_score"].mean().item()
        ),
        "decision_risk_mean": decision_risk.mean().item(),
        "decision_risk_p95": torch.quantile(
            decision_risk,
            0.95,
        ).item(),
        "decision_risk_violation_rate": (
            decision_risk.ge(1.0).float().mean().item()
        ),
    }


@torch.inference_mode()
def evaluate_score_set(
    base_model,
    candidate_model,
    base_module,
    candidate_module,
    dataloader: DataLoader,
    device: torch.device,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Measure decision and modified-module activation effects."""

    base_model.eval()
    candidate_model.eval()
    base_outputs = []
    candidate_outputs = []
    base_handle = base_module.register_forward_hook(
        _capture_tensor_output(base_outputs)
    )
    candidate_handle = candidate_module.register_forward_hook(
        _capture_tensor_output(candidate_outputs)
    )
    values = {
        "output_kl": [],
        "abs_delta_score": [],
        "decision_risk": [],
    }
    activation_error_sum = 0.0
    base_activation_square_sum = 0.0
    activation_numel = 0

    _synchronize(device)
    start_time = time.perf_counter()

    try:
        for images, _ in dataloader:
            images = images.to(device)
            base_outputs.clear()
            candidate_outputs.clear()

            base_logits = base_model(images)
            candidate_logits = candidate_model(images)

            if (
                len(base_outputs) != 1
                or len(candidate_outputs) != 1
            ):
                raise RuntimeError(
                    "Expected exactly one hook output per model forward."
                )

            base_activation = base_outputs[0]
            candidate_activation = candidate_outputs[0]

            if base_activation.shape != candidate_activation.shape:
                raise ValueError(
                    "Modified-module activation shape mismatch: "
                    f"{base_activation.shape} vs "
                    f"{candidate_activation.shape}."
                )

            activation_error_sum += (
                candidate_activation - base_activation
            ).square().sum().item()
            base_activation_square_sum += (
                base_activation.square().sum().item()
            )
            activation_numel += base_activation.numel()

            batch_metrics = binary_decision_metrics(
                base_logits=base_logits,
                candidate_logits=candidate_logits,
                eps=eps,
            )

            for name in values:
                values[name].append(
                    batch_metrics[name].detach().cpu()
                )
    finally:
        base_handle.remove()
        candidate_handle.remove()

    _synchronize(device)
    runtime_seconds = time.perf_counter() - start_time

    if activation_numel == 0:
        raise RuntimeError("Score loader produced no samples.")

    activation_mse = (
        activation_error_sum / activation_numel
    )
    base_activation_mean_square = (
        base_activation_square_sum / activation_numel
    )

    return {
        "activation_rel_mse": (
            activation_mse
            / (base_activation_mean_square + eps)
        ),
        **_summarize_decision_metrics(values),
        "scoring_runtime_seconds": runtime_seconds,
    }


@torch.inference_mode()
def evaluate_oracle_set(
    base_model,
    candidate_model,
    dataloader: DataLoader,
    device: torch.device,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Measure held-out teacher agreement and binary task accuracy."""

    base_model.eval()
    candidate_model.eval()
    values = {
        "abs_delta_score": [],
        "decision_risk": [],
        "flip": [],
        "base_pred": [],
        "candidate_pred": [],
        "labels": [],
    }

    _synchronize(device)
    start_time = time.perf_counter()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        base_logits = base_model(images)
        candidate_logits = candidate_model(images)
        batch_metrics = binary_decision_metrics(
            base_logits=base_logits,
            candidate_logits=candidate_logits,
            eps=eps,
        )

        for name in values:
            tensor = (
                labels
                if name == "labels"
                else batch_metrics[name]
            )
            values[name].append(tensor.detach().cpu())

    _synchronize(device)
    runtime_seconds = time.perf_counter() - start_time

    if not values["labels"]:
        raise RuntimeError("Oracle loader produced no samples.")

    concatenated = {
        name: torch.cat(tensors)
        for name, tensors in values.items()
    }
    decision_risk = concatenated["decision_risk"]
    candidate_pred = concatenated["candidate_pred"]
    base_pred = concatenated["base_pred"]

    return {
        "oracle_flip_rate": (
            concatenated["flip"].mean().item()
        ),
        "oracle_teacher_agreement": (
            candidate_pred.eq(base_pred).float().mean().item()
        ),
        "oracle_accuracy": (
            candidate_pred.eq(
                concatenated["labels"]
            ).float().mean().item()
        ),
        "oracle_abs_delta_score_mean": (
            concatenated["abs_delta_score"].mean().item()
        ),
        "oracle_decision_risk_mean": (
            decision_risk.mean().item()
        ),
        "oracle_decision_risk_p95": torch.quantile(
            decision_risk,
            0.95,
        ).item(),
        "oracle_decision_risk_violation_rate": (
            decision_risk.ge(1.0).float().mean().item()
        ),
        "oracle_runtime_seconds": runtime_seconds,
    }


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
