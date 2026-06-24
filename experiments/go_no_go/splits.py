"""Deterministic class-balanced score and oracle split utilities."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class CalibrationSplits:
    """Disjoint loaders and exact dataset-relative indices."""

    score_loaders: dict[int, DataLoader]
    oracle_loader: DataLoader
    score_indices: dict[int, tuple[int, ...]]
    oracle_indices: tuple[int, ...]
    score_class_counts: dict[int, dict[int, int]]
    oracle_class_counts: dict[int, int]


def extract_binary_labels(
    dataset: Dataset,
) -> tuple[int, ...]:
    """Read remapped labels without applying image transforms when possible."""

    if all(
        hasattr(dataset, attribute)
        for attribute in (
            "base_dataset",
            "source_indices",
            "label_map",
        )
    ):
        targets = dataset.base_dataset.targets
        labels = tuple(
            int(
                dataset.label_map[
                    int(targets[source_index])
                ]
            )
            for source_index in dataset.source_indices
        )
    else:
        labels = tuple(
            int(dataset[index][1])
            for index in range(len(dataset))
        )

    unique_labels = sorted(set(labels))

    if unique_labels != [0, 1]:
        raise ValueError(
            "Expected a binary dataset with remapped labels 0 and 1; "
            f"found {unique_labels}."
        )

    return labels


def _select_balanced_indices(
    indices_by_class: dict[int, list[int]],
    total_size: int,
    generator: torch.Generator,
) -> tuple[int, ...]:
    class_ids = sorted(indices_by_class)

    if total_size <= 0:
        raise ValueError("Split sizes must be positive.")

    if total_size % len(class_ids) != 0:
        raise ValueError(
            "Class-balanced split sizes must be divisible by "
            f"the number of classes ({len(class_ids)})."
        )

    per_class_size = total_size // len(class_ids)
    selected = []

    for class_id in class_ids:
        available = indices_by_class[class_id]

        if per_class_size > len(available):
            raise ValueError(
                f"Requested {per_class_size} samples for class "
                f"{class_id}, but only {len(available)} are available."
            )

        permutation = torch.randperm(
            len(available),
            generator=generator,
        )[:per_class_size].tolist()
        selected.extend(
            available[position]
            for position in permutation
        )

    mixed_order = torch.randperm(
        len(selected),
        generator=generator,
    ).tolist()

    return tuple(
        selected[position]
        for position in mixed_order
    )


def _class_counts(
    indices: Sequence[int],
    labels: Sequence[int],
) -> dict[int, int]:
    counts = {0: 0, 1: 0}

    for index in indices:
        counts[int(labels[index])] += 1

    return counts


def _build_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(
        Subset(dataset, list(indices)),
        shuffle=False,
        **loader_kwargs,
    )


def build_class_balanced_calibration_splits(
    dataset: Dataset,
    score_size: int,
    oracle_size: int,
    score_seeds: Sequence[int],
    oracle_seed: int,
    batch_size: int,
    num_workers: int = 0,
) -> CalibrationSplits:
    """Build a fixed oracle and seed-varying, non-overlapping score sets."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    normalized_score_seeds = tuple(
        int(seed) for seed in score_seeds
    )

    if not normalized_score_seeds:
        raise ValueError("At least one score seed is required.")

    if len(set(normalized_score_seeds)) != len(
        normalized_score_seeds
    ):
        raise ValueError("score_seeds must be unique.")

    labels = extract_binary_labels(dataset)
    indices_by_class = {
        class_id: [
            index
            for index, label in enumerate(labels)
            if label == class_id
        ]
        for class_id in (0, 1)
    }

    oracle_generator = torch.Generator().manual_seed(
        oracle_seed
    )
    oracle_indices = _select_balanced_indices(
        indices_by_class=indices_by_class,
        total_size=oracle_size,
        generator=oracle_generator,
    )
    oracle_index_set = set(oracle_indices)
    score_source_by_class = {
        class_id: [
            index
            for index in class_indices
            if index not in oracle_index_set
        ]
        for class_id, class_indices in indices_by_class.items()
    }

    score_indices = {}
    score_class_counts = {}
    score_loaders = {}

    for score_seed in normalized_score_seeds:
        score_generator = torch.Generator().manual_seed(
            score_seed
        )
        indices = _select_balanced_indices(
            indices_by_class=score_source_by_class,
            total_size=score_size,
            generator=score_generator,
        )
        overlap = set(indices) & oracle_index_set

        if overlap:
            raise RuntimeError(
                "Score and oracle sets unexpectedly overlap: "
                f"{len(overlap)} indices."
            )

        score_indices[score_seed] = indices
        score_class_counts[score_seed] = _class_counts(
            indices,
            labels,
        )
        score_loaders[score_seed] = _build_loader(
            dataset=dataset,
            indices=indices,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    if len(normalized_score_seeds) > 1:
        unique_score_sets = {
            indices
            for indices in score_indices.values()
        }

        if len(unique_score_sets) != len(score_indices):
            raise RuntimeError(
                "Different score seeds produced identical score sets."
            )

    return CalibrationSplits(
        score_loaders=score_loaders,
        oracle_loader=_build_loader(
            dataset=dataset,
            indices=oracle_indices,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        score_indices=score_indices,
        oracle_indices=oracle_indices,
        score_class_counts=score_class_counts,
        oracle_class_counts=_class_counts(
            oracle_indices,
            labels,
        ),
    )


def _source_indices(
    dataset: Dataset,
    dataset_indices: Sequence[int],
) -> list[int]:
    if hasattr(dataset, "source_indices"):
        return [
            int(dataset.source_indices[index])
            for index in dataset_indices
        ]

    return [int(index) for index in dataset_indices]


def save_split_indices(
    splits: CalibrationSplits,
    dataset: Dataset,
    output_dir: Path,
    source_split: str,
    oracle_seed: int,
) -> tuple[Path, Path]:
    """Persist exact dataset-relative and original-source indices."""

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "split_indices.npz"
    json_path = output_dir / "split_indices.json"

    for output_path in (npz_path, json_path):
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )

    arrays = {
        "class_ids": np.asarray([0, 1], dtype=np.int64),
        "oracle_dataset_indices": np.asarray(
            splits.oracle_indices,
            dtype=np.int64,
        ),
        "oracle_source_indices": np.asarray(
            _source_indices(
                dataset,
                splits.oracle_indices,
            ),
            dtype=np.int64,
        ),
        "oracle_class_counts": np.asarray(
            [
                splits.oracle_class_counts[class_id]
                for class_id in (0, 1)
            ],
            dtype=np.int64,
        ),
    }
    sidecar = {
        "source_split": source_split,
        "oracle_seed": int(oracle_seed),
        "oracle": {
            "size": len(splits.oracle_indices),
            "class_counts": {
                str(key): value
                for key, value in (
                    splits.oracle_class_counts.items()
                )
            },
            "dataset_indices": list(splits.oracle_indices),
            "source_indices": _source_indices(
                dataset,
                splits.oracle_indices,
            ),
        },
        "score_sets": {},
    }

    for score_seed, indices in splits.score_indices.items():
        prefix = f"score_seed_{score_seed}"
        source_indices = _source_indices(
            dataset,
            indices,
        )
        arrays[f"{prefix}_dataset_indices"] = np.asarray(
            indices,
            dtype=np.int64,
        )
        arrays[f"{prefix}_source_indices"] = np.asarray(
            source_indices,
            dtype=np.int64,
        )
        arrays[f"{prefix}_class_counts"] = np.asarray(
            [
                splits.score_class_counts[
                    score_seed
                ][class_id]
                for class_id in (0, 1)
            ],
            dtype=np.int64,
        )
        sidecar["score_sets"][str(score_seed)] = {
            "size": len(indices),
            "class_counts": {
                str(key): value
                for key, value in (
                    splits.score_class_counts[
                        score_seed
                    ].items()
                )
            },
            "oracle_overlap_count": len(
                set(indices) & set(splits.oracle_indices)
            ),
            "dataset_indices": list(indices),
            "source_indices": source_indices,
        }

    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(
        json.dumps(
            sidecar,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return npz_path, json_path
