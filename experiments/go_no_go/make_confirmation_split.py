"""Create a fresh class-balanced confirmation split from the train pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch

from experiments.go_no_go.adapters import (
    build_binary_eval_dataset,
)
from experiments.go_no_go.splits import extract_binary_labels


SOURCE_SPLIT = "train"
SCHEMA_VERSION = "go_no_go_confirmation_split_v1"
SCORE_DATASET_KEY = re.compile(
    r"^score_seed_(.+)_dataset_indices$"
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-split",
        type=Path,
        default=Path(
            "results/go_no_go/split_indices.npz"
        ),
    )
    parser.add_argument(
        "--canonical-metadata",
        type=Path,
        default=Path(
            "results/go_no_go/split_indices.json"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3030,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/go_no_go_confirmation_v1"
        ),
    )

    return parser.parse_args()


def validate_size(size: int) -> int:
    if size <= 0:
        raise ValueError("size must be positive.")

    if size % 2 != 0:
        raise ValueError(
            "size must be even for exact binary class balance."
        )

    return size // 2


def confirmation_output_paths(
    output_dir: Path,
) -> tuple[Path, Path]:
    npz_path = (
        output_dir / "confirmation_split_indices.npz"
    )
    json_path = (
        output_dir / "confirmation_split_indices.json"
    )

    for output_path in (npz_path, json_path):
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing confirmation file: "
                f"{output_path}"
            )

    return npz_path, json_path


def load_canonical_metadata(
    metadata_path: Path,
) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if metadata.get("source_split") != SOURCE_SPLIT:
        raise ValueError(
            "Canonical metadata must use source_split='train'."
        )

    return metadata


def load_canonical_indices(
    split_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    score_sets = {}

    with np.load(
        split_path,
        allow_pickle=False,
    ) as split_data:
        if "oracle_dataset_indices" not in split_data:
            raise ValueError(
                f"{split_path} lacks oracle_dataset_indices."
            )

        development_oracle = np.asarray(
            split_data["oracle_dataset_indices"],
            dtype=np.int64,
        )

        for key in split_data.files:
            match = SCORE_DATASET_KEY.match(key)

            if match is None:
                continue

            score_seed = match.group(1)
            score_sets[score_seed] = np.asarray(
                split_data[key],
                dtype=np.int64,
            )

    if development_oracle.ndim != 1:
        raise ValueError(
            "oracle_dataset_indices must be one-dimensional."
        )

    if not score_sets:
        raise ValueError(
            "Canonical split contains no score-set dataset indices."
        )

    for name, indices in {
        "development_oracle": development_oracle,
        **{
            f"score_seed_{seed}": indices
            for seed, indices in score_sets.items()
        },
    }.items():
        if indices.ndim != 1:
            raise ValueError(
                f"{name} indices must be one-dimensional."
            )

        if len(indices) != len(np.unique(indices)):
            raise ValueError(
                f"{name} contains duplicate indices."
            )

    return development_oracle, score_sets


def validate_canonical_consistency(
    metadata: dict,
    development_oracle: np.ndarray,
    score_sets: dict[str, np.ndarray],
) -> None:
    oracle_metadata = metadata.get("oracle", {})
    metadata_oracle_indices = np.asarray(
        oracle_metadata.get("dataset_indices", []),
        dtype=np.int64,
    )

    if not np.array_equal(
        development_oracle,
        metadata_oracle_indices,
    ):
        raise ValueError(
            "Canonical NPZ and JSON development-oracle indices differ."
        )

    metadata_score_sets = metadata.get("score_sets", {})

    for score_seed, indices in score_sets.items():
        if score_seed not in metadata_score_sets:
            raise ValueError(
                f"Canonical JSON lacks score seed {score_seed}."
            )

        metadata_indices = np.asarray(
            metadata_score_sets[score_seed].get(
                "dataset_indices",
                [],
            ),
            dtype=np.int64,
        )

        if not np.array_equal(indices, metadata_indices):
            raise ValueError(
                "Canonical NPZ and JSON score-set indices differ "
                f"for seed {score_seed}."
            )


def exclusion_summary(
    development_oracle: np.ndarray,
    score_sets: dict[str, np.ndarray],
) -> tuple[set[int], dict]:
    development_set = set(
        int(index) for index in development_oracle
    )
    score_set_values = {
        score_seed: set(int(index) for index in indices)
        for score_seed, indices in score_sets.items()
    }
    score_union = set().union(
        *score_set_values.values()
    )
    excluded_union = development_set | score_union
    counts = {
        "development_oracle": len(development_set),
        "score_sets": {
            score_seed: len(indices)
            for score_seed, indices in score_set_values.items()
        },
        "score_set_union": len(score_union),
        "total_excluded_union": len(excluded_union),
    }

    return excluded_union, counts


def select_confirmation_indices(
    labels: tuple[int, ...],
    excluded_indices: set[int],
    per_class_size: int,
    seed: int,
) -> np.ndarray:
    available_by_class = {
        class_id: [
            index
            for index, label in enumerate(labels)
            if label == class_id
            and index not in excluded_indices
        ]
        for class_id in (0, 1)
    }
    generator = torch.Generator().manual_seed(seed)
    selected = []

    for class_id in (0, 1):
        available = available_by_class[class_id]

        if len(available) < per_class_size:
            raise ValueError(
                f"Class {class_id} has only {len(available)} "
                f"eligible samples; {per_class_size} are required."
            )

        positions = torch.randperm(
            len(available),
            generator=generator,
        )[:per_class_size].tolist()
        selected.extend(
            available[position]
            for position in positions
        )

    mixed_order = torch.randperm(
        len(selected),
        generator=generator,
    ).tolist()

    return np.asarray(
        [selected[position] for position in mixed_order],
        dtype=np.int64,
    )


def source_indices_for_dataset(
    dataset,
    dataset_indices: np.ndarray,
) -> np.ndarray | None:
    if not hasattr(dataset, "source_indices"):
        return None

    return np.asarray(
        [
            int(dataset.source_indices[int(index)])
            for index in dataset_indices
        ],
        dtype=np.int64,
    )


def validate_confirmation(
    confirmation_indices: np.ndarray,
    labels: tuple[int, ...],
    requested_size: int,
    development_oracle: np.ndarray,
    score_sets: dict[str, np.ndarray],
) -> tuple[dict[int, int], dict]:
    if len(confirmation_indices) != requested_size:
        raise RuntimeError(
            f"Confirmation size is {len(confirmation_indices)}; "
            f"expected {requested_size}."
        )

    if len(confirmation_indices) != len(
        np.unique(confirmation_indices)
    ):
        raise RuntimeError(
            "Confirmation split contains duplicate indices."
        )

    class_counts = {
        class_id: int(
            sum(
                labels[int(index)] == class_id
                for index in confirmation_indices
            )
        )
        for class_id in (0, 1)
    }
    expected_per_class = requested_size // 2

    if class_counts != {
        0: expected_per_class,
        1: expected_per_class,
    }:
        raise RuntimeError(
            "Confirmation split is not exactly class-balanced: "
            f"{class_counts}."
        )

    confirmation_set = set(
        int(index) for index in confirmation_indices
    )
    overlap_counts = {
        "development_oracle": len(
            confirmation_set
            & set(int(index) for index in development_oracle)
        ),
        "score_sets": {
            score_seed: len(
                confirmation_set
                & set(int(index) for index in indices)
            )
            for score_seed, indices in score_sets.items()
        },
    }

    if overlap_counts["development_oracle"] != 0:
        raise RuntimeError(
            "Confirmation split overlaps the development oracle."
        )

    nonzero_score_overlaps = {
        score_seed: count
        for score_seed, count in (
            overlap_counts["score_sets"].items()
        )
        if count != 0
    }

    if nonzero_score_overlaps:
        raise RuntimeError(
            "Confirmation split overlaps score sets: "
            f"{nonzero_score_overlaps}."
        )

    return class_counts, overlap_counts


def save_confirmation(
    *,
    npz_path: Path,
    json_path: Path,
    canonical_split_path: Path,
    canonical_metadata_path: Path,
    confirmation_indices: np.ndarray,
    source_indices: np.ndarray | None,
    seed: int,
    requested_size: int,
    class_counts: dict[int, int],
    excluded_counts: dict,
    overlap_counts: dict,
) -> None:
    arrays = {
        "class_ids": np.asarray([0, 1], dtype=np.int64),
        "dataset_indices": confirmation_indices,
        "class_counts": np.asarray(
            [class_counts[0], class_counts[1]],
            dtype=np.int64,
        ),
    }

    if source_indices is not None:
        arrays["source_indices"] = source_indices

    npz_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with npz_path.open("xb") as file:
        np.savez_compressed(file, **arrays)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_split": SOURCE_SPLIT,
        "confirmation_seed": int(seed),
        "requested_size": int(requested_size),
        "class_counts": {
            str(class_id): count
            for class_id, count in class_counts.items()
        },
        "excluded_index_counts": excluded_counts,
        "overlap_counts": overlap_counts,
        "canonical_split_path": str(
            canonical_split_path.resolve()
        ),
        "canonical_metadata_path": str(
            canonical_metadata_path.resolve()
        ),
        "source_indices_available": (
            source_indices is not None
        ),
    }

    with json_path.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def main() -> None:
    args = parse_arguments()
    per_class_size = validate_size(args.size)
    npz_path, json_path = confirmation_output_paths(
        args.output_dir
    )
    canonical_metadata = load_canonical_metadata(
        args.canonical_metadata
    )
    development_oracle, score_sets = (
        load_canonical_indices(args.canonical_split)
    )
    validate_canonical_consistency(
        metadata=canonical_metadata,
        development_oracle=development_oracle,
        score_sets=score_sets,
    )
    excluded_indices, excluded_counts = exclusion_summary(
        development_oracle=development_oracle,
        score_sets=score_sets,
    )
    dataset = build_binary_eval_dataset(
        source_split=SOURCE_SPLIT
    )
    labels = extract_binary_labels(dataset)

    if excluded_indices and (
        min(excluded_indices) < 0
        or max(excluded_indices) >= len(dataset)
    ):
        raise ValueError(
            "Canonical excluded indices are outside the train pool."
        )

    confirmation_indices = select_confirmation_indices(
        labels=labels,
        excluded_indices=excluded_indices,
        per_class_size=per_class_size,
        seed=args.seed,
    )
    source_indices = source_indices_for_dataset(
        dataset=dataset,
        dataset_indices=confirmation_indices,
    )
    class_counts, overlap_counts = validate_confirmation(
        confirmation_indices=confirmation_indices,
        labels=labels,
        requested_size=args.size,
        development_oracle=development_oracle,
        score_sets=score_sets,
    )
    save_confirmation(
        npz_path=npz_path,
        json_path=json_path,
        canonical_split_path=args.canonical_split,
        canonical_metadata_path=args.canonical_metadata,
        confirmation_indices=confirmation_indices,
        source_indices=source_indices,
        seed=args.seed,
        requested_size=args.size,
        class_counts=class_counts,
        excluded_counts=excluded_counts,
        overlap_counts=overlap_counts,
    )
    print(
        "First 10 dataset indices: "
        f"{confirmation_indices[:10].tolist()}"
    )
    print(
        "First 10 source indices: "
        + (
            str(source_indices[:10].tolist())
            if source_indices is not None
            else "unavailable"
        )
    )
    print(f"Saved confirmation indices: {npz_path}")
    print(f"Saved confirmation metadata: {json_path}")


if __name__ == "__main__":
    main()
