"""Collect signed samplewise score deltas for single-action candidates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from config import ExperimentConfig
from utils import get_device, set_seed

from experiments.go_no_go import DEFAULT_RESULTS_DIR
from experiments.go_no_go.adapters import (
    apply_existing_quantization_inplace,
    build_binary_eval_dataset,
    get_checkpoint_default_path,
    load_reference_model,
)
from experiments.go_no_go.run_single_action_benchmark import (
    enumerate_candidates,
)


OUTPUT_SCHEMA_VERSION = "go_no_go_score_delta_vectors_v1"
EXPECTED_SCORE_SAMPLE_COUNT = 512
EXPECTED_CANDIDATE_COUNT = 63
VALIDATION_ATOL = 1e-5
VALIDATION_RTOL = 1e-4


def parse_arguments():
    config = ExperimentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=get_checkpoint_default_path(),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--split-indices",
        type=Path,
        default=None,
        help=(
            "Optional explicit split_indices.npz path. Defaults to "
            "<benchmark-dir>/split_indices.npz."
        ),
    )
    parser.add_argument(
        "--score-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.batch_size,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("results") / "go_no_go_vectors_v1"
        ),
    )

    return parser.parse_args()


def validate_arguments(args) -> tuple[int, ...]:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    score_seeds = tuple(
        int(seed) for seed in args.score_seeds
    )

    if not score_seeds:
        raise ValueError("At least one score seed is required.")

    if len(set(score_seeds)) != len(score_seeds):
        raise ValueError("score_seeds must be unique.")

    return score_seeds


def resolve_split_paths(args) -> tuple[Path, Path]:
    split_indices_path = (
        args.split_indices
        if args.split_indices is not None
        else args.benchmark_dir / "split_indices.npz"
    )
    sibling_json_path = split_indices_path.with_suffix(
        ".json"
    )
    benchmark_json_path = (
        args.benchmark_dir / "split_indices.json"
    )

    if sibling_json_path.exists():
        split_json_path = sibling_json_path
    else:
        split_json_path = benchmark_json_path

    if not split_indices_path.exists():
        raise FileNotFoundError(split_indices_path)

    if not split_json_path.exists():
        raise FileNotFoundError(split_json_path)

    return split_indices_path, split_json_path


def output_paths(
    output_dir: Path,
    score_seeds: tuple[int, ...],
) -> dict[int, tuple[Path, Path]]:
    paths = {
        score_seed: (
            output_dir
            / f"score_delta_vectors_seed{score_seed}.npz",
            output_dir
            / f"score_delta_vectors_seed{score_seed}.json",
        )
        for score_seed in score_seeds
    }

    for npz_path, json_path in paths.values():
        for path in (npz_path, json_path):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {path}"
                )

    return paths


def load_split_metadata(
    split_json_path: Path,
) -> dict:
    with split_json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    source_split = metadata.get("source_split")

    if source_split not in {"train", "test"}:
        raise ValueError(
            "split_indices.json must identify source_split as "
            "'train' or 'test'."
        )

    return metadata


def load_split_indices(
    split_indices_path: Path,
    score_seeds: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray | None]]:
    records = {}

    with np.load(
        split_indices_path,
        allow_pickle=False,
    ) as split_data:
        for score_seed in score_seeds:
            dataset_key = (
                f"score_seed_{score_seed}_dataset_indices"
            )
            source_key = (
                f"score_seed_{score_seed}_source_indices"
            )

            if dataset_key not in split_data:
                raise ValueError(
                    f"{split_indices_path} lacks {dataset_key}."
                )

            dataset_indices = np.asarray(
                split_data[dataset_key],
                dtype=np.int64,
            )
            source_indices = (
                np.asarray(
                    split_data[source_key],
                    dtype=np.int64,
                )
                if source_key in split_data
                else None
            )

            if dataset_indices.ndim != 1:
                raise ValueError(
                    f"{dataset_key} must be one-dimensional."
                )

            if len(dataset_indices) != len(
                np.unique(dataset_indices)
            ):
                raise ValueError(
                    f"{dataset_key} contains duplicate indices."
                )

            if (
                source_indices is not None
                and source_indices.shape
                != dataset_indices.shape
            ):
                raise ValueError(
                    f"{source_key} shape does not match {dataset_key}."
                )

            records[score_seed] = {
                "dataset_indices": dataset_indices,
                "source_indices": source_indices,
            }

    return records


def load_benchmark_csvs(
    benchmark_dir: Path,
    score_seeds: tuple[int, ...],
) -> dict[int, dict]:
    records = {}
    required_columns = {
        "score_seed",
        "layer_name",
        "action",
        "abs_delta_score_mean",
    }

    for score_seed in score_seeds:
        benchmark_path = (
            benchmark_dir
            / f"single_action_metrics_seed{score_seed}.csv"
        )

        if not benchmark_path.exists():
            raise FileNotFoundError(benchmark_path)

        dataframe = pd.read_csv(benchmark_path)
        missing_columns = required_columns - set(
            dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"{benchmark_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        seeds = dataframe["score_seed"].unique()

        if len(seeds) != 1 or int(seeds[0]) != score_seed:
            raise ValueError(
                f"{benchmark_path} does not contain only "
                f"score_seed={score_seed}."
            )

        candidate_ids = (
            dataframe["layer_name"].astype(str)
            + "|"
            + dataframe["action"].astype(str)
        ).tolist()

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(
                f"{benchmark_path} has duplicate candidate IDs."
            )

        records[score_seed] = {
            "path": benchmark_path,
            "dataframe": dataframe,
            "candidate_ids": candidate_ids,
            "layer_names": (
                dataframe["layer_name"].astype(str).tolist()
            ),
            "action_names": (
                dataframe["action"].astype(str).tolist()
            ),
        }

    reference_ids = records[score_seeds[0]][
        "candidate_ids"
    ]

    for score_seed in score_seeds[1:]:
        if records[score_seed]["candidate_ids"] != reference_ids:
            raise ValueError(
                "Candidate ID ordering differs between score seeds "
                f"{score_seeds[0]} and {score_seed}."
            )

    return records


def validate_split_sidecar(
    split_metadata: dict,
    split_records: dict[int, dict[str, np.ndarray | None]],
) -> None:
    score_sets = split_metadata.get("score_sets", {})

    for score_seed, split_record in split_records.items():
        sidecar_record = score_sets.get(str(score_seed))

        if sidecar_record is None:
            raise ValueError(
                f"split_indices.json lacks score seed {score_seed}."
            )

        sidecar_indices = np.asarray(
            sidecar_record.get("dataset_indices", []),
            dtype=np.int64,
        )

        if not np.array_equal(
            sidecar_indices,
            split_record["dataset_indices"],
        ):
            raise ValueError(
                f"NPZ/JSON dataset indices differ for seed {score_seed}."
            )


def build_score_loaders(
    dataset,
    split_records: dict[int, dict[str, np.ndarray | None]],
    batch_size: int,
) -> dict[int, DataLoader]:
    loaders = {}

    for score_seed, split_record in split_records.items():
        dataset_indices = split_record[
            "dataset_indices"
        ]

        if len(dataset_indices) != EXPECTED_SCORE_SAMPLE_COUNT:
            raise ValueError(
                f"score_seed={score_seed} has {len(dataset_indices)} "
                "samples; expected exactly "
                f"{EXPECTED_SCORE_SAMPLE_COUNT}."
            )

        if (
            dataset_indices.min() < 0
            or dataset_indices.max() >= len(dataset)
        ):
            raise ValueError(
                f"score_seed={score_seed} indices are outside the "
                "source dataset."
            )

        loaders[score_seed] = DataLoader(
            Subset(dataset, dataset_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


@torch.inference_mode()
def collect_binary_scores(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    predicted_classes = []

    for images, _ in dataloader:
        images = images.to(device)
        logits = model(images)

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(
                "Expected binary logits with shape [N, 2]."
            )

        batch_scores = logits[:, 1] - logits[:, 0]
        scores.append(
            batch_scores.float().cpu().numpy()
        )
        predicted_classes.append(
            logits.argmax(dim=1).cpu().numpy()
        )

    if not scores:
        raise RuntimeError("Score loader produced no samples.")

    return (
        np.concatenate(scores).astype(
            np.float32,
            copy=False,
        ),
        np.concatenate(predicted_classes).astype(
            np.int64,
            copy=False,
        ),
    )


def validate_candidate_order(
    reference_model,
    benchmark_records: dict[int, dict],
    score_seeds: tuple[int, ...],
) -> tuple[list[str], list[str], list[str]]:
    enumerated = enumerate_candidates(reference_model)
    candidate_ids = [
        f"{layer_name}|{action_name}"
        for layer_name, action_name in enumerated
    ]
    source_candidate_ids = benchmark_records[
        score_seeds[0]
    ]["candidate_ids"]

    if candidate_ids != source_candidate_ids:
        raise ValueError(
            "Current model/action enumeration does not exactly match "
            "the source benchmark CSV ordering."
        )

    if len(candidate_ids) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"Found {len(candidate_ids)} candidates; expected exactly "
            f"{EXPECTED_CANDIDATE_COUNT}."
        )

    layer_names = [layer_name for layer_name, _ in enumerated]
    action_names = [action_name for _, action_name in enumerated]

    return candidate_ids, layer_names, action_names


def validate_delta_means(
    delta_scores: np.ndarray,
    benchmark: pd.DataFrame,
    candidate_ids: list[str],
    score_seed: int,
) -> dict[str, float]:
    computed_means = np.abs(delta_scores).mean(
        axis=1,
        dtype=np.float32,
    ).astype(np.float64)
    expected_means = benchmark[
        "abs_delta_score_mean"
    ].to_numpy(dtype=np.float64)
    absolute_errors = np.abs(
        computed_means - expected_means
    )
    relative_errors = absolute_errors / np.maximum(
        np.abs(expected_means),
        VALIDATION_ATOL,
    )
    close = np.isclose(
        computed_means,
        expected_means,
        atol=VALIDATION_ATOL,
        rtol=VALIDATION_RTOL,
    )
    maximum_absolute_error = float(
        absolute_errors.max(initial=0.0)
    )
    maximum_relative_error = float(
        relative_errors.max(initial=0.0)
    )

    print(
        f"score_seed={score_seed}: maximum absolute validation "
        f"error={maximum_absolute_error:.8g}, maximum relative "
        f"validation error={maximum_relative_error:.8g}"
    )

    if not close.all():
        worst_index = int(absolute_errors.argmax())
        raise RuntimeError(
            "Saved float32 delta validation failed for "
            f"score_seed={score_seed}, "
            f"candidate={candidate_ids[worst_index]}, "
            f"computed={computed_means[worst_index]:.9g}, "
            f"expected={expected_means[worst_index]:.9g}, "
            f"absolute_error={absolute_errors[worst_index]:.9g}."
        )

    return {
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "absolute_tolerance": VALIDATION_ATOL,
        "relative_tolerance": VALIDATION_RTOL,
    }


def save_seed_outputs(
    *,
    npz_path: Path,
    json_path: Path,
    checkpoint_path: Path,
    benchmark_path: Path,
    split_source: str,
    score_seed: int,
    candidate_ids: list[str],
    layer_names: list[str],
    action_names: list[str],
    split_record: dict[str, np.ndarray | None],
    baseline_score: np.ndarray,
    baseline_predicted_class: np.ndarray,
    delta_scores: np.ndarray,
    validation: dict[str, float],
) -> None:
    arrays = {
        "candidate_ids": np.asarray(
            candidate_ids,
            dtype=np.str_,
        ),
        "dataset_indices": split_record[
            "dataset_indices"
        ],
        "baseline_score": baseline_score,
        "baseline_margin": np.abs(
            baseline_score
        ).astype(np.float32, copy=False),
        "baseline_predicted_class": (
            baseline_predicted_class
        ),
        "delta_scores": delta_scores,
        "action_names": np.asarray(
            action_names,
            dtype=np.str_,
        ),
        "layer_names": np.asarray(
            layer_names,
            dtype=np.str_,
        ),
    }
    source_indices = split_record["source_indices"]

    if source_indices is not None:
        arrays["source_indices"] = source_indices

    npz_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with npz_path.open("xb") as file:
        np.savez_compressed(file, **arrays)

    metadata = {
        "checkpoint_path": str(
            checkpoint_path.resolve()
        ),
        "score_seed": score_seed,
        "score_sample_count": int(
            baseline_score.shape[0]
        ),
        "candidate_count": len(candidate_ids),
        "split_source": split_source,
        "benchmark_csv_path": str(
            benchmark_path.resolve()
        ),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "validation": validation,
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
            allow_nan=False,
        )
        file.write("\n")


def main() -> None:
    args = parse_arguments()
    score_seeds = validate_arguments(args)
    paths_by_seed = output_paths(
        output_dir=args.output_dir,
        score_seeds=score_seeds,
    )
    split_indices_path, split_json_path = (
        resolve_split_paths(args)
    )
    split_metadata = load_split_metadata(
        split_json_path
    )
    split_records = load_split_indices(
        split_indices_path=split_indices_path,
        score_seeds=score_seeds,
    )
    validate_split_sidecar(
        split_metadata=split_metadata,
        split_records=split_records,
    )
    benchmark_records = load_benchmark_csvs(
        benchmark_dir=args.benchmark_dir,
        score_seeds=score_seeds,
    )
    deterministic_seed = int(
        split_metadata.get("oracle_seed", 2026)
    )
    set_seed(deterministic_seed)
    device = get_device()
    dataset = build_binary_eval_dataset(
        source_split=split_metadata["source_split"]
    )
    score_loaders = build_score_loaders(
        dataset=dataset,
        split_records=split_records,
        batch_size=args.batch_size,
    )
    reference_model = load_reference_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    candidate_ids, layer_names, action_names = (
        validate_candidate_order(
            reference_model=reference_model,
            benchmark_records=benchmark_records,
            score_seeds=score_seeds,
        )
    )
    baseline_scores = {}
    baseline_predicted_classes = {}

    for score_seed in score_seeds:
        baseline_score, baseline_predicted_class = (
            collect_binary_scores(
                model=reference_model,
                dataloader=score_loaders[score_seed],
                device=device,
            )
        )

        if len(baseline_score) != EXPECTED_SCORE_SAMPLE_COUNT:
            raise RuntimeError(
                f"Baseline produced {len(baseline_score)} samples "
                f"for score_seed={score_seed}; expected "
                f"{EXPECTED_SCORE_SAMPLE_COUNT}."
            )

        baseline_scores[score_seed] = baseline_score
        baseline_predicted_classes[
            score_seed
        ] = baseline_predicted_class

    delta_scores_by_seed = {
        score_seed: np.empty(
            (
                EXPECTED_CANDIDATE_COUNT,
                EXPECTED_SCORE_SAMPLE_COUNT,
            ),
            dtype=np.float32,
        )
        for score_seed in score_seeds
    }

    for candidate_index, (
        layer_name,
        action_name,
    ) in enumerate(
        zip(layer_names, action_names),
        start=1,
    ):
        print(
            f"[{candidate_index}/{EXPECTED_CANDIDATE_COUNT}] "
            f"{layer_name}|{action_name}"
        )
        candidate_model = None
        target_module = None

        try:
            candidate_model = copy.deepcopy(
                reference_model
            ).cpu().eval()
            target_module = candidate_model.get_submodule(
                layer_name
            )
            apply_existing_quantization_inplace(
                module=target_module,
                action_name=action_name,
            )
            candidate_model = candidate_model.to(
                device
            ).eval()

            for score_seed in score_seeds:
                candidate_score, _ = collect_binary_scores(
                    model=candidate_model,
                    dataloader=score_loaders[score_seed],
                    device=device,
                )
                delta_scores_by_seed[score_seed][
                    candidate_index - 1
                ] = (
                    candidate_score
                    - baseline_scores[score_seed]
                ).astype(np.float32, copy=False)
        finally:
            target_module = None

            if candidate_model is not None:
                del candidate_model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    validations = {}

    for score_seed in score_seeds:
        validations[score_seed] = validate_delta_means(
            delta_scores=delta_scores_by_seed[
                score_seed
            ],
            benchmark=benchmark_records[
                score_seed
            ]["dataframe"],
            candidate_ids=candidate_ids,
            score_seed=score_seed,
        )

    for score_seed in score_seeds:
        npz_path, json_path = paths_by_seed[score_seed]
        save_seed_outputs(
            npz_path=npz_path,
            json_path=json_path,
            checkpoint_path=args.checkpoint,
            benchmark_path=benchmark_records[
                score_seed
            ]["path"],
            split_source=split_metadata["source_split"],
            score_seed=score_seed,
            candidate_ids=candidate_ids,
            layer_names=layer_names,
            action_names=action_names,
            split_record=split_records[score_seed],
            baseline_score=baseline_scores[score_seed],
            baseline_predicted_class=(
                baseline_predicted_classes[score_seed]
            ),
            delta_scores=delta_scores_by_seed[
                score_seed
            ],
            validation=validations[score_seed],
        )
        print(f"Saved: {npz_path}")
        print(f"Metadata: {json_path}")

    print(
        "Validation complete: score sample count=512, "
        "candidate count=63, candidate ordering matches source CSVs."
    )
    print(f"Device used: {device}")


if __name__ == "__main__":
    main()
