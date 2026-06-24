"""Evaluate fixed scalar and vector plans on the confirmation split only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from additive_planner import (
    ACTION_TO_BITS,
    action_to_bits,
    constant_parameter_storage_bytes,
    weight_storage_bytes,
)
from config import ExperimentConfig
from quantization import (
    build_mixed_quantized_model,
    list_quantizable_layers,
)
from utils import get_device, set_seed

from experiments.go_no_go.adapters import (
    build_binary_eval_dataset,
    get_checkpoint_default_path,
    load_reference_model,
)
from experiments.go_no_go.metrics import evaluate_oracle_set
from experiments.go_no_go.splits import extract_binary_labels


REQUESTED_MEMORY_SAVING_RATIOS = (
    0.70,
    0.80,
    0.82,
    0.84,
    0.85,
    0.86,
)

PRIMARY_SCALAR_METRICS = (
    "weight_rel_l2",
    "activation_rel_mse",
    "output_kl_mean",
    "abs_delta_score_mean",
    "decision_risk_mean",
    "decision_risk_p95",
)

SECONDARY_SCALAR_METRIC = "decision_risk_violation_rate"

COMPACT_SELECTIONS = (
    "vector_signed_mean_risk",
    "vector_signed_p95_risk",
    "decision_risk_p95",
    "weight_rel_l2",
    "output_kl_mean",
)

RESULT_COLUMNS = (
    "plan_id",
    "plan_kind",
    "selected_score_metric",
    "vector_objective",
    "requested_memory_saving_ratio",
    "actual_memory_saving_ratio",
    "comparison_role",
    "confirmation_accuracy",
    "confirmation_teacher_agreement",
    "confirmation_flip_rate",
    "confirmation_abs_delta_score_mean",
    "confirmation_decision_risk_mean",
    "confirmation_decision_risk_p95",
    "confirmation_decision_risk_violation_rate",
    "runtime_seconds",
    "num_fp32",
    "num_fp16",
    "num_int8",
    "num_int4",
    "source_plan_path",
)

ALLOWED_PLAN_FIELDS = {
    "plan_id",
    "plan_kind",
    "selected_score_metric",
    "vector_objective",
    "requested_memory_saving_ratio",
    "actual_memory_saving_ratio",
    "actual_total_memory_bytes",
    "fp32_total_memory_bytes",
    "layer_actions",
    "per_layer_action_bytes",
    "action_counts",
}


def parse_arguments():
    config = ExperimentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=get_checkpoint_default_path(),
    )
    parser.add_argument(
        "--confirmation-split",
        type=Path,
        default=Path(
            "results/go_no_go_confirmation_v1/"
            "confirmation_split_indices.npz"
        ),
    )
    parser.add_argument(
        "--confirmation-metadata",
        type=Path,
        default=Path(
            "results/go_no_go_confirmation_v1/"
            "confirmation_split_indices.json"
        ),
    )
    parser.add_argument(
        "--vector-plan-dir",
        type=Path,
        default=Path(
            "results/go_no_go_vector_plans_v1"
        ),
    )
    parser.add_argument(
        "--scalar-plan-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("results/go_no_go_planner_v1"),
            Path("results/go_no_go_planner_v2_stress"),
        ],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.batch_size,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/go_no_go_confirmation_eval_v1"
        ),
    )

    return parser.parse_args()


def validate_arguments(args) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")


def preflight_outputs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "results_csv": (
            output_dir / "confirmation_plan_results.csv"
        ),
        "results_json": (
            output_dir / "confirmation_plan_results.json"
        ),
        "primary_csv": (
            output_dir / "confirmation_primary_summary.csv"
        ),
    }

    for output_path in paths.values():
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )

    return paths


def load_confirmation_metadata(
    metadata_path: Path,
) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if metadata.get("source_split") != "train":
        raise ValueError(
            "Confirmation metadata must use source_split='train'."
        )

    if int(metadata.get("requested_size", -1)) != 2000:
        raise ValueError(
            "Confirmation metadata must report exactly 2000 samples."
        )

    expected_counts = {"0": 1000, "1": 1000}

    if metadata.get("class_counts") != expected_counts:
        raise ValueError(
            "Confirmation metadata is not exactly class-balanced."
        )

    overlaps = metadata.get("overlap_counts", {})

    if int(overlaps.get("development_oracle", -1)) != 0:
        raise ValueError(
            "Confirmation metadata reports development-oracle overlap."
        )

    score_overlaps = overlaps.get("score_sets", {})

    if not score_overlaps or any(
        int(count) != 0 for count in score_overlaps.values()
    ):
        raise ValueError(
            "Confirmation metadata reports score-set overlap."
        )

    return metadata


def load_confirmation_indices(
    split_path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    with np.load(
        split_path,
        allow_pickle=False,
    ) as split_data:
        required = {
            "dataset_indices",
            "class_counts",
        }
        missing = required - set(split_data.files)

        if missing:
            raise ValueError(
                f"{split_path} is missing arrays: {sorted(missing)}"
            )

        dataset_indices = np.asarray(
            split_data["dataset_indices"],
            dtype=np.int64,
        )
        class_counts = np.asarray(
            split_data["class_counts"],
            dtype=np.int64,
        )
        source_indices = (
            np.asarray(
                split_data["source_indices"],
                dtype=np.int64,
            )
            if "source_indices" in split_data
            else None
        )

    if dataset_indices.shape != (2000,):
        raise ValueError(
            "Confirmation split must contain exactly 2000 indices."
        )

    if len(dataset_indices) != len(np.unique(dataset_indices)):
        raise ValueError(
            "Confirmation split contains duplicate indices."
        )

    if not np.array_equal(
        class_counts,
        np.asarray([1000, 1000], dtype=np.int64),
    ):
        raise ValueError(
            "Confirmation NPZ class counts are not [1000, 1000]."
        )

    if (
        source_indices is not None
        and source_indices.shape != dataset_indices.shape
    ):
        raise ValueError(
            "Confirmation source_indices shape is inconsistent."
        )

    return dataset_indices, source_indices, class_counts


def build_confirmation_loader(
    dataset_indices: np.ndarray,
    source_indices: np.ndarray | None,
    batch_size: int,
) -> DataLoader:
    dataset = build_binary_eval_dataset(
        source_split="train"
    )

    if (
        dataset_indices.min() < 0
        or dataset_indices.max() >= len(dataset)
    ):
        raise ValueError(
            "Confirmation indices are outside the binary train pool."
        )

    labels = extract_binary_labels(dataset)
    observed_counts = {
        class_id: sum(
            labels[int(index)] == class_id
            for index in dataset_indices
        )
        for class_id in (0, 1)
    }

    if observed_counts != {0: 1000, 1: 1000}:
        raise ValueError(
            "Confirmation dataset labels are not exactly balanced."
        )

    if (
        source_indices is not None
        and hasattr(dataset, "source_indices")
    ):
        expected_source_indices = np.asarray(
            [
                dataset.source_indices[int(index)]
                for index in dataset_indices
            ],
            dtype=np.int64,
        )

        if not np.array_equal(
            source_indices,
            expected_source_indices,
        ):
            raise ValueError(
                "Confirmation source indices do not match the dataset."
            )

    return DataLoader(
        Subset(dataset, dataset_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def read_plan_structure(plan_path: Path) -> dict:
    """Retain structural plan fields only; saved evaluation/objectives are unused."""

    with plan_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_plan = json.load(file)

    return {
        key: raw_plan[key]
        for key in ALLOWED_PLAN_FIELDS
        if key in raw_plan
    }


def ratio_is_requested(value: float) -> bool:
    return any(
        math.isclose(
            value,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for expected in REQUESTED_MEMORY_SAVING_RATIOS
    )


def plan_fingerprint(plan: dict) -> tuple:
    return (
        plan["plan_kind"],
        plan.get("selected_score_metric", ""),
        plan.get("vector_objective", ""),
        plan.get("requested_memory_saving_ratio"),
        plan["actual_total_memory_bytes"],
        tuple(sorted(plan["layer_actions"].items())),
    )


def add_deduplicated_plan(
    plans_by_id: dict[str, dict],
    plan: dict,
    source_path: Path,
) -> None:
    plan_id = str(plan["plan_id"])

    if plan_id in plans_by_id:
        existing = plans_by_id[plan_id]

        if plan_fingerprint(existing) != plan_fingerprint(plan):
            raise ValueError(
                f"Conflicting duplicate plan_id: {plan_id}"
            )

        existing["source_plan_paths"].append(
            str(source_path.resolve())
        )
        return

    plan["source_plan_paths"] = [
        str(source_path.resolve())
    ]
    plans_by_id[plan_id] = plan


def validate_structural_plan(
    plan: dict,
    plan_path: Path,
) -> None:
    required = {
        "plan_id",
        "plan_kind",
        "actual_memory_saving_ratio",
        "actual_total_memory_bytes",
        "layer_actions",
    }
    missing = required - set(plan)

    if missing:
        raise ValueError(
            f"{plan_path} is missing structural fields: "
            f"{sorted(missing)}"
        )

    if not plan["layer_actions"]:
        raise ValueError(f"{plan_path} has no layer actions.")


def load_selected_plans(
    vector_plan_dir: Path,
    scalar_plan_dirs: list[Path],
) -> list[dict]:
    plans_by_id = {}
    vector_paths = sorted(
        vector_plan_dir.glob("plan_*.json")
    )

    if not vector_paths:
        raise FileNotFoundError(
            f"No vector plan JSON files found in {vector_plan_dir}."
        )

    for plan_path in vector_paths:
        plan = read_plan_structure(plan_path)
        validate_structural_plan(plan, plan_path)

        if plan["plan_kind"] != "vector_beam":
            raise ValueError(
                f"Unexpected non-vector plan in {plan_path}."
            )

        if "vector_objective" not in plan:
            raise ValueError(
                f"{plan_path} lacks vector_objective."
            )

        ratio = float(plan["requested_memory_saving_ratio"])

        if not ratio_is_requested(ratio):
            raise ValueError(
                f"Vector plan has unexpected requested ratio: {ratio}"
            )

        plan["comparison_role"] = "primary"
        add_deduplicated_plan(
            plans_by_id,
            plan,
            plan_path,
        )

    for scalar_plan_dir in scalar_plan_dirs:
        scalar_paths = sorted(
            scalar_plan_dir.glob("plan_*.json")
        )

        if not scalar_paths:
            raise FileNotFoundError(
                f"No scalar plan JSON files found in {scalar_plan_dir}."
            )

        for plan_path in scalar_paths:
            plan = read_plan_structure(plan_path)
            validate_structural_plan(plan, plan_path)
            plan_kind = plan["plan_kind"]

            if plan_kind == "optimized":
                ratio = float(
                    plan["requested_memory_saving_ratio"]
                )

                if not ratio_is_requested(ratio):
                    continue

                score_metric = plan.get(
                    "selected_score_metric"
                )

                if score_metric in PRIMARY_SCALAR_METRICS:
                    plan["comparison_role"] = "primary"
                elif score_metric == SECONDARY_SCALAR_METRIC:
                    plan["comparison_role"] = "secondary"
                else:
                    raise ValueError(
                        f"Unexpected scalar score metric in {plan_path}: "
                        f"{score_metric}"
                    )
            elif plan_kind == "uniform_anchor":
                plan["comparison_role"] = "secondary"
            else:
                raise ValueError(
                    f"Unexpected scalar plan kind in {plan_path}: "
                    f"{plan_kind}"
                )

            add_deduplicated_plan(
                plans_by_id,
                plan,
                plan_path,
            )

    plans = list(plans_by_id.values())

    if len(plans) != len(
        {plan["plan_id"] for plan in plans}
    ):
        raise RuntimeError(
            "Duplicate plan IDs remain after deduplication."
        )

    return plans


def model_memory_accounting(
    reference_model,
) -> tuple[list[str], int, int]:
    layer_names = list_quantizable_layers(
        reference_model
    )
    constant_bytes = constant_parameter_storage_bytes(
        model=reference_model,
        layer_names=layer_names,
    )
    fp32_total_bytes = (
        constant_bytes
        + sum(
            weight_storage_bytes(
                weight_numel=(
                    reference_model.get_submodule(
                        layer_name
                    ).weight.numel()
                ),
                bits=32,
            )
            for layer_name in layer_names
        )
    )

    return layer_names, constant_bytes, fp32_total_bytes


def validate_plan_memory(
    plan: dict,
    reference_model,
    layer_names: list[str],
    constant_parameter_bytes: int,
    fp32_total_memory_bytes: int,
) -> tuple[dict[str, int], dict[str, int], int, float]:
    layer_actions = {
        str(layer_name): str(action_name)
        for layer_name, action_name in (
            plan["layer_actions"].items()
        )
    }

    if set(layer_actions) != set(layer_names):
        missing = set(layer_names) - set(layer_actions)
        extra = set(layer_actions) - set(layer_names)
        raise ValueError(
            f"Plan {plan['plan_id']} layer mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    if len(layer_actions) != len(layer_names):
        raise ValueError(
            f"Plan {plan['plan_id']} does not have unique layer actions."
        )

    per_layer_bytes = {}
    counts = {action: 0 for action in ACTION_TO_BITS}

    for layer_name in layer_names:
        action_name = layer_actions[layer_name]

        if action_name not in ACTION_TO_BITS:
            raise ValueError(
                f"Plan {plan['plan_id']} has invalid action "
                f"{action_name}."
            )

        bits = action_to_bits(action_name)
        weight_numel = (
            reference_model.get_submodule(
                layer_name
            ).weight.numel()
        )
        per_layer_bytes[layer_name] = weight_storage_bytes(
            weight_numel=weight_numel,
            bits=bits,
        )
        counts[action_name] += 1

    actual_total_bytes = (
        constant_parameter_bytes
        + sum(per_layer_bytes.values())
    )

    if actual_total_bytes != int(
        plan["actual_total_memory_bytes"]
    ):
        raise ValueError(
            f"Plan {plan['plan_id']} stored memory does not match "
            "exact action accounting."
        )

    if (
        "fp32_total_memory_bytes" in plan
        and int(plan["fp32_total_memory_bytes"])
        != fp32_total_memory_bytes
    ):
        raise ValueError(
            f"Plan {plan['plan_id']} FP32 memory metadata differs."
        )

    stored_per_layer = plan.get("per_layer_action_bytes")

    if stored_per_layer is not None and {
        str(key): int(value)
        for key, value in stored_per_layer.items()
    } != per_layer_bytes:
        raise ValueError(
            f"Plan {plan['plan_id']} per-layer bytes differ."
        )

    actual_saving_ratio = (
        1.0
        - actual_total_bytes / fp32_total_memory_bytes
    )

    if not math.isclose(
        actual_saving_ratio,
        float(plan["actual_memory_saving_ratio"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Plan {plan['plan_id']} saving ratio differs from "
            "exact accounting."
        )

    return (
        layer_actions,
        counts,
        actual_total_bytes,
        actual_saving_ratio,
    )


def evaluate_plan(
    plan: dict,
    reference_model,
    confirmation_loader: DataLoader,
    device: torch.device,
    layer_names: list[str],
    constant_parameter_bytes: int,
    fp32_total_memory_bytes: int,
) -> dict:
    (
        layer_actions,
        counts,
        _,
        actual_saving_ratio,
    ) = validate_plan_memory(
        plan=plan,
        reference_model=reference_model,
        layer_names=layer_names,
        constant_parameter_bytes=(
            constant_parameter_bytes
        ),
        fp32_total_memory_bytes=(
            fp32_total_memory_bytes
        ),
    )
    layer_bits = {
        layer_name: action_to_bits(action_name)
        for layer_name, action_name in layer_actions.items()
    }
    candidate_model = build_mixed_quantized_model(
        fp32_model=reference_model,
        layer_bits=layer_bits,
        default_bits=32,
        device=device,
    )

    try:
        measured = evaluate_oracle_set(
            base_model=reference_model,
            candidate_model=candidate_model,
            dataloader=confirmation_loader,
            device=device,
        )
    finally:
        del candidate_model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    requested_ratio = plan.get(
        "requested_memory_saving_ratio"
    )

    return {
        "plan_id": plan["plan_id"],
        "plan_kind": plan["plan_kind"],
        "selected_score_metric": plan.get(
            "selected_score_metric",
            "",
        ),
        "vector_objective": plan.get(
            "vector_objective",
            "",
        ),
        "requested_memory_saving_ratio": (
            float(requested_ratio)
            if requested_ratio is not None
            else None
        ),
        "actual_memory_saving_ratio": (
            actual_saving_ratio
        ),
        "comparison_role": plan["comparison_role"],
        "confirmation_accuracy": measured[
            "oracle_accuracy"
        ],
        "confirmation_teacher_agreement": measured[
            "oracle_teacher_agreement"
        ],
        "confirmation_flip_rate": measured[
            "oracle_flip_rate"
        ],
        "confirmation_abs_delta_score_mean": measured[
            "oracle_abs_delta_score_mean"
        ],
        "confirmation_decision_risk_mean": measured[
            "oracle_decision_risk_mean"
        ],
        "confirmation_decision_risk_p95": measured[
            "oracle_decision_risk_p95"
        ],
        "confirmation_decision_risk_violation_rate": (
            measured[
                "oracle_decision_risk_violation_rate"
            ]
        ),
        "runtime_seconds": measured[
            "oracle_runtime_seconds"
        ],
        "num_fp32": counts["fp32"],
        "num_fp16": counts["fp16"],
        "num_int8": counts["int8"],
        "num_int4": counts["int4"],
        "source_plan_path": ";".join(
            plan["source_plan_paths"]
        ),
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    rows: list[dict],
    output_paths: dict[str, Path],
    confirmation_split_path: Path,
    confirmation_metadata_path: Path,
    confirmation_metadata: dict,
    device: torch.device,
) -> None:
    output_paths["results_csv"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    write_csv(rows, output_paths["results_csv"])
    primary_rows = [
        row
        for row in rows
        if row["comparison_role"] == "primary"
    ]
    write_csv(primary_rows, output_paths["primary_csv"])
    payload = {
        "schema_version": (
            "go_no_go_confirmation_plan_evaluation_v1"
        ),
        "confirmation_split_provenance": {
            "split_path": str(
                confirmation_split_path.resolve()
            ),
            "metadata_path": str(
                confirmation_metadata_path.resolve()
            ),
            "source_split": confirmation_metadata[
                "source_split"
            ],
            "confirmation_seed": confirmation_metadata[
                "confirmation_seed"
            ],
            "sample_count": confirmation_metadata[
                "requested_size"
            ],
            "class_counts": confirmation_metadata[
                "class_counts"
            ],
            "overlap_counts": confirmation_metadata[
                "overlap_counts"
            ],
        },
        "device": str(device),
        "data_access_policy": {
            "confirmation_split_only": True,
            "development_oracle_metrics_used": False,
            "score_delta_vectors_read": False,
            "test_split_read": False,
            "planner_objectives_used_for_selection": False,
        },
        "statistical_significance_claimed": False,
        "results": rows,
    }

    with output_paths["results_json"].open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file.write("\n")


def selection_label(row: pd.Series) -> str:
    return (
        str(row["vector_objective"])
        if row["vector_objective"]
        else str(row["selected_score_metric"])
    )


def print_analysis_tables(rows: list[dict]) -> None:
    results = pd.DataFrame(rows)
    results["selection"] = results.apply(
        selection_label,
        axis=1,
    )

    for ratio in REQUESTED_MEMORY_SAVING_RATIOS:
        selected = results[
            results["comparison_role"].eq("primary")
            & results[
                "requested_memory_saving_ratio"
            ].eq(ratio)
        ].sort_values(
            by=[
                "confirmation_flip_rate",
                "confirmation_decision_risk_p95",
            ],
            ascending=True,
            kind="stable",
        )
        columns = [
            "plan_kind",
            "selection",
            "actual_memory_saving_ratio",
            "confirmation_flip_rate",
            "confirmation_decision_risk_p95",
            "confirmation_accuracy",
        ]
        print(
            "\nPrimary plans at requested memory saving "
            f"{ratio:.2%}"
        )
        print(selected[columns].to_string(index=False))

    compact = results[
        results["comparison_role"].eq("primary")
        & results["selection"].isin(COMPACT_SELECTIONS)
    ].sort_values(
        by=[
            "requested_memory_saving_ratio",
            "confirmation_flip_rate",
            "confirmation_decision_risk_p95",
        ],
        ascending=True,
        kind="stable",
    )
    compact_columns = [
        "requested_memory_saving_ratio",
        "plan_kind",
        "selection",
        "actual_memory_saving_ratio",
        "confirmation_flip_rate",
        "confirmation_decision_risk_p95",
        "confirmation_accuracy",
    ]
    print("\nCompact vector/scalar confirmation comparison")
    print(compact[compact_columns].to_string(index=False))
    print(
        "\nThese confirmation results are descriptive; no "
        "statistical significance is claimed."
    )


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    output_paths = preflight_outputs(args.output_dir)
    confirmation_metadata = load_confirmation_metadata(
        args.confirmation_metadata
    )
    (
        confirmation_indices,
        confirmation_source_indices,
        _,
    ) = load_confirmation_indices(
        args.confirmation_split
    )
    print(
        "Confirmation provenance: "
        f"source={confirmation_metadata['source_split']}, "
        f"seed={confirmation_metadata['confirmation_seed']}, "
        f"samples={confirmation_metadata['requested_size']}, "
        f"schema={confirmation_metadata['schema_version']}"
    )
    print(
        "Confirmation disjointness: development oracle overlap=0; "
        "all score-set overlaps=0 (validated from metadata)."
    )
    confirmation_loader = build_confirmation_loader(
        dataset_indices=confirmation_indices,
        source_indices=confirmation_source_indices,
        batch_size=args.batch_size,
    )
    plans = load_selected_plans(
        vector_plan_dir=args.vector_plan_dir,
        scalar_plan_dirs=args.scalar_plan_dirs,
    )
    deterministic_seed = int(
        confirmation_metadata["confirmation_seed"]
    )
    set_seed(deterministic_seed)
    device = get_device()
    reference_model = load_reference_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    (
        layer_names,
        constant_parameter_bytes,
        fp32_total_memory_bytes,
    ) = model_memory_accounting(reference_model)
    rows = []

    for plan_index, plan in enumerate(plans, start=1):
        print(
            f"[{plan_index}/{len(plans)}] "
            f"Evaluating {plan['plan_id']}"
        )
        rows.append(
            evaluate_plan(
                plan=plan,
                reference_model=reference_model,
                confirmation_loader=confirmation_loader,
                device=device,
                layer_names=layer_names,
                constant_parameter_bytes=(
                    constant_parameter_bytes
                ),
                fp32_total_memory_bytes=(
                    fp32_total_memory_bytes
                ),
            )
        )

    if len(rows) != len({row["plan_id"] for row in rows}):
        raise RuntimeError(
            "Duplicate plan IDs found after evaluation."
        )

    save_results(
        rows=rows,
        output_paths=output_paths,
        confirmation_split_path=args.confirmation_split,
        confirmation_metadata_path=(
            args.confirmation_metadata
        ),
        confirmation_metadata=confirmation_metadata,
        device=device,
    )
    print_analysis_tables(rows)
    print(f"\nDevice used: {device}")
    print(f"Saved confirmation evaluation: {args.output_dir}")


if __name__ == "__main__":
    main()
