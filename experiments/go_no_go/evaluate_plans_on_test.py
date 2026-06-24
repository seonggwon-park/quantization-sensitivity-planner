"""Run locked, evaluation-only plan measurements on binary CIFAR-10 test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from additive_planner import action_to_bits
from config import ExperimentConfig
from quantization import build_mixed_quantized_model
from utils import get_device, set_seed

from experiments.go_no_go.adapters import (
    build_binary_eval_dataset,
    get_checkpoint_default_path,
    load_reference_model,
)
from experiments.go_no_go.evaluate_plans_on_confirmation import (
    COMPACT_SELECTIONS,
    REQUESTED_MEMORY_SAVING_RATIOS,
    load_selected_plans,
    model_memory_accounting,
    validate_plan_memory,
)
from experiments.go_no_go.metrics import evaluate_oracle_set
from experiments.go_no_go.splits import extract_binary_labels


RESULT_COLUMNS = (
    "plan_id",
    "plan_kind",
    "selection",
    "requested_memory_saving_ratio",
    "actual_memory_saving_ratio",
    "comparison_role",
    "test_accuracy",
    "test_teacher_agreement",
    "test_flip_rate",
    "test_abs_delta_score_mean",
    "test_decision_risk_mean",
    "test_decision_risk_p95",
    "test_decision_risk_violation_rate",
    "runtime_seconds",
    "num_fp32",
    "num_fp16",
    "num_int8",
    "num_int4",
    "source_plan_path",
)


def parse_arguments():
    config = ExperimentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=get_checkpoint_default_path(),
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
        default=Path("results/go_no_go_test_eval_v1"),
    )

    return parser.parse_args()


def validate_arguments(args) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")


def preflight_outputs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "results_csv": output_dir / "test_plan_results.csv",
        "results_json": output_dir / "test_plan_results.json",
        "primary_csv": output_dir / "test_primary_summary.csv",
    }

    for output_path in paths.values():
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )

    return paths


def build_full_test_loader(
    batch_size: int,
) -> tuple[DataLoader, int, dict[int, int]]:
    dataset = build_binary_eval_dataset(
        source_split="test"
    )
    labels = extract_binary_labels(dataset)
    class_counts = {
        class_id: sum(
            label == class_id for label in labels
        )
        for class_id in (0, 1)
    }
    expected_source_indices = [
        index
        for index, original_label in enumerate(
            dataset.base_dataset.targets
        )
        if int(original_label) in dataset.class_ids
    ]
    expected_count = len(expected_source_indices)

    if len(dataset) != expected_count:
        raise ValueError(
            "Binary test dataset size differs from direct CIFAR target "
            "filtering."
        )

    if tuple(dataset.source_indices) != tuple(
        expected_source_indices
    ):
        raise ValueError(
            "Binary test dataset source indices differ from project "
            "class filtering."
        )

    if sum(class_counts.values()) != expected_count:
        raise ValueError(
            "Binary test class counts do not sum to expected size."
        )

    if class_counts[0] != class_counts[1]:
        raise ValueError(
            "Binary CIFAR-10 test classes are not balanced."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, expected_count, class_counts


def plan_selection(plan: dict) -> str:
    vector_objective = plan.get("vector_objective")

    if vector_objective:
        return str(vector_objective)

    selected_score_metric = plan.get(
        "selected_score_metric"
    )

    if not selected_score_metric:
        raise ValueError(
            f"Plan {plan['plan_id']} lacks a selection label."
        )

    return str(selected_score_metric)


def evaluate_plan_on_test(
    plan: dict,
    reference_model,
    test_loader: DataLoader,
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
            dataloader=test_loader,
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
        "selection": plan_selection(plan),
        "requested_memory_saving_ratio": (
            float(requested_ratio)
            if requested_ratio is not None
            else None
        ),
        "actual_memory_saving_ratio": (
            actual_saving_ratio
        ),
        "comparison_role": plan["comparison_role"],
        "test_accuracy": measured["oracle_accuracy"],
        "test_teacher_agreement": measured[
            "oracle_teacher_agreement"
        ],
        "test_flip_rate": measured["oracle_flip_rate"],
        "test_abs_delta_score_mean": measured[
            "oracle_abs_delta_score_mean"
        ],
        "test_decision_risk_mean": measured[
            "oracle_decision_risk_mean"
        ],
        "test_decision_risk_p95": measured[
            "oracle_decision_risk_p95"
        ],
        "test_decision_risk_violation_rate": measured[
            "oracle_decision_risk_violation_rate"
        ],
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
    test_sample_count: int,
    test_class_counts: dict[int, int],
    checkpoint_path: Path,
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
        "schema_version": "go_no_go_locked_test_evaluation_v1",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "test_dataset": {
            "source_split": "test",
            "full_binary_test_set": True,
            "subsampled": False,
            "sample_count": test_sample_count,
            "class_counts": {
                str(class_id): count
                for class_id, count in (
                    test_class_counts.items()
                )
            },
        },
        "device": str(device),
        "data_access_policy": {
            "locked_test_evaluation_only": True,
            "test_used_for_plan_generation": False,
            "test_used_for_method_selection": False,
            "development_oracle_metrics_used": False,
            "confirmation_metrics_used": False,
            "score_delta_vectors_read": False,
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


def print_test_tables(rows: list[dict]) -> None:
    results = pd.DataFrame(rows)

    for ratio in REQUESTED_MEMORY_SAVING_RATIOS:
        selected = results[
            results["comparison_role"].eq("primary")
            & results[
                "requested_memory_saving_ratio"
            ].eq(ratio)
        ].sort_values(
            by=[
                "test_flip_rate",
                "test_decision_risk_p95",
            ],
            ascending=True,
            kind="stable",
        )
        columns = [
            "plan_kind",
            "selection",
            "actual_memory_saving_ratio",
            "test_flip_rate",
            "test_decision_risk_p95",
            "test_accuracy",
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
            "test_flip_rate",
            "test_decision_risk_p95",
        ],
        ascending=True,
        kind="stable",
    )
    compact_columns = [
        "requested_memory_saving_ratio",
        "plan_kind",
        "selection",
        "actual_memory_saving_ratio",
        "test_flip_rate",
        "test_decision_risk_p95",
        "test_accuracy",
    ]
    print("\nCompact locked-test vector/scalar comparison")
    print(compact[compact_columns].to_string(index=False))
    print(
        "\nLOCKED TEST EVALUATION ONLY: these results must not be "
        "used to retune objectives, beam settings, budgets, baselines, "
        "or method choices."
    )
    print(
        "Results are descriptive; no statistical significance is "
        "claimed."
    )


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    output_paths = preflight_outputs(args.output_dir)
    plans = load_selected_plans(
        vector_plan_dir=args.vector_plan_dir,
        scalar_plan_dirs=args.scalar_plan_dirs,
    )
    test_loader, test_sample_count, test_class_counts = (
        build_full_test_loader(args.batch_size)
    )
    print(f"Full binary test sample count: {test_sample_count}")
    print(f"Full binary test class counts: {test_class_counts}")
    config = ExperimentConfig()
    set_seed(config.seed)
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
            evaluate_plan_on_test(
                plan=plan,
                reference_model=reference_model,
                test_loader=test_loader,
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
        test_sample_count=test_sample_count,
        test_class_counts=test_class_counts,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    print_test_tables(rows)
    print(f"\nDevice used: {device}")
    print(f"Saved locked test evaluation: {args.output_dir}")


if __name__ == "__main__":
    main()
