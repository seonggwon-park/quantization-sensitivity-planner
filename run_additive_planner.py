from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from additive_planner import (
    build_layer_action_options,
    minimum_total_memory_bytes,
    solve_additive_plan,
)
from config import ExperimentConfig
from data import build_dataloaders, make_fixed_subset_loader
from experiment_logger import record_experiment
from metrics import compare_binary_models
from model import load_binary_resnet18_checkpoint
from quantization import (
    build_mixed_quantized_model,
    estimate_parameter_memory_mb,
    list_quantizable_layers,
)
from utils import get_device, set_seed


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ExperimentConfig().checkpoint_path,
    )

    parser.add_argument(
        "--risk-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--risk-metric",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--risk-label",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--memory-saving-ratios",
        type=float,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--evaluation-split",
        choices=["validation", "test"],
        default="test",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--memory-quantum-kb",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--results-output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--allocations-output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def normalize_label(label: str) -> str:
    normalized = (
        label.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if not normalized:
        raise ValueError(
            "risk_label must not be empty."
        )

    return normalized


def save_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            "No rows to save."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def mb_to_bytes(memory_mb: float) -> int:
    return int(memory_mb * 1024 * 1024)


def bytes_to_mb(memory_bytes: int) -> float:
    return memory_bytes / (1024 * 1024)


def format_budget_key(
    memory_saving_ratio: float,
) -> str:
    basis_points = int(
        round(memory_saving_ratio * 10_000)
    )

    return f"save_{basis_points:04d}bp"


def main():
    args = parse_arguments()

    if args.memory_quantum_kb <= 0:
        raise ValueError(
            "memory_quantum_kb must be positive."
        )

    for ratio in args.memory_saving_ratios:
        if ratio < 0 or ratio >= 1:
            raise ValueError(
                "memory-saving ratios must be in [0, 1)."
            )

    config = ExperimentConfig()
    config.create_directories()

    risk_label = normalize_label(
        args.risk_label
    )

    memory_quantum_bytes = (
        args.memory_quantum_kb * 1024
    )

    set_seed(config.seed)
    device = get_device()

    dataloaders = build_dataloaders(config)

    evaluation_loader = make_fixed_subset_loader(
        original_loader=dataloaders[
            args.evaluation_split
        ],
        max_samples=args.max_samples,
        seed=config.seed,
    )

    fp32_model, _ = load_binary_resnet18_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    layer_names = list_quantizable_layers(
        fp32_model
    )

    options_by_layer = build_layer_action_options(
        model=fp32_model,
        risk_csv=args.risk_csv,
        risk_metric=args.risk_metric,
        memory_quantum_bytes=memory_quantum_bytes,
    )

    fp32_memory_mb = estimate_parameter_memory_mb(
        model=fp32_model,
        layer_bits={},
        default_bits=32,
    )

    minimum_memory_bytes = minimum_total_memory_bytes(
        model=fp32_model,
        options_by_layer=options_by_layer,
    )

    minimum_memory_mb = bytes_to_mb(
        minimum_memory_bytes
    )

    maximum_saving_ratio = (
        1.0 - minimum_memory_mb / fp32_memory_mb
    )

    if args.results_output is None:
        results_output = (
            config.result_dir
            / f"{risk_label}_additive_planner_results.csv"
        )
    else:
        results_output = args.results_output

    if args.allocations_output is None:
        allocations_output = (
            config.result_dir
            / f"{risk_label}_additive_planner_allocations.csv"
        )
    else:
        allocations_output = args.allocations_output

    result_rows = []
    allocation_rows = []

    for memory_saving_ratio in (
        args.memory_saving_ratios
    ):
        target_memory_mb = (
            fp32_memory_mb
            * (1.0 - memory_saving_ratio)
        )

        if target_memory_mb < minimum_memory_mb:
            raise ValueError(
                f"Requested saving ratio "
                f"{memory_saving_ratio:.6f} is infeasible. "
                f"Maximum feasible saving ratio is "
                f"{maximum_saving_ratio:.6f}."
            )

        print(
            "\n"
            f"Planning for target saving ratio: "
            f"{memory_saving_ratio:.4f}"
        )

        planner_start = time.perf_counter()

        plan = solve_additive_plan(
            model=fp32_model,
            options_by_layer=options_by_layer,
            target_total_memory_bytes=mb_to_bytes(
                target_memory_mb
            ),
            memory_quantum_bytes=memory_quantum_bytes,
        )

        planner_runtime_seconds = (
            time.perf_counter() - planner_start
        )

        quantized_model = build_mixed_quantized_model(
            fp32_model=fp32_model,
            layer_bits=plan.layer_bits,
            default_bits=4,
            device=device,
        )

        metrics = compare_binary_models(
            fp32_model=fp32_model,
            quantized_model=quantized_model,
            dataloader=evaluation_loader,
            device=device,
        )

        actual_memory_mb = estimate_parameter_memory_mb(
            model=fp32_model,
            layer_bits=plan.layer_bits,
            default_bits=4,
        )

        actual_memory_saving_ratio = (
            1.0 - actual_memory_mb / fp32_memory_mb
        )

        action_counts = {
            action: 0
            for action in (
                "fp32",
                "fp16",
                "int8",
                "int4",
            )
        }

        for option in plan.selected_options.values():
            action_counts[option.action] += 1

        upgraded_layers = [
            f"{layer_name}:{option.action}"
            for layer_name, option in (
                plan.selected_options.items()
            )
            if option.action != "int4"
        ]

        result_row = {
            "risk_label": risk_label,
            "risk_csv": str(args.risk_csv),
            "risk_metric": args.risk_metric,
            "evaluation_split": args.evaluation_split,
            "requested_memory_saving_ratio": (
                memory_saving_ratio
            ),
            "target_total_memory_mb": target_memory_mb,
            "actual_total_memory_mb": actual_memory_mb,
            "actual_memory_saving_ratio": (
                actual_memory_saving_ratio
            ),
            "memory_budget_gap_mb": (
                target_memory_mb - actual_memory_mb
            ),
            "planner_objective": plan.objective_value,
            "planner_runtime_seconds": (
                planner_runtime_seconds
            ),
            "memory_quantum_kb": args.memory_quantum_kb,
            "fp32_layer_count": action_counts["fp32"],
            "fp16_layer_count": action_counts["fp16"],
            "int8_layer_count": action_counts["int8"],
            "int4_layer_count": action_counts["int4"],
            "upgraded_layers": ",".join(
                upgraded_layers
            ),
            **metrics,
        }

        result_rows.append(result_row)

        for layer_name in layer_names:
            option = plan.selected_options[layer_name]

            allocation_rows.append(
                {
                    "risk_label": risk_label,
                    "requested_memory_saving_ratio": (
                        memory_saving_ratio
                    ),
                    "layer": layer_name,
                    "action": option.action,
                    "bits": option.bits,
                    "risk_value": option.risk,
                    "weight_numel": option.weight_numel,
                    "estimated_weight_memory_mb": (
                        option.exact_weight_bytes
                        / (1024 * 1024)
                    ),
                }
            )

        save_csv(
            rows=result_rows,
            output_path=results_output,
        )

        save_csv(
            rows=allocation_rows,
            output_path=allocations_output,
        )

        print(
            f"Objective: {plan.objective_value:.6f}"
        )

        print(
            "Actual memory saving: "
            f"{actual_memory_saving_ratio:.4%}"
        )

        print(
            "Flip rate: "
            f"{metrics['flip_rate']:.4%}"
        )

        print(
            "Upgraded layers: "
            f"{upgraded_layers}"
        )

        del quantized_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_metrics = {}

    for row in result_rows:
        budget_key = format_budget_key(
            row["requested_memory_saving_ratio"]
        )

        summary_metrics[
            f"{budget_key}_flip_rate"
        ] = row["flip_rate"]

        summary_metrics[
            f"{budget_key}_quantized_accuracy"
        ] = row["quantized_accuracy"]

        summary_metrics[
            f"{budget_key}_actual_memory_saving_ratio"
        ] = row[
            "actual_memory_saving_ratio"
        ]

        summary_metrics[
            f"{budget_key}_planner_objective"
        ] = row["planner_objective"]

    record_experiment(
        run_name=(
            f"{risk_label}_additive_"
            "mixed_precision_planner"
        ),
        config={
            "task": config.class_names,
            "model": "binary ResNet-18",
            "risk_label": risk_label,
            "risk_csv": str(args.risk_csv),
            "risk_metric": args.risk_metric,
            "evaluation_split": args.evaluation_split,
            "max_samples": args.max_samples,
            "requested_memory_saving_ratios": (
                args.memory_saving_ratios
            ),
            "memory_quantum_kb": (
                args.memory_quantum_kb
            ),
            "planner": (
                "multiple-choice knapsack dynamic programming "
                "with additive layer-action risk"
            ),
            "quantization": (
                "mixed weight-only fake quantization, "
                "per-output-channel symmetric"
            ),
        },
        metrics=summary_metrics,
        artifacts={
            "results_csv": str(results_output),
            "allocations_csv": str(allocations_output),
        },
    )

    print(
        f"\nSaved planner results: "
        f"{results_output}"
    )

    print(
        f"Saved planner allocations: "
        f"{allocations_output}"
    )


if __name__ == "__main__":
    main()