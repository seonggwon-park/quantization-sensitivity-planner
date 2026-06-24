"""Compare rank-normalized score metrics with the existing additive solver."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from additive_planner import (
    QUANTIZED_ACTIONS,
    build_layer_action_options,
    constant_parameter_storage_bytes,
    minimum_total_memory_bytes,
    solve_additive_plan,
    weight_storage_bytes,
)
from config import ExperimentConfig
from quantization import (
    build_mixed_quantized_model,
    list_quantizable_layers,
)
from utils import get_device, set_seed

from experiments.go_no_go import DEFAULT_RESULTS_DIR
from experiments.go_no_go.adapters import (
    build_binary_eval_dataset,
    get_checkpoint_default_path,
    load_reference_model,
)
from experiments.go_no_go.metrics import evaluate_oracle_set


PLANNER_SCORE_METRICS = (
    "weight_rel_l2",
    "activation_rel_mse",
    "output_kl_mean",
    "abs_delta_score_mean",
    "decision_risk_mean",
    "decision_risk_p95",
    "decision_risk_violation_rate",
)

ANCHOR_ACTIONS = (
    "fp16",
    "int8",
    "int4",
)

RANK_RISK_COLUMN = "rank_normalized_risk"

SUMMARY_COLUMNS = (
    "plan_id",
    "plan_kind",
    "selected_score_metric",
    "anchor_action",
    "requested_memory_saving_ratio",
    "target_total_memory_bytes",
    "actual_total_memory_bytes",
    "actual_memory_saving_ratio",
    "meets_requested_budget",
    "planner_objective",
    "planner_runtime_seconds",
    "fp32_layer_count",
    "fp16_layer_count",
    "int8_layer_count",
    "int4_layer_count",
    "oracle_flip_rate",
    "oracle_teacher_agreement",
    "oracle_accuracy",
    "oracle_abs_delta_score_mean",
    "oracle_decision_risk_mean",
    "oracle_decision_risk_p95",
    "oracle_decision_risk_violation_rate",
    "oracle_runtime_seconds",
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
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory containing "
            "single_action_metrics_seed*.csv."
        ),
    )
    parser.add_argument(
        "--split-indices",
        type=Path,
        default=(
            DEFAULT_RESULTS_DIR / "split_indices.npz"
        ),
    )
    parser.add_argument(
        "--memory-saving-ratios",
        type=float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--memory-quantum-kb",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.batch_size,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def budget_key(memory_saving_ratio: float) -> str:
    basis_points = int(
        round(memory_saving_ratio * 10_000)
    )

    return f"save_{basis_points:04d}bp"


def validate_arguments(args) -> list[float]:
    if args.memory_quantum_kb <= 0:
        raise ValueError(
            "memory_quantum_kb must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    ratios = [
        float(ratio)
        for ratio in args.memory_saving_ratios
    ]

    if len(set(ratios)) != len(ratios):
        raise ValueError(
            "memory-saving ratios must be unique."
        )

    for ratio in ratios:
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError(
                "memory-saving ratios must be in [0, 1)."
            )

    keys = [budget_key(ratio) for ratio in ratios]

    if len(set(keys)) != len(keys):
        raise ValueError(
            "Requested ratios collide after basis-point filename "
            "normalization."
        )

    return ratios


def discover_benchmark_paths(
    benchmark_dir: Path,
) -> list[Path]:
    paths = sorted(
        benchmark_dir.glob(
            "single_action_metrics_seed*.csv"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No single_action_metrics_seed*.csv files found in "
            f"{benchmark_dir}."
        )

    return paths


def load_benchmark_metrics(
    benchmark_paths: list[Path],
) -> pd.DataFrame:
    """Load score rows into memory without changing source CSVs."""

    required_columns = {
        "score_seed",
        "layer_name",
        "action",
        *PLANNER_SCORE_METRICS,
    }
    frames = []
    seen_seeds = set()

    for benchmark_path in benchmark_paths:
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

        if len(seeds) != 1:
            raise ValueError(
                f"{benchmark_path} must contain one score seed."
            )

        score_seed = int(seeds[0])

        if score_seed in seen_seeds:
            raise ValueError(
                f"Duplicate benchmark for score_seed={score_seed}."
            )

        seen_seeds.add(score_seed)
        selected = dataframe[
            [
                "score_seed",
                "layer_name",
                "action",
                *PLANNER_SCORE_METRICS,
            ]
        ].copy()

        if selected.duplicated(
            subset=["layer_name", "action"]
        ).any():
            raise ValueError(
                f"{benchmark_path} has duplicate layer/action rows."
            )

        frames.append(selected)

    combined = pd.concat(frames, ignore_index=True)
    expected_candidates = None

    for score_seed, selected in combined.groupby(
        "score_seed",
        sort=True,
    ):
        candidate_set = set(
            zip(
                selected["layer_name"],
                selected["action"],
            )
        )

        if expected_candidates is None:
            expected_candidates = candidate_set
        elif candidate_set != expected_candidates:
            raise ValueError(
                f"Candidate set differs for score_seed={score_seed}."
            )

    for score_metric in PLANNER_SCORE_METRICS:
        values = combined[score_metric].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Score metric {score_metric} has non-finite values."
            )

        if score_metric != "output_kl_mean" and (
            values < 0.0
        ).any():
            raise ValueError(
                f"Score metric {score_metric} has negative values."
            )

    tiny_negative_kl = (
        combined["output_kl_mean"].ge(-1e-8)
        & combined["output_kl_mean"].lt(0.0)
    )
    combined.loc[
        tiny_negative_kl,
        "output_kl_mean",
    ] = 0.0

    if combined["output_kl_mean"].lt(0.0).any():
        raise ValueError(
            "output_kl_mean contains values below -1e-8."
        )

    return combined


def rank_normalize_metric(
    benchmark: pd.DataFrame,
    score_metric: str,
) -> pd.DataFrame:
    """Rank per seed, then average percentile risk by layer/action."""

    if score_metric not in PLANNER_SCORE_METRICS:
        raise ValueError(
            f"Unsupported planner score metric: {score_metric}"
        )

    ranked = benchmark[
        [
            "score_seed",
            "layer_name",
            "action",
            score_metric,
        ]
    ].copy()
    ranked[RANK_RISK_COLUMN] = ranked.groupby(
        "score_seed",
        sort=False,
    )[score_metric].rank(
        method="average",
        ascending=True,
        pct=True,
    )

    aggregated = (
        ranked.groupby(
            ["layer_name", "action"],
            sort=False,
            as_index=False,
        )[RANK_RISK_COLUMN]
        .mean()
        .rename(columns={"layer_name": "layer"})
    )

    return aggregated


def build_oracle_loader(
    split_indices_path: Path,
    batch_size: int,
) -> DataLoader:
    """Reuse the saved, unchanged TRAIN oracle subset."""

    with np.load(split_indices_path) as split_data:
        if "oracle_dataset_indices" not in split_data:
            raise ValueError(
                f"{split_indices_path} lacks oracle_dataset_indices."
            )

        oracle_indices = split_data[
            "oracle_dataset_indices"
        ].astype(np.int64).tolist()

    if not oracle_indices:
        raise ValueError("Oracle index set is empty.")

    if len(oracle_indices) != len(set(oracle_indices)):
        raise ValueError("Oracle index set contains duplicates.")

    dataset = build_binary_eval_dataset(
        source_split="train"
    )

    if min(oracle_indices) < 0 or max(oracle_indices) >= len(
        dataset
    ):
        raise ValueError(
            "Oracle index set is outside the calibration pool."
        )

    return DataLoader(
        Subset(dataset, oracle_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def fp32_total_storage_bytes(
    model,
    layer_names: list[str],
) -> int:
    constant_bytes = constant_parameter_storage_bytes(
        model=model,
        layer_names=layer_names,
    )
    weight_bytes = sum(
        weight_storage_bytes(
            model.get_submodule(layer_name).weight.numel(),
            32,
        )
        for layer_name in layer_names
    )

    return constant_bytes + weight_bytes


def uniform_storage_bytes(
    model,
    layer_names: list[str],
    bits: int,
) -> int:
    constant_bytes = constant_parameter_storage_bytes(
        model=model,
        layer_names=layer_names,
    )
    weight_bytes = sum(
        weight_storage_bytes(
            model.get_submodule(layer_name).weight.numel(),
            bits,
        )
        for layer_name in layer_names
    )

    return constant_bytes + weight_bytes


def action_counts(
    layer_actions: dict[str, str],
) -> dict[str, int]:
    counts = {
        "fp32": 0,
        "fp16": 0,
        "int8": 0,
        "int4": 0,
    }

    for action_name in layer_actions.values():
        counts[action_name] += 1

    return counts


def evaluate_assignment(
    reference_model,
    layer_bits: dict[str, int],
    oracle_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    candidate_model = build_mixed_quantized_model(
        fp32_model=reference_model,
        layer_bits=layer_bits,
        default_bits=32,
        device=device,
    )

    try:
        return evaluate_oracle_set(
            base_model=reference_model,
            candidate_model=candidate_model,
            dataloader=oracle_loader,
            device=device,
        )
    finally:
        del candidate_model

        if device.type == "cuda":
            torch.cuda.empty_cache()


def make_summary_row(
    *,
    plan_id: str,
    plan_kind: str,
    selected_score_metric: str,
    anchor_action: str | None,
    requested_memory_saving_ratio: float,
    target_total_memory_bytes: int,
    actual_total_memory_bytes: int,
    fp32_total_bytes: int,
    planner_objective: float,
    planner_runtime_seconds: float,
    counts: dict[str, int],
    oracle_metrics: dict[str, float],
) -> dict:
    actual_saving_ratio = (
        1.0
        - actual_total_memory_bytes / fp32_total_bytes
    )

    return {
        "plan_id": plan_id,
        "plan_kind": plan_kind,
        "selected_score_metric": selected_score_metric,
        "anchor_action": anchor_action or "",
        "requested_memory_saving_ratio": (
            requested_memory_saving_ratio
        ),
        "target_total_memory_bytes": (
            target_total_memory_bytes
        ),
        "actual_total_memory_bytes": (
            actual_total_memory_bytes
        ),
        "actual_memory_saving_ratio": actual_saving_ratio,
        "meets_requested_budget": (
            actual_total_memory_bytes
            <= target_total_memory_bytes
        ),
        "planner_objective": planner_objective,
        "planner_runtime_seconds": planner_runtime_seconds,
        "fp32_layer_count": counts["fp32"],
        "fp16_layer_count": counts["fp16"],
        "int8_layer_count": counts["int8"],
        "int4_layer_count": counts["int4"],
        **oracle_metrics,
    }


def optimized_plan_payload(
    *,
    plan_id: str,
    score_metric: str,
    memory_saving_ratio: float,
    target_total_memory_bytes: int,
    fp32_total_bytes: int,
    memory_quantum_bytes: int,
    benchmark_paths: list[Path],
    plan,
    oracle_metrics: dict[str, float],
) -> dict:
    layer_actions = {
        layer_name: option.action
        for layer_name, option in plan.selected_options.items()
    }

    return {
        "plan_id": plan_id,
        "plan_kind": "optimized",
        "selected_score_metric": score_metric,
        "action_space": [
            "fp32",
            *QUANTIZED_ACTIONS,
        ],
        "fp32_policy": {
            "risk": 0.0,
            "bits": 32,
        },
        "rank_normalization": (
            "per-score-seed ascending average percentile rank; "
            "then mean rank by layer/action"
        ),
        "interaction_updates": False,
        "solver": (
            "additive_planner.solve_additive_plan"
        ),
        "benchmark_csvs": [
            str(path) for path in benchmark_paths
        ],
        "requested_memory_saving_ratio": (
            memory_saving_ratio
        ),
        "target_total_memory_bytes": (
            target_total_memory_bytes
        ),
        "fp32_total_memory_bytes": fp32_total_bytes,
        "actual_total_memory_bytes": (
            plan.actual_total_memory_bytes
        ),
        "actual_memory_saving_ratio": (
            1.0
            - plan.actual_total_memory_bytes
            / fp32_total_bytes
        ),
        "memory_quantum_bytes": memory_quantum_bytes,
        "planner_objective": plan.objective_value,
        "layer_actions": layer_actions,
        "layer_options": {
            layer_name: {
                "action": option.action,
                "bits": option.bits,
                "rank_normalized_risk": option.risk,
                "weight_numel": option.weight_numel,
                "exact_weight_bytes": option.exact_weight_bytes,
            }
            for layer_name, option in (
                plan.selected_options.items()
            )
        },
        "oracle_metrics": oracle_metrics,
    }


def anchor_plan_payload(
    *,
    plan_id: str,
    action_name: str,
    bits: int,
    layer_names: list[str],
    actual_total_memory_bytes: int,
    fp32_total_bytes: int,
    memory_saving_ratios: list[float],
    oracle_metrics: dict[str, float],
) -> dict:
    return {
        "plan_id": plan_id,
        "plan_kind": "uniform_anchor",
        "selected_score_metric": "uniform_anchor",
        "anchor_action": action_name,
        "action_space": [
            "fp32",
            *QUANTIZED_ACTIONS,
        ],
        "fp32_policy": {
            "risk": 0.0,
            "bits": 32,
        },
        "rank_normalization": None,
        "interaction_updates": False,
        "solver": None,
        "compared_at_memory_saving_ratios": (
            memory_saving_ratios
        ),
        "actual_total_memory_bytes": (
            actual_total_memory_bytes
        ),
        "actual_memory_saving_ratio": (
            1.0
            - actual_total_memory_bytes
            / fp32_total_bytes
        ),
        "layer_actions": {
            layer_name: action_name
            for layer_name in layer_names
        },
        "bits": bits,
        "oracle_metrics": oracle_metrics,
    }


def planned_output_paths(
    output_dir: Path,
    memory_saving_ratios: list[float],
) -> list[Path]:
    paths = [
        output_dir / "planner_comparison_summary.csv"
    ]

    for action_name in ANCHOR_ACTIONS:
        paths.append(
            output_dir
            / f"plan_uniform_{action_name}_anchor.json"
        )

    for ratio in memory_saving_ratios:
        key = budget_key(ratio)

        for score_metric in PLANNER_SCORE_METRICS:
            paths.append(
                output_dir
                / f"plan_{score_metric}_{key}.json"
            )

    return paths


def preflight_outputs(output_paths: list[Path]) -> None:
    for output_path in output_paths:
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )


def save_outputs(
    summary_rows: list[dict],
    plan_payloads: dict[Path, dict],
    summary_path: Path,
) -> None:
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    for output_path, payload in plan_payloads.items():
        with output_path.open(
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


def print_budget_tables(
    summary_rows: list[dict],
    memory_saving_ratios: list[float],
) -> None:
    summary = pd.DataFrame(summary_rows)

    for ratio in memory_saving_ratios:
        selected = summary[
            summary["requested_memory_saving_ratio"].eq(
                ratio
            )
        ].sort_values(
            by=[
                "oracle_flip_rate",
                "oracle_decision_risk_p95",
            ],
            ascending=True,
            kind="stable",
        )
        columns = [
            "plan_kind",
            "selected_score_metric",
            "anchor_action",
            "actual_memory_saving_ratio",
            "meets_requested_budget",
            "oracle_flip_rate",
            "oracle_decision_risk_p95",
            "oracle_accuracy",
        ]
        print(
            "\nBudget: requested memory saving "
            f"{ratio:.2%}"
        )
        print(selected[columns].to_string(index=False))


def main() -> None:
    args = parse_arguments()
    memory_saving_ratios = validate_arguments(args)
    output_paths = planned_output_paths(
        output_dir=args.output_dir,
        memory_saving_ratios=memory_saving_ratios,
    )
    preflight_outputs(output_paths)
    benchmark_paths = discover_benchmark_paths(
        args.benchmark_dir
    )
    benchmark = load_benchmark_metrics(
        benchmark_paths
    )
    score_seeds = sorted(
        int(seed)
        for seed in benchmark["score_seed"].unique()
    )
    set_seed(score_seeds[0])
    device = get_device()
    oracle_loader = build_oracle_loader(
        split_indices_path=args.split_indices,
        batch_size=args.batch_size,
    )
    reference_model = load_reference_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    layer_names = list_quantizable_layers(
        reference_model
    )
    fp32_total_bytes = fp32_total_storage_bytes(
        model=reference_model,
        layer_names=layer_names,
    )
    memory_quantum_bytes = (
        args.memory_quantum_kb * 1024
    )
    summary_rows = []
    plan_payloads = {}
    anchor_records = []

    for action_name in ANCHOR_ACTIONS:
        bits = {
            "fp16": 16,
            "int8": 8,
            "int4": 4,
        }[action_name]
        layer_bits = {
            layer_name: bits
            for layer_name in layer_names
        }
        oracle_metrics = evaluate_assignment(
            reference_model=reference_model,
            layer_bits=layer_bits,
            oracle_loader=oracle_loader,
            device=device,
        )
        actual_total_bytes = uniform_storage_bytes(
            model=reference_model,
            layer_names=layer_names,
            bits=bits,
        )
        plan_id = f"uniform_{action_name}_anchor"
        anchor_records.append(
            {
                "plan_id": plan_id,
                "action_name": action_name,
                "bits": bits,
                "actual_total_bytes": actual_total_bytes,
                "counts": action_counts(
                    {
                        layer_name: action_name
                        for layer_name in layer_names
                    }
                ),
                "oracle_metrics": oracle_metrics,
            }
        )
        plan_path = (
            args.output_dir
            / f"plan_uniform_{action_name}_anchor.json"
        )
        plan_payloads[plan_path] = anchor_plan_payload(
            plan_id=plan_id,
            action_name=action_name,
            bits=bits,
            layer_names=layer_names,
            actual_total_memory_bytes=(
                actual_total_bytes
            ),
            fp32_total_bytes=fp32_total_bytes,
            memory_saving_ratios=(
                memory_saving_ratios
            ),
            oracle_metrics=oracle_metrics,
        )

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for score_metric in PLANNER_SCORE_METRICS:
            normalized_risks = rank_normalize_metric(
                benchmark=benchmark,
                score_metric=score_metric,
            )
            risk_csv = (
                temp_dir
                / f"{score_metric}_rank_normalized.csv"
            )
            normalized_risks.to_csv(
                risk_csv,
                index=False,
            )
            options_by_layer = build_layer_action_options(
                model=reference_model,
                risk_csv=risk_csv,
                risk_metric=RANK_RISK_COLUMN,
                memory_quantum_bytes=(
                    memory_quantum_bytes
                ),
            )
            minimum_bytes = minimum_total_memory_bytes(
                model=reference_model,
                options_by_layer=options_by_layer,
            )

            for ratio in memory_saving_ratios:
                target_total_bytes = int(
                    fp32_total_bytes * (1.0 - ratio)
                )

                if target_total_bytes < minimum_bytes:
                    maximum_saving = (
                        1.0
                        - minimum_bytes / fp32_total_bytes
                    )
                    raise ValueError(
                        f"Requested saving ratio {ratio:.6f} is "
                        "infeasible. Maximum feasible saving is "
                        f"{maximum_saving:.6f}."
                    )

                planner_start = time.perf_counter()
                plan = solve_additive_plan(
                    model=reference_model,
                    options_by_layer=options_by_layer,
                    target_total_memory_bytes=(
                        target_total_bytes
                    ),
                    memory_quantum_bytes=(
                        memory_quantum_bytes
                    ),
                )
                planner_runtime_seconds = (
                    time.perf_counter() - planner_start
                )
                layer_actions = {
                    layer_name: option.action
                    for layer_name, option in (
                        plan.selected_options.items()
                    )
                }
                oracle_metrics = evaluate_assignment(
                    reference_model=reference_model,
                    layer_bits=plan.layer_bits,
                    oracle_loader=oracle_loader,
                    device=device,
                )
                plan_id = (
                    f"{score_metric}_{budget_key(ratio)}"
                )
                summary_rows.append(
                    make_summary_row(
                        plan_id=plan_id,
                        plan_kind="optimized",
                        selected_score_metric=score_metric,
                        anchor_action=None,
                        requested_memory_saving_ratio=ratio,
                        target_total_memory_bytes=(
                            target_total_bytes
                        ),
                        actual_total_memory_bytes=(
                            plan.actual_total_memory_bytes
                        ),
                        fp32_total_bytes=fp32_total_bytes,
                        planner_objective=(
                            plan.objective_value
                        ),
                        planner_runtime_seconds=(
                            planner_runtime_seconds
                        ),
                        counts=action_counts(layer_actions),
                        oracle_metrics=oracle_metrics,
                    )
                )
                plan_path = (
                    args.output_dir
                    / f"plan_{plan_id}.json"
                )
                plan_payloads[plan_path] = (
                    optimized_plan_payload(
                        plan_id=plan_id,
                        score_metric=score_metric,
                        memory_saving_ratio=ratio,
                        target_total_memory_bytes=(
                            target_total_bytes
                        ),
                        fp32_total_bytes=fp32_total_bytes,
                        memory_quantum_bytes=(
                            memory_quantum_bytes
                        ),
                        benchmark_paths=benchmark_paths,
                        plan=plan,
                        oracle_metrics=oracle_metrics,
                    )
                )

    for ratio in memory_saving_ratios:
        target_total_bytes = int(
            fp32_total_bytes * (1.0 - ratio)
        )

        for anchor in anchor_records:
            summary_rows.append(
                make_summary_row(
                    plan_id=anchor["plan_id"],
                    plan_kind="uniform_anchor",
                    selected_score_metric=(
                        "uniform_anchor"
                    ),
                    anchor_action=anchor["action_name"],
                    requested_memory_saving_ratio=ratio,
                    target_total_memory_bytes=(
                        target_total_bytes
                    ),
                    actual_total_memory_bytes=(
                        anchor["actual_total_bytes"]
                    ),
                    fp32_total_bytes=fp32_total_bytes,
                    planner_objective=math.nan,
                    planner_runtime_seconds=0.0,
                    counts=anchor["counts"],
                    oracle_metrics=anchor[
                        "oracle_metrics"
                    ],
                )
            )

    summary_path = (
        args.output_dir
        / "planner_comparison_summary.csv"
    )
    save_outputs(
        summary_rows=summary_rows,
        plan_payloads=plan_payloads,
        summary_path=summary_path,
    )
    print_budget_tables(
        summary_rows=summary_rows,
        memory_saving_ratios=memory_saving_ratios,
    )
    print(f"\nSaved planner outputs: {args.output_dir}")
    print(
        "Interaction updates: disabled "
        "(single-action additive risks only)."
    )


if __name__ == "__main__":
    main()
