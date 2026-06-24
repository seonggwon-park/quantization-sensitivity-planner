"""Offline held-out check of existing additive-planner allocations."""

import argparse
import csv
import math
from pathlib import Path

import pandas as pd

from experiments.go_no_go import DEFAULT_RESULTS_DIR
from experiments.go_no_go.metrics import (
    spearman_rank_correlation,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=(
            DEFAULT_RESULTS_DIR
            / "single_action_benchmark.csv"
        ),
    )
    parser.add_argument(
        "--allocations",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metric",
        default="p95_margin_risk",
    )
    parser.add_argument(
        "--benchmark-split",
        choices=["ranking", "holdout"],
        default="holdout",
    )
    parser.add_argument(
        "--group-column",
        default="requested_memory_saving_ratio",
    )
    parser.add_argument(
        "--min-spearman",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_RESULTS_DIR
            / "planner_eval.csv"
        ),
    )

    return parser.parse_args()


def save_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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


def main() -> None:
    args = parse_arguments()

    if not -1.0 <= args.min_spearman <= 1.0:
        raise ValueError(
            "min_spearman must be between -1 and 1."
        )

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {args.output}"
        )

    benchmark = pd.read_csv(args.benchmark)
    allocations = pd.read_csv(args.allocations)
    benchmark_required = {
        "data_split",
        "layer",
        "action",
        args.metric,
    }
    allocation_required = {
        args.group_column,
        "layer",
        "action",
        "risk_value",
    }

    missing_benchmark = benchmark_required - set(
        benchmark.columns
    )
    missing_allocations = allocation_required - set(
        allocations.columns
    )

    if missing_benchmark:
        raise ValueError(
            f"Benchmark CSV is missing columns: {missing_benchmark}"
        )

    if missing_allocations:
        raise ValueError(
            "Allocation CSV is missing columns: "
            f"{missing_allocations}"
        )

    measured = benchmark[
        benchmark["data_split"] == args.benchmark_split
    ]
    duplicate_rows = measured.duplicated(
        subset=["layer", "action"],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Benchmark split has duplicate layer/action rows."
        )

    measured_lookup = measured.set_index(
        ["layer", "action"]
    )[args.metric].to_dict()
    rows = []

    for group_value, group in allocations.groupby(
        args.group_column,
        sort=False,
    ):
        predicted_values = []
        heldout_values = []

        for allocation in group.itertuples(index=False):
            layer_name = str(
                getattr(allocation, "layer")
            )
            action_name = str(
                getattr(allocation, "action")
            ).lower()
            predicted_risk = float(
                getattr(allocation, "risk_value")
            )

            if action_name == "fp32":
                heldout_risk = 0.0
            else:
                key = (layer_name, action_name)

                if key not in measured_lookup:
                    raise ValueError(
                        f"No {args.benchmark_split} benchmark value "
                        f"for {key}."
                    )

                heldout_risk = float(
                    measured_lookup[key]
                )

            predicted_values.append(predicted_risk)
            heldout_values.append(heldout_risk)

        correlation = spearman_rank_correlation(
            predicted_values,
            heldout_values,
        )
        passed = (
            math.isfinite(correlation)
            and correlation >= args.min_spearman
        )
        predicted_total = sum(predicted_values)
        heldout_total = sum(heldout_values)

        rows.append(
            {
                args.group_column: group_value,
                "metric": args.metric,
                "benchmark_split": args.benchmark_split,
                "num_allocated_layers": len(group),
                "planner_additive_risk": predicted_total,
                "heldout_additive_risk": heldout_total,
                "additive_risk_gap": (
                    heldout_total - predicted_total
                ),
                "layer_action_spearman": correlation,
                "min_spearman": args.min_spearman,
                "decision": "GO" if passed else "NO_GO",
            }
        )

    if not rows:
        raise RuntimeError(
            "No planner allocation groups were available."
        )

    save_csv(rows, args.output)
    print(f"Saved planner evaluation: {args.output}")


if __name__ == "__main__":
    main()

