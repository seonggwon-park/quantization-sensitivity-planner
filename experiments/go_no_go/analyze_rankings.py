"""Compare ranking-split and holdout-split single-action rankings."""

import argparse
import csv
import math
from pathlib import Path

import pandas as pd

from experiments.go_no_go import DEFAULT_RESULTS_DIR
from experiments.go_no_go.metrics import (
    ranked_labels,
    spearman_rank_correlation,
    top_k_overlap_rate,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_RESULTS_DIR
            / "single_action_benchmark.csv"
        ),
    )
    parser.add_argument(
        "--metric",
        default="p95_margin_risk",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-spearman",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min-top-k-overlap",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_RESULTS_DIR
            / "ranking_analysis.csv"
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


def validate_thresholds(args) -> None:
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not -1.0 <= args.min_spearman <= 1.0:
        raise ValueError(
            "min_spearman must be between -1 and 1."
        )

    if not 0.0 <= args.min_top_k_overlap <= 1.0:
        raise ValueError(
            "min_top_k_overlap must be between 0 and 1."
        )


def main() -> None:
    args = parse_arguments()
    validate_thresholds(args)

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {args.output}"
        )

    dataframe = pd.read_csv(args.input)
    required_columns = {
        "data_split",
        "layer",
        "action",
        args.metric,
    }
    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Benchmark CSV is missing columns: {missing_columns}"
        )

    duplicate_rows = dataframe.duplicated(
        subset=["data_split", "layer", "action"],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Benchmark CSV contains duplicate split/layer/action rows."
        )

    rows = []

    for action_name in dataframe["action"].drop_duplicates():
        selected = dataframe[
            dataframe["action"] == action_name
        ]
        pivot = selected.pivot(
            index="layer",
            columns="data_split",
            values=args.metric,
        )

        if not {"ranking", "holdout"}.issubset(
            pivot.columns
        ):
            raise ValueError(
                f"Action {action_name} lacks ranking or holdout rows."
            )

        paired = pivot[["ranking", "holdout"]]

        if paired.isna().any().any():
            raise ValueError(
                f"Action {action_name} has unpaired layer measurements."
            )

        labels = paired.index.tolist()
        ranking_values = paired["ranking"].tolist()
        holdout_values = paired["holdout"].tolist()
        correlation = spearman_rank_correlation(
            ranking_values,
            holdout_values,
        )
        ranking_order = ranked_labels(
            labels,
            ranking_values,
        )
        holdout_order = ranked_labels(
            labels,
            holdout_values,
        )
        overlap_rate = top_k_overlap_rate(
            ranking_order,
            holdout_order,
            args.top_k,
        )
        passed = (
            math.isfinite(correlation)
            and correlation >= args.min_spearman
            and overlap_rate >= args.min_top_k_overlap
        )

        rows.append(
            {
                "action": action_name,
                "metric": args.metric,
                "num_layers": len(labels),
                "spearman_rank_correlation": correlation,
                "top_k": min(args.top_k, len(labels)),
                "top_k_overlap_rate": overlap_rate,
                "min_spearman": args.min_spearman,
                "min_top_k_overlap": (
                    args.min_top_k_overlap
                ),
                "decision": "GO" if passed else "NO_GO",
                "ranking_order": ",".join(ranking_order),
                "holdout_order": ",".join(holdout_order),
            }
        )

    if not rows:
        raise RuntimeError("No action rankings were available.")

    save_csv(rows, args.output)
    overall = (
        "GO"
        if all(row["decision"] == "GO" for row in rows)
        else "NO_GO"
    )
    print(f"Overall decision: {overall}")
    print(f"Saved ranking analysis: {args.output}")


if __name__ == "__main__":
    main()

