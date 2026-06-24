"""Analyze score-metric rankings against held-out oracle diagnostics."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from experiments.go_no_go import DEFAULT_RESULTS_DIR


SCORE_METRICS = (
    "weight_rel_l2",
    "activation_rel_mse",
    "output_kl_mean",
    "abs_delta_score_mean",
    "decision_risk_mean",
    "decision_risk_p95",
    "decision_risk_violation_rate",
)

PRIMARY_ORACLE_TARGETS = (
    "oracle_decision_risk_p95",
    "oracle_decision_risk_mean",
    "oracle_abs_delta_score_mean",
)

SECONDARY_ORACLE_TARGETS = (
    "oracle_flip_rate",
)

ORACLE_TARGETS = (
    *PRIMARY_ORACLE_TARGETS,
    *SECONDARY_ORACLE_TARGETS,
)

SPARSE_FLIP_WARNING = (
    "WARNING: Flip ranking is sparse and tie-dominated; "
    "treat flip correlation as secondary."
)

PLOT_FILENAMES = (
    "kendall_tau_oracle_decision_risk_p95.png",
    "top10_recall_oracle_decision_risk_p95.png",
    "kendall_tau_oracle_flip_rate_secondary_sparse.png",
    "scatter_score_metrics_vs_oracle_decision_risk_p95.png",
    "scatter_score_metrics_vs_oracle_flip_rate_secondary_sparse.png",
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    return parser.parse_args()


def _candidate_id(
    layer_name: str,
    action_name: str,
) -> str:
    return f"{layer_name}|{action_name}"


def _warn(
    message: str,
    warning_messages: list[str],
) -> None:
    print(message)

    if message not in warning_messages:
        warning_messages.append(message)


def _required_columns() -> set[str]:
    return {
        "score_seed",
        "layer_name",
        "action",
        *SCORE_METRICS,
        *ORACLE_TARGETS,
    }


def load_seed_metrics(
    input_dir: Path,
    warning_messages: list[str],
) -> pd.DataFrame:
    """Load complete seed files without mutating source CSVs."""

    input_paths = sorted(
        input_dir.glob(
            "single_action_metrics_seed*.csv"
        )
    )

    if not input_paths:
        raise FileNotFoundError(
            "No single_action_metrics_seed*.csv files found in "
            f"{input_dir}."
        )

    required_columns = _required_columns()
    seed_frames = []
    seen_seeds = set()

    for input_path in input_paths:
        dataframe = pd.read_csv(input_path)
        missing_columns = required_columns - set(
            dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"{input_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        seeds = dataframe["score_seed"].unique()

        if len(seeds) != 1:
            raise ValueError(
                f"{input_path} must contain exactly one score seed."
            )

        score_seed = int(seeds[0])

        if score_seed in seen_seeds:
            raise ValueError(
                f"Multiple input files contain score_seed={score_seed}."
            )

        seen_seeds.add(score_seed)
        selected = dataframe[
            [
                "score_seed",
                "layer_name",
                "action",
                *SCORE_METRICS,
                *ORACLE_TARGETS,
            ]
        ].copy()
        selected["candidate_id"] = [
            _candidate_id(layer_name, action_name)
            for layer_name, action_name in zip(
                selected["layer_name"],
                selected["action"],
            )
        ]

        if selected["candidate_id"].duplicated().any():
            raise ValueError(
                f"{input_path} contains duplicate layer/action candidates."
            )

        seed_frames.append(selected)

    combined = pd.concat(
        seed_frames,
        ignore_index=True,
    )

    for column in (*SCORE_METRICS, *ORACLE_TARGETS):
        values = combined[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Column {column} contains NaN or infinite values."
            )

    tiny_negative_kl = (
        combined["output_kl_mean"].ge(-1e-8)
        & combined["output_kl_mean"].lt(0.0)
    )
    tiny_negative_count = int(tiny_negative_kl.sum())

    if tiny_negative_count:
        combined.loc[
            tiny_negative_kl,
            "output_kl_mean",
        ] = 0.0
        _warn(
            "WARNING: Clamped "
            f"{tiny_negative_count} tiny negative output_kl_mean "
            "values to zero in analysis memory only.",
            warning_messages,
        )

    materially_negative_kl = combined[
        "output_kl_mean"
    ].lt(-1e-8)

    if materially_negative_kl.any():
        _warn(
            "WARNING: output_kl_mean contains values below -1e-8; "
            "these were not clamped.",
            warning_messages,
        )

    validate_candidate_sets(combined)
    validate_oracle_consistency(combined)

    return combined


def validate_candidate_sets(
    dataframe: pd.DataFrame,
) -> None:
    expected_candidates = None

    for score_seed, selected in dataframe.groupby(
        "score_seed",
        sort=True,
    ):
        candidates = set(selected["candidate_id"])

        if expected_candidates is None:
            expected_candidates = candidates
        elif candidates != expected_candidates:
            missing = expected_candidates - candidates
            extra = candidates - expected_candidates
            raise ValueError(
                f"score_seed={score_seed} candidate mismatch; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )


def validate_oracle_consistency(
    dataframe: pd.DataFrame,
) -> None:
    for oracle_target in ORACLE_TARGETS:
        inconsistent = dataframe.groupby(
            "candidate_id"
        )[oracle_target].nunique(dropna=False)

        if inconsistent.gt(1).any():
            candidates = inconsistent[
                inconsistent.gt(1)
            ].index.tolist()
            raise ValueError(
                f"Oracle target {oracle_target} differs across score "
                f"seeds for candidates: {candidates}"
            )


def _is_constant(values: pd.Series) -> bool:
    return values.nunique(dropna=False) <= 1


def _tied_candidate_count(values: pd.Series) -> int:
    return int(values.duplicated(keep=False).sum())


def _sorted_candidate_ids(
    dataframe: pd.DataFrame,
    metric: str,
) -> list[str]:
    ranked = dataframe.sort_values(
        by=[metric, "candidate_id"],
        ascending=[False, True],
        kind="stable",
    )

    return ranked["candidate_id"].tolist()


def _cutoff_tie_count(
    dataframe: pd.DataFrame,
    metric: str,
    top_k: int,
) -> int:
    ranked = dataframe.sort_values(
        by=[metric, "candidate_id"],
        ascending=[False, True],
        kind="stable",
    )
    cutoff_value = ranked.iloc[top_k - 1][metric]

    return int(ranked[metric].eq(cutoff_value).sum())


def top_k_risky_action_recall(
    dataframe: pd.DataFrame,
    score_metric: str,
    oracle_target: str,
    top_k: int,
) -> float:
    """Return exact-k intersection recall with deterministic tie-breaking."""

    if len(dataframe) < top_k:
        raise ValueError(
            f"Need at least {top_k} candidates, found {len(dataframe)}."
        )

    predicted = set(
        _sorted_candidate_ids(
            dataframe,
            score_metric,
        )[:top_k]
    )
    oracle = set(
        _sorted_candidate_ids(
            dataframe,
            oracle_target,
        )[:top_k]
    )

    return len(predicted & oracle) / top_k


def compute_by_seed(
    dataframe: pd.DataFrame,
    warning_messages: list[str],
) -> pd.DataFrame:
    """Compute all score/target rank diagnostics for every seed."""

    result_rows = []
    boundary_tie_comparisons = 0

    for score_seed, selected in dataframe.groupby(
        "score_seed",
        sort=True,
    ):
        selected = selected.copy()
        constant_score_metrics = {
            metric
            for metric in SCORE_METRICS
            if _is_constant(selected[metric])
        }
        constant_oracle_targets = {
            target
            for target in ORACLE_TARGETS
            if _is_constant(selected[target])
        }

        for metric in sorted(constant_score_metrics):
            _warn(
                f"WARNING: score_seed={int(score_seed)} metric "
                f"{metric} is constant; rank diagnostics are NaN.",
                warning_messages,
            )

        for target in sorted(constant_oracle_targets):
            _warn(
                f"WARNING: score_seed={int(score_seed)} oracle target "
                f"{target} is constant; rank diagnostics are NaN.",
                warning_messages,
            )

        for score_metric in SCORE_METRICS:
            score_values = selected[score_metric]

            for oracle_target in ORACLE_TARGETS:
                oracle_values = selected[oracle_target]
                is_constant = (
                    score_metric in constant_score_metrics
                    or oracle_target in constant_oracle_targets
                )
                score_top5_ties = _cutoff_tie_count(
                    selected,
                    score_metric,
                    5,
                )
                oracle_top5_ties = _cutoff_tie_count(
                    selected,
                    oracle_target,
                    5,
                )
                score_top10_ties = _cutoff_tie_count(
                    selected,
                    score_metric,
                    10,
                )
                oracle_top10_ties = _cutoff_tie_count(
                    selected,
                    oracle_target,
                    10,
                )

                if any(
                    tie_count > 1
                    for tie_count in (
                        score_top5_ties,
                        oracle_top5_ties,
                        score_top10_ties,
                        oracle_top10_ties,
                    )
                ):
                    boundary_tie_comparisons += 1

                if is_constant:
                    spearman = math.nan
                    kendall_tau_b = math.nan
                    top5_recall = math.nan
                    top10_recall = math.nan
                else:
                    spearman = float(
                        spearmanr(
                            score_values,
                            oracle_values,
                        ).statistic
                    )
                    kendall_tau_b = float(
                        kendalltau(
                            score_values,
                            oracle_values,
                            variant="b",
                        ).statistic
                    )
                    top5_recall = top_k_risky_action_recall(
                        selected,
                        score_metric,
                        oracle_target,
                        5,
                    )
                    top10_recall = top_k_risky_action_recall(
                        selected,
                        score_metric,
                        oracle_target,
                        10,
                    )

                result_rows.append(
                    {
                        "score_seed": int(score_seed),
                        "score_metric": score_metric,
                        "oracle_target": oracle_target,
                        "oracle_target_role": (
                            "primary"
                            if oracle_target
                            in PRIMARY_ORACLE_TARGETS
                            else "secondary_sparse"
                        ),
                        "candidate_count": len(selected),
                        "score_unique_values": int(
                            score_values.nunique(dropna=False)
                        ),
                        "oracle_unique_values": int(
                            oracle_values.nunique(dropna=False)
                        ),
                        "score_tied_candidate_count": (
                            _tied_candidate_count(score_values)
                        ),
                        "oracle_tied_candidate_count": (
                            _tied_candidate_count(oracle_values)
                        ),
                        "spearman_correlation": spearman,
                        "kendall_tau_b": kendall_tau_b,
                        "top5_risky_action_recall": top5_recall,
                        "top10_risky_action_recall": top10_recall,
                        "score_top5_cutoff_tie_count": (
                            score_top5_ties
                        ),
                        "oracle_top5_cutoff_tie_count": (
                            oracle_top5_ties
                        ),
                        "score_top10_cutoff_tie_count": (
                            score_top10_ties
                        ),
                        "oracle_top10_cutoff_tie_count": (
                            oracle_top10_ties
                        ),
                    }
                )

    if boundary_tie_comparisons:
        _warn(
            "WARNING: "
            f"{boundary_tie_comparisons} metric/target comparisons "
            "have a tie at a top-k cutoff; exact-k recall uses "
            "candidate_id as a deterministic tie-breaker, and tie "
            "counts are retained in the per-seed output.",
            warning_messages,
        )

    return pd.DataFrame(result_rows)


def build_summary(
    by_seed: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate diagnostic means and sample standard deviations by seed."""

    return (
        by_seed.groupby(
            [
                "score_metric",
                "oracle_target",
                "oracle_target_role",
            ],
            sort=False,
            as_index=False,
        )
        .agg(
            score_seed_count=("score_seed", "nunique"),
            spearman_mean=("spearman_correlation", "mean"),
            spearman_std=("spearman_correlation", "std"),
            kendall_tau_b_mean=("kendall_tau_b", "mean"),
            kendall_tau_b_std=("kendall_tau_b", "std"),
            top5_recall_mean=(
                "top5_risky_action_recall",
                "mean",
            ),
            top5_recall_std=(
                "top5_risky_action_recall",
                "std",
            ),
            top10_recall_mean=(
                "top10_risky_action_recall",
                "mean",
            ),
            top10_recall_std=(
                "top10_risky_action_recall",
                "std",
            ),
        )
    )


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def plot_summary_bar(
    summary: pd.DataFrame,
    oracle_target: str,
    value_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    selected = (
        summary[
            summary["oracle_target"] == oracle_target
        ]
        .set_index("score_metric")
        .reindex(SCORE_METRICS)
    )
    values = selected[value_column].to_numpy(dtype=float)
    errors = selected[std_column].fillna(0.0).to_numpy(
        dtype=float
    )
    positions = np.arange(len(SCORE_METRICS))
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        positions,
        values,
        yerr=errors,
        capsize=4,
        color="#4C78A8",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(
        positions,
        [_metric_label(metric) for metric in SCORE_METRICS],
        rotation=35,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _top_candidates(
    dataframe: pd.DataFrame,
    oracle_target: str,
    count: int,
) -> list[str]:
    reference_seed = dataframe["score_seed"].min()
    reference = dataframe[
        dataframe["score_seed"] == reference_seed
    ]

    return _sorted_candidate_ids(
        reference,
        oracle_target,
    )[:count]


def plot_scatter_grid(
    dataframe: pd.DataFrame,
    oracle_target: str,
    title: str,
    output_path: Path,
) -> None:
    top_three = _top_candidates(
        dataframe,
        oracle_target,
        3,
    )
    seeds = sorted(dataframe["score_seed"].unique())
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(16, 13),
    )
    flat_axes = axes.flatten()

    for metric_index, score_metric in enumerate(
        SCORE_METRICS
    ):
        axis = flat_axes[metric_index]

        for color_index, score_seed in enumerate(seeds):
            selected = dataframe[
                dataframe["score_seed"] == score_seed
            ]
            axis.scatter(
                selected[score_metric],
                selected[oracle_target],
                s=24,
                alpha=0.5,
                color=colors(color_index),
                label=f"score seed {score_seed}",
            )

        highlighted = dataframe[
            dataframe["candidate_id"].isin(top_three)
        ]
        axis.scatter(
            highlighted[score_metric],
            highlighted[oracle_target],
            s=90,
            facecolors="none",
            edgecolors="#D62728",
            linewidths=1.7,
            marker="o",
            label="top 3 oracle risk",
        )

        for candidate_id in top_three:
            candidate_rows = highlighted[
                highlighted["candidate_id"] == candidate_id
            ]
            axis.annotate(
                candidate_id,
                (
                    candidate_rows[score_metric].mean(),
                    candidate_rows[oracle_target].iloc[0],
                ),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )

        axis.set_xlabel(_metric_label(score_metric))
        axis.set_ylabel(_metric_label(oracle_target))
        axis.grid(alpha=0.2)

    for unused_axis in flat_axes[len(SCORE_METRICS) :]:
        unused_axis.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(seeds) + 1,
    )
    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_plots(
    dataframe: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_summary_bar(
        summary=summary,
        oracle_target="oracle_decision_risk_p95",
        value_column="kendall_tau_b_mean",
        std_column="kendall_tau_b_std",
        ylabel="Mean Kendall tau-b across score seeds",
        title=(
            "Score metric ranking vs held-out oracle decision risk p95"
        ),
        output_path=(
            output_dir
            / "kendall_tau_oracle_decision_risk_p95.png"
        ),
    )
    plot_summary_bar(
        summary=summary,
        oracle_target="oracle_decision_risk_p95",
        value_column="top10_recall_mean",
        std_column="top10_recall_std",
        ylabel="Mean Top-10 risky-action recall",
        title=(
            "Top-10 recall vs held-out oracle decision risk p95"
        ),
        output_path=(
            output_dir
            / "top10_recall_oracle_decision_risk_p95.png"
        ),
    )
    plot_summary_bar(
        summary=summary,
        oracle_target="oracle_flip_rate",
        value_column="kendall_tau_b_mean",
        std_column="kendall_tau_b_std",
        ylabel="Mean Kendall tau-b across score seeds",
        title=(
            "Score metric ranking vs oracle flip rate "
            "(secondary / sparse)"
        ),
        output_path=(
            output_dir
            / "kendall_tau_oracle_flip_rate_secondary_sparse.png"
        ),
    )
    plot_scatter_grid(
        dataframe=dataframe,
        oracle_target="oracle_decision_risk_p95",
        title=(
            "Score metrics vs held-out oracle decision risk p95"
        ),
        output_path=(
            output_dir
            / "scatter_score_metrics_vs_oracle_decision_risk_p95.png"
        ),
    )
    plot_scatter_grid(
        dataframe=dataframe,
        oracle_target="oracle_flip_rate",
        title=(
            "Score metrics vs oracle flip rate (secondary / sparse)"
        ),
        output_path=(
            output_dir
            / "scatter_score_metrics_vs_oracle_flip_rate_secondary_sparse.png"
        ),
    )


def print_ranking_table(
    summary: pd.DataFrame,
    oracle_target: str,
    title: str,
) -> None:
    selected = summary[
        summary["oracle_target"] == oracle_target
    ].sort_values(
        by="kendall_tau_b_mean",
        ascending=False,
        na_position="last",
    )
    columns = [
        "score_metric",
        "kendall_tau_b_mean",
        "kendall_tau_b_std",
        "spearman_mean",
        "top5_recall_mean",
        "top10_recall_mean",
    ]
    print(f"\n{title}")
    print(selected[columns].to_string(index=False))


def print_top_oracle_candidates(
    dataframe: pd.DataFrame,
) -> None:
    oracle_target = "oracle_decision_risk_p95"
    top_ten = _top_candidates(
        dataframe,
        oracle_target,
        10,
    )
    reference_seed = dataframe["score_seed"].min()
    reference = dataframe[
        dataframe["score_seed"] == reference_seed
    ].set_index("candidate_id")

    print(
        "\nD. Top 10 oracle-risk candidates and tie-aware "
        "average rank under every score metric"
    )

    for score_seed, selected in dataframe.groupby(
        "score_seed",
        sort=True,
    ):
        rank_table = selected.set_index("candidate_id")
        output = pd.DataFrame(
            {
                "candidate_id": top_ten,
                oracle_target: [
                    reference.loc[
                        candidate_id,
                        oracle_target,
                    ]
                    for candidate_id in top_ten
                ],
            }
        )

        for score_metric in SCORE_METRICS:
            ranks = rank_table[score_metric].rank(
                ascending=False,
                method="average",
            )
            output[f"{score_metric}_rank"] = [
                ranks.loc[candidate_id]
                for candidate_id in top_ten
            ]

        print(f"\nScore seed {int(score_seed)}")
        print(output.to_string(index=False))


def _preflight_outputs(
    output_dir: Path,
) -> tuple[Path, Path]:
    by_seed_path = (
        output_dir / "metric_ranking_by_seed.csv"
    )
    summary_path = (
        output_dir / "metric_ranking_summary.csv"
    )
    output_paths = [
        by_seed_path,
        summary_path,
        *(
            output_dir / filename
            for filename in PLOT_FILENAMES
        ),
    ]

    for output_path in output_paths:
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )

    return by_seed_path, summary_path


def main() -> None:
    args = parse_arguments()
    by_seed_path, summary_path = _preflight_outputs(
        args.output_dir
    )
    warning_messages = []
    dataframe = load_seed_metrics(
        input_dir=args.input_dir,
        warning_messages=warning_messages,
    )
    oracle_reference = dataframe[
        dataframe["score_seed"]
        == dataframe["score_seed"].min()
    ]
    nonzero_flip_count = int(
        oracle_reference["oracle_flip_rate"].ne(0.0).sum()
    )

    if nonzero_flip_count < 5:
        _warn(
            SPARSE_FLIP_WARNING,
            warning_messages,
        )

    by_seed = compute_by_seed(
        dataframe=dataframe,
        warning_messages=warning_messages,
    )
    summary = build_summary(by_seed)
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    by_seed.to_csv(
        by_seed_path,
        index=False,
        mode="x",
    )
    summary.to_csv(
        summary_path,
        index=False,
        mode="x",
    )
    generate_plots(
        dataframe=dataframe,
        summary=summary,
        output_dir=args.output_dir,
    )

    print_ranking_table(
        summary=summary,
        oracle_target="oracle_decision_risk_p95",
        title=(
            "A. Ranking by mean Kendall tau-b against "
            "oracle_decision_risk_p95"
        ),
    )
    print_ranking_table(
        summary=summary,
        oracle_target="oracle_flip_rate",
        title=(
            "B. Ranking by mean Kendall tau-b against "
            "oracle_flip_rate (secondary / sparse)"
        ),
    )
    print("\nC. Warning summary")

    if warning_messages:
        for message in warning_messages:
            print(f"- {message}")
    else:
        print("- No warnings.")

    print(
        "- These diagnostics are descriptive; no statistical "
        "significance is claimed."
    )
    print_top_oracle_candidates(dataframe)
    print(f"\nSaved per-seed analysis: {by_seed_path}")
    print(f"Saved summary analysis: {summary_path}")
    print(
        "Saved plots: "
        + ", ".join(
            str(args.output_dir / filename)
            for filename in PLOT_FILENAMES
        )
    )


if __name__ == "__main__":
    main()
