"""Analyze samplewise vector-additive proxies for saved planner plans."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


PROXY_NAMES = (
    "vector_signed_mean_risk",
    "vector_signed_p95_risk",
    "vector_signed_violation_rate",
    "vector_abs_sum_mean_risk",
    "vector_abs_sum_p95_risk",
    "scalar_additive_mean_risk",
    "scalar_additive_p95_risk",
)

ORACLE_TARGETS = (
    "oracle_decision_risk_p95",
    "oracle_decision_risk_mean",
    "oracle_abs_delta_score_mean",
    "oracle_flip_rate",
)

PRIMARY_ORACLE_TARGETS = (
    "oracle_decision_risk_p95",
    "oracle_decision_risk_mean",
    "oracle_abs_delta_score_mean",
)

OPTIMIZED_SCORE_METRICS = (
    "weight_rel_l2",
    "activation_rel_mse",
    "output_kl_mean",
    "abs_delta_score_mean",
    "decision_risk_mean",
    "decision_risk_p95",
    "decision_risk_violation_rate",
)

VIOLATION_SCORE_METRIC = "decision_risk_violation_rate"
EPSILON = 1e-12

PLOT_FILENAMES = (
    "pooled_kendall_p95_comparison.png",
    "pooled_kendall_mean_comparison.png",
    "per_budget_kendall_p95.png",
    "scatter_vector_signed_p95_by_budget.png",
    "scatter_scalar_additive_p95_by_budget.png",
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=Path("results/go_no_go_vectors_v1"),
    )
    parser.add_argument(
        "--planner-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("results/go_no_go_planner_v1"),
            Path("results/go_no_go_planner_v2_stress"),
        ],
    )
    parser.add_argument(
        "--score-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/go_no_go_vector_analysis_v1"
        ),
    )

    return parser.parse_args()


def validate_score_seeds(
    score_seeds: list[int],
) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in score_seeds)

    if not normalized:
        raise ValueError("At least one score seed is required.")

    if len(set(normalized)) != len(normalized):
        raise ValueError("score_seeds must be unique.")

    return normalized


def preflight_outputs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "by_plan": output_dir / "vector_proxy_by_plan.csv",
        "ranking": (
            output_dir / "vector_proxy_ranking_summary.csv"
        ),
    }

    for filename in PLOT_FILENAMES:
        paths[filename] = output_dir / filename

    for output_path in paths.values():
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )

    return paths


def load_vector_archives(
    vector_dir: Path,
    score_seeds: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    archives = {}
    reference_candidate_ids = None

    for score_seed in score_seeds:
        vector_path = (
            vector_dir
            / f"score_delta_vectors_seed{score_seed}.npz"
        )

        if not vector_path.exists():
            raise FileNotFoundError(vector_path)

        with np.load(
            vector_path,
            allow_pickle=False,
        ) as archive:
            required_arrays = {
                "candidate_ids",
                "baseline_margin",
                "delta_scores",
                "action_names",
                "layer_names",
            }
            missing_arrays = required_arrays - set(
                archive.files
            )

            if missing_arrays:
                raise ValueError(
                    f"{vector_path} is missing arrays: "
                    f"{sorted(missing_arrays)}"
                )

            record = {
                name: np.asarray(archive[name]).copy()
                for name in required_arrays
            }

        candidate_ids = record["candidate_ids"].astype(
            str
        )
        layer_names = record["layer_names"].astype(str)
        action_names = record["action_names"].astype(str)
        reconstructed_ids = np.asarray(
            [
                f"{layer_name}|{action_name}"
                for layer_name, action_name in zip(
                    layer_names,
                    action_names,
                )
            ]
        )

        if not np.array_equal(
            candidate_ids,
            reconstructed_ids,
        ):
            raise ValueError(
                f"{vector_path} candidate ID arrays disagree."
            )

        baseline_margin = record["baseline_margin"].astype(
            np.float64
        )
        delta_scores = record["delta_scores"].astype(
            np.float64
        )

        if baseline_margin.ndim != 1:
            raise ValueError(
                f"{vector_path} baseline_margin must be 1-D."
            )

        if delta_scores.shape != (
            len(candidate_ids),
            len(baseline_margin),
        ):
            raise ValueError(
                f"{vector_path} delta_scores shape is inconsistent."
            )

        if (
            not np.isfinite(baseline_margin).all()
            or not np.isfinite(delta_scores).all()
        ):
            raise ValueError(
                f"{vector_path} contains non-finite values."
            )

        if (baseline_margin < 0.0).any():
            raise ValueError(
                f"{vector_path} has negative baseline margins."
            )

        if reference_candidate_ids is None:
            reference_candidate_ids = candidate_ids.tolist()
        elif candidate_ids.tolist() != reference_candidate_ids:
            raise ValueError(
                "Candidate ordering differs between vector seeds."
            )

        archives[score_seed] = {
            "candidate_ids": candidate_ids,
            "baseline_margin": baseline_margin,
            "delta_scores": delta_scores,
        }

    return archives


def _anchor_signature(plan: dict) -> tuple:
    return tuple(
        sorted(
            (str(layer), str(action))
            for layer, action in plan["layer_actions"].items()
        )
    )


def _validate_oracle_metrics(
    plan: dict,
    plan_path: Path,
) -> None:
    oracle_metrics = plan.get("oracle_metrics", {})
    missing_targets = set(ORACLE_TARGETS) - set(
        oracle_metrics
    )

    if missing_targets:
        raise ValueError(
            f"{plan_path} lacks oracle metrics: "
            f"{sorted(missing_targets)}"
        )

    for target in ORACLE_TARGETS:
        if not math.isfinite(float(oracle_metrics[target])):
            raise ValueError(
                f"{plan_path} has non-finite {target}."
            )


def _merge_anchor(
    existing: dict,
    duplicate: dict,
    duplicate_path: Path,
) -> None:
    for target in ORACLE_TARGETS:
        first = float(existing["oracle_metrics"][target])
        second = float(duplicate["oracle_metrics"][target])

        if not math.isclose(
            first,
            second,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Duplicate uniform anchors disagree on "
                f"{target}: {first} vs {second}."
            )

    first_saving = float(
        existing["actual_memory_saving_ratio"]
    )
    second_saving = float(
        duplicate["actual_memory_saving_ratio"]
    )

    if not math.isclose(
        first_saving,
        second_saving,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Duplicate uniform anchors disagree on actual savings."
        )

    budgets = set(
        float(value)
        for value in existing.get(
            "compared_at_memory_saving_ratios",
            [],
        )
    )
    budgets.update(
        float(value)
        for value in duplicate.get(
            "compared_at_memory_saving_ratios",
            [],
        )
    )
    existing["compared_at_memory_saving_ratios"] = sorted(
        budgets
    )
    existing["source_plan_paths"].append(
        str(duplicate_path.resolve())
    )


def load_plans(
    planner_dirs: list[Path],
) -> list[dict]:
    optimized_plans = []
    anchors_by_signature = {}
    seen_optimized_ids = set()

    for planner_dir in planner_dirs:
        plan_paths = sorted(planner_dir.glob("plan_*.json"))

        if not plan_paths:
            raise FileNotFoundError(
                f"No plan_*.json files found in {planner_dir}."
            )

        for plan_path in plan_paths:
            with plan_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                plan = json.load(file)

            required_keys = {
                "plan_id",
                "plan_kind",
                "selected_score_metric",
                "actual_memory_saving_ratio",
                "layer_actions",
                "oracle_metrics",
            }
            missing_keys = required_keys - set(plan)

            if missing_keys:
                raise ValueError(
                    f"{plan_path} is missing keys: "
                    f"{sorted(missing_keys)}"
                )

            if not plan["layer_actions"]:
                raise ValueError(
                    f"{plan_path} has no layer actions."
                )

            _validate_oracle_metrics(plan, plan_path)
            plan["source_plan_paths"] = [
                str(plan_path.resolve())
            ]

            if plan["plan_kind"] == "optimized":
                if "requested_memory_saving_ratio" not in plan:
                    raise ValueError(
                        f"{plan_path} lacks requested budget."
                    )

                score_metric = plan[
                    "selected_score_metric"
                ]

                if score_metric not in OPTIMIZED_SCORE_METRICS:
                    raise ValueError(
                        f"{plan_path} has unsupported score metric "
                        f"{score_metric}."
                    )

                if plan["plan_id"] in seen_optimized_ids:
                    raise ValueError(
                        f"Duplicate optimized plan ID: {plan['plan_id']}"
                    )

                seen_optimized_ids.add(plan["plan_id"])
                optimized_plans.append(plan)
            elif plan["plan_kind"] == "uniform_anchor":
                signature = _anchor_signature(plan)

                if signature in anchors_by_signature:
                    _merge_anchor(
                        anchors_by_signature[signature],
                        plan,
                        plan_path,
                    )
                else:
                    plan[
                        "compared_at_memory_saving_ratios"
                    ] = sorted(
                        float(value)
                        for value in plan.get(
                            "compared_at_memory_saving_ratios",
                            [],
                        )
                    )
                    anchors_by_signature[signature] = plan
            else:
                raise ValueError(
                    f"Unsupported plan_kind in {plan_path}: "
                    f"{plan['plan_kind']}"
                )

    return [
        *optimized_plans,
        *anchors_by_signature.values(),
    ]


def compute_seed_proxies(
    layer_actions: dict[str, str],
    vector_archive: dict[str, np.ndarray],
) -> dict[str, float]:
    candidate_ids = vector_archive[
        "candidate_ids"
    ].astype(str)
    candidate_lookup = {
        candidate_id: index
        for index, candidate_id in enumerate(candidate_ids)
    }
    selected_indices = []

    for layer_name, action_name in sorted(
        layer_actions.items()
    ):
        if action_name == "fp32":
            continue

        candidate_id = f"{layer_name}|{action_name}"

        if candidate_id not in candidate_lookup:
            raise ValueError(
                f"Plan action {candidate_id} is absent from vectors."
            )

        selected_indices.append(
            candidate_lookup[candidate_id]
        )

    baseline_margin = vector_archive[
        "baseline_margin"
    ].astype(np.float64)
    denominator = baseline_margin + EPSILON

    if selected_indices:
        selected_deltas = vector_archive[
            "delta_scores"
        ][selected_indices].astype(np.float64)
        summed_delta = selected_deltas.sum(axis=0)
        absolute_sum = np.abs(selected_deltas).sum(axis=0)
        per_action_risk = (
            np.abs(selected_deltas)
            / denominator[np.newaxis, :]
        )
        scalar_additive_mean = per_action_risk.mean(
            axis=1
        ).sum()
        scalar_additive_p95 = np.quantile(
            per_action_risk,
            0.95,
            axis=1,
        ).sum()
    else:
        summed_delta = np.zeros_like(baseline_margin)
        absolute_sum = np.zeros_like(baseline_margin)
        scalar_additive_mean = 0.0
        scalar_additive_p95 = 0.0

    signed_risk = np.abs(summed_delta) / denominator
    absolute_sum_risk = absolute_sum / denominator

    return {
        "vector_signed_mean_risk": float(
            signed_risk.mean()
        ),
        "vector_signed_p95_risk": float(
            np.quantile(signed_risk, 0.95)
        ),
        "vector_signed_violation_rate": float(
            np.mean(signed_risk >= 1.0)
        ),
        "vector_abs_sum_mean_risk": float(
            absolute_sum_risk.mean()
        ),
        "vector_abs_sum_p95_risk": float(
            np.quantile(absolute_sum_risk, 0.95)
        ),
        "scalar_additive_mean_risk": float(
            scalar_additive_mean
        ),
        "scalar_additive_p95_risk": float(
            scalar_additive_p95
        ),
    }


def build_plan_proxy_table(
    plans: list[dict],
    vector_archives: dict[int, dict[str, np.ndarray]],
    score_seeds: tuple[int, ...],
) -> pd.DataFrame:
    rows = []

    for plan in plans:
        seed_proxies = {
            score_seed: compute_seed_proxies(
                layer_actions=plan["layer_actions"],
                vector_archive=vector_archives[score_seed],
            )
            for score_seed in score_seeds
        }
        selected_score_metric = plan[
            "selected_score_metric"
        ]
        is_primary = (
            plan["plan_kind"] == "optimized"
            and selected_score_metric
            != VIOLATION_SCORE_METRIC
        )

        if plan["plan_kind"] != "optimized":
            exclusion_reason = "uniform_anchor"
        elif selected_score_metric == VIOLATION_SCORE_METRIC:
            exclusion_reason = (
                "violation_metric_actual_saving_mismatch"
            )
        else:
            exclusion_reason = ""

        row = {
            "plan_id": plan["plan_id"],
            "plan_kind": plan["plan_kind"],
            "selected_score_metric": selected_score_metric,
            "requested_memory_saving_ratio": (
                float(plan["requested_memory_saving_ratio"])
                if plan["plan_kind"] == "optimized"
                else math.nan
            ),
            "compared_at_memory_saving_ratios": ";".join(
                str(value)
                for value in plan.get(
                    "compared_at_memory_saving_ratios",
                    [],
                )
            ),
            "actual_memory_saving_ratio": float(
                plan["actual_memory_saving_ratio"]
            ),
            "primary_included": is_primary,
            "secondary_included": True,
            "primary_exclusion_reason": exclusion_reason,
            "selected_non_fp32_action_count": sum(
                action_name != "fp32"
                for action_name in plan[
                    "layer_actions"
                ].values()
            ),
            "source_plan_paths": ";".join(
                plan["source_plan_paths"]
            ),
        }

        for oracle_target in ORACLE_TARGETS:
            row[oracle_target] = float(
                plan["oracle_metrics"][oracle_target]
            )

        for proxy_name in PROXY_NAMES:
            values = np.asarray(
                [
                    seed_proxies[seed][proxy_name]
                    for seed in score_seeds
                ],
                dtype=np.float64,
            )
            row[
                f"{proxy_name}_mean_across_seeds"
            ] = float(values.mean())
            row[
                f"{proxy_name}_std_across_seeds"
            ] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else math.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def expand_comparison_rows(
    by_plan: pd.DataFrame,
    comparison_set: str,
) -> pd.DataFrame:
    if comparison_set == "primary":
        return by_plan[
            by_plan["primary_included"]
        ].copy()

    if comparison_set != "secondary_all":
        raise ValueError(
            f"Unknown comparison set: {comparison_set}"
        )

    optimized = by_plan[
        by_plan["plan_kind"] == "optimized"
    ].copy()
    anchor_rows = []
    anchors = by_plan[
        by_plan["plan_kind"] == "uniform_anchor"
    ]

    for _, anchor in anchors.iterrows():
        budgets = [
            float(value)
            for value in str(
                anchor[
                    "compared_at_memory_saving_ratios"
                ]
            ).split(";")
            if value
        ]

        for budget in budgets:
            expanded = anchor.copy()
            expanded[
                "requested_memory_saving_ratio"
            ] = budget
            anchor_rows.append(expanded)

    if anchor_rows:
        anchors_expanded = pd.DataFrame(anchor_rows)
        return pd.concat(
            [optimized, anchors_expanded],
            ignore_index=True,
        )

    return optimized


def _is_constant(values: pd.Series) -> bool:
    return values.nunique(dropna=False) <= 1


def _tied_value_count(values: pd.Series) -> int:
    return int(values.duplicated(keep=False).sum())


def _correlations(
    first: pd.Series,
    second: pd.Series,
) -> tuple[float, float]:
    if _is_constant(first) or _is_constant(second):
        return math.nan, math.nan

    return (
        float(spearmanr(first, second).statistic),
        float(
            kendalltau(
                first,
                second,
                variant="b",
            ).statistic
        ),
    )


def _sorted_plan_ids(
    dataframe: pd.DataFrame,
    metric_column: str,
) -> list[str]:
    return dataframe.sort_values(
        by=[metric_column, "plan_id"],
        ascending=[False, True],
        kind="stable",
    )["plan_id"].tolist()


def _top_k_recall(
    dataframe: pd.DataFrame,
    proxy_column: str,
    oracle_target: str,
    top_k: int,
) -> float:
    if len(dataframe) < top_k:
        return math.nan

    if (
        _is_constant(dataframe[proxy_column])
        or _is_constant(dataframe[oracle_target])
    ):
        return math.nan

    predicted = set(
        _sorted_plan_ids(dataframe, proxy_column)[:top_k]
    )
    oracle = set(
        _sorted_plan_ids(dataframe, oracle_target)[:top_k]
    )

    return len(predicted & oracle) / top_k


def _cutoff_tie_count(
    dataframe: pd.DataFrame,
    metric_column: str,
    top_k: int,
) -> int:
    if len(dataframe) < top_k:
        return 0

    ranked = dataframe.sort_values(
        by=[metric_column, "plan_id"],
        ascending=[False, True],
        kind="stable",
    )
    cutoff = ranked.iloc[top_k - 1][metric_column]

    return int(ranked[metric_column].eq(cutoff).sum())


def build_ranking_summary(
    by_plan: pd.DataFrame,
    warning_messages: list[str],
) -> pd.DataFrame:
    result_rows = []
    boundary_tie_count = 0

    for comparison_set in (
        "primary",
        "secondary_all",
    ):
        comparison = expand_comparison_rows(
            by_plan,
            comparison_set,
        )
        budgets = sorted(
            float(value)
            for value in comparison[
                "requested_memory_saving_ratio"
            ].dropna().unique()
        )

        for proxy_name in PROXY_NAMES:
            proxy_column = (
                f"{proxy_name}_mean_across_seeds"
            )

            for oracle_target in ORACLE_TARGETS:
                per_budget_rows = []
                pooled_proxy_ranks = []
                pooled_oracle_ranks = []

                for budget in budgets:
                    selected = comparison[
                        comparison[
                            "requested_memory_saving_ratio"
                        ].eq(budget)
                    ].copy()
                    proxy_constant = _is_constant(
                        selected[proxy_column]
                    )
                    oracle_constant = _is_constant(
                        selected[oracle_target]
                    )
                    spearman, kendall_tau_b = _correlations(
                        selected[proxy_column],
                        selected[oracle_target],
                    )
                    top1_recall = _top_k_recall(
                        selected,
                        proxy_column,
                        oracle_target,
                        1,
                    )
                    top3_recall = _top_k_recall(
                        selected,
                        proxy_column,
                        oracle_target,
                        3,
                    )
                    proxy_top1_ties = _cutoff_tie_count(
                        selected,
                        proxy_column,
                        1,
                    )
                    oracle_top1_ties = _cutoff_tie_count(
                        selected,
                        oracle_target,
                        1,
                    )
                    proxy_top3_ties = _cutoff_tie_count(
                        selected,
                        proxy_column,
                        3,
                    )
                    oracle_top3_ties = _cutoff_tie_count(
                        selected,
                        oracle_target,
                        3,
                    )

                    if any(
                        count > 1
                        for count in (
                            proxy_top1_ties,
                            oracle_top1_ties,
                            proxy_top3_ties,
                            oracle_top3_ties,
                        )
                    ):
                        boundary_tie_count += 1

                    warning = ""

                    if proxy_constant or oracle_constant:
                        warning = (
                            "constant proxy"
                            if proxy_constant
                            else "constant oracle target"
                        )
                        warning_messages.append(
                            "WARNING: "
                            f"{comparison_set}, budget={budget}, "
                            f"{proxy_name} vs {oracle_target}: "
                            f"{warning}."
                        )

                    per_budget_row = {
                        "comparison_set": comparison_set,
                        "oracle_target_role": (
                            "primary"
                            if oracle_target
                            in PRIMARY_ORACLE_TARGETS
                            else "secondary_sparse"
                        ),
                        "analysis_scope": "per_budget",
                        "requested_memory_saving_ratio": budget,
                        "proxy_name": proxy_name,
                        "oracle_target": oracle_target,
                        "plan_count": len(selected),
                        "budget_count": 1,
                        "spearman": spearman,
                        "kendall_tau_b": kendall_tau_b,
                        "top1_recall": top1_recall,
                        "top3_recall": top3_recall,
                        "proxy_tied_value_count": (
                            _tied_value_count(
                                selected[proxy_column]
                            )
                        ),
                        "oracle_tied_value_count": (
                            _tied_value_count(
                                selected[oracle_target]
                            )
                        ),
                        "proxy_top1_cutoff_tie_count": (
                            proxy_top1_ties
                        ),
                        "oracle_top1_cutoff_tie_count": (
                            oracle_top1_ties
                        ),
                        "proxy_top3_cutoff_tie_count": (
                            proxy_top3_ties
                        ),
                        "oracle_top3_cutoff_tie_count": (
                            oracle_top3_ties
                        ),
                        "proxy_constant_budget_count": int(
                            proxy_constant
                        ),
                        "oracle_constant_budget_count": int(
                            oracle_constant
                        ),
                        "warning": warning,
                    }
                    per_budget_rows.append(per_budget_row)
                    result_rows.append(per_budget_row)
                    pooled_proxy_ranks.append(
                        selected[proxy_column].rank(
                            method="average",
                            ascending=True,
                        ).to_numpy(dtype=float)
                    )
                    pooled_oracle_ranks.append(
                        selected[oracle_target].rank(
                            method="average",
                            ascending=True,
                        ).to_numpy(dtype=float)
                    )

                concatenated_proxy_ranks = np.concatenate(
                    pooled_proxy_ranks
                )
                concatenated_oracle_ranks = np.concatenate(
                    pooled_oracle_ranks
                )
                pooled_spearman, pooled_kendall = (
                    _correlations(
                        pd.Series(concatenated_proxy_ranks),
                        pd.Series(concatenated_oracle_ranks),
                    )
                )
                top1_values = np.asarray(
                    [row["top1_recall"] for row in per_budget_rows],
                    dtype=float,
                )
                top3_values = np.asarray(
                    [row["top3_recall"] for row in per_budget_rows],
                    dtype=float,
                )
                result_rows.append(
                    {
                        "comparison_set": comparison_set,
                        "oracle_target_role": (
                            "primary"
                            if oracle_target
                            in PRIMARY_ORACLE_TARGETS
                            else "secondary_sparse"
                        ),
                        "analysis_scope": (
                            "pooled_within_budget"
                        ),
                        "requested_memory_saving_ratio": math.nan,
                        "proxy_name": proxy_name,
                        "oracle_target": oracle_target,
                        "plan_count": sum(
                            row["plan_count"]
                            for row in per_budget_rows
                        ),
                        "budget_count": len(budgets),
                        "spearman": pooled_spearman,
                        "kendall_tau_b": pooled_kendall,
                        "top1_recall": float(
                            np.nanmean(top1_values)
                        ),
                        "top3_recall": float(
                            np.nanmean(top3_values)
                        ),
                        "proxy_tied_value_count": sum(
                            row["proxy_tied_value_count"]
                            for row in per_budget_rows
                        ),
                        "oracle_tied_value_count": sum(
                            row["oracle_tied_value_count"]
                            for row in per_budget_rows
                        ),
                        "proxy_top1_cutoff_tie_count": sum(
                            row[
                                "proxy_top1_cutoff_tie_count"
                            ]
                            for row in per_budget_rows
                        ),
                        "oracle_top1_cutoff_tie_count": sum(
                            row[
                                "oracle_top1_cutoff_tie_count"
                            ]
                            for row in per_budget_rows
                        ),
                        "proxy_top3_cutoff_tie_count": sum(
                            row[
                                "proxy_top3_cutoff_tie_count"
                            ]
                            for row in per_budget_rows
                        ),
                        "oracle_top3_cutoff_tie_count": sum(
                            row[
                                "oracle_top3_cutoff_tie_count"
                            ]
                            for row in per_budget_rows
                        ),
                        "proxy_constant_budget_count": sum(
                            row[
                                "proxy_constant_budget_count"
                            ]
                            for row in per_budget_rows
                        ),
                        "oracle_constant_budget_count": sum(
                            row[
                                "oracle_constant_budget_count"
                            ]
                            for row in per_budget_rows
                        ),
                        "warning": (
                            "one or more constant budgets"
                            if any(row["warning"] for row in per_budget_rows)
                            else ""
                        ),
                    }
                )

    if boundary_tie_count:
        warning_messages.append(
            "WARNING: "
            f"{boundary_tie_count} per-budget comparisons have a "
            "top-k boundary tie; exact-k recall uses plan_id as a "
            "deterministic tie-breaker."
        )

    return pd.DataFrame(result_rows)


def save_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataframe.to_csv(
        output_path,
        index=False,
        mode="x",
    )


def _pooled_rows(
    ranking: pd.DataFrame,
    oracle_target: str,
    proxies: tuple[str, ...],
) -> pd.DataFrame:
    return ranking[
        ranking["analysis_scope"].eq(
            "pooled_within_budget"
        )
        & ranking["oracle_target"].eq(oracle_target)
        & ranking["proxy_name"].isin(proxies)
    ].copy()


def plot_pooled_bar(
    ranking: pd.DataFrame,
    oracle_target: str,
    proxies: tuple[str, ...],
    title: str,
    output_path: Path,
) -> None:
    selected = _pooled_rows(
        ranking,
        oracle_target,
        proxies,
    )
    positions = np.arange(len(proxies))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 6))

    for offset, comparison_set in enumerate(
        ("primary", "secondary_all")
    ):
        values = (
            selected[
                selected["comparison_set"].eq(
                    comparison_set
                )
            ]
            .set_index("proxy_name")
            .reindex(proxies)["kendall_tau_b"]
            .to_numpy(dtype=float)
        )
        axis.bar(
            positions + (offset - 0.5) * width,
            values,
            width=width,
            label=(
                "primary: optimized non-violation"
                if comparison_set == "primary"
                else "secondary: all plans"
            ),
            hatch="//" if comparison_set == "secondary_all" else None,
        )

    axis.set_xticks(
        positions,
        [proxy.replace("_", " ") for proxy in proxies],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Pooled within-budget Kendall tau-b")
    axis.set_title(title)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_budget_lines(
    ranking: pd.DataFrame,
    output_path: Path,
) -> None:
    proxies = (
        "scalar_additive_p95_risk",
        "vector_signed_p95_risk",
        "vector_abs_sum_p95_risk",
    )
    selected = ranking[
        ranking["analysis_scope"].eq("per_budget")
        & ranking["oracle_target"].eq(
            "oracle_decision_risk_p95"
        )
        & ranking["proxy_name"].isin(proxies)
    ]
    figure, axis = plt.subplots(figsize=(12, 7))

    for proxy in proxies:
        for comparison_set, linestyle in (
            ("primary", "-"),
            ("secondary_all", "--"),
        ):
            line = selected[
                selected["proxy_name"].eq(proxy)
                & selected["comparison_set"].eq(
                    comparison_set
                )
            ].sort_values("requested_memory_saving_ratio")
            axis.plot(
                line["requested_memory_saving_ratio"],
                line["kendall_tau_b"],
                marker="o",
                linestyle=linestyle,
                label=(
                    f"{proxy} ({'primary' if comparison_set == 'primary' else 'secondary'})"
                ),
            )

    axis.set_xlabel("Requested memory-saving ratio")
    axis.set_ylabel("Within-budget Kendall tau-b")
    axis.set_title(
        "Per-budget proxy ranking vs oracle decision risk p95"
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_budget_scatters(
    by_plan: pd.DataFrame,
    proxy_name: str,
    output_path: Path,
) -> None:
    comparison = expand_comparison_rows(
        by_plan,
        "secondary_all",
    )
    budgets = sorted(
        comparison[
            "requested_memory_saving_ratio"
        ].unique()
    )
    proxy_column = f"{proxy_name}_mean_across_seeds"
    column_count = 4
    row_count = math.ceil(len(budgets) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(17, 4.5 * row_count),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for budget_index, budget in enumerate(budgets):
        axis = flat_axes[budget_index]
        selected = comparison[
            comparison[
                "requested_memory_saving_ratio"
            ].eq(budget)
        ]
        primary = selected[selected["primary_included"]]
        secondary = selected[
            ~selected["primary_included"]
        ]
        axis.scatter(
            primary[proxy_column],
            primary["oracle_decision_risk_p95"],
            label="primary",
            marker="o",
            alpha=0.8,
        )
        axis.scatter(
            secondary[proxy_column],
            secondary["oracle_decision_risk_p95"],
            label="secondary only",
            marker="x",
            alpha=0.85,
        )
        axis.set_title(f"Requested saving {budget:.0%}")
        axis.set_xlabel(proxy_name.replace("_", " "))
        axis.set_ylabel("oracle decision risk p95")
        axis.grid(alpha=0.2)

    for unused_axis in flat_axes[len(budgets) :]:
        unused_axis.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
    )
    figure.suptitle(
        f"{proxy_name} vs oracle decision risk p95 by budget"
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_plots(
    by_plan: pd.DataFrame,
    ranking: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    plot_pooled_bar(
        ranking=ranking,
        oracle_target="oracle_decision_risk_p95",
        proxies=(
            "scalar_additive_p95_risk",
            "vector_signed_p95_risk",
            "vector_abs_sum_p95_risk",
        ),
        title=(
            "Pooled within-budget p95 proxy ranking vs "
            "oracle decision risk p95"
        ),
        output_path=output_paths[
            "pooled_kendall_p95_comparison.png"
        ],
    )
    plot_pooled_bar(
        ranking=ranking,
        oracle_target="oracle_decision_risk_mean",
        proxies=(
            "scalar_additive_mean_risk",
            "vector_signed_mean_risk",
            "vector_abs_sum_mean_risk",
        ),
        title=(
            "Pooled within-budget mean proxy ranking vs "
            "oracle decision risk mean"
        ),
        output_path=output_paths[
            "pooled_kendall_mean_comparison.png"
        ],
    )
    plot_budget_lines(
        ranking=ranking,
        output_path=output_paths[
            "per_budget_kendall_p95.png"
        ],
    )
    plot_budget_scatters(
        by_plan=by_plan,
        proxy_name="vector_signed_p95_risk",
        output_path=output_paths[
            "scatter_vector_signed_p95_by_budget.png"
        ],
    )
    plot_budget_scatters(
        by_plan=by_plan,
        proxy_name="scalar_additive_p95_risk",
        output_path=output_paths[
            "scatter_scalar_additive_p95_by_budget.png"
        ],
    )


def print_final_report(
    ranking: pd.DataFrame,
    warning_messages: list[str],
) -> None:
    target = "oracle_decision_risk_p95"
    proxies = (
        "scalar_additive_p95_risk",
        "vector_signed_p95_risk",
    )
    main_table = ranking[
        ranking["analysis_scope"].eq(
            "pooled_within_budget"
        )
        & ranking["oracle_target"].eq(target)
        & ranking["proxy_name"].isin(proxies)
    ][
        [
            "comparison_set",
            "proxy_name",
            "spearman",
            "kendall_tau_b",
            "top1_recall",
            "top3_recall",
            "proxy_tied_value_count",
            "oracle_tied_value_count",
        ]
    ].sort_values(
        ["comparison_set", "proxy_name"]
    )
    print(
        "\nA. Scalar-additive vs vector-signed p95 "
        "against oracle decision risk p95"
    )
    print(main_table.to_string(index=False))
    primary = main_table[
        main_table["comparison_set"].eq("primary")
    ].set_index("proxy_name")
    scalar_tau = primary.loc[
        "scalar_additive_p95_risk",
        "kendall_tau_b",
    ]
    vector_tau = primary.loc[
        "vector_signed_p95_risk",
        "kendall_tau_b",
    ]
    difference = vector_tau - scalar_tau
    print(
        "\nB. Vector-signed pooled tau-b comparison"
    )

    if math.isfinite(difference):
        print(
            "vector_signed_p95_risk "
            + ("improves" if difference > 0.0 else "does not improve")
            + " pooled within-budget Kendall tau-b over "
            "scalar_additive_p95_risk by "
            f"{difference:.6f}."
        )
    else:
        print("Comparison is undefined because a tau-b value is NaN.")

    budget_table = ranking[
        ranking["comparison_set"].eq("primary")
        & ranking["analysis_scope"].eq("per_budget")
        & ranking["oracle_target"].eq(target)
        & ranking["proxy_name"].isin(proxies)
    ].pivot(
        index="requested_memory_saving_ratio",
        columns="proxy_name",
        values="kendall_tau_b",
    ).reset_index()
    budget_table["vector_minus_scalar_tau_b"] = (
        budget_table["vector_signed_p95_risk"]
        - budget_table["scalar_additive_p95_risk"]
    )
    print("\nC. Budget-by-budget primary Kendall tau-b")
    print(budget_table.to_string(index=False))
    print("\nD. Warnings and limitations")
    unique_warnings = list(dict.fromkeys(warning_messages))

    if unique_warnings:
        for warning in unique_warnings:
            print(f"- {warning}")
    else:
        print("- No constant-value or top-k tie warnings.")

    print(
        "- decision_risk_violation_rate plans and uniform anchors "
        "are secondary-only; primary comparisons use optimized "
        "non-violation plans."
    )
    print(
        "- oracle_flip_rate is sparse and is treated as a secondary "
        "diagnostic target."
    )
    print(
        "- Vector summation tests a linear additive proxy; it does not "
        "model interaction updates or establish causality."
    )
    print(
        "- Results are descriptive; no statistical significance is "
        "claimed."
    )


def main() -> None:
    args = parse_arguments()
    score_seeds = validate_score_seeds(args.score_seeds)
    output_paths = preflight_outputs(args.output_dir)
    vector_archives = load_vector_archives(
        vector_dir=args.vector_dir,
        score_seeds=score_seeds,
    )
    plans = load_plans(args.planner_dirs)
    by_plan = build_plan_proxy_table(
        plans=plans,
        vector_archives=vector_archives,
        score_seeds=score_seeds,
    )
    warning_messages = []
    ranking = build_ranking_summary(
        by_plan=by_plan,
        warning_messages=warning_messages,
    )
    save_csv(by_plan, output_paths["by_plan"])
    save_csv(ranking, output_paths["ranking"])
    generate_plots(
        by_plan=by_plan,
        ranking=ranking,
        output_paths=output_paths,
    )
    print_final_report(
        ranking=ranking,
        warning_messages=warning_messages,
    )
    print(f"\nSaved vector analysis: {args.output_dir}")


if __name__ == "__main__":
    main()
