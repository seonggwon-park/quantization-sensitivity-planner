"""Descriptively aggregate the three completed frozen reproducibility runs.

This module reads only per-seed artifacts beneath the supplied reproducibility
root. It performs no inference, method selection, retuning, or significance
testing. Locked test results are the primary aggregate; confirmation results
are retained as supportive context only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_SEEDS = (101, 202, 303)
BUDGETS = (0.70, 0.80, 0.82, 0.84, 0.85, 0.86)
VECTOR_MAIN = "vector_signed_mean_risk"
VECTOR_ABLATION = "vector_signed_p95_risk"
SCALAR_BASELINES = (
    "weight_rel_l2",
    "activation_rel_mse",
    "output_kl_mean",
    "abs_delta_score_mean",
    "decision_risk_mean",
    "decision_risk_p95",
)
METHODS = (VECTOR_MAIN, VECTOR_ABLATION, *SCALAR_BASELINES)
COMPARISON_EPSILON = 1e-12
NEAR_PARETO_SAVING_TOLERANCE = 0.001
POST_HOC_ENVELOPE_LABEL = "post-hoc scalar oracle envelope"
LOCKED_TEST_STATEMENT = (
    "Locked test evaluation is descriptive only and must not be used for "
    "method, objective, budget, baseline, or hyperparameter changes."
)

CONFIRMATION_METRICS = (
    "confirmation_accuracy",
    "confirmation_teacher_agreement",
    "confirmation_flip_rate",
    "confirmation_abs_delta_score_mean",
    "confirmation_decision_risk_mean",
    "confirmation_decision_risk_p95",
    "confirmation_decision_risk_violation_rate",
)
TEST_METRICS = (
    "test_accuracy",
    "test_teacher_agreement",
    "test_flip_rate",
    "test_abs_delta_score_mean",
    "test_decision_risk_mean",
    "test_decision_risk_p95",
    "test_decision_risk_violation_rate",
)

PER_SEED_COLUMNS = (
    "seed",
    "requested_memory_saving_ratio",
    "plan_id",
    "plan_kind",
    "selection",
    "comparison_role",
    "actual_memory_saving_ratio",
    *CONFIRMATION_METRICS,
    "confirmation_runtime_seconds",
    *TEST_METRICS,
    "test_runtime_seconds",
    "training_final_validation_accuracy",
    "training_manifest_path",
    "training_log_path",
    "confirmation_summary_path",
    "test_summary_path",
    "confirmation_source_plan_path",
    "test_source_plan_path",
    "training_checkpoint_path",
)

PAIRWISE_COLUMNS = (
    "seed",
    "requested_memory_saving_ratio",
    "scalar_baseline",
    "vector_selection",
    "vector_actual_memory_saving_ratio",
    "scalar_actual_memory_saving_ratio",
    "saving_delta",
    "vector_test_p95",
    "scalar_test_p95",
    "p95_delta",
    "p95_relative_reduction",
    "vector_test_flip_rate",
    "scalar_test_flip_rate",
    "flip_delta",
    "vector_test_accuracy",
    "scalar_test_accuracy",
    "accuracy_delta",
    "vector_lower_p95",
    "vector_no_worse_flip",
    "vector_no_worse_accuracy",
    "vector_pareto_wins",
    "vector_near_pareto_wins",
)

SUMMARY_COLUMNS = (
    "section",
    "analysis_label",
    "summary_scope",
    "evaluated_method",
    "reference_method",
    "scalar_baseline",
    "seed",
    "requested_memory_saving_ratio",
    "n_cells",
    "expected_cell_count",
    "p95_win_count",
    "p95_win_rate",
    "relative_p95_reduction_mean",
    "relative_p95_reduction_median",
    "relative_p95_reduction_min",
    "relative_p95_reduction_max",
    "mean_p95_delta",
    "mean_saving_delta",
    "flip_win_count",
    "flip_tie_count",
    "flip_loss_count",
    "accuracy_win_count",
    "accuracy_tie_count",
    "accuracy_loss_count",
    "strict_pareto_win_count",
    "near_pareto_win_count",
    "descriptive_only",
    "notes",
)

TRAINING_SUMMARY_COLUMNS = (
    "seed",
    "training_seed",
    "data_split_seed",
    "validation_fraction",
    "final_epoch",
    "final_train_accuracy",
    "final_train_loss",
    "final_validation_accuracy",
    "final_validation_loss",
    "checkpoint_path",
    "test_data_accessed_during_training",
    "shared_data_split_across_seeds",
    "training_manifest_path",
    "training_log_path",
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create descriptive aggregate tables and figures for the frozen "
            "binary ResNet-18 reproducibility runs."
        )
    )
    parser.add_argument(
        "--repro-root",
        type=Path,
        default=Path("results/repro_v1"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(EXPECTED_SEEDS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/repro_v1/aggregate_v1"),
    )
    return parser.parse_args(argv)


def require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    source_path: Path,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_path} is missing columns: {missing}")


def nested_value(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Manifest lacks required field: {'.'.join(keys)}")
        value = value[key]
    return value


def normalize_budget(value: Any, source_path: Path) -> float:
    numeric = float(value)
    matches = [budget for budget in BUDGETS if abs(numeric - budget) <= 1e-9]
    if len(matches) != 1:
        raise ValueError(
            f"{source_path} contains non-frozen budget value {numeric}."
        )
    return matches[0]


def normalize_confirmation_selection(frame: pd.DataFrame) -> pd.Series:
    vector = frame["vector_objective"].fillna("").astype(str).str.strip()
    scalar = frame["selected_score_metric"].fillna("").astype(str).str.strip()
    selection = vector.where(vector.ne(""), scalar)
    if selection.eq("").any():
        raise ValueError("Confirmation summary contains an unlabeled plan.")
    return selection


def validate_method_grid(
    frame: pd.DataFrame,
    source_path: Path,
    seed: int,
) -> None:
    observed_methods = set(frame["selection"])
    if observed_methods != set(METHODS):
        raise ValueError(
            f"{source_path} method set differs for seed {seed}: "
            f"observed={sorted(observed_methods)}, expected={sorted(METHODS)}"
        )
    observed_budgets = set(frame["requested_memory_saving_ratio"])
    if observed_budgets != set(BUDGETS):
        raise ValueError(
            f"{source_path} budget set differs for seed {seed}: "
            f"observed={sorted(observed_budgets)}, expected={list(BUDGETS)}"
        )
    keys = frame[["selection", "requested_memory_saving_ratio"]]
    if keys.duplicated().any():
        duplicate_rows = frame.loc[
            keys.duplicated(keep=False),
            ["selection", "requested_memory_saving_ratio"],
        ]
        raise ValueError(
            f"{source_path} has duplicate method-budget cells:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )
    expected_rows = len(METHODS) * len(BUDGETS)
    if len(frame) != expected_rows:
        raise ValueError(
            f"{source_path} has {len(frame)} rows; expected {expected_rows}."
        )
    if not frame["comparison_role"].eq("primary").all():
        raise ValueError(f"{source_path} contains non-primary rows.")


def load_evaluation_summary(
    path: Path,
    seed: int,
    kind: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    common = (
        "plan_id",
        "plan_kind",
        "requested_memory_saving_ratio",
        "actual_memory_saving_ratio",
        "comparison_role",
        "runtime_seconds",
        "source_plan_path",
    )
    if kind == "test":
        require_columns(frame, (*common, "selection", *TEST_METRICS), path)
    elif kind == "confirmation":
        require_columns(
            frame,
            (
                *common,
                "selected_score_metric",
                "vector_objective",
                *CONFIRMATION_METRICS,
            ),
            path,
        )
        frame["selection"] = normalize_confirmation_selection(frame)
    else:
        raise ValueError(f"Unknown evaluation kind: {kind}")

    frame["selection"] = frame["selection"].astype(str).str.strip()
    frame["requested_memory_saving_ratio"] = frame[
        "requested_memory_saving_ratio"
    ].map(lambda value: normalize_budget(value, path))
    validate_method_grid(frame, path, seed)
    return frame


def load_training_run(
    repro_root: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_root = repro_root / f"seed_{seed}"
    manifest_path = seed_root / "training_manifest.json"
    log_path = seed_root / "training_log.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("training_seed", -1)) != seed:
        raise ValueError(
            f"{manifest_path} training_seed does not match seed {seed}."
        )
    if manifest.get("test_data_accessed") is not False:
        raise ValueError(
            f"{manifest_path} does not confirm test_data_accessed=false."
        )
    if nested_value(
        manifest,
        "resolved_settings",
        "shared_data_partition_across_training_seeds",
    ) is not True:
        raise ValueError(
            f"{manifest_path} does not confirm a shared data partition."
        )

    training_log = pd.read_csv(log_path)
    require_columns(
        training_log,
        (
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        ),
        log_path,
    )
    if training_log.empty or training_log["epoch"].duplicated().any():
        raise ValueError(f"{log_path} has an invalid epoch log.")
    training_log = training_log.sort_values("epoch")
    final = training_log.iloc[-1]
    final_epoch = int(final["epoch"])
    expected_final_epoch = int(
        nested_value(manifest, "resolved_settings", "final_epoch")
    )
    if final_epoch != expected_final_epoch:
        raise ValueError(
            f"{log_path} ends at epoch {final_epoch}, expected "
            f"{expected_final_epoch}."
        )

    record = {
        "seed": seed,
        "training_seed": int(manifest["training_seed"]),
        "data_split_seed": int(manifest["data_split_seed"]),
        "validation_fraction": float(manifest["validation_fraction"]),
        "final_epoch": final_epoch,
        "final_train_accuracy": float(final["train_accuracy"]),
        "final_train_loss": float(final["train_loss"]),
        "final_validation_accuracy": float(final["validation_accuracy"]),
        "final_validation_loss": float(final["validation_loss"]),
        "checkpoint_path": str(manifest["output_checkpoint_path"]),
        "test_data_accessed_during_training": bool(
            manifest["test_data_accessed"]
        ),
        "shared_data_split_across_seeds": True,
        "training_manifest_path": str(manifest_path.resolve()),
        "training_log_path": str(log_path.resolve()),
    }
    return manifest, record


def recipe_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    resolved_settings = manifest["resolved_settings"]
    return {
        "architecture": nested_value(
            resolved_settings, "model", "architecture"
        ),
        "pretrained_initialization_family": nested_value(
            resolved_settings, "model", "pretrained_weights"
        ),
        "data_split_seed": manifest["data_split_seed"],
        "validation_fraction": manifest["validation_fraction"],
        "epoch_count": nested_value(resolved_settings, "epoch_count"),
        "optimizer": nested_value(resolved_settings, "optimizer"),
        "scheduler": nested_value(resolved_settings, "scheduler"),
        "loss": nested_value(resolved_settings, "loss"),
        "transforms": nested_value(resolved_settings, "transforms"),
        "dataset": {
            key: nested_value(resolved_settings, "dataset", key)
            for key in (
                "dataset_class",
                "source",
                "source_split",
                "validation_fraction",
            )
        },
    }


def validate_shared_recipe(manifests: dict[int, dict[str, Any]]) -> dict[str, Any]:
    reference_seed = EXPECTED_SEEDS[0]
    reference = recipe_fingerprint(manifests[reference_seed])
    reference_json = json.dumps(reference, sort_keys=True)
    for seed, manifest in manifests.items():
        candidate = recipe_fingerprint(manifest)
        if json.dumps(candidate, sort_keys=True) != reference_json:
            raise ValueError(
                f"Seed {seed} does not share the frozen recipe used by seed "
                f"{reference_seed}."
            )
    return reference


def build_per_seed_results(
    repro_root: Path,
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    combined_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    manifests: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        seed_root = repro_root / f"seed_{seed}"
        manifest, training = load_training_run(repro_root, seed)
        manifests[seed] = manifest
        training_rows.append(training)

        confirmation_path = (
            seed_root
            / "confirmation_eval"
            / "confirmation_primary_summary.csv"
        )
        test_path = seed_root / "test_eval" / "test_primary_summary.csv"
        confirmation = load_evaluation_summary(
            confirmation_path, seed, "confirmation"
        )
        test = load_evaluation_summary(test_path, seed, "test")

        join_keys = ["selection", "requested_memory_saving_ratio"]
        confirmation_fields = [
            *join_keys,
            "plan_id",
            "plan_kind",
            "actual_memory_saving_ratio",
            "comparison_role",
            *CONFIRMATION_METRICS,
            "runtime_seconds",
            "source_plan_path",
        ]
        test_fields = [
            *join_keys,
            "plan_id",
            "plan_kind",
            "actual_memory_saving_ratio",
            "comparison_role",
            *TEST_METRICS,
            "runtime_seconds",
            "source_plan_path",
        ]
        merged = confirmation[confirmation_fields].merge(
            test[test_fields],
            on=join_keys,
            how="outer",
            validate="one_to_one",
            suffixes=("_confirmation", "_test"),
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(
                f"Seed {seed} confirmation/test method grids do not match."
            )
        for field in ("plan_id", "plan_kind", "comparison_role"):
            if not merged[f"{field}_confirmation"].eq(
                merged[f"{field}_test"]
            ).all():
                raise ValueError(
                    f"Seed {seed} confirmation/test {field} values differ."
                )
        saving_error = (
            merged["actual_memory_saving_ratio_confirmation"]
            - merged["actual_memory_saving_ratio_test"]
        ).abs()
        if (saving_error > 1e-12).any():
            raise ValueError(
                f"Seed {seed} confirmation/test actual savings differ."
            )

        for _, row in merged.iterrows():
            output = {
                "seed": seed,
                "requested_memory_saving_ratio": float(
                    row["requested_memory_saving_ratio"]
                ),
                "plan_id": row["plan_id_test"],
                "plan_kind": row["plan_kind_test"],
                "selection": row["selection"],
                "comparison_role": row["comparison_role_test"],
                "actual_memory_saving_ratio": float(
                    row["actual_memory_saving_ratio_test"]
                ),
                "confirmation_runtime_seconds": float(
                    row["runtime_seconds_confirmation"]
                ),
                "test_runtime_seconds": float(row["runtime_seconds_test"]),
                "training_final_validation_accuracy": training[
                    "final_validation_accuracy"
                ],
                "training_manifest_path": training[
                    "training_manifest_path"
                ],
                "training_log_path": training["training_log_path"],
                "confirmation_summary_path": str(
                    confirmation_path.resolve()
                ),
                "test_summary_path": str(test_path.resolve()),
                "confirmation_source_plan_path": row[
                    "source_plan_path_confirmation"
                ],
                "test_source_plan_path": row["source_plan_path_test"],
                "training_checkpoint_path": training["checkpoint_path"],
            }
            for metric in CONFIRMATION_METRICS:
                output[metric] = float(row[metric])
            for metric in TEST_METRICS:
                output[metric] = float(row[metric])
            combined_rows.append(output)

    per_seed = pd.DataFrame(combined_rows, columns=PER_SEED_COLUMNS)
    training_summary = pd.DataFrame(
        training_rows, columns=TRAINING_SUMMARY_COLUMNS
    )
    expected_rows = len(seeds) * len(BUDGETS) * len(METHODS)
    if len(per_seed) != expected_rows:
        raise RuntimeError(
            f"Aggregated {len(per_seed)} cells; expected {expected_rows}."
        )
    if per_seed[
        ["seed", "requested_memory_saving_ratio", "selection"]
    ].duplicated().any():
        raise RuntimeError("Aggregated seed-budget-method keys are not unique.")
    shared_recipe = validate_shared_recipe(manifests)
    return per_seed, training_summary, shared_recipe


def comparison_row(
    candidate: pd.Series,
    reference: pd.Series,
    scalar_baseline: str,
) -> dict[str, Any]:
    candidate_saving = float(candidate["actual_memory_saving_ratio"])
    reference_saving = float(reference["actual_memory_saving_ratio"])
    candidate_p95 = float(candidate["test_decision_risk_p95"])
    reference_p95 = float(reference["test_decision_risk_p95"])
    candidate_flip = float(candidate["test_flip_rate"])
    reference_flip = float(reference["test_flip_rate"])
    candidate_accuracy = float(candidate["test_accuracy"])
    reference_accuracy = float(reference["test_accuracy"])
    lower_p95 = candidate_p95 < reference_p95
    relative_reduction = (
        (reference_p95 - candidate_p95) / reference_p95
        if reference_p95 != 0.0
        else np.nan
    )
    return {
        "seed": int(candidate["seed"]),
        "requested_memory_saving_ratio": float(
            candidate["requested_memory_saving_ratio"]
        ),
        "scalar_baseline": scalar_baseline,
        "vector_selection": str(candidate["selection"]),
        "vector_actual_memory_saving_ratio": candidate_saving,
        "scalar_actual_memory_saving_ratio": reference_saving,
        "saving_delta": candidate_saving - reference_saving,
        "vector_test_p95": candidate_p95,
        "scalar_test_p95": reference_p95,
        "p95_delta": candidate_p95 - reference_p95,
        "p95_relative_reduction": relative_reduction,
        "vector_test_flip_rate": candidate_flip,
        "scalar_test_flip_rate": reference_flip,
        "flip_delta": candidate_flip - reference_flip,
        "vector_test_accuracy": candidate_accuracy,
        "scalar_test_accuracy": reference_accuracy,
        "accuracy_delta": candidate_accuracy - reference_accuracy,
        "vector_lower_p95": lower_p95,
        "vector_no_worse_flip": candidate_flip <= reference_flip,
        "vector_no_worse_accuracy": candidate_accuracy >= reference_accuracy,
        "vector_pareto_wins": (
            lower_p95 and candidate_saving >= reference_saving
        ),
        "vector_near_pareto_wins": (
            lower_p95
            and candidate_saving
            >= reference_saving - NEAR_PARETO_SAVING_TOLERANCE
        ),
    }


def build_main_pairwise(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = per_seed.set_index(
        ["seed", "requested_memory_saving_ratio", "selection"],
        verify_integrity=True,
    )
    for seed in EXPECTED_SEEDS:
        for budget in BUDGETS:
            candidate = indexed.loc[(seed, budget, VECTOR_MAIN)]
            for baseline in SCALAR_BASELINES:
                reference = indexed.loc[(seed, budget, baseline)]
                rows.append(comparison_row(candidate, reference, baseline))
    pairwise = pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)
    expected = len(EXPECTED_SEEDS) * len(BUDGETS) * len(SCALAR_BASELINES)
    if len(pairwise) != expected:
        raise RuntimeError(f"Built {len(pairwise)} pairs; expected {expected}.")
    return pairwise


def relation_counts(values: pd.Series, lower_is_better: bool) -> tuple[int, int, int]:
    numeric = values.to_numpy(dtype=float)
    ties = np.isclose(numeric, 0.0, atol=COMPARISON_EPSILON, rtol=0.0)
    if lower_is_better:
        wins = numeric < -COMPARISON_EPSILON
        losses = numeric > COMPARISON_EPSILON
    else:
        wins = numeric > COMPARISON_EPSILON
        losses = numeric < -COMPARISON_EPSILON
    return int(wins.sum()), int(ties.sum()), int(losses.sum())


def summarize_comparisons(
    frame: pd.DataFrame,
    section: str,
    analysis_label: str,
    summary_scope: str,
    evaluated_method: str,
    reference_method: str,
    scalar_baseline: str,
    expected_cell_count: int,
    seed: int | None = None,
    budget: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"Cannot summarize an empty comparison: {analysis_label}")
    flip_wins, flip_ties, flip_losses = relation_counts(
        frame["flip_delta"], lower_is_better=True
    )
    accuracy_wins, accuracy_ties, accuracy_losses = relation_counts(
        frame["accuracy_delta"], lower_is_better=False
    )
    relative = frame["p95_relative_reduction"]
    return {
        "section": section,
        "analysis_label": analysis_label,
        "summary_scope": summary_scope,
        "evaluated_method": evaluated_method,
        "reference_method": reference_method,
        "scalar_baseline": scalar_baseline,
        "seed": seed,
        "requested_memory_saving_ratio": budget,
        "n_cells": len(frame),
        "expected_cell_count": expected_cell_count,
        "p95_win_count": int(frame["vector_lower_p95"].sum()),
        "p95_win_rate": float(frame["vector_lower_p95"].mean()),
        "relative_p95_reduction_mean": float(relative.mean()),
        "relative_p95_reduction_median": float(relative.median()),
        "relative_p95_reduction_min": float(relative.min()),
        "relative_p95_reduction_max": float(relative.max()),
        "mean_p95_delta": float(frame["p95_delta"].mean()),
        "mean_saving_delta": float(frame["saving_delta"].mean()),
        "flip_win_count": flip_wins,
        "flip_tie_count": flip_ties,
        "flip_loss_count": flip_losses,
        "accuracy_win_count": accuracy_wins,
        "accuracy_tie_count": accuracy_ties,
        "accuracy_loss_count": accuracy_losses,
        "strict_pareto_win_count": int(frame["vector_pareto_wins"].sum()),
        "near_pareto_win_count": int(
            frame["vector_near_pareto_wins"].sum()
        ),
        "descriptive_only": True,
        "notes": notes,
    }


def build_envelope_comparisons(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in EXPECTED_SEEDS:
        for budget in BUDGETS:
            cell = per_seed[
                per_seed["seed"].eq(seed)
                & per_seed["requested_memory_saving_ratio"].eq(budget)
            ]
            candidate = cell[cell["selection"].eq(VECTOR_MAIN)].iloc[0]
            scalars = cell[cell["selection"].isin(SCALAR_BASELINES)].sort_values(
                [
                    "test_decision_risk_p95",
                    "actual_memory_saving_ratio",
                    "selection",
                ],
                ascending=[True, False, True],
                kind="mergesort",
            )
            reference = scalars.iloc[0]
            row = comparison_row(candidate, reference, str(reference["selection"]))
            row["envelope_selected_scalar"] = str(reference["selection"])
            rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_comparisons(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = per_seed.set_index(
        ["seed", "requested_memory_saving_ratio", "selection"],
        verify_integrity=True,
    )
    for seed in EXPECTED_SEEDS:
        for budget in BUDGETS:
            ablation = indexed.loc[(seed, budget, VECTOR_ABLATION)]
            main = indexed.loc[(seed, budget, VECTOR_MAIN)]
            rows.append(comparison_row(ablation, main, VECTOR_MAIN))
    return pd.DataFrame(rows)


def append_scoped_summaries(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    section_prefix: str,
    label: str,
    evaluated_method: str,
    reference_method: str,
    scalar_baseline: str,
    notes: str,
) -> None:
    rows.append(
        summarize_comparisons(
            frame,
            section=f"{section_prefix}_overall",
            analysis_label=label,
            summary_scope="all_seed_budget_pairs",
            evaluated_method=evaluated_method,
            reference_method=reference_method,
            scalar_baseline=scalar_baseline,
            expected_cell_count=18,
            notes=notes,
        )
    )
    for budget in BUDGETS:
        subset = frame[frame["requested_memory_saving_ratio"].eq(budget)]
        rows.append(
            summarize_comparisons(
                subset,
                section=f"{section_prefix}_per_budget",
                analysis_label=label,
                summary_scope="per_budget_across_three_seeds",
                evaluated_method=evaluated_method,
                reference_method=reference_method,
                scalar_baseline=scalar_baseline,
                expected_cell_count=3,
                budget=budget,
                notes=notes,
            )
        )
    for seed in EXPECTED_SEEDS:
        subset = frame[frame["seed"].eq(seed)]
        rows.append(
            summarize_comparisons(
                subset,
                section=f"{section_prefix}_per_seed",
                analysis_label=label,
                summary_scope="per_seed_across_six_budgets",
                evaluated_method=evaluated_method,
                reference_method=reference_method,
                scalar_baseline=scalar_baseline,
                expected_cell_count=6,
                seed=seed,
                notes=notes,
            )
        )


def build_aggregate_summary(
    pairwise: pd.DataFrame,
    envelope: pd.DataFrame,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_notes = (
        "Primary locked-test descriptive comparison; seed-budget cells are "
        "repeated measurements, not independent statistical samples."
    )
    for baseline in SCALAR_BASELINES:
        baseline_pairs = pairwise[pairwise["scalar_baseline"].eq(baseline)]
        rows.append(
            summarize_comparisons(
                baseline_pairs,
                section="A_overall_by_scalar_baseline",
                analysis_label="vector main versus fixed scalar baseline",
                summary_scope="all_seed_budget_pairs",
                evaluated_method=VECTOR_MAIN,
                reference_method=baseline,
                scalar_baseline=baseline,
                expected_cell_count=18,
                notes=primary_notes,
            )
        )
        for budget in BUDGETS:
            subset = baseline_pairs[
                baseline_pairs["requested_memory_saving_ratio"].eq(budget)
            ]
            rows.append(
                summarize_comparisons(
                    subset,
                    section="B_per_budget_by_scalar_baseline",
                    analysis_label="vector main versus fixed scalar baseline",
                    summary_scope="per_budget_across_three_seeds",
                    evaluated_method=VECTOR_MAIN,
                    reference_method=baseline,
                    scalar_baseline=baseline,
                    expected_cell_count=3,
                    budget=budget,
                    notes=primary_notes,
                )
            )
        for seed in EXPECTED_SEEDS:
            subset = baseline_pairs[baseline_pairs["seed"].eq(seed)]
            rows.append(
                summarize_comparisons(
                    subset,
                    section="C_per_seed_by_scalar_baseline",
                    analysis_label="vector main versus fixed scalar baseline",
                    summary_scope="per_seed_across_six_budgets",
                    evaluated_method=VECTOR_MAIN,
                    reference_method=baseline,
                    scalar_baseline=baseline,
                    expected_cell_count=6,
                    seed=seed,
                    notes=primary_notes,
                )
            )

    append_scoped_summaries(
        rows=rows,
        frame=envelope,
        section_prefix="D_post_hoc_scalar_oracle_envelope",
        label=POST_HOC_ENVELOPE_LABEL,
        evaluated_method=VECTOR_MAIN,
        reference_method="best_scalar_by_locked_test_p95",
        scalar_baseline="best_scalar_envelope",
        notes=(
            "Post-hoc scalar oracle envelope selected using locked test p95; "
            "descriptive diagnostic only and not a primary comparison."
        ),
    )
    append_scoped_summaries(
        rows=rows,
        frame=ablation,
        section_prefix="E_signed_p95_ablation",
        label="signed-p95 ablation only",
        evaluated_method=VECTOR_ABLATION,
        reference_method=VECTOR_MAIN,
        scalar_baseline="",
        notes=(
            "Ablation-only comparison of vector signed-p95 against the frozen "
            "vector signed-mean main method."
        ),
    )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def method_display_name(method: str) -> str:
    labels = {
        VECTOR_MAIN: "Vector signed mean (main)",
        VECTOR_ABLATION: "Vector signed p95 (ablation)",
        "weight_rel_l2": "Weight relative L2",
        "activation_rel_mse": "Activation relative MSE",
        "output_kl_mean": "Output KL mean",
        "abs_delta_score_mean": "Absolute score delta mean",
        "decision_risk_mean": "Decision risk mean",
        "decision_risk_p95": "Decision risk p95",
    }
    return labels[method]


def plot_test_p95_by_budget_seed(
    per_seed: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = plt.get_cmap("tab10")
    color_by_method = {method: colors(index) for index, method in enumerate(METHODS)}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for axis, seed in zip(axes, EXPECTED_SEEDS):
        seed_data = per_seed[per_seed["seed"].eq(seed)]
        for method in METHODS:
            values = seed_data[seed_data["selection"].eq(method)].sort_values(
                "requested_memory_saving_ratio"
            )
            axis.plot(
                values["requested_memory_saving_ratio"],
                values["test_decision_risk_p95"],
                marker="o" if method.startswith("vector_") else ".",
                linewidth=2.6 if method == VECTOR_MAIN else 1.4,
                linestyle="--" if method == VECTOR_ABLATION else "-",
                color=color_by_method[method],
                label=method_display_name(method),
            )
        axis.set_title(f"Training seed {seed}")
        axis.set_xlabel("Requested memory saving ratio")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Locked test decision-risk p95")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Locked test p95 by budget and training seed")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_relative_reduction_heatmap(
    pairwise: pd.DataFrame,
    output_path: Path,
) -> None:
    pivot = pairwise.pivot_table(
        index="scalar_baseline",
        columns="requested_memory_saving_ratio",
        values="p95_relative_reduction",
        aggfunc="mean",
    ).reindex(index=SCALAR_BASELINES, columns=BUDGETS)
    values = pivot.to_numpy(dtype=float)
    bound = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1e-9)
    fig, axis = plt.subplots(figsize=(10, 5.8))
    image = axis.imshow(values, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:.1%}",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_xticks(range(len(BUDGETS)), [f"{value:.2f}" for value in BUDGETS])
    axis.set_yticks(
        range(len(SCALAR_BASELINES)),
        [method_display_name(value) for value in SCALAR_BASELINES],
    )
    axis.set_xlabel("Requested memory saving ratio")
    axis.set_title("Vector-main relative test-p95 reduction (mean across seeds)")
    fig.colorbar(image, ax=axis, label="Relative p95 reduction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_test_p95_saving_scatter(
    per_seed: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = plt.get_cmap("tab10")
    fig, axis = plt.subplots(figsize=(10, 6.5))
    for index, method in enumerate(METHODS):
        values = per_seed[per_seed["selection"].eq(method)]
        vector_method = method.startswith("vector_")
        axis.scatter(
            values["actual_memory_saving_ratio"],
            values["test_decision_risk_p95"],
            label=method_display_name(method),
            color=colors(index),
            marker="o" if vector_method else "x",
            s=55 if vector_method else 42,
            alpha=0.8,
        )
    axis.set_xlabel("Actual memory saving ratio")
    axis.set_ylabel("Locked test decision-risk p95")
    axis.set_title("Locked test p95 versus actual saving")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_win_rate(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    overall = summary[
        summary["section"].eq("A_overall_by_scalar_baseline")
    ].set_index("scalar_baseline").reindex(SCALAR_BASELINES)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(
        [method_display_name(value) for value in SCALAR_BASELINES],
        overall["p95_win_rate"],
        color="#3977b8",
    )
    axis.bar_label(bars, labels=[f"{value:.0%}" for value in overall["p95_win_rate"]])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Vector-main locked-test p95 win rate")
    axis.set_title("Descriptive p95 win rate across 18 seed-budget cells")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_flip_accuracy_tradeoff(
    pairwise: pd.DataFrame,
    output_path: Path,
) -> None:
    means = pairwise.groupby("scalar_baseline", sort=False)[
        ["flip_delta", "accuracy_delta"]
    ].mean().reindex(SCALAR_BASELINES)
    labels = [method_display_name(value) for value in SCALAR_BASELINES]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    axes[0].bar(positions, means["flip_delta"], color="#d95f5f")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Mean flip-rate delta")
    axes[0].set_ylabel("Vector main minus scalar")
    axes[1].bar(positions, means["accuracy_delta"], color="#4b9b70")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Mean accuracy delta")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Vector-main flip and accuracy trade-offs versus scalar baselines")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "per_seed": output_dir / "per_seed_budget_results.csv",
        "pairwise": output_dir / "vector_vs_scalar_pairwise.csv",
        "aggregate": output_dir / "aggregate_summary.csv",
        "training": output_dir / "training_summary.csv",
        "p95_lines": output_dir / "test_p95_by_budget_seed.png",
        "heatmap": output_dir / "vector_main_relative_reduction_heatmap.png",
        "scatter": output_dir / "test_p95_vs_actual_saving_scatter.png",
        "win_rate": output_dir / "vector_main_win_rate_by_baseline.png",
        "tradeoff": output_dir / "flip_accuracy_tradeoff_summary.png",
    }


def preflight_outputs(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing aggregate outputs: "
            + ", ".join(existing)
        )


def print_tables(
    summary: pd.DataFrame,
    shared_recipe: dict[str, Any],
) -> None:
    overall = summary[
        summary["section"].eq("A_overall_by_scalar_baseline")
    ]
    print("\nMain result table: vector signed-mean versus scalar baselines")
    print(
        overall[
            [
                "scalar_baseline",
                "p95_win_count",
                "expected_cell_count",
                "p95_win_rate",
                "relative_p95_reduction_mean",
                "mean_p95_delta",
                "mean_saving_delta",
                "strict_pareto_win_count",
                "near_pareto_win_count",
            ]
        ].to_string(index=False)
    )

    per_budget = summary[
        summary["section"].eq("B_per_budget_by_scalar_baseline")
    ].pivot(
        index="requested_memory_saving_ratio",
        columns="scalar_baseline",
        values="p95_win_rate",
    ).reindex(index=BUDGETS, columns=SCALAR_BASELINES)
    print("\nPer-budget p95 win-rate table across three seeds")
    print(per_budget.to_string())

    per_seed = summary[
        summary["section"].eq("C_per_seed_by_scalar_baseline")
    ].pivot(
        index="seed",
        columns="scalar_baseline",
        values="p95_win_rate",
    ).reindex(index=EXPECTED_SEEDS, columns=SCALAR_BASELINES)
    print("\nPer-seed p95 win-rate table across six budgets")
    print(per_seed.to_string())

    envelope = summary[
        summary["section"].str.startswith(
            "D_post_hoc_scalar_oracle_envelope"
        )
    ]
    print(f"\n{POST_HOC_ENVELOPE_LABEL} (not a primary comparison)")
    print(
        envelope[
            [
                "summary_scope",
                "seed",
                "requested_memory_saving_ratio",
                "n_cells",
                "p95_win_count",
                "p95_win_rate",
                "relative_p95_reduction_mean",
                "mean_saving_delta",
            ]
        ].to_string(index=False)
    )

    ablation = summary[
        summary["section"].str.startswith("E_signed_p95_ablation")
    ]
    print("\nSigned-p95 ablation table (ablation only)")
    print(
        ablation[
            [
                "summary_scope",
                "seed",
                "requested_memory_saving_ratio",
                "n_cells",
                "p95_win_count",
                "p95_win_rate",
                "relative_p95_reduction_mean",
                "mean_saving_delta",
            ]
        ].to_string(index=False)
    )

    print("\nLimitations")
    print(
        "- The 18 seed-budget cells are descriptive repeated measurements, "
        "not 18 independent statistical samples."
    )
    print("- No p-values or claims of statistical significance are made.")
    print(
        "- The three seeds share architecture "
        f"({shared_recipe['architecture']}), pretrained initialization family "
        f"({shared_recipe['pretrained_initialization_family']}), data split "
        f"(seed {shared_recipe['data_split_seed']}), and frozen training recipe."
    )
    print(
        "- Confirmation results are supportive only; locked test results are "
        "the primary aggregate."
    )
    print(f"- {LOCKED_TEST_STATEMENT}")
    print(
        f"- The {POST_HOC_ENVELOPE_LABEL} uses locked test outcomes and is "
        "reported only as a post-hoc descriptive diagnostic."
    )
    print("- Only three independently seeded training runs are summarized.")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    seeds = tuple(sorted(int(seed) for seed in args.seeds))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(
            f"Frozen aggregate requires seeds {EXPECTED_SEEDS}; got {seeds}."
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates.")

    repro_root = args.repro_root.resolve()
    output_dir = args.output_dir.resolve()
    if not repro_root.is_dir():
        raise FileNotFoundError(repro_root)
    if output_dir == repro_root or repro_root not in output_dir.parents:
        raise ValueError(
            "--output-dir must be a new descendant of --repro-root so the "
            "analysis cannot write into pilot result directories."
        )
    paths = output_paths(output_dir)
    preflight_outputs(paths)

    per_seed, training_summary, shared_recipe = build_per_seed_results(
        repro_root, seeds
    )
    pairwise = build_main_pairwise(per_seed)
    envelope = build_envelope_comparisons(per_seed)
    ablation = build_ablation_comparisons(per_seed)
    aggregate = build_aggregate_summary(pairwise, envelope, ablation)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(paths["per_seed"], index=False)
    pairwise.to_csv(paths["pairwise"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    training_summary.to_csv(paths["training"], index=False)
    plot_test_p95_by_budget_seed(per_seed, paths["p95_lines"])
    plot_relative_reduction_heatmap(pairwise, paths["heatmap"])
    plot_test_p95_saving_scatter(per_seed, paths["scatter"])
    plot_win_rate(aggregate, paths["win_rate"])
    plot_flip_accuracy_tradeoff(pairwise, paths["tradeoff"])

    print_tables(aggregate, shared_recipe)
    print("\nSaved descriptive frozen-run aggregate")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
