"""Heuristic mixed-precision beam search over signed score-delta vectors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from additive_planner import (
    QUANTIZED_ACTIONS,
    action_to_bits,
    weight_storage_bytes,
)


ACTIONS = (
    "fp32",
    "fp16",
    "int8",
    "int4",
)

VECTOR_OBJECTIVES = (
    "vector_signed_mean_risk",
    "vector_signed_p95_risk",
)

OBJECTIVE_FILE_LABELS = {
    "vector_signed_mean_risk": "vector_signed_mean",
    "vector_signed_p95_risk": "vector_signed_p95",
}

VECTOR_FILE_PATTERN = re.compile(
    r"^score_delta_vectors_seed(-?\d+)\.npz$"
)

EPSILON = 1e-12
SOLVER_NAME = "deterministic_memory_diverse_vector_beam_search"

SUMMARY_COLUMNS = (
    "plan_id",
    "vector_objective",
    "requested_memory_saving_ratio",
    "actual_memory_saving_ratio",
    "objective_mean",
    "objective_std",
    "objective_per_seed",
    "num_fp32",
    "num_fp16",
    "num_int8",
    "num_int4",
    "selected_search_pass",
    "beam_width",
    "max_states_per_memory_bin",
)


@dataclass(frozen=True)
class VectorBundle:
    score_seeds: tuple[int, ...]
    source_paths: tuple[Path, ...]
    candidate_ids: tuple[str, ...]
    layer_names: tuple[str, ...]
    action_names: tuple[str, ...]
    baseline_margins: np.ndarray
    delta_by_candidate: dict[str, np.ndarray]


@dataclass(frozen=True)
class LayerMemory:
    weight_numel: int
    action_bytes: dict[str, int]


@dataclass(frozen=True)
class MemoryAccounting:
    layer_order: tuple[str, ...]
    layers: dict[str, LayerMemory]
    constant_parameter_bytes: int
    fp32_total_memory_bytes: int
    metadata_paths: tuple[Path, ...]


@dataclass(frozen=True)
class BeamState:
    actions: tuple[str, ...]
    weight_bytes: int
    summed_deltas: np.ndarray
    heuristic_objective: float


@dataclass(frozen=True)
class PassResult:
    search_pass: str
    selected_layer_order: tuple[str, ...]
    layer_actions: dict[str, str]
    actual_total_memory_bytes: int
    objective_per_seed: dict[int, float]
    objective_mean: float
    objective_std: float
    action_string: str


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=Path("results/go_no_go_vectors_v1"),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("results/go_no_go"),
    )
    parser.add_argument(
        "--memory-saving-ratios",
        type=float,
        nargs="+",
        default=[0.70, 0.80, 0.82, 0.84, 0.85, 0.86],
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-states-per-memory-bin",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--memory-quantum-kb",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--include-original-order",
        action="store_true",
        help="Run a third pass in original model module order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/go_no_go_vector_plans_v1"),
    )

    return parser.parse_args()


def budget_key(memory_saving_ratio: float) -> str:
    basis_points = int(
        round(memory_saving_ratio * 10_000)
    )

    return f"save_{basis_points:04d}bp"


def validate_arguments(args) -> tuple[float, ...]:
    if args.beam_width <= 0:
        raise ValueError("beam_width must be positive.")

    if args.max_states_per_memory_bin <= 0:
        raise ValueError(
            "max_states_per_memory_bin must be positive."
        )

    if args.memory_quantum_kb <= 0:
        raise ValueError(
            "memory_quantum_kb must be positive."
        )

    ratios = tuple(
        float(ratio)
        for ratio in args.memory_saving_ratios
    )

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


def planned_output_paths(
    output_dir: Path,
    memory_saving_ratios: tuple[float, ...],
) -> dict[str, Path]:
    paths = {
        "summary": output_dir / "vector_planner_summary.csv"
    }

    for ratio in memory_saving_ratios:
        key = budget_key(ratio)

        for objective_name in VECTOR_OBJECTIVES:
            label = OBJECTIVE_FILE_LABELS[objective_name]
            plan_id = f"{label}_{key}"
            paths[plan_id] = output_dir / f"plan_{plan_id}.json"

    for output_path in paths.values():
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )

    return paths


def discover_vector_paths(
    vector_dir: Path,
) -> list[tuple[int, Path]]:
    discovered = []

    for path in vector_dir.glob(
        "score_delta_vectors_seed*.npz"
    ):
        match = VECTOR_FILE_PATTERN.match(path.name)

        if match is None:
            continue

        discovered.append((int(match.group(1)), path))

    discovered.sort(key=lambda item: item[0])

    if not discovered:
        raise FileNotFoundError(
            f"No score-delta vector NPZ files found in {vector_dir}."
        )

    seeds = [seed for seed, _ in discovered]

    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate vector files for one score seed.")

    return discovered


def load_vector_bundle(vector_dir: Path) -> VectorBundle:
    discovered = discover_vector_paths(vector_dir)
    source_paths = []
    score_seeds = []
    baseline_margins = []
    deltas_by_seed = []
    reference_candidate_ids = None
    reference_layer_names = None
    reference_action_names = None

    for score_seed, source_path in discovered:
        with np.load(
            source_path,
            allow_pickle=False,
        ) as archive:
            required_arrays = {
                "candidate_ids",
                "layer_names",
                "action_names",
                "baseline_margin",
                "delta_scores",
            }
            missing = required_arrays - set(archive.files)

            if missing:
                raise ValueError(
                    f"{source_path} is missing arrays: {sorted(missing)}"
                )

            candidate_ids = archive["candidate_ids"].astype(str)
            layer_names = archive["layer_names"].astype(str)
            action_names = archive["action_names"].astype(str)
            margin = archive["baseline_margin"].astype(
                np.float64
            )
            delta_scores = archive["delta_scores"].astype(
                np.float64
            )

        if candidate_ids.ndim != 1:
            raise ValueError(
                f"{source_path} candidate_ids must be 1-D."
            )

        if not (
            candidate_ids.shape
            == layer_names.shape
            == action_names.shape
        ):
            raise ValueError(
                f"{source_path} candidate metadata shapes differ."
            )

        if delta_scores.shape != (
            len(candidate_ids),
            len(margin),
        ):
            raise ValueError(
                f"{source_path} delta_scores shape is inconsistent."
            )

        if (
            not np.isfinite(margin).all()
            or not np.isfinite(delta_scores).all()
        ):
            raise ValueError(
                f"{source_path} contains non-finite vector values."
            )

        if (margin < 0.0).any():
            raise ValueError(
                f"{source_path} contains negative margins."
            )

        reconstructed_ids = np.asarray(
            [
                f"{layer_name}|{action_name}"
                for layer_name, action_name in zip(
                    layer_names,
                    action_names,
                )
            ]
        )

        if not np.array_equal(candidate_ids, reconstructed_ids):
            raise ValueError(
                f"{source_path} candidate IDs do not match pairs."
            )

        for action_name in action_names:
            if action_name not in QUANTIZED_ACTIONS:
                raise ValueError(
                    f"Invalid vector action: {action_name}"
                )

        if len(candidate_ids) != len(set(candidate_ids.tolist())):
            raise ValueError(
                f"{source_path} contains duplicate candidate IDs."
            )

        if reference_candidate_ids is None:
            reference_candidate_ids = candidate_ids
            reference_layer_names = layer_names
            reference_action_names = action_names
        elif not (
            np.array_equal(candidate_ids, reference_candidate_ids)
            and np.array_equal(layer_names, reference_layer_names)
            and np.array_equal(action_names, reference_action_names)
        ):
            raise ValueError(
                "Vector NPZ candidate IDs or ordering differ by seed."
            )

        source_paths.append(source_path.resolve())
        score_seeds.append(score_seed)
        baseline_margins.append(margin)
        deltas_by_seed.append(delta_scores)

    margin_shapes = {
        margin.shape for margin in baseline_margins
    }

    if len(margin_shapes) != 1:
        raise ValueError(
            "Score seeds have different sample counts."
        )

    layer_to_actions = {}

    for layer_name, action_name in zip(
        reference_layer_names,
        reference_action_names,
    ):
        layer_to_actions.setdefault(layer_name, set()).add(
            action_name
        )

    expected_quantized_actions = set(QUANTIZED_ACTIONS)

    for layer_name, actions in layer_to_actions.items():
        if actions != expected_quantized_actions:
            raise ValueError(
                f"Layer {layer_name} has vector actions {sorted(actions)}; "
                f"expected {sorted(expected_quantized_actions)}."
            )

    stacked_deltas = np.stack(deltas_by_seed, axis=0)
    delta_by_candidate = {
        candidate_id: stacked_deltas[:, candidate_index, :]
        for candidate_index, candidate_id in enumerate(
            reference_candidate_ids.tolist()
        )
    }

    return VectorBundle(
        score_seeds=tuple(score_seeds),
        source_paths=tuple(source_paths),
        candidate_ids=tuple(reference_candidate_ids.tolist()),
        layer_names=tuple(reference_layer_names.tolist()),
        action_names=tuple(reference_action_names.tolist()),
        baseline_margins=np.stack(
            baseline_margins,
            axis=0,
        ),
        delta_by_candidate=delta_by_candidate,
    )


def ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def load_memory_accounting(
    benchmark_dir: Path,
    vectors: VectorBundle,
) -> MemoryAccounting:
    metadata_columns = [
        "layer_name",
        "action",
        "bits",
        "layer_numel",
        "whole_model_saving",
    ]
    metadata_frames = []
    metadata_paths = []

    for score_seed in vectors.score_seeds:
        metadata_path = (
            benchmark_dir
            / f"single_action_metrics_seed{score_seed}.csv"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        frame = pd.read_csv(
            metadata_path,
            usecols=metadata_columns,
        )
        metadata_candidate_ids = (
            frame["layer_name"].astype(str)
            + "|"
            + frame["action"].astype(str)
        ).tolist()

        if metadata_candidate_ids != list(vectors.candidate_ids):
            raise ValueError(
                f"{metadata_path} ordering differs from vector NPZ files."
            )

        metadata_frames.append(frame)
        metadata_paths.append(metadata_path.resolve())

    reference = metadata_frames[0]

    for other in metadata_frames[1:]:
        if not reference.equals(other):
            raise ValueError(
                "Layer memory metadata differs between score seeds."
            )

    total_parameter_estimates = []

    for row in reference.itertuples(index=False):
        action_name = str(row.action)
        bits = int(row.bits)

        if bits != action_to_bits(action_name):
            raise ValueError(
                f"Bit metadata mismatch for {row.layer_name}|{action_name}."
            )

        saving_ratio = float(row.whole_model_saving)

        if saving_ratio <= 0.0:
            raise ValueError(
                "whole_model_saving must be positive for quantized actions."
            )

        total_parameter_estimates.append(
            int(
                round(
                    int(row.layer_numel)
                    * (32 - bits)
                    / (32 * saving_ratio)
                )
            )
        )

    if len(set(total_parameter_estimates)) != 1:
        raise ValueError(
            "Benchmark metadata does not reconstruct one total "
            "parameter count."
        )

    total_parameter_numel = total_parameter_estimates[0]
    layer_order = ordered_unique(vectors.layer_names)
    layers = {}

    for layer_name in layer_order:
        selected = reference[
            reference["layer_name"].eq(layer_name)
        ]
        numel_values = selected["layer_numel"].unique()

        if len(numel_values) != 1:
            raise ValueError(
                f"Inconsistent layer_numel for {layer_name}."
            )

        weight_numel = int(numel_values[0])
        action_bytes = {
            action_name: weight_storage_bytes(
                weight_numel=weight_numel,
                bits=action_to_bits(action_name),
            )
            for action_name in ACTIONS
        }
        layers[layer_name] = LayerMemory(
            weight_numel=weight_numel,
            action_bytes=action_bytes,
        )

    quantizable_weight_numel = sum(
        layer.weight_numel for layer in layers.values()
    )
    constant_parameter_numel = (
        total_parameter_numel - quantizable_weight_numel
    )

    if constant_parameter_numel < 0:
        raise ValueError(
            "Quantizable weights exceed reconstructed model parameters."
        )

    constant_parameter_bytes = (
        constant_parameter_numel * 4
    )
    fp32_total_memory_bytes = (
        constant_parameter_bytes
        + sum(
            layer.action_bytes["fp32"]
            for layer in layers.values()
        )
    )

    if fp32_total_memory_bytes != total_parameter_numel * 4:
        raise RuntimeError(
            "FP32 memory reconstruction failed exact accounting."
        )

    return MemoryAccounting(
        layer_order=layer_order,
        layers=layers,
        constant_parameter_bytes=constant_parameter_bytes,
        fp32_total_memory_bytes=fp32_total_memory_bytes,
        metadata_paths=tuple(metadata_paths),
    )


def objective_details(
    summed_deltas: np.ndarray,
    baseline_margins: np.ndarray,
    objective_name: str,
    score_seeds: tuple[int, ...],
) -> tuple[dict[int, float], float, float]:
    normalized_risk = (
        np.abs(summed_deltas)
        / (baseline_margins + EPSILON)
    )

    if objective_name == "vector_signed_mean_risk":
        per_seed_values = normalized_risk.mean(axis=1)
    elif objective_name == "vector_signed_p95_risk":
        per_seed_values = np.quantile(
            normalized_risk,
            0.95,
            axis=1,
        )
    else:
        raise ValueError(
            f"Unsupported vector objective: {objective_name}"
        )

    objective_per_seed = {
        score_seed: float(value)
        for score_seed, value in zip(
            score_seeds,
            per_seed_values,
        )
    }
    objective_mean = float(per_seed_values.mean())
    objective_std = (
        float(per_seed_values.std(ddof=1))
        if len(per_seed_values) > 1
        else 0.0
    )

    return objective_per_seed, objective_mean, objective_std


def partial_objective(
    summed_deltas: np.ndarray,
    vectors: VectorBundle,
    objective_name: str,
) -> float:
    _, objective_mean, _ = objective_details(
        summed_deltas=summed_deltas,
        baseline_margins=vectors.baseline_margins,
        objective_name=objective_name,
        score_seeds=vectors.score_seeds,
    )

    return objective_mean


def state_sort_key(
    state: BeamState,
    constant_parameter_bytes: int,
) -> tuple[float, int, str]:
    return (
        state.heuristic_objective,
        constant_parameter_bytes + state.weight_bytes,
        ",".join(state.actions),
    )


def retain_diverse_states(
    states: list[BeamState],
    beam_width: int,
    max_states_per_memory_bin: int,
    memory_quantum_bytes: int,
    constant_parameter_bytes: int,
) -> list[BeamState]:
    states_by_bin = {}

    for state in states:
        total_partial_bytes = (
            constant_parameter_bytes + state.weight_bytes
        )
        memory_bin = total_partial_bytes // memory_quantum_bytes
        states_by_bin.setdefault(memory_bin, []).append(state)

    retained_by_bin = {}

    for memory_bin, bin_states in states_by_bin.items():
        retained_by_bin[memory_bin] = sorted(
            bin_states,
            key=lambda state: state_sort_key(
                state,
                constant_parameter_bytes,
            ),
        )[:max_states_per_memory_bin]

    selected = []

    for within_bin_rank in range(
        max_states_per_memory_bin
    ):
        rank_candidates = [
            bin_states[within_bin_rank]
            for bin_states in retained_by_bin.values()
            if len(bin_states) > within_bin_rank
        ]
        rank_candidates.sort(
            key=lambda state: state_sort_key(
                state,
                constant_parameter_bytes,
            )
        )
        remaining_capacity = beam_width - len(selected)

        if remaining_capacity <= 0:
            break

        selected.extend(
            rank_candidates[:remaining_capacity]
        )

    return selected


def action_delta(
    vectors: VectorBundle,
    layer_name: str,
    action_name: str,
) -> np.ndarray | None:
    if action_name == "fp32":
        return None

    candidate_id = f"{layer_name}|{action_name}"

    if candidate_id not in vectors.delta_by_candidate:
        raise ValueError(
            f"Missing vector candidate: {candidate_id}"
        )

    return vectors.delta_by_candidate[candidate_id]


def pass_result_sort_key(
    result: PassResult,
) -> tuple[float, int, str]:
    return (
        result.objective_mean,
        result.actual_total_memory_bytes,
        result.action_string,
    )


def run_search_pass(
    *,
    search_pass: str,
    selected_layer_order: tuple[str, ...],
    original_layer_order: tuple[str, ...],
    vectors: VectorBundle,
    memory: MemoryAccounting,
    objective_name: str,
    target_total_memory_bytes: int,
    beam_width: int,
    max_states_per_memory_bin: int,
    memory_quantum_bytes: int,
) -> PassResult:
    target_weight_bytes = (
        target_total_memory_bytes
        - memory.constant_parameter_bytes
    )

    if target_weight_bytes < 0:
        raise ValueError(
            "Target memory is below constant parameter storage."
        )

    minimum_bytes_by_layer = [
        min(memory.layers[layer_name].action_bytes.values())
        for layer_name in selected_layer_order
    ]
    suffix_minimum_bytes = [0] * (
        len(selected_layer_order) + 1
    )

    for index in range(
        len(selected_layer_order) - 1,
        -1,
        -1,
    ):
        suffix_minimum_bytes[index] = (
            suffix_minimum_bytes[index + 1]
            + minimum_bytes_by_layer[index]
        )

    if suffix_minimum_bytes[0] > target_weight_bytes:
        raise ValueError(
            "Requested target is infeasible even with all INT4 actions."
        )

    zero_deltas = np.zeros_like(
        vectors.baseline_margins,
        dtype=np.float64,
    )
    beam = [
        BeamState(
            actions=(),
            weight_bytes=0,
            summed_deltas=zero_deltas,
            heuristic_objective=0.0,
        )
    ]

    for layer_index, layer_name in enumerate(
        selected_layer_order
    ):
        expanded_states = []
        minimum_remaining_bytes = suffix_minimum_bytes[
            layer_index + 1
        ]

        for state in beam:
            for action_name in ACTIONS:
                new_weight_bytes = (
                    state.weight_bytes
                    + memory.layers[
                        layer_name
                    ].action_bytes[action_name]
                )

                if (
                    new_weight_bytes
                    + minimum_remaining_bytes
                    > target_weight_bytes
                ):
                    continue

                delta = action_delta(
                    vectors,
                    layer_name,
                    action_name,
                )
                new_summed_deltas = (
                    state.summed_deltas
                    if delta is None
                    else state.summed_deltas + delta
                )
                heuristic_objective = partial_objective(
                    summed_deltas=new_summed_deltas,
                    vectors=vectors,
                    objective_name=objective_name,
                )
                expanded_states.append(
                    BeamState(
                        actions=(
                            *state.actions,
                            action_name,
                        ),
                        weight_bytes=new_weight_bytes,
                        summed_deltas=new_summed_deltas,
                        heuristic_objective=(
                            heuristic_objective
                        ),
                    )
                )

        if not expanded_states:
            raise RuntimeError(
                f"Beam became empty after layer {layer_name}."
            )

        beam = retain_diverse_states(
            states=expanded_states,
            beam_width=beam_width,
            max_states_per_memory_bin=(
                max_states_per_memory_bin
            ),
            memory_quantum_bytes=memory_quantum_bytes,
            constant_parameter_bytes=(
                memory.constant_parameter_bytes
            ),
        )

    complete_results = []

    for state in beam:
        if len(state.actions) != len(selected_layer_order):
            raise RuntimeError(
                "Complete beam state has missing layer actions."
            )

        layer_actions = dict(
            zip(selected_layer_order, state.actions)
        )
        action_string = ",".join(
            layer_actions[layer_name]
            for layer_name in original_layer_order
        )
        objective_per_seed, objective_mean, objective_std = (
            objective_details(
                summed_deltas=state.summed_deltas,
                baseline_margins=(
                    vectors.baseline_margins
                ),
                objective_name=objective_name,
                score_seeds=vectors.score_seeds,
            )
        )
        complete_results.append(
            PassResult(
                search_pass=search_pass,
                selected_layer_order=selected_layer_order,
                layer_actions=layer_actions,
                actual_total_memory_bytes=(
                    memory.constant_parameter_bytes
                    + state.weight_bytes
                ),
                objective_per_seed=objective_per_seed,
                objective_mean=objective_mean,
                objective_std=objective_std,
                action_string=action_string,
            )
        )

    return min(
        complete_results,
        key=pass_result_sort_key,
    )


def search_orders(
    memory: MemoryAccounting,
    include_original_order: bool,
) -> list[tuple[str, tuple[str, ...]]]:
    original_index = {
        layer_name: index
        for index, layer_name in enumerate(memory.layer_order)
    }
    descending = tuple(
        sorted(
            memory.layer_order,
            key=lambda layer_name: (
                -memory.layers[layer_name].weight_numel,
                original_index[layer_name],
            ),
        )
    )
    ascending = tuple(
        sorted(
            memory.layer_order,
            key=lambda layer_name: (
                memory.layers[layer_name].weight_numel,
                original_index[layer_name],
            ),
        )
    )
    orders = [
        ("descending_parameter_count", descending),
        ("ascending_parameter_count", ascending),
    ]

    if include_original_order:
        orders.append(
            ("original_model_module_order", memory.layer_order)
        )

    return orders


def selected_summed_deltas(
    layer_actions: dict[str, str],
    vectors: VectorBundle,
) -> np.ndarray:
    summed = np.zeros_like(
        vectors.baseline_margins,
        dtype=np.float64,
    )

    for layer_name, action_name in layer_actions.items():
        delta = action_delta(
            vectors,
            layer_name,
            action_name,
        )

        if delta is not None:
            summed = summed + delta

    return summed


def validate_complete_plan(
    *,
    result: PassResult,
    objective_name: str,
    target_total_memory_bytes: int,
    vectors: VectorBundle,
    memory: MemoryAccounting,
) -> None:
    if set(result.layer_actions) != set(memory.layer_order):
        raise RuntimeError(
            "Plan does not contain exactly one entry for every layer."
        )

    if len(result.layer_actions) != len(memory.layer_order):
        raise RuntimeError("Plan layer actions are not unique.")

    for action_name in result.layer_actions.values():
        if action_name not in ACTIONS:
            raise RuntimeError(
                f"Plan contains invalid action: {action_name}"
            )

    recomputed_memory = (
        memory.constant_parameter_bytes
        + sum(
            memory.layers[layer_name].action_bytes[
                action_name
            ]
            for layer_name, action_name in (
                result.layer_actions.items()
            )
        )
    )

    if recomputed_memory != result.actual_total_memory_bytes:
        raise RuntimeError(
            "Exact memory accounting does not match beam state."
        )

    if recomputed_memory > target_total_memory_bytes:
        raise RuntimeError(
            "Selected plan exceeds target total memory."
        )

    summed_deltas = selected_summed_deltas(
        result.layer_actions,
        vectors,
    )
    per_seed, objective_mean, objective_std = objective_details(
        summed_deltas=summed_deltas,
        baseline_margins=vectors.baseline_margins,
        objective_name=objective_name,
        score_seeds=vectors.score_seeds,
    )

    if not math.isclose(
        objective_mean,
        result.objective_mean,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Recomputed vector objective does not match selected state."
        )

    if not math.isclose(
        objective_std,
        result.objective_std,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Recomputed objective std does not match selected state."
        )

    for score_seed in vectors.score_seeds:
        if not math.isclose(
            per_seed[score_seed],
            result.objective_per_seed[score_seed],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Recomputed per-seed objective does not match."
            )


def action_counts(
    layer_actions: dict[str, str],
) -> dict[str, int]:
    return {
        action_name: sum(
            selected_action == action_name
            for selected_action in layer_actions.values()
        )
        for action_name in ACTIONS
    }


def plan_payload(
    *,
    plan_id: str,
    objective_name: str,
    result: PassResult,
    requested_memory_saving_ratio: float,
    target_total_memory_bytes: int,
    memory: MemoryAccounting,
    vectors: VectorBundle,
    search_passes_run: list[str],
    beam_width: int,
    max_states_per_memory_bin: int,
    memory_quantum_bytes: int,
) -> dict:
    counts = action_counts(result.layer_actions)
    actual_memory_saving_ratio = (
        1.0
        - result.actual_total_memory_bytes
        / memory.fp32_total_memory_bytes
    )

    return {
        "plan_id": plan_id,
        "plan_kind": "vector_beam",
        "vector_objective": objective_name,
        "objective_per_seed": {
            str(seed): value
            for seed, value in (
                result.objective_per_seed.items()
            )
        },
        "objective_mean": result.objective_mean,
        "objective_std": result.objective_std,
        "requested_memory_saving_ratio": (
            requested_memory_saving_ratio
        ),
        "actual_memory_saving_ratio": (
            actual_memory_saving_ratio
        ),
        "fp32_total_memory_bytes": (
            memory.fp32_total_memory_bytes
        ),
        "actual_total_memory_bytes": (
            result.actual_total_memory_bytes
        ),
        "target_total_memory_bytes": (
            target_total_memory_bytes
        ),
        "constant_parameter_bytes": (
            memory.constant_parameter_bytes
        ),
        "layer_actions": {
            layer_name: result.layer_actions[layer_name]
            for layer_name in memory.layer_order
        },
        "per_layer_action_bytes": {
            layer_name: memory.layers[
                layer_name
            ].action_bytes[result.layer_actions[layer_name]]
            for layer_name in memory.layer_order
        },
        "action_counts": counts,
        "vector_source_npz_paths": [
            str(path) for path in vectors.source_paths
        ],
        "benchmark_memory_metadata_csvs": [
            str(path) for path in memory.metadata_paths
        ],
        "score_seeds": list(vectors.score_seeds),
        "beam_width": beam_width,
        "max_states_per_memory_bin": (
            max_states_per_memory_bin
        ),
        "memory_quantum_bytes": memory_quantum_bytes,
        "selected_layer_order": list(
            result.selected_layer_order
        ),
        "selected_search_pass": result.search_pass,
        "search_passes_run": search_passes_run,
        "solver_name": SOLVER_NAME,
        "heuristic_search": True,
        "globally_optimal": False,
        "optimality_guarantee": "none",
        "interaction_updates": False,
        "rank_normalization": False,
        "input_scope": (
            "score_delta_vector_npz_and_benchmark_memory_metadata_only"
        ),
        "confirmation_split_read": False,
        "development_oracle_metrics_read": False,
        "test_data_read": False,
        "tie_breaking": (
            "objective, total_memory_bytes, lexicographic action string"
        ),
    }


def summary_row(
    payload: dict,
) -> dict:
    counts = payload["action_counts"]

    return {
        "plan_id": payload["plan_id"],
        "vector_objective": payload["vector_objective"],
        "requested_memory_saving_ratio": payload[
            "requested_memory_saving_ratio"
        ],
        "actual_memory_saving_ratio": payload[
            "actual_memory_saving_ratio"
        ],
        "objective_mean": payload["objective_mean"],
        "objective_std": payload["objective_std"],
        "objective_per_seed": json.dumps(
            payload["objective_per_seed"],
            sort_keys=True,
        ),
        "num_fp32": counts["fp32"],
        "num_fp16": counts["fp16"],
        "num_int8": counts["int8"],
        "num_int4": counts["int4"],
        "selected_search_pass": payload[
            "selected_search_pass"
        ],
        "beam_width": payload["beam_width"],
        "max_states_per_memory_bin": payload[
            "max_states_per_memory_bin"
        ],
    }


def save_outputs(
    output_dir: Path,
    output_paths: dict[str, Path],
    payloads: dict[str, dict],
    summary_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for plan_id, payload in payloads.items():
        with output_paths[plan_id].open(
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

    with output_paths["summary"].open(
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


def print_budget_tables(
    summary_rows: list[dict],
    memory_saving_ratios: tuple[float, ...],
) -> None:
    summary = pd.DataFrame(summary_rows)

    for ratio in memory_saving_ratios:
        selected = summary[
            summary["requested_memory_saving_ratio"].eq(
                ratio
            )
        ].sort_values(
            ["objective_mean", "actual_memory_saving_ratio"],
            ascending=[True, False],
            kind="stable",
        )
        columns = [
            "vector_objective",
            "objective_mean",
            "actual_memory_saving_ratio",
            "num_fp32",
            "num_fp16",
            "num_int8",
            "num_int4",
            "selected_search_pass",
        ]
        print(
            "\nRequested memory saving: "
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
    vectors = load_vector_bundle(args.vector_dir)
    memory = load_memory_accounting(
        benchmark_dir=args.benchmark_dir,
        vectors=vectors,
    )
    orders = search_orders(
        memory=memory,
        include_original_order=args.include_original_order,
    )
    search_pass_names = [name for name, _ in orders]
    memory_quantum_bytes = args.memory_quantum_kb * 1024
    payloads = {}
    summary_rows = []

    for memory_saving_ratio in memory_saving_ratios:
        target_total_memory_bytes = int(
            memory.fp32_total_memory_bytes
            * (1.0 - memory_saving_ratio)
        )

        for objective_name in VECTOR_OBJECTIVES:
            pass_results = [
                run_search_pass(
                    search_pass=search_pass,
                    selected_layer_order=layer_order,
                    original_layer_order=memory.layer_order,
                    vectors=vectors,
                    memory=memory,
                    objective_name=objective_name,
                    target_total_memory_bytes=(
                        target_total_memory_bytes
                    ),
                    beam_width=args.beam_width,
                    max_states_per_memory_bin=(
                        args.max_states_per_memory_bin
                    ),
                    memory_quantum_bytes=(
                        memory_quantum_bytes
                    ),
                )
                for search_pass, layer_order in orders
            ]
            selected_result = min(
                pass_results,
                key=pass_result_sort_key,
            )
            validate_complete_plan(
                result=selected_result,
                objective_name=objective_name,
                target_total_memory_bytes=(
                    target_total_memory_bytes
                ),
                vectors=vectors,
                memory=memory,
            )
            label = OBJECTIVE_FILE_LABELS[objective_name]
            plan_id = (
                f"{label}_{budget_key(memory_saving_ratio)}"
            )
            payload = plan_payload(
                plan_id=plan_id,
                objective_name=objective_name,
                result=selected_result,
                requested_memory_saving_ratio=(
                    memory_saving_ratio
                ),
                target_total_memory_bytes=(
                    target_total_memory_bytes
                ),
                memory=memory,
                vectors=vectors,
                search_passes_run=search_pass_names,
                beam_width=args.beam_width,
                max_states_per_memory_bin=(
                    args.max_states_per_memory_bin
                ),
                memory_quantum_bytes=memory_quantum_bytes,
            )
            payloads[plan_id] = payload
            summary_rows.append(summary_row(payload))

    save_outputs(
        output_dir=args.output_dir,
        output_paths=output_paths,
        payloads=payloads,
        summary_rows=summary_rows,
    )
    print_budget_tables(
        summary_rows=summary_rows,
        memory_saving_ratios=memory_saving_ratios,
    )
    print(f"\nSaved vector beam plans: {args.output_dir}")
    print(
        "Search status: deterministic heuristic beam search; "
        "global optimality is not guaranteed."
    )
    print(
        "Input policy confirmed: no confirmation split, development "
        "oracle metrics, test data, or interaction updates were read."
    )


if __name__ == "__main__":
    main()
