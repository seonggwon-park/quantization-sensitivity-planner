from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn

from quantization import list_quantizable_layers


ACTION_TO_BITS = {
    "fp32": 32,
    "fp16": 16,
    "int8": 8,
    "int4": 4,
}

QUANTIZED_ACTIONS = (
    "fp16",
    "int8",
    "int4",
)


@dataclass(frozen=True)
class LayerActionOption:
    layer: str
    action: str
    bits: int
    risk: float
    weight_numel: int
    exact_weight_bytes: int
    scaled_cost_units: int


@dataclass
class AdditivePlan:
    layer_bits: dict[str, int]
    selected_options: dict[str, LayerActionOption]
    objective_value: float
    target_total_memory_bytes: int
    actual_total_memory_bytes: int
    scaled_total_memory_bytes: int
    constant_parameter_bytes: int
    used_capacity_units: int
    capacity_units: int


def action_to_bits(action: str) -> int:
    if action not in ACTION_TO_BITS:
        raise ValueError(
            f"Unsupported action: {action}"
        )

    return ACTION_TO_BITS[action]


def weight_storage_bytes(
    weight_numel: int,
    bits: int,
) -> int:
    """
    Storage estimate for one weight tensor.

    ceil is used so INT4 tensors with an odd parameter count
    still have a safe byte estimate.
    """

    return math.ceil(
        weight_numel * bits / 8
    )


def constant_parameter_storage_bytes(
    model: nn.Module,
    layer_names: list[str],
) -> int:
    """
    Count all parameters that remain FP32 regardless of action:
    Conv/Linear biases, BatchNorm parameters, etc.
    """

    quantized_weight_names = {
        f"{layer_name}.weight"
        for layer_name in layer_names
    }

    total_bits = 0

    for parameter_name, parameter in model.named_parameters():
        if parameter_name not in quantized_weight_names:
            total_bits += parameter.numel() * 32

    return total_bits // 8


def build_layer_action_options(
    model: nn.Module,
    risk_csv: Path,
    risk_metric: str,
    memory_quantum_bytes: int,
) -> dict[str, list[LayerActionOption]]:
    """
    Construct layer-action options from a validation risk table.

    FP32 is added automatically:
    - risk = 0
    - bits = 32
    """

    if memory_quantum_bytes <= 0:
        raise ValueError(
            "memory_quantum_bytes must be positive."
        )

    dataframe = pd.read_csv(risk_csv)

    required_columns = {
        "layer",
        "action",
        risk_metric,
    }

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Risk CSV is missing columns: "
            f"{missing_columns}"
        )

    layer_names = list_quantizable_layers(model)

    selected = dataframe[
        dataframe["action"].isin(QUANTIZED_ACTIONS)
    ].copy()

    duplicate_rows = selected.duplicated(
        subset=["layer", "action"],
        keep=False,
    )

    if duplicate_rows.any():
        duplicated = selected.loc[
            duplicate_rows,
            ["layer", "action"],
        ]

        raise ValueError(
            "Risk CSV has duplicate layer-action rows:\n"
            f"{duplicated.to_string(index=False)}"
        )

    lookup = selected.set_index(
        ["layer", "action"]
    )[risk_metric].to_dict()

    options_by_layer = {}

    for layer_name in layer_names:
        module = model.get_submodule(layer_name)

        if not isinstance(
            module,
            (nn.Conv2d, nn.Linear),
        ):
            raise TypeError(
                f"{layer_name} is not Conv2d or Linear."
            )

        weight_numel = module.weight.numel()

        layer_options = []

        for action in ("fp32", *QUANTIZED_ACTIONS):
            bits = action_to_bits(action)

            if action == "fp32":
                risk = 0.0
            else:
                key = (layer_name, action)

                if key not in lookup:
                    raise ValueError(
                        f"Missing risk value for {key}."
                    )

                risk = float(lookup[key])

                if not math.isfinite(risk) or risk < 0:
                    raise ValueError(
                        f"Invalid risk for {key}: {risk}"
                    )

            exact_bytes = weight_storage_bytes(
                weight_numel=weight_numel,
                bits=bits,
            )

            scaled_cost_units = math.ceil(
                exact_bytes / memory_quantum_bytes
            )

            layer_options.append(
                LayerActionOption(
                    layer=layer_name,
                    action=action,
                    bits=bits,
                    risk=risk,
                    weight_numel=weight_numel,
                    exact_weight_bytes=exact_bytes,
                    scaled_cost_units=scaled_cost_units,
                )
            )

        options_by_layer[layer_name] = layer_options

    return options_by_layer


def minimum_total_memory_bytes(
    model: nn.Module,
    options_by_layer: dict[
        str,
        list[LayerActionOption],
    ],
) -> int:
    """
    Minimum total parameter memory allowed by the action set.
    Normally this is uniform INT4.
    """

    layer_names = list(options_by_layer)

    constant_bytes = constant_parameter_storage_bytes(
        model=model,
        layer_names=layer_names,
    )

    minimum_weight_bytes = sum(
        min(
            option.exact_weight_bytes
            for option in options_by_layer[layer_name]
        )
        for layer_name in layer_names
    )

    return constant_bytes + minimum_weight_bytes


def solve_additive_plan(
    model: nn.Module,
    options_by_layer: dict[
        str,
        list[LayerActionOption],
    ],
    target_total_memory_bytes: int,
    memory_quantum_bytes: int,
) -> AdditivePlan:
    """
    Solve a multiple-choice knapsack problem.

    The solver is exact under the chosen memory quantum.
    A 1 KB quantum makes the plan slightly conservative:
    actual memory never exceeds the requested budget.
    """

    if memory_quantum_bytes <= 0:
        raise ValueError(
            "memory_quantum_bytes must be positive."
        )

    layer_names = list(options_by_layer)

    constant_bytes = constant_parameter_storage_bytes(
        model=model,
        layer_names=layer_names,
    )

    available_weight_bytes = (
        target_total_memory_bytes - constant_bytes
    )

    if available_weight_bytes <= 0:
        raise ValueError(
            "Memory budget is smaller than constant "
            "FP32 parameter storage."
        )

    minimum_exact_weight_bytes = sum(
        min(
            option.exact_weight_bytes
            for option in options_by_layer[layer_name]
        )
        for layer_name in layer_names
    )

    if available_weight_bytes < minimum_exact_weight_bytes:
        raise ValueError(
            "Requested memory budget is infeasible even "
            "for the lowest-bit allocation."
        )

    capacity_units = (
        available_weight_bytes
        // memory_quantum_bytes
    )

    minimum_scaled_weight_units = sum(
        min(
            option.scaled_cost_units
            for option in options_by_layer[layer_name]
        )
        for layer_name in layer_names
    )

    if minimum_scaled_weight_units > capacity_units:
        raise ValueError(
            "The selected memory quantum is too coarse for "
            "this budget. Use a smaller quantum."
        )

    dp = np.full(
        capacity_units + 1,
        np.inf,
        dtype=np.float64,
    )

    dp[0] = 0.0

    chosen_action_indices = np.full(
        (
            len(layer_names),
            capacity_units + 1,
        ),
        fill_value=-1,
        dtype=np.int16,
    )

    previous_capacity = np.full(
        (
            len(layer_names),
            capacity_units + 1,
        ),
        fill_value=-1,
        dtype=np.int32,
    )

    for layer_index, layer_name in enumerate(
        layer_names
    ):
        layer_options = options_by_layer[layer_name]

        next_dp = np.full(
            capacity_units + 1,
            np.inf,
            dtype=np.float64,
        )

        for action_index, option in enumerate(
            layer_options
        ):
            cost = option.scaled_cost_units

            if cost > capacity_units:
                continue

            candidate_values = (
                dp[: capacity_units + 1 - cost]
                + option.risk
            )

            target_values = next_dp[cost:]

            improved = candidate_values < target_values

            if not improved.any():
                continue

            previous_indices = np.flatnonzero(
                improved
            )

            target_values[improved] = candidate_values[
                improved
            ]

            updated_capacities = (
                cost + previous_indices
            )

            chosen_action_indices[
                layer_index,
                updated_capacities,
            ] = action_index

            previous_capacity[
                layer_index,
                updated_capacities,
            ] = previous_indices

        dp = next_dp

    if not np.isfinite(dp).any():
        raise RuntimeError(
            "No feasible planner solution was found."
        )

    used_capacity_units = int(np.argmin(dp))

    selected_options = {}
    current_capacity = used_capacity_units

    for layer_index in range(
        len(layer_names) - 1,
        -1,
        -1,
    ):
        action_index = chosen_action_indices[
            layer_index,
            current_capacity,
        ]

        if action_index < 0:
            raise RuntimeError(
                "Failed to reconstruct planner solution."
            )

        layer_name = layer_names[layer_index]

        option = options_by_layer[layer_name][
            action_index
        ]

        selected_options[layer_name] = option

        current_capacity = previous_capacity[
            layer_index,
            current_capacity,
        ]

    selected_options = {
        layer_name: selected_options[layer_name]
        for layer_name in layer_names
    }

    layer_bits = {
        layer_name: option.bits
        for layer_name, option in selected_options.items()
    }

    actual_weight_bytes = sum(
        option.exact_weight_bytes
        for option in selected_options.values()
    )

    scaled_weight_bytes = sum(
        option.scaled_cost_units
        * memory_quantum_bytes
        for option in selected_options.values()
    )

    actual_total_memory_bytes = (
        constant_bytes + actual_weight_bytes
    )

    scaled_total_memory_bytes = (
        constant_bytes + scaled_weight_bytes
    )

    if actual_total_memory_bytes > target_total_memory_bytes:
        raise RuntimeError(
            "Planner violated the requested memory budget."
        )

    return AdditivePlan(
        layer_bits=layer_bits,
        selected_options=selected_options,
        objective_value=float(dp[used_capacity_units]),
        target_total_memory_bytes=(
            target_total_memory_bytes
        ),
        actual_total_memory_bytes=(
            actual_total_memory_bytes
        ),
        scaled_total_memory_bytes=(
            scaled_total_memory_bytes
        ),
        constant_parameter_bytes=constant_bytes,
        used_capacity_units=used_capacity_units,
        capacity_units=capacity_units,
    )