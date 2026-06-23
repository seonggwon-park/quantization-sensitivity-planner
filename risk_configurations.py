from pathlib import Path

import pandas as pd


def bits_to_label(bits: int) -> str:
    if bits == 32:
        return "fp32"

    if bits == 16:
        return "fp16"

    return f"int{bits}"


def load_layer_ranking(
    ranking_csv: Path,
    action: str = "int4",
    risk_metric: str = "p95_margin_risk",
) -> list[str]:
    """
    Load a layer ranking from a single-layer sweep CSV.
    """

    dataframe = pd.read_csv(ranking_csv)

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
            f"Ranking CSV is missing columns: "
            f"{missing_columns}"
        )

    selected = dataframe[
        dataframe["action"] == action
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No rows found for action={action}"
        )

    if selected["layer"].duplicated().any():
        raise ValueError(
            "Expected one row per layer for the selected action."
        )

    ranked = selected.sort_values(
        by=risk_metric,
        ascending=False,
        kind="stable",
    )

    return ranked["layer"].tolist()


def make_full_assignment(
    layer_names: list[str],
    default_bits: int,
    protected_layers: list[str],
    protected_bits: int,
) -> dict[str, int]:
    """
    Example:
        default_bits = 4
        protected_layers = ["conv1"]
        protected_bits = 32

    means:
        conv1 -> FP32
        every other quantizable layer -> INT4
    """

    assignment = {
        layer_name: default_bits
        for layer_name in layer_names
    }

    for layer_name in protected_layers:
        if layer_name not in assignment:
            raise ValueError(
                f"Unknown layer in configuration: {layer_name}"
            )

        assignment[layer_name] = protected_bits

    return assignment


def build_oracle_protection_configurations(
    layer_names: list[str],
    ranked_layers: list[str],
    protect_counts: list[int],
    default_bits: int = 4,
    protected_bits: int = 32,
    include_bottom_controls: bool = False,
) -> list[dict]:
    """
    Build manually interpretable mixed-precision controls.

    "oracle" means that the ranking comes from empirical
    single-layer sweep results, not from a cheap proxy metric.
    """

    layer_set = set(layer_names)

    ranked_layers = [
        layer_name
        for layer_name in ranked_layers
        if layer_name in layer_set
    ]

    if len(ranked_layers) != len(layer_names):
        missing_layers = layer_set - set(ranked_layers)

        if missing_layers:
            raise ValueError(
                "Ranking does not contain all quantizable layers: "
                f"{missing_layers}"
            )

    unique_counts = list(
        dict.fromkeys(protect_counts)
    )

    configurations = []

    default_label = bits_to_label(default_bits)
    protected_label = bits_to_label(protected_bits)

    for count in unique_counts:
        if count < 0 or count > len(layer_names):
            raise ValueError(
                f"Invalid protect count: {count}"
            )

        if count == 0:
            configurations.append(
                {
                    "name": f"uniform_{default_label}",
                    "selection_policy": "uniform",
                    "protected_layers": [],
                    "layer_bits": make_full_assignment(
                        layer_names=layer_names,
                        default_bits=default_bits,
                        protected_layers=[],
                        protected_bits=protected_bits,
                    ),
                }
            )

            continue

        top_layers = ranked_layers[:count]

        configurations.append(
            {
                "name": (
                    f"oracle_top{count}_"
                    f"{protected_label}"
                ),
                "selection_policy": "oracle_high_risk",
                "protected_layers": top_layers,
                "layer_bits": make_full_assignment(
                    layer_names=layer_names,
                    default_bits=default_bits,
                    protected_layers=top_layers,
                    protected_bits=protected_bits,
                ),
            }
        )

        if include_bottom_controls:
            bottom_layers = ranked_layers[-count:]

            configurations.append(
                {
                    "name": (
                        f"oracle_bottom{count}_"
                        f"{protected_label}"
                    ),
                    "selection_policy": "oracle_low_risk_control",
                    "protected_layers": bottom_layers,
                    "layer_bits": make_full_assignment(
                        layer_names=layer_names,
                        default_bits=default_bits,
                        protected_layers=bottom_layers,
                        protected_bits=protected_bits,
                    ),
                }
            )

    return configurations