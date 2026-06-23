from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import ExperimentConfig


ACTION_ORDER = ["fp16", "int8", "int4"]


def plot_heatmap(
    dataframe: pd.DataFrame,
    metric_name: str,
    output_path: Path,
):
    """
    Plot layer × action heatmap.
    """

    available_actions = [
        action
        for action in ACTION_ORDER
        if action in dataframe["action"].unique()
    ]

    layer_order = dataframe["layer"].drop_duplicates()

    pivot = dataframe.pivot(
        index="layer",
        columns="action",
        values=metric_name,
    )

    pivot = pivot.reindex(
        index=layer_order,
        columns=available_actions,
    )

    figure_height = max(6, len(pivot.index) * 0.35)

    figure, axis = plt.subplots(
        figsize=(8, figure_height)
    )

    image = axis.imshow(
        pivot.values,
        aspect="auto",
    )

    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns)

    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels(pivot.index)

    axis.set_title(
        f"Layer-wise {metric_name}"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(metric_name)

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


def plot_top_int4_layers(
    dataframe: pd.DataFrame,
    output_path: Path,
):
    """
    Plot top-10 most risky layers under INT4.
    """

    int4_dataframe = dataframe[
        dataframe["action"] == "int4"
    ].copy()

    top_layers = int4_dataframe.nlargest(
        10,
        "p95_margin_risk",
    ).sort_values(
        "p95_margin_risk",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.barh(
        top_layers["layer"],
        top_layers["p95_margin_risk"],
    )

    axis.set_xlabel("P95 Margin-Normalized Risk")
    axis.set_ylabel("Layer")

    axis.set_title(
        "Top-10 Sensitive Layers Under INT4"
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


def main():
    config = ExperimentConfig()

    input_path = (
        config.result_dir / "single_layer_sweep.csv"
    )

    dataframe = pd.read_csv(input_path)

    plot_heatmap(
        dataframe=dataframe,
        metric_name="mean_margin_risk",
        output_path=(
            config.figure_dir
            / "mean_margin_risk_heatmap.png"
        ),
    )

    plot_heatmap(
        dataframe=dataframe,
        metric_name="flip_rate",
        output_path=(
            config.figure_dir
            / "flip_rate_heatmap.png"
        ),
    )

    plot_top_int4_layers(
        dataframe=dataframe,
        output_path=(
            config.figure_dir
            / "top_int4_sensitive_layers.png"
        ),
    )

    print(
        f"Saved figures to: {config.figure_dir}"
    )


if __name__ == "__main__":
    main()