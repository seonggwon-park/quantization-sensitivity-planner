import argparse
import csv
from pathlib import Path

import torch

from config import ExperimentConfig
from data import build_dataloaders, make_fixed_subset_loader
from experiment_logger import record_experiment
from metrics import compare_binary_models
from model import load_binary_resnet18_checkpoint
from quantization import (
    build_mixed_quantized_model,
    estimate_parameter_memory_mb,
    list_quantizable_layers,
)
from risk_configurations import (
    build_ranked_protection_configurations,
    load_layer_ranking,
)
from utils import get_device, set_seed


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ExperimentConfig().checkpoint_path,
    )

    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=(
            ExperimentConfig().result_dir
            / "validation_single_layer_sweep.csv"
        ),
    )

    parser.add_argument(
        "--ranking-action",
        type=str,
        default="int4",
    )

    parser.add_argument(
        "--risk-metric",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ranking-label",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--evaluation-split",
        choices=["validation", "test"],
        default="test",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--protect-counts",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4],
    )

    parser.add_argument(
        "--default-bits",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--protected-bits",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--include-bottom-controls",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
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

    with open(
        output_path,
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


def count_weight_parameters(
    model,
    layer_names: list[str],
) -> int:
    total = 0

    for layer_name in layer_names:
        module = model.get_submodule(layer_name)
        total += module.weight.numel()

    return total


def normalize_label(label: str) -> str:
    normalized = (
        label.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if not normalized:
        raise ValueError(
            "ranking_label must not be empty."
        )

    return normalized


def main():
    args = parse_arguments()

    config = ExperimentConfig()
    config.create_directories()

    ranking_label = normalize_label(
        args.ranking_label
    )

    set_seed(config.seed)
    device = get_device()

    dataloaders = build_dataloaders(config)

    evaluation_loader = make_fixed_subset_loader(
        original_loader=dataloaders[
            args.evaluation_split
        ],
        max_samples=args.max_samples,
        seed=config.seed,
    )

    fp32_model, _ = load_binary_resnet18_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    layer_names = list_quantizable_layers(
        fp32_model
    )

    ranked_layers = load_layer_ranking(
        ranking_csv=args.ranking_csv,
        action=args.ranking_action,
        risk_metric=args.risk_metric,
    )

    configurations = (
        build_ranked_protection_configurations(
            layer_names=layer_names,
            ranked_layers=ranked_layers,
            protect_counts=args.protect_counts,
            ranking_label=ranking_label,
            default_bits=args.default_bits,
            protected_bits=args.protected_bits,
            include_bottom_controls=(
                args.include_bottom_controls
            ),
        )
    )

    if args.output is None:
        output_path = (
            config.result_dir
            / f"{ranking_label}_mixed_controls.csv"
        )
    else:
        output_path = args.output

    fp32_memory_mb = estimate_parameter_memory_mb(
        model=fp32_model,
        layer_bits={},
        default_bits=32,
    )

    total_quantizable_weight_parameters = (
        count_weight_parameters(
            model=fp32_model,
            layer_names=layer_names,
        )
    )

    rows = []

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):
        config_name = configuration["name"]
        protected_layers = configuration[
            "protected_layers"
        ]
        layer_bits = configuration["layer_bits"]

        print(
            f"\n[{index}/{len(configurations)}] "
            f"{config_name}"
        )

        quantized_model = build_mixed_quantized_model(
            fp32_model=fp32_model,
            layer_bits=layer_bits,
            default_bits=args.default_bits,
            device=device,
        )

        metrics = compare_binary_models(
            fp32_model=fp32_model,
            quantized_model=quantized_model,
            dataloader=evaluation_loader,
            device=device,
        )

        estimated_memory_mb = (
            estimate_parameter_memory_mb(
                model=fp32_model,
                layer_bits=layer_bits,
                default_bits=args.default_bits,
            )
        )

        protected_weight_parameters = (
            count_weight_parameters(
                model=fp32_model,
                layer_names=protected_layers,
            )
        )

        row = {
            "config_name": config_name,
            "selection_policy": configuration[
                "selection_policy"
            ],
            "ranking_label": ranking_label,
            "ranking_csv": str(args.ranking_csv),
            "ranking_action": args.ranking_action,
            "risk_metric": args.risk_metric,
            "evaluation_split": args.evaluation_split,

            "default_bits": args.default_bits,
            "protected_bits": args.protected_bits,

            "protected_layer_count": len(
                protected_layers
            ),
            "protected_layers": ",".join(
                protected_layers
            ),

            "protected_weight_parameters": (
                protected_weight_parameters
            ),
            "protected_weight_parameter_ratio": (
                protected_weight_parameters
                / total_quantizable_weight_parameters
            ),

            "fp32_parameter_memory_mb": fp32_memory_mb,
            "estimated_parameter_memory_mb": (
                estimated_memory_mb
            ),
            "memory_saving_mb": (
                fp32_memory_mb - estimated_memory_mb
            ),
            "memory_saving_ratio": (
                1.0
                - estimated_memory_mb / fp32_memory_mb
            ),

            **metrics,
        }

        rows.append(row)
        save_csv(rows, output_path)

        print(row)

        del quantized_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_metrics = {}

    for row in rows:
        prefix = row["config_name"]

        summary_metrics[
            f"{prefix}_flip_rate"
        ] = row["flip_rate"]

        summary_metrics[
            f"{prefix}_quantized_accuracy"
        ] = row["quantized_accuracy"]

        summary_metrics[
            f"{prefix}_memory_saving_ratio"
        ] = row["memory_saving_ratio"]

        summary_metrics[
            f"{prefix}_mean_margin_risk"
        ] = row["mean_margin_risk"]

    record_experiment(
        run_name=(
            f"{ranking_label}_ranked_"
            "mixed_precision_controls"
        ),
        config={
            "task": config.class_names,
            "model": "binary ResNet-18",
            "ranking_label": ranking_label,
            "ranking_csv": str(args.ranking_csv),
            "ranking_action": args.ranking_action,
            "risk_metric": args.risk_metric,
            "evaluation_split": args.evaluation_split,
            "max_samples": args.max_samples,
            "protect_counts": args.protect_counts,
            "default_bits": args.default_bits,
            "protected_bits": args.protected_bits,
            "include_bottom_controls": (
                args.include_bottom_controls
            ),
            "quantization": (
                "mixed weight-only fake quantization, "
                "per-output-channel symmetric"
            ),
        },
        metrics=summary_metrics,
        artifacts={
            "csv": str(output_path),
        },
    )

    print(f"\nSaved controls: {output_path}")


if __name__ == "__main__":
    main()