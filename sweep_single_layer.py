import argparse
import csv
from pathlib import Path

import torch

from experiment_logger import record_experiment
from config import ExperimentConfig
from data import build_dataloaders, make_fixed_subset_loader
from metrics import compare_binary_models
from model import load_binary_resnet18_checkpoint
from quantization import (
    build_single_layer_quantized_model,
    estimate_parameter_memory_mb,
    list_quantizable_layers,
    relative_l2_weight_error,
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
        "--max-samples",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[16, 8, 4],
    )

    parser.add_argument(
        "--layers",
        type=str,
        nargs="*",
        default=None,
    )

    return parser.parse_args()


def save_csv(rows, output_path: Path):
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


def bit_to_action_name(bits: int) -> str:
    mapping = {
        16: "fp16",
        8: "int8",
        4: "int4",
    }

    if bits not in mapping:
        raise ValueError(
            f"Unsupported sweep bit-width: {bits}"
        )

    return mapping[bits]


def main():
    args = parse_arguments()

    config = ExperimentConfig()
    config.create_directories()

    set_seed(config.seed)
    device = get_device()

    dataloaders = build_dataloaders(config)

    analysis_loader = make_fixed_subset_loader(
        original_loader=dataloaders["test"],
        max_samples=args.max_samples,
        seed=config.seed,
    )

    fp32_model, _ = load_binary_resnet18_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    all_layer_names = list_quantizable_layers(
        fp32_model
    )

    if args.layers is None or len(args.layers) == 0:
        target_layer_names = all_layer_names
    else:
        target_layer_names = args.layers

        unknown_layers = set(target_layer_names) - set(
            all_layer_names
        )

        if unknown_layers:
            raise ValueError(
                f"Unknown layers: {unknown_layers}\n"
                f"Available layers: {all_layer_names}"
            )

    fp32_memory_mb = estimate_parameter_memory_mb(
        fp32_model,
        layer_bits={},
    )

    rows = []

    for layer_index, layer_name in enumerate(
        target_layer_names
    ):
        for bits in args.bits:
            action = bit_to_action_name(bits)

            print(
                f"\n[{layer_index + 1}/{len(target_layer_names)}] "
                f"layer={layer_name}, action={action}"
            )

            quantized_model = (
                build_single_layer_quantized_model(
                    fp32_model=fp32_model,
                    layer_name=layer_name,
                    bits=bits,
                    device=device,
                )
            )

            metrics = compare_binary_models(
                fp32_model=fp32_model,
                quantized_model=quantized_model,
                dataloader=analysis_loader,
                device=device,
            )

            original_module = fp32_model.get_submodule(
                layer_name
            )

            quantized_module = (
                quantized_model.get_submodule(
                    layer_name
                )
            )

            weight_error = relative_l2_weight_error(
                original_weight=original_module.weight.detach(),
                quantized_weight=quantized_module.weight.detach(),
            )

            layer_bits = {
                layer_name: bits,
            }

            estimated_memory_mb = (
                estimate_parameter_memory_mb(
                    fp32_model,
                    layer_bits=layer_bits,
                )
            )

            row = {
                "layer_index": layer_index,
                "layer": layer_name,
                "action": action,
                "bits": bits,

                "relative_l2_weight_error": weight_error,

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

            print(row)

            output_path = (
                config.result_dir
                / "single_layer_sweep.csv"
            )

            save_csv(rows, output_path)

            del quantized_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_path = (
        config.result_dir / "single_layer_sweep.csv"
    )

    print(f"\nSaved final sweep: {output_path}")
    worst_risk_row = max(
    rows,
    key=lambda row: row["p95_margin_risk"],
)

    worst_flip_row = max(
        rows,
        key=lambda row: row["flip_rate"],
    )

    record_experiment(
        run_name="single_layer_quantization_sweep",
        config={
            "task": config.class_names,
            "model": "binary ResNet-18",
            "max_samples": args.max_samples,
            "requested_bits": args.bits,
            "num_layers": len(target_layer_names),
            "quantization": (
                "single-layer weight-only fake quantization, "
                "per-output-channel symmetric"
            ),
        },
        metrics={
            "num_experiments": len(rows),
            "highest_p95_risk_layer": worst_risk_row["layer"],
            "highest_p95_risk_action": worst_risk_row["action"],
            "highest_p95_margin_risk": worst_risk_row[
                "p95_margin_risk"
            ],
            "highest_flip_rate_layer": worst_flip_row["layer"],
            "highest_flip_rate_action": worst_flip_row["action"],
            "highest_flip_rate": worst_flip_row[
                "flip_rate"
            ],
        },
        artifacts={
            "csv": str(output_path),
        },
    )


if __name__ == "__main__":
    main()