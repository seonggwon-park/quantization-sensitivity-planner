import argparse
import csv
from pathlib import Path

from config import ExperimentConfig
from data import build_dataloaders, make_fixed_subset_loader
from metrics import compare_binary_models
from model import load_binary_resnet18_checkpoint
from quantization import (
    build_uniform_quantized_model,
    estimate_parameter_memory_mb,
    list_quantizable_layers,
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

    quantizable_layers = list_quantizable_layers(
        fp32_model
    )

    fp32_memory_mb = estimate_parameter_memory_mb(
        fp32_model,
        layer_bits={},
    )

    action_to_bits = {
        "fp32": 32,
        "fp16": 16,
        "int8": 8,
        "int4": 4,
    }

    rows = []

    for action, bits in action_to_bits.items():
        print(f"Running baseline: {action}")

        if bits == 32:
            candidate_model = fp32_model
        else:
            candidate_model = build_uniform_quantized_model(
                fp32_model=fp32_model,
                bits=bits,
                device=device,
            )

        metrics = compare_binary_models(
            fp32_model=fp32_model,
            quantized_model=candidate_model,
            dataloader=analysis_loader,
            device=device,
        )

        layer_bits = {
            layer_name: bits
            for layer_name in quantizable_layers
        }

        memory_mb = estimate_parameter_memory_mb(
            fp32_model,
            layer_bits=layer_bits,
        )

        row = {
            "action": action,
            "bits": bits,
            "estimated_parameter_memory_mb": memory_mb,
            "memory_saving_mb": fp32_memory_mb - memory_mb,
            "memory_saving_ratio": (
                1.0 - memory_mb / fp32_memory_mb
            ),
            **metrics,
        }

        rows.append(row)

        print(row)

    output_path = (
        config.result_dir / "uniform_baselines.csv"
    )

    save_csv(rows, output_path)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()