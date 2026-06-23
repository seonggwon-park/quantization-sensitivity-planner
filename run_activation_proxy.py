import argparse
import csv
import time
from pathlib import Path

from activation_proxy import collect_local_activation_proxy
from config import ExperimentConfig
from data import build_dataloaders, make_fixed_subset_loader
from experiment_logger import record_experiment
from model import load_binary_resnet18_checkpoint
from quantization import list_quantizable_layers
from utils import get_device, set_seed


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ExperimentConfig().checkpoint_path,
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[4],
    )

    parser.add_argument(
        "--layers",
        type=str,
        nargs="*",
        default=None,
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
    if not rows:
        raise RuntimeError(
            "No activation proxy rows to save."
        )

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


def resolve_output_path(
    args,
    config: ExperimentConfig,
) -> Path:
    if args.output is not None:
        return args.output

    return (
        config.result_dir
        / f"{args.split}_local_activation_proxy.csv"
    )


def main():
    args = parse_arguments()

    config = ExperimentConfig()
    config.create_directories()

    set_seed(config.seed)
    device = get_device()

    dataloaders = build_dataloaders(config)

    calibration_loader = make_fixed_subset_loader(
        original_loader=dataloaders[args.split],
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

    output_path = resolve_output_path(
        args=args,
        config=config,
    )

    print(
        "\nCollecting local activation reconstruction "
        "proxy..."
    )

    print(
        f"Split: {args.split} | "
        f"Samples: {args.max_samples} | "
        f"Layers: {len(target_layer_names)} | "
        f"Bits: {args.bits}"
    )

    start_time = time.perf_counter()

    rows = collect_local_activation_proxy(
        model=fp32_model,
        dataloader=calibration_loader,
        layer_names=target_layer_names,
        bits=args.bits,
        device=device,
        data_split=args.split,
    )

    runtime_seconds = time.perf_counter() - start_time

    save_csv(
        rows=rows,
        output_path=output_path,
    )

    highest_p95_row = max(
        rows,
        key=lambda row: row[
            "p95_relative_activation_error"
        ],
    )

    highest_mean_row = max(
        rows,
        key=lambda row: row[
            "mean_relative_activation_error"
        ],
    )

    record_experiment(
        run_name="local_activation_proxy",
        config={
            "task": config.class_names,
            "model": "binary ResNet-18",
            "data_split": args.split,
            "max_samples": args.max_samples,
            "requested_bits": args.bits,
            "num_layers": len(target_layer_names),
            "proxy": (
                "forward-only local module-output "
                "reconstruction error"
            ),
            "quantization": (
                "weight-only fake quantization, "
                "per-output-channel symmetric"
            ),
        },
        metrics={
            "num_layer_action_pairs": len(rows),
            "proxy_runtime_seconds": runtime_seconds,
            "highest_p95_layer": highest_p95_row[
                "layer"
            ],
            "highest_p95_action": highest_p95_row[
                "action"
            ],
            "highest_p95_relative_activation_error": (
                highest_p95_row[
                    "p95_relative_activation_error"
                ]
            ),
            "highest_mean_layer": highest_mean_row[
                "layer"
            ],
            "highest_mean_action": highest_mean_row[
                "action"
            ],
            "highest_mean_relative_activation_error": (
                highest_mean_row[
                    "mean_relative_activation_error"
                ]
            ),
        },
        artifacts={
            "csv": str(output_path),
        },
    )

    print(
        f"\nSaved activation proxy CSV: "
        f"{output_path}"
    )

    print(
        f"Runtime: {runtime_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()