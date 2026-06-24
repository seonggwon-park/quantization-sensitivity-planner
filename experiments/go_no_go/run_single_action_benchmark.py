"""Run isolated one-layer, one-action quantization measurements."""

import argparse
import copy
import csv
from pathlib import Path

import torch

from additive_planner import (
    QUANTIZED_ACTIONS,
    action_to_bits,
)
from config import ExperimentConfig
from quantization import (
    estimate_parameter_memory_mb,
    list_quantizable_layers,
    relative_l2_weight_error,
)
from utils import get_device, set_seed

from experiments.go_no_go import DEFAULT_RESULTS_DIR
from experiments.go_no_go.adapters import (
    apply_existing_quantization_inplace,
    build_binary_eval_dataset,
    get_checkpoint_default_path,
    load_reference_model,
)
from experiments.go_no_go.metrics import evaluate_model_pair
from experiments.go_no_go.splits import (
    build_disjoint_eval_loaders,
)


def parse_arguments():
    config = ExperimentConfig()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=get_checkpoint_default_path(),
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=QUANTIZED_ACTIONS,
        default=list(QUANTIZED_ACTIONS),
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--ranking-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.batch_size,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=config.num_workers,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.seed,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_RESULTS_DIR
            / "single_action_benchmark.csv"
        ),
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

    with output_path.open(
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


def resolve_target_layers(
    reference_model,
    requested_layers: list[str] | None,
) -> list[str]:
    all_layers = list_quantizable_layers(
        reference_model
    )

    if not requested_layers:
        return all_layers

    unknown_layers = set(requested_layers) - set(
        all_layers
    )

    if unknown_layers:
        raise ValueError(
            f"Unknown layers: {sorted(unknown_layers)}. "
            f"Available layers: {all_layers}"
        )

    return list(dict.fromkeys(requested_layers))


def main() -> None:
    args = parse_arguments()

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {args.output}"
        )

    set_seed(args.seed)
    device = get_device()

    reference_model = load_reference_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    dataset = build_binary_eval_dataset()
    split_loaders = build_disjoint_eval_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        ranking_fraction=args.ranking_fraction,
        max_samples=args.max_samples,
    )
    target_layers = resolve_target_layers(
        reference_model=reference_model,
        requested_layers=args.layers,
    )

    fp32_memory_mb = estimate_parameter_memory_mb(
        model=reference_model,
        layer_bits={},
        default_bits=32,
    )
    rows = []
    total_actions = len(target_layers) * len(args.actions)
    action_index = 0

    for layer_index, layer_name in enumerate(
        target_layers
    ):
        for action_name in args.actions:
            action_index += 1
            bits = action_to_bits(action_name)
            print(
                f"[{action_index}/{total_actions}] "
                f"layer={layer_name}, action={action_name}"
            )

            candidate_model = copy.deepcopy(
                reference_model
            ).cpu().eval()
            target_module = candidate_model.get_submodule(
                layer_name
            )
            apply_existing_quantization_inplace(
                module=target_module,
                action_name=action_name,
            )
            candidate_model = candidate_model.to(
                device
            ).eval()

            reference_module = reference_model.get_submodule(
                layer_name
            )
            candidate_module = candidate_model.get_submodule(
                layer_name
            )
            weight_error = relative_l2_weight_error(
                original_weight=(
                    reference_module.weight.detach()
                ),
                quantized_weight=(
                    candidate_module.weight.detach()
                ),
            )
            estimated_memory_mb = (
                estimate_parameter_memory_mb(
                    model=reference_model,
                    layer_bits={layer_name: bits},
                    default_bits=32,
                )
            )

            common = {
                "checkpoint": str(args.checkpoint),
                "seed": args.seed,
                "ranking_fraction": args.ranking_fraction,
                "layer_index": layer_index,
                "layer": layer_name,
                "action": action_name,
                "bits": bits,
                "weight_numel": (
                    reference_module.weight.numel()
                ),
                "relative_l2_weight_error": weight_error,
                "fp32_parameter_memory_mb": fp32_memory_mb,
                "estimated_parameter_memory_mb": (
                    estimated_memory_mb
                ),
                "memory_saving_mb": (
                    fp32_memory_mb - estimated_memory_mb
                ),
            }

            for split_name, dataloader in (
                ("ranking", split_loaders.ranking),
                ("holdout", split_loaders.holdout),
            ):
                comparison = evaluate_model_pair(
                    reference_model=reference_model,
                    candidate_model=candidate_model,
                    dataloader=dataloader,
                    device=device,
                )
                rows.append(
                    {
                        **common,
                        "data_split": split_name,
                        **comparison,
                    }
                )

            save_csv(rows, args.output)
            del candidate_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError(
            "No single-action measurements were executed."
        )

    print(f"Saved benchmark: {args.output}")


if __name__ == "__main__":
    main()

