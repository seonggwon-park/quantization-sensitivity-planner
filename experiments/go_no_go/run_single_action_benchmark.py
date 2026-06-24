"""Run isolated one-layer, one-action Go/No-Go measurements."""

import argparse
import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn

from additive_planner import (
    QUANTIZED_ACTIONS,
    action_to_bits,
)
from config import ExperimentConfig
from quantization import list_quantizable_layers
from utils import get_device, set_seed

from experiments.go_no_go import DEFAULT_RESULTS_DIR
from experiments.go_no_go.adapters import (
    apply_existing_quantization_inplace,
    build_binary_eval_dataset,
    get_checkpoint_default_path,
    load_reference_model,
)
from experiments.go_no_go.metrics import (
    evaluate_oracle_set,
    evaluate_score_set,
    weight_relative_l2,
)
from experiments.go_no_go.splits import (
    build_class_balanced_calibration_splits,
    save_split_indices,
)


RESULT_COLUMNS = (
    "run_id",
    "score_seed",
    "oracle_seed",
    "layer_name",
    "action",
    "bits",
    "layer_numel",
    "whole_model_saving",
    "quantizable_weight_saving",
    "weight_rel_l2",
    "activation_rel_mse",
    "output_kl_mean",
    "abs_delta_score_mean",
    "decision_risk_mean",
    "decision_risk_p95",
    "decision_risk_violation_rate",
    "scoring_runtime_seconds",
    "oracle_flip_rate",
    "oracle_teacher_agreement",
    "oracle_accuracy",
    "oracle_abs_delta_score_mean",
    "oracle_decision_risk_mean",
    "oracle_decision_risk_p95",
    "oracle_decision_risk_violation_rate",
    "oracle_runtime_seconds",
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
        "--score-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--oracle-size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--score-seeds",
        type=int,
        nargs="+",
        default=[0],
    )
    parser.add_argument(
        "--oracle-seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.batch_size,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
    )

    return parser.parse_args()


def enumerate_candidates(
    reference_model: nn.Module,
) -> list[tuple[str, str]]:
    """Enumerate Conv2d/Linear weights crossed with existing actions."""

    candidates = []

    for layer_name in list_quantizable_layers(
        reference_model
    ):
        module = reference_model.get_submodule(layer_name)

        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            raise TypeError(
                f"Quantizable layer {layer_name} is not Conv2d or Linear."
            )

        for action_name in QUANTIZED_ACTIONS:
            candidates.append((layer_name, action_name))

    return candidates


def theoretical_storage_savings(
    model: nn.Module,
    layer_name: str,
    bits: int,
) -> tuple[float, float]:
    """Return theoretical saving ratios; no runtime claim is implied."""

    module = model.get_submodule(layer_name)
    layer_numel = module.weight.numel()
    total_parameter_numel = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    total_quantizable_weight_numel = sum(
        model.get_submodule(name).weight.numel()
        for name in list_quantizable_layers(model)
    )
    saved_bits = layer_numel * (32 - bits)

    return (
        saved_bits / (total_parameter_numel * 32),
        saved_bits
        / (total_quantizable_weight_numel * 32),
    )


def make_run_id(
    score_seed: int,
    layer_name: str,
    action_name: str,
) -> str:
    return (
        f"score_seed={score_seed}|"
        f"layer={layer_name}|action={action_name}"
    )


def assert_unique_rows(
    rows: list[dict],
) -> None:
    keys = [
        (
            int(row["score_seed"]),
            str(row["layer_name"]),
            str(row["action"]),
        )
        for row in rows
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate (score_seed, layer_name, action) rows "
            "are about to be written."
        )


def write_results(
    rows: list[dict],
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {output_path}"
        )

    assert_unique_rows(rows)

    for row in rows:
        if tuple(row) != RESULT_COLUMNS:
            raise ValueError(
                "Result row does not exactly match the required schema."
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)


def _preflight_output_paths(
    output_dir: Path,
    score_seeds: list[int],
) -> dict[int, Path]:
    output_paths = {
        score_seed: (
            output_dir
            / f"single_action_metrics_seed{score_seed}.csv"
        )
        for score_seed in score_seeds
    }
    protected_paths = [
        output_dir / "split_indices.npz",
        output_dir / "split_indices.json",
        *output_paths.values(),
    ]

    for output_path in protected_paths:
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )

    return output_paths


def main() -> None:
    args = parse_arguments()

    if args.max_candidates is not None and args.max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")

    if len(set(args.score_seeds)) != len(args.score_seeds):
        raise ValueError("score_seeds must be unique.")

    output_paths = _preflight_output_paths(
        output_dir=args.output_dir,
        score_seeds=args.score_seeds,
    )
    set_seed(args.oracle_seed)
    device = get_device()
    dataset = build_binary_eval_dataset(
        source_split="train"
    )
    splits = build_class_balanced_calibration_splits(
        dataset=dataset,
        score_size=args.score_size,
        oracle_size=args.oracle_size,
        score_seeds=args.score_seeds,
        oracle_seed=args.oracle_seed,
        batch_size=args.batch_size,
        num_workers=0,
    )
    reference_model = load_reference_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    base_evaluation_model = copy.deepcopy(
        reference_model
    ).to(device).eval()
    all_candidates = enumerate_candidates(
        reference_model
    )
    selected_candidates = (
        all_candidates
        if args.max_candidates is None
        else all_candidates[: args.max_candidates]
    )
    rows_by_seed = {
        score_seed: []
        for score_seed in args.score_seeds
    }

    for candidate_index, (
        layer_name,
        action_name,
    ) in enumerate(selected_candidates, start=1):
        print(
            f"[{candidate_index}/{len(selected_candidates)}] "
            f"layer={layer_name}, action={action_name}"
        )
        candidate_model = None
        candidate_module = None

        try:
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
            candidate_module = candidate_model.get_submodule(
                layer_name
            )
            reference_module = reference_model.get_submodule(
                layer_name
            )
            base_evaluation_module = (
                base_evaluation_model.get_submodule(
                    layer_name
                )
            )
            bits = action_to_bits(action_name)
            whole_model_saving, quantizable_weight_saving = (
                theoretical_storage_savings(
                    model=reference_model,
                    layer_name=layer_name,
                    bits=bits,
                )
            )
            weight_rel_l2 = weight_relative_l2(
                base_weight=reference_module.weight.detach(),
                candidate_weight=(
                    candidate_module.weight.detach()
                ),
            )
            score_metrics_by_seed = {}

            for score_seed in args.score_seeds:
                score_metrics_by_seed[score_seed] = (
                    evaluate_score_set(
                        base_model=base_evaluation_model,
                        candidate_model=candidate_model,
                        base_module=base_evaluation_module,
                        candidate_module=candidate_module,
                        dataloader=(
                            splits.score_loaders[score_seed]
                        ),
                        device=device,
                    )
                )

            oracle_metrics = evaluate_oracle_set(
                base_model=base_evaluation_model,
                candidate_model=candidate_model,
                dataloader=splits.oracle_loader,
                device=device,
            )

            for score_seed in args.score_seeds:
                score_metrics = score_metrics_by_seed[
                    score_seed
                ]
                row = {
                    "run_id": make_run_id(
                        score_seed=score_seed,
                        layer_name=layer_name,
                        action_name=action_name,
                    ),
                    "score_seed": score_seed,
                    "oracle_seed": args.oracle_seed,
                    "layer_name": layer_name,
                    "action": action_name,
                    "bits": bits,
                    "layer_numel": (
                        reference_module.weight.numel()
                    ),
                    "whole_model_saving": (
                        whole_model_saving
                    ),
                    "quantizable_weight_saving": (
                        quantizable_weight_saving
                    ),
                    "weight_rel_l2": weight_rel_l2,
                    "activation_rel_mse": score_metrics[
                        "activation_rel_mse"
                    ],
                    "output_kl_mean": score_metrics[
                        "output_kl_mean"
                    ],
                    "abs_delta_score_mean": score_metrics[
                        "abs_delta_score_mean"
                    ],
                    "decision_risk_mean": score_metrics[
                        "decision_risk_mean"
                    ],
                    "decision_risk_p95": score_metrics[
                        "decision_risk_p95"
                    ],
                    "decision_risk_violation_rate": (
                        score_metrics[
                            "decision_risk_violation_rate"
                        ]
                    ),
                    "scoring_runtime_seconds": score_metrics[
                        "scoring_runtime_seconds"
                    ],
                    "oracle_flip_rate": oracle_metrics[
                        "oracle_flip_rate"
                    ],
                    "oracle_teacher_agreement": oracle_metrics[
                        "oracle_teacher_agreement"
                    ],
                    "oracle_accuracy": oracle_metrics[
                        "oracle_accuracy"
                    ],
                    "oracle_abs_delta_score_mean": (
                        oracle_metrics[
                            "oracle_abs_delta_score_mean"
                        ]
                    ),
                    "oracle_decision_risk_mean": (
                        oracle_metrics[
                            "oracle_decision_risk_mean"
                        ]
                    ),
                    "oracle_decision_risk_p95": (
                        oracle_metrics[
                            "oracle_decision_risk_p95"
                        ]
                    ),
                    "oracle_decision_risk_violation_rate": (
                        oracle_metrics[
                            "oracle_decision_risk_violation_rate"
                        ]
                    ),
                    "oracle_runtime_seconds": oracle_metrics[
                        "oracle_runtime_seconds"
                    ],
                }
                rows_by_seed[score_seed].append(row)
        finally:
            target_module = None
            candidate_module = None

            if candidate_model is not None:
                del candidate_model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_rows = [
        row
        for score_seed in args.score_seeds
        for row in rows_by_seed[score_seed]
    ]
    expected_row_count = (
        len(selected_candidates) * len(args.score_seeds)
    )

    if len(all_rows) != expected_row_count:
        raise RuntimeError(
            f"Expected {expected_row_count} rows, got {len(all_rows)}."
        )

    assert_unique_rows(all_rows)
    save_split_indices(
        splits=splits,
        dataset=dataset,
        output_dir=args.output_dir,
        source_split="train",
        oracle_seed=args.oracle_seed,
    )

    for score_seed in args.score_seeds:
        write_results(
            rows=rows_by_seed[score_seed],
            output_path=output_paths[score_seed],
        )

    overlap_count = sum(
        len(
            set(indices)
            & set(splits.oracle_indices)
        )
        for indices in splits.score_indices.values()
    )
    print(f"Discovered candidate count: {len(all_candidates)}")
    print(f"Expected row count: {expected_row_count}")
    print(f"Actual row count: {len(all_rows)}")
    print(
        "Output CSV path: "
        + ", ".join(
            str(output_paths[seed])
            for seed in args.score_seeds
        )
    )
    print("Duplicate key check result: PASS")
    print(f"Score/oracle overlap count: {overlap_count}")
    print(f"Device used: {device}")


if __name__ == "__main__":
    main()
