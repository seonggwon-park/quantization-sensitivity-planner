"""Train one seed of the frozen binary ResNet-18 reproducibility protocol.

The pilot checkpoint is inspected only to validate the binary task. Its weights
are never used to initialize a reproducibility run. This module intentionally
constructs only CIFAR-10 ``train=True`` datasets; test data is not accessed.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import platform
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

import data as project_data
import model as project_model
from config import ExperimentConfig
from data import BinaryCIFAR10, build_transforms, split_train_validation_indices
from metrics import evaluate_classifier
from model import build_binary_resnet18, save_checkpoint
from utils import get_device


SCHEMA_VERSION = "go_no_go_frozen_binary_training_v1"
PROTOCOL_NAME = "binary_resnet18_repro_v1"
FROZEN_TRAINING_SEEDS = (101, 202, 303)

TRAINING_LOG_COLUMNS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one final-epoch binary ResNet-18 checkpoint under the "
            "frozen three-seed reproducibility protocol."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Required protocol seed; must be one of 101, 202, or 303.",
    )
    parser.add_argument(
        "--data-split-seed",
        type=int,
        default=2026,
        help=(
            "Fixed train/validation partition seed shared by all training "
            "seeds (default: 2026)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Fixed number of training epochs; no best-epoch selection is used.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=ExperimentConfig().checkpoint_path,
        help=(
            "Pilot checkpoint inspected only for task metadata and head shape; "
            "its weights are not loaded into the training model."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/repro_v1"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Defaults to results/repro_v1/seed_<seed>.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic PyTorch execution (enabled by default).",
    )
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help=(
            "Explicitly permit nondeterministic operations. With deterministic "
            "mode, PyTorch warns instead of failing for such operations."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the complete protocol without training or writing.",
    )
    return parser.parse_args(argv)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_output(arguments: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def collect_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
        "git_commit_hash": _git_output(["rev-parse", "HEAD"]),
        "git_branch": _git_output(["branch", "--show-current"]),
        "git_dirty_status": _git_output(["status", "--porcelain"])
        or "clean",
    }


def _normalized_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    return re.sub(r"\s+", "", source)


def _require_source_evidence(
    path: Path,
    required_fragments: dict[str, str],
) -> None:
    normalized = _normalized_source(path)
    missing = [
        label
        for label, fragment in required_fragments.items()
        if re.sub(r"\s+", "", fragment) not in normalized
    ]
    if missing:
        raise RuntimeError(
            f"Could not resolve the frozen recipe from {path}: missing source "
            f"evidence for {missing}. Refusing to invent replacement settings."
        )


def inspect_recipe_sources() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    training_entrypoint = (repository_root / "train.py").resolve()
    data_source = Path(
        inspect.getsourcefile(project_data.build_dataloaders) or ""
    ).resolve()
    model_source = Path(
        inspect.getsourcefile(project_model.build_binary_resnet18) or ""
    ).resolve()
    config_source = Path(
        inspect.getsourcefile(ExperimentConfig) or ""
    ).resolve()

    for path in (training_entrypoint, data_source, model_source, config_source):
        if not path.is_file():
            raise RuntimeError(f"Required recipe source file was not found: {path}")

    _require_source_evidence(
        training_entrypoint,
        {
            "ImageNet-pretrained model construction": (
                "build_binary_resnet18(pretrained=True)"
            ),
            "AdamW optimizer": "torch.optim.AdamW(",
            "optimizer learning rate": "lr=config.learning_rate",
            "optimizer weight decay": "weight_decay=config.weight_decay",
            "cosine scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR(",
            "scheduler epoch convention": "T_max=config.epochs",
            "cross-entropy training loss": "functional.cross_entropy(",
            "one-through-epochs convention": (
                "for epoch in range(1, config.epochs + 1)"
            ),
        },
    )
    _require_source_evidence(
        data_source,
        {
            "project transforms": "build_transforms(config.image_size)",
            "project train-validation split": (
                "split_train_validation_indices("
            ),
            "binary dataset adapter": "BinaryCIFAR10(",
            "training augmentation": "transforms.RandomHorizontalFlip()",
        },
    )
    _require_source_evidence(
        model_source,
        {
            "torchvision ResNet-18": "model = resnet18(weights=weights)",
            "two-output classifier head": (
                "model.fc = nn.Linear(original_feature_dim, 2)"
            ),
        },
    )

    training_log = repository_root / "docs" / "experiment_log.md"
    readmes = sorted(
        str(path.resolve())
        for path in repository_root.rglob("README*")
        if ".venv" not in path.parts and "data" not in path.parts
    )

    return {
        "original_training_entrypoint": str(training_entrypoint),
        "config_source": str(config_source),
        "data_source": str(data_source),
        "model_source": str(model_source),
        "training_log": (
            str(training_log.resolve()) if training_log.is_file() else None
        ),
        "readme_files_inspected": readmes,
    }


def inspect_reference_checkpoint(
    checkpoint_path: Path,
    expected_class_ids: tuple[int, int],
    expected_class_names: tuple[str, str],
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Reference checkpoint does not exist: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not safely inspect the reference checkpoint with "
            "weights_only=True."
        ) from error

    if not isinstance(checkpoint, dict):
        raise TypeError("Reference checkpoint must contain a dictionary payload.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Reference checkpoint is missing model_state_dict.")

    class_ids = tuple(checkpoint.get("class_ids", ()))
    class_names = tuple(checkpoint.get("class_names", ()))
    if class_ids != expected_class_ids or class_names != expected_class_names:
        raise ValueError(
            "Reference checkpoint binary mapping does not match ExperimentConfig: "
            f"checkpoint={class_ids}/{class_names}, "
            f"config={expected_class_ids}/{expected_class_names}."
        )

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("model_state_dict must be a dictionary.")
    if "fc.weight" not in state_dict or "fc.bias" not in state_dict:
        raise KeyError("Reference checkpoint is missing the binary fc head.")
    if tuple(state_dict["fc.weight"].shape) != (2, 512):
        raise ValueError(
            "Reference checkpoint fc.weight is not the expected binary "
            f"ResNet-18 shape: {tuple(state_dict['fc.weight'].shape)}"
        )
    if tuple(state_dict["fc.bias"].shape) != (2,):
        raise ValueError(
            "Reference checkpoint fc.bias is not the expected binary shape."
        )

    metadata = {
        key: _jsonable(value)
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }
    return {
        "path": str(checkpoint_path.resolve()),
        "metadata": metadata,
        "state_dict_entries": len(state_dict),
        "fc_weight_shape": list(state_dict["fc.weight"].shape),
        "fc_bias_shape": list(state_dict["fc.bias"].shape),
        "usage_policy": (
            "pilot metadata/shape validation only; pilot weights are never "
            "loaded into the newly trained model"
        ),
    }


def _callable_defaults(callable_object: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name, parameter in inspect.signature(callable_object).parameters.items():
        if name in {"params", "optimizer", "T_max"}:
            continue
        if parameter.default is inspect.Parameter.empty:
            continue
        defaults[name] = _jsonable(parameter.default)
    return defaults


def resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else Path("results/repro_v1") / f"seed_{args.seed}"
    )
    return {
        "checkpoint": (
            args.checkpoint_dir
            / f"resnet18_binary_seed{args.seed}_final.pt"
        ),
        "manifest": run_dir / "training_manifest.json",
        "training_log": run_dir / "training_log.csv",
    }


def resolve_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed not in FROZEN_TRAINING_SEEDS:
        raise ValueError(
            f"--seed must be one of {FROZEN_TRAINING_SEEDS}; got {args.seed}."
        )
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")
    if not args.deterministic and not args.allow_nondeterministic:
        raise ValueError(
            "--no-deterministic requires explicit --allow-nondeterministic."
        )

    config = ExperimentConfig()
    critical_values = {
        "class_ids": config.class_ids,
        "class_names": config.class_names,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "validation_fraction": config.validation_fraction,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    unresolved = [key for key, value in critical_values.items() if value is None]
    if unresolved:
        raise RuntimeError(
            f"Critical ExperimentConfig settings are unresolved: {unresolved}"
        )
    if len(config.class_ids) != 2 or len(config.class_names) != 2:
        raise ValueError("The frozen protocol requires exactly two classes.")
    if config.image_size <= 0 or config.batch_size <= 0:
        raise ValueError("image_size and batch_size must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("Invalid optimizer learning rate or weight decay.")

    source_discovery = inspect_recipe_sources()
    reference = inspect_reference_checkpoint(
        checkpoint_path=args.reference_checkpoint,
        expected_class_ids=config.class_ids,
        expected_class_names=config.class_names,
    )
    train_transform, validation_transform = build_transforms(config.image_size)
    environment = collect_environment()
    output_paths = resolve_output_paths(args)

    optimizer_defaults = _callable_defaults(torch.optim.AdamW)
    optimizer_defaults.update(
        {
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    )
    scheduler_defaults = _callable_defaults(
        torch.optim.lr_scheduler.CosineAnnealingLR
    )
    scheduler_defaults["T_max"] = args.epochs

    binary_mapping = {
        str(binary_label): {
            "cifar10_class_id": int(class_id),
            "class_name": class_name,
        }
        for binary_label, (class_id, class_name) in enumerate(
            zip(config.class_ids, config.class_names)
        )
    }

    setting_provenance = {
        "seed": "required --seed CLI argument; restricted to 101/202/303",
        "data_split_seed": (
            "--data-split-seed CLI argument; fixed default 2026 and separate "
            "from all training RNGs"
        ),
        "epochs": "required --epochs CLI argument; final epoch is saved",
        "architecture": (
            f"{source_discovery['model_source']}: build_binary_resnet18"
        ),
        "pretraining": (
            f"{source_discovery['original_training_entrypoint']}: "
            "build_binary_resnet18(pretrained=True)"
        ),
        "class_mapping": (
            f"{source_discovery['config_source']}: ExperimentConfig; "
            "cross-checked against pilot checkpoint metadata"
        ),
        "data_and_transforms": (
            f"{source_discovery['data_source']}: BinaryCIFAR10, "
            "build_transforms, split_train_validation_indices"
        ),
        "batch_size_num_workers_validation_fraction": (
            f"{source_discovery['config_source']}: ExperimentConfig"
        ),
        "optimizer": (
            f"{source_discovery['original_training_entrypoint']}: AdamW; "
            f"lr/weight_decay from {source_discovery['config_source']}"
        ),
        "scheduler": (
            f"{source_discovery['original_training_entrypoint']}: "
            "CosineAnnealingLR with T_max=epochs"
        ),
        "loss": (
            f"{source_discovery['original_training_entrypoint']}: "
            "torch.nn.functional.cross_entropy"
        ),
        "corroborating_training_log": source_discovery["training_log"],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "allowed_training_seeds": list(FROZEN_TRAINING_SEEDS),
        "training_seed": args.seed,
        "data_split_seed": args.data_split_seed,
        "validation_fraction": config.validation_fraction,
        "shared_data_partition_across_training_seeds": True,
        "data_partition_statement": (
            "The same train/validation partition is used for all allowed "
            "training seeds."
        ),
        "epoch_count": args.epochs,
        "final_epoch": args.epochs,
        "model": {
            "architecture": "torchvision ResNet-18 with Linear(512, 2) head",
            "factory": "model.build_binary_resnet18",
            "pretrained": True,
            "pretrained_weights": str(ResNet18_Weights.DEFAULT),
            "initialization": (
                "fresh ImageNet-pretrained construction after per-run seeding; "
                "reference checkpoint weights are not loaded"
            ),
        },
        "binary_class_mapping": binary_mapping,
        "dataset": {
            "dataset_class": "data.BinaryCIFAR10",
            "source": "torchvision.datasets.CIFAR10",
            "root": str(config.data_dir),
            "source_split": "train",
            "training_and_validation_only": True,
            "test_data_accessed": False,
            "validation_fraction": config.validation_fraction,
            "data_split_seed": args.data_split_seed,
            "shared_across_all_allowed_training_seeds": True,
            "shared_partition_statement": (
                "The same train/validation partition is used for all allowed "
                "training seeds."
            ),
            "split_function": "data.split_train_validation_indices",
            "training_augmentation": True,
            "post_training_partition_exclusions": None,
            "eligible_binary_pool_policy": (
                "all selected-class CIFAR-10 train examples are partitioned "
                "between training and validation"
            ),
        },
        "transforms": {
            "builder": "data.build_transforms",
            "image_size": config.image_size,
            "train": repr(train_transform),
            "validation": repr(validation_transform),
            "normalization_mean": list(project_data.IMAGENET_MEAN),
            "normalization_std": list(project_data.IMAGENET_STD),
        },
        "dataloader": {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "train_shuffle": True,
            "validation_shuffle": False,
            "pin_memory": environment["cuda_available"],
            "persistent_workers": config.num_workers > 0,
            "worker_seed_function": (
                "torch.initial_seed() modulo 2**32 -> Python random and NumPy"
            ),
            "seeded_torch_generators": True,
        },
        "loss": {
            "name": "torch.nn.functional.cross_entropy",
            "reduction": "mean",
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            **optimizer_defaults,
        },
        "scheduler": {
            "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
            **scheduler_defaults,
            "step_timing": "after validation at the end of every epoch",
        },
        "checkpoint_policy": {
            "selection": "fixed final epoch",
            "validation_best_selection": False,
            "test_selection": False,
            "test_evaluation_during_training": False,
        },
        "determinism": {
            "requested": args.deterministic,
            "strict": args.deterministic and not args.allow_nondeterministic,
            "allow_nondeterministic": args.allow_nondeterministic,
            "cudnn_deterministic": args.deterministic,
            "cudnn_benchmark": False,
            "deterministic_algorithms": args.deterministic,
            "deterministic_algorithms_warn_only": (
                args.deterministic and args.allow_nondeterministic
            ),
            "cublas_workspace_config": (
                ":4096:8" if args.deterministic else None
            ),
        },
        "reference_checkpoint": reference,
        "source_discovery": source_discovery,
        "setting_provenance": setting_provenance,
        "environment": environment,
        "output_paths": {
            key: str(path.resolve()) for key, path in output_paths.items()
        },
    }


def print_resolved_protocol(
    args: argparse.Namespace,
    resolved: dict[str, Any],
) -> None:
    print("Resolved frozen binary training protocol")
    print(json.dumps(resolved, indent=2, ensure_ascii=False))
    print("\nCLI arguments")
    print(json.dumps(_jsonable(vars(args)), indent=2, ensure_ascii=False))
    print("\nSeed separation")
    print(f"  training seed: {resolved['training_seed']}")
    print(f"  fixed data split seed: {resolved['data_split_seed']}")
    print(
        "  train/validation partition shared across all runs: "
        f"{resolved['shared_data_partition_across_training_seeds']}"
    )
    print("\nDiscovered original training entrypoint")
    print(resolved["source_discovery"]["original_training_entrypoint"])
    print("\nCritical setting provenance")
    for setting, source in resolved["setting_provenance"].items():
        print(f"  {setting}: {source}")
    print("\nExpected output paths")
    for label, path in resolved["output_paths"].items():
        print(f"  {label}: {path}")


def configure_reproducibility(
    seed: int,
    deterministic: bool,
    allow_nondeterministic: bool,
) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

    if not hasattr(torch, "use_deterministic_algorithms"):
        if deterministic:
            raise RuntimeError(
                "This PyTorch version cannot enable deterministic algorithms."
            )
    else:
        torch.use_deterministic_algorithms(
            deterministic,
            warn_only=deterministic and allow_nondeterministic,
        )

    return {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else deterministic and allow_nondeterministic
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "allow_nondeterministic": allow_nondeterministic,
    }


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _class_counts(
    dataset: BinaryCIFAR10,
) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for source_index in dataset.source_indices:
        original_label = int(dataset.base_dataset.targets[source_index])
        binary_label = dataset.label_map[original_label]
        counts[binary_label] += 1
    return counts


def build_train_validation_loaders(
    config: ExperimentConfig,
    training_seed: int,
    data_split_seed: int,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    train_transform, validation_transform = build_transforms(config.image_size)
    train_indices, validation_indices = split_train_validation_indices(
        root=str(config.data_dir),
        class_ids=config.class_ids,
        validation_fraction=config.validation_fraction,
        seed=data_split_seed,
    )

    if set(train_indices).intersection(validation_indices):
        raise RuntimeError("Training and validation source indices overlap.")

    train_dataset = BinaryCIFAR10(
        root=str(config.data_dir),
        train=True,
        class_ids=config.class_ids,
        transform=train_transform,
        source_indices=train_indices,
        download=False,
    )
    validation_dataset = BinaryCIFAR10(
        root=str(config.data_dir),
        train=True,
        class_ids=config.class_ids,
        transform=validation_transform,
        source_indices=validation_indices,
        download=False,
    )

    eligible_indices = {
        index
        for index, label in enumerate(train_dataset.base_dataset.targets)
        if int(label) in config.class_ids
    }
    partition_union = set(train_indices).union(validation_indices)
    if partition_union != eligible_indices:
        raise RuntimeError(
            "The train/validation partition does not cover the complete binary "
            "CIFAR-10 training pool."
        )

    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_dataloader_worker,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_generator = torch.Generator().manual_seed(training_seed)
    validation_generator = torch.Generator().manual_seed(training_seed + 1)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=validation_generator,
        **loader_kwargs,
    )

    dataset_summary = {
        "eligible_binary_train_pool_size": len(eligible_indices),
        "training_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "training_seed": training_seed,
        "data_split_seed": data_split_seed,
        "validation_fraction": config.validation_fraction,
        "shared_data_partition_across_training_seeds": True,
        "training_class_counts": _class_counts(train_dataset),
        "validation_class_counts": _class_counts(validation_dataset),
        "train_validation_overlap_count": 0,
        "union_covers_full_eligible_train_pool": True,
        "test_data_accessed": False,
    }
    return {
        "train": train_loader,
        "validation": validation_loader,
    }, dataset_summary


def _ensure_outputs_absent(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing reproducibility outputs: "
            + ", ".join(existing)
        )


def _write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(
            _jsonable(manifest),
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run the same cross-entropy epoch loop defined by the original train.py."""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    if total_samples == 0:
        raise RuntimeError("The training dataloader produced no samples.")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def _raise_determinism_context(
    error: RuntimeError,
    allow_nondeterministic: bool,
) -> None:
    if "determin" in str(error).lower() and not allow_nondeterministic:
        raise RuntimeError(
            "A nondeterministic operation was rejected by the frozen protocol. "
            "Investigate the operation, or explicitly pass "
            "--allow-nondeterministic to permit PyTorch warning-only behavior."
        ) from error
    raise error


def run_training(
    args: argparse.Namespace,
    resolved: dict[str, Any],
) -> None:
    output_paths = resolve_output_paths(args)
    _ensure_outputs_absent(output_paths)

    applied_determinism = configure_reproducibility(
        seed=args.seed,
        deterministic=args.deterministic,
        allow_nondeterministic=args.allow_nondeterministic,
    )
    device = get_device()

    config = ExperimentConfig()
    config.seed = args.seed
    config.epochs = args.epochs
    dataloaders, dataset_summary = build_train_validation_loaders(
        config=config,
        training_seed=args.seed,
        data_split_seed=args.data_split_seed,
    )
    print(f"Device: {device}")
    print("Resolved dataset summary:")
    print(json.dumps(dataset_summary, indent=2))
    print("Test data accessed: false")

    model = build_binary_resnet18(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        try:
            train_metrics = train_one_epoch(
                model=model,
                dataloader=dataloaders["train"],
                optimizer=optimizer,
                device=device,
            )
            validation_metrics = evaluate_classifier(
                model=model,
                dataloader=dataloaders["validation"],
                device=device,
            )
        except RuntimeError as error:
            _raise_determinism_context(error, args.allow_nondeterministic)

        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        log_rows.append(row)
        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"lr={learning_rate:.8g} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f}"
        )

    completed_at = _timestamp()
    checkpoint_metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "training_seed": args.seed,
        "data_split_seed": args.data_split_seed,
        "validation_fraction": config.validation_fraction,
        "data_partition_statement": (
            "The same train/validation partition is used for all allowed "
            "training seeds."
        ),
        "epoch_count": args.epochs,
        "final_epoch": args.epochs,
        "model_architecture": resolved["model"],
        "class_ids": config.class_ids,
        "class_names": config.class_names,
        "binary_class_mapping": resolved["binary_class_mapping"],
        "optimizer_configuration": resolved["optimizer"],
        "scheduler_configuration": resolved["scheduler"],
        "transform_configuration": resolved["transforms"],
        "dataset_source": {
            **resolved["dataset"],
            "resolved_counts": dataset_summary,
        },
        "deterministic_settings": applied_determinism,
        "git_commit_hash": resolved["environment"]["git_commit_hash"],
        "timestamp": completed_at,
        "test_data_accessed": False,
        "checkpoint_selection": "fixed final epoch",
        "final_training_metrics": log_rows[-1],
    }
    save_checkpoint(
        model=model,
        checkpoint_path=output_paths["checkpoint"],
        metadata=checkpoint_metadata,
    )
    _write_training_log(output_paths["training_log"], log_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "timestamp": completed_at,
        "training_seed": args.seed,
        "data_split_seed": args.data_split_seed,
        "validation_fraction": config.validation_fraction,
        "data_partition_statement": (
            "The same train/validation partition is used for all allowed "
            "training seeds."
        ),
        "cli_arguments": _jsonable(vars(args)),
        "resolved_settings": resolved,
        "applied_deterministic_settings": applied_determinism,
        "environment_versions": resolved["environment"],
        "dataset_summary": dataset_summary,
        "output_checkpoint_path": str(output_paths["checkpoint"].resolve()),
        "training_log_path": str(output_paths["training_log"].resolve()),
        "final_epoch_metrics": log_rows[-1],
        "test_data_accessed": False,
    }
    _write_manifest(output_paths["manifest"], manifest)

    print(f"Final-epoch checkpoint: {output_paths['checkpoint'].resolve()}")
    print(f"Training manifest: {output_paths['manifest'].resolve()}")
    print(f"Training log: {output_paths['training_log'].resolve()}")
    print("Test data accessed: false")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    resolved = resolve_protocol(args)
    print_resolved_protocol(args, resolved)

    if args.dry_run:
        print("\nDry run complete: no training performed and no files written.")
        return

    run_training(args, resolved)


if __name__ == "__main__":
    main()
