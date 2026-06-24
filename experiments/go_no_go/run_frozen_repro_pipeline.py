"""Orchestrate the fixed post-training compression protocol for one checkpoint.

All scientific computation is delegated to the existing benchmark, planner,
vector, and evaluator modules. The benchmark's native CLI cannot accept a
canonical split path, so an explicit subprocess worker injects loaders from the
canonical NPZ into its existing ``main`` function. It never generates a split.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

import numpy as np


STAGES = (
    "benchmark",
    "scalar_plans",
    "vectors",
    "vector_plans",
    "confirmation_eval",
    "test_eval",
)
FIXED_SCORE_SEEDS = (0, 1, 2)
FIXED_MEMORY_SAVING_RATIOS = (0.70, 0.80, 0.82, 0.84, 0.85, 0.86)
FIXED_BEAM_WIDTH = 512
FIXED_MAX_STATES_PER_MEMORY_BIN = 8
FIXED_MEMORY_QUANTUM_KB = 1
PROTOCOL_STATEMENT = (
    "No confirmation or test result is used for method, objective, or "
    "hyperparameter selection."
)
SCHEMA_VERSION = "go_no_go_frozen_repro_pipeline_v1"

DEFAULT_CANONICAL_SPLIT = Path("results/go_no_go/split_indices.npz")
DEFAULT_CANONICAL_METADATA = Path("results/go_no_go/split_indices.json")
DEFAULT_CONFIRMATION_SPLIT = Path(
    "results/go_no_go_confirmation_v1/confirmation_split_indices.npz"
)
DEFAULT_CONFIRMATION_METADATA = Path(
    "results/go_no_go_confirmation_v1/confirmation_split_indices.json"
)

PILOT_RESULT_DIRECTORIES = (
    Path("results/go_no_go"),
    Path("results/go_no_go_planner_v1"),
    Path("results/go_no_go_planner_v2_stress"),
    Path("results/go_no_go_vectors_v1"),
    Path("results/go_no_go_vector_plans_v1"),
    Path("results/go_no_go_confirmation_v1"),
    Path("results/go_no_go_confirmation_eval_v1"),
    Path("results/go_no_go_test_eval_v1"),
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed post-training compression protocol for one "
            "independently trained binary ResNet-18 checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--canonical-split",
        type=Path,
        default=DEFAULT_CANONICAL_SPLIT,
    )
    parser.add_argument(
        "--canonical-metadata",
        type=Path,
        default=DEFAULT_CANONICAL_METADATA,
    )
    parser.add_argument(
        "--confirmation-split",
        type=Path,
        default=DEFAULT_CONFIRMATION_SPLIT,
    )
    parser.add_argument(
        "--confirmation-metadata",
        type=Path,
        default=DEFAULT_CONFIRMATION_METADATA,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--score-seeds",
        type=int,
        nargs="+",
        default=list(FIXED_SCORE_SEEDS),
    )
    parser.add_argument(
        "--memory-saving-ratios",
        type=float,
        nargs="+",
        default=list(FIXED_MEMORY_SAVING_RATIOS),
    )
    parser.add_argument("--beam-width", type=int, default=FIXED_BEAM_WIDTH)
    parser.add_argument(
        "--max-states-per-memory-bin",
        type=int,
        default=FIXED_MAX_STATES_PER_MEMORY_BIN,
    )
    parser.add_argument(
        "--memory-quantum-kb",
        type=int,
        default=FIXED_MEMORY_QUANTUM_KB,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all", *STAGES),
        default=["all"],
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from a compatible pipeline manifest while preserving "
            "completed, unrequested stages."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parse_benchmark_worker_arguments(
    argv: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--canonical-split", type=Path, required=True)
    parser.add_argument("--canonical-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--score-seeds", type=int, nargs="+", required=True)
    return parser.parse_args(argv)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def paths_overlap(first: Path, second: Path) -> bool:
    first = resolved(first)
    second = resolved(second)
    return first == second or first in second.parents or second in first.parents


def normalize_stages(stage_arguments: Sequence[str]) -> tuple[str, ...]:
    if "all" in stage_arguments:
        if len(stage_arguments) != 1:
            raise ValueError("--stages all cannot be combined with stage names.")
        return STAGES
    if len(set(stage_arguments)) != len(stage_arguments):
        raise ValueError("--stages contains duplicate stage names.")
    requested = set(stage_arguments)
    return tuple(stage for stage in STAGES if stage in requested)


def stage_directories(output_root: Path) -> dict[str, Path]:
    return {stage: output_root / stage for stage in STAGES}


def summary_paths(output_root: Path) -> dict[str, Path]:
    directories = stage_directories(output_root)
    return {
        "scalar_plan_summary": (
            directories["scalar_plans"] / "planner_comparison_summary.csv"
        ),
        "vector_plan_summary": (
            directories["vector_plans"] / "vector_planner_summary.csv"
        ),
        "confirmation_primary_summary": (
            directories["confirmation_eval"]
            / "confirmation_primary_summary.csv"
        ),
        "test_primary_summary": (
            directories["test_eval"] / "test_primary_summary.csv"
        ),
    }


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_canonical_split(
    split_path: Path,
    metadata_path: Path,
    score_seeds: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    require_file(split_path, "Canonical split NPZ")
    require_file(metadata_path, "Canonical split metadata")
    with np.load(split_path, allow_pickle=False) as split_data:
        arrays = {name: split_data[name].copy() for name in split_data.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    required_arrays = {
        "class_ids",
        "oracle_dataset_indices",
        "oracle_source_indices",
        "oracle_class_counts",
    }
    for score_seed in score_seeds:
        prefix = f"score_seed_{score_seed}"
        required_arrays.update(
            {
                f"{prefix}_dataset_indices",
                f"{prefix}_source_indices",
                f"{prefix}_class_counts",
            }
        )
    missing = sorted(required_arrays.difference(arrays))
    if missing:
        raise KeyError(
            f"Canonical split is missing required arrays: {missing}"
        )
    if arrays["class_ids"].tolist() != [0, 1]:
        raise ValueError("Canonical split class_ids must be exactly [0, 1].")
    if metadata.get("source_split") != "train":
        raise ValueError("Canonical metadata must use source_split='train'.")
    metadata_seeds = {int(seed) for seed in metadata.get("score_sets", {})}
    if metadata_seeds != set(score_seeds):
        raise ValueError(
            "Canonical metadata score seeds do not exactly match the fixed "
            f"protocol: {sorted(metadata_seeds)} vs {list(score_seeds)}."
        )

    oracle_indices = arrays["oracle_dataset_indices"].astype(np.int64)
    if oracle_indices.size == 0 or oracle_indices.size % 2:
        raise ValueError("Canonical oracle indices must have positive even size.")
    if len(set(oracle_indices.tolist())) != oracle_indices.size:
        raise ValueError("Canonical oracle indices contain duplicates.")
    oracle_index_set = set(oracle_indices.tolist())

    metadata_oracle = metadata.get("oracle", {})
    if metadata_oracle.get("dataset_indices") != oracle_indices.tolist():
        raise ValueError("Canonical NPZ and JSON oracle indices differ.")
    if metadata_oracle.get("source_indices") != arrays[
        "oracle_source_indices"
    ].astype(np.int64).tolist():
        raise ValueError("Canonical NPZ and JSON oracle source indices differ.")

    for score_seed in score_seeds:
        prefix = f"score_seed_{score_seed}"
        score_indices = arrays[
            f"{prefix}_dataset_indices"
        ].astype(np.int64)
        if score_indices.size == 0 or score_indices.size % 2:
            raise ValueError(
                f"Score seed {score_seed} must have positive even size."
            )
        if len(set(score_indices.tolist())) != score_indices.size:
            raise ValueError(f"Score seed {score_seed} contains duplicates.")
        if oracle_index_set.intersection(score_indices.tolist()):
            raise ValueError(
                f"Score seed {score_seed} overlaps the canonical oracle set."
            )
        metadata_score = metadata["score_sets"].get(str(score_seed), {})
        if metadata_score.get("dataset_indices") != score_indices.tolist():
            raise ValueError(
                f"Canonical NPZ and JSON differ for score seed {score_seed}."
            )
        if metadata_score.get("source_indices") != arrays[
            f"{prefix}_source_indices"
        ].astype(np.int64).tolist():
            raise ValueError(
                "Canonical NPZ and JSON source indices differ for score seed "
                f"{score_seed}."
            )
    return arrays, metadata


def validate_confirmation_inputs(
    split_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    require_file(split_path, "Confirmation split NPZ")
    require_file(metadata_path, "Confirmation split metadata")
    with np.load(split_path, allow_pickle=False) as split_data:
        if "dataset_indices" not in split_data.files:
            raise KeyError("Confirmation split lacks dataset_indices.")
        count = int(split_data["dataset_indices"].size)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_split") != "train":
        raise ValueError("Confirmation metadata must use source_split='train'.")
    if count != int(metadata.get("requested_size", -1)):
        raise ValueError(
            "Confirmation NPZ size does not match its requested_size metadata."
        )
    return {
        "sample_count": count,
        "source_split": metadata.get("source_split"),
        "schema_version": metadata.get("schema_version"),
    }


def validate_fixed_protocol(args: argparse.Namespace) -> tuple[str, ...]:
    stages = normalize_stages(args.stages)
    if not args.run_name.strip():
        raise ValueError("--run-name cannot be empty.")
    if any(separator in args.run_name for separator in ("/", "\\")):
        raise ValueError("--run-name must be a name, not a path.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if tuple(args.score_seeds) != FIXED_SCORE_SEEDS:
        raise ValueError(
            f"Frozen score seeds are {FIXED_SCORE_SEEDS}; got "
            f"{tuple(args.score_seeds)}."
        )
    supplied_ratios = tuple(float(value) for value in args.memory_saving_ratios)
    if len(supplied_ratios) != len(FIXED_MEMORY_SAVING_RATIOS) or any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(
            supplied_ratios, FIXED_MEMORY_SAVING_RATIOS
        )
    ):
        raise ValueError(
            "Frozen memory-saving ratios are "
            f"{FIXED_MEMORY_SAVING_RATIOS}; got {supplied_ratios}."
        )
    fixed_settings = {
        "beam_width": (args.beam_width, FIXED_BEAM_WIDTH),
        "max_states_per_memory_bin": (
            args.max_states_per_memory_bin,
            FIXED_MAX_STATES_PER_MEMORY_BIN,
        ),
        "memory_quantum_kb": (
            args.memory_quantum_kb,
            FIXED_MEMORY_QUANTUM_KB,
        ),
    }
    changed = {
        name: actual
        for name, (actual, expected) in fixed_settings.items()
        if actual != expected
    }
    if changed:
        raise ValueError(
            f"Frozen vector settings cannot be retuned: {changed}."
        )
    return stages


def validate_output_isolation(output_root: Path) -> None:
    outputs = [
        output_root / "pipeline_manifest.json",
        output_root / "pipeline_commands.txt",
        *stage_directories(output_root).values(),
    ]
    overlaps = []
    for output_path in outputs:
        for pilot_directory in PILOT_RESULT_DIRECTORIES:
            if paths_overlap(output_path, pilot_directory):
                overlaps.append(
                    f"{resolved(output_path)} <-> {resolved(pilot_directory)}"
                )
    if overlaps:
        raise ValueError(
            "Pipeline outputs overlap protected pilot result directories:\n"
            + "\n".join(overlaps)
        )


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def extend_values(command: list[str], flag: str, values: Sequence[Any]) -> None:
    command.append(flag)
    command.extend(str(value) for value in values)


def build_stage_commands(
    args: argparse.Namespace,
    stages: Sequence[str],
) -> dict[str, list[str]]:
    python = sys.executable
    directories = stage_directories(args.output_root)
    commands: dict[str, list[str]] = {}

    benchmark = [
        python,
        "-m",
        "experiments.go_no_go.run_frozen_repro_pipeline",
        "--_canonical-benchmark-worker",
        "--checkpoint",
        str(args.checkpoint),
        "--canonical-split",
        str(args.canonical_split),
        "--canonical-metadata",
        str(args.canonical_metadata),
        "--output-dir",
        str(directories["benchmark"]),
        "--batch-size",
        str(args.batch_size),
    ]
    extend_values(benchmark, "--score-seeds", args.score_seeds)
    commands["benchmark"] = benchmark

    scalar = [
        python,
        "-m",
        "experiments.go_no_go.planner_eval",
        "--checkpoint",
        str(args.checkpoint),
        "--benchmark-dir",
        str(directories["benchmark"]),
        "--split-indices",
        str(args.canonical_split),
        "--memory-quantum-kb",
        str(args.memory_quantum_kb),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(directories["scalar_plans"]),
    ]
    extend_values(
        scalar, "--memory-saving-ratios", args.memory_saving_ratios
    )
    commands["scalar_plans"] = scalar

    vectors = [
        python,
        "-m",
        "experiments.go_no_go.collect_score_delta_vectors",
        "--checkpoint",
        str(args.checkpoint),
        "--benchmark-dir",
        str(directories["benchmark"]),
        "--split-indices",
        str(args.canonical_split),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(directories["vectors"]),
    ]
    extend_values(vectors, "--score-seeds", args.score_seeds)
    commands["vectors"] = vectors

    vector_plans = [
        python,
        "-m",
        "experiments.go_no_go.vector_beam_planner",
        "--vector-dir",
        str(directories["vectors"]),
        "--benchmark-dir",
        str(directories["benchmark"]),
        "--beam-width",
        str(args.beam_width),
        "--max-states-per-memory-bin",
        str(args.max_states_per_memory_bin),
        "--memory-quantum-kb",
        str(args.memory_quantum_kb),
        "--output-dir",
        str(directories["vector_plans"]),
    ]
    extend_values(
        vector_plans,
        "--memory-saving-ratios",
        args.memory_saving_ratios,
    )
    commands["vector_plans"] = vector_plans

    commands["confirmation_eval"] = [
        python,
        "-m",
        "experiments.go_no_go.evaluate_plans_on_confirmation",
        "--checkpoint",
        str(args.checkpoint),
        "--confirmation-split",
        str(args.confirmation_split),
        "--confirmation-metadata",
        str(args.confirmation_metadata),
        "--vector-plan-dir",
        str(directories["vector_plans"]),
        "--scalar-plan-dirs",
        str(directories["scalar_plans"]),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(directories["confirmation_eval"]),
    ]
    commands["test_eval"] = [
        python,
        "-m",
        "experiments.go_no_go.evaluate_plans_on_test",
        "--checkpoint",
        str(args.checkpoint),
        "--vector-plan-dir",
        str(directories["vector_plans"]),
        "--scalar-plan-dirs",
        str(directories["scalar_plans"]),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(directories["test_eval"]),
    ]
    return {stage: commands[stage] for stage in stages}


def stage_required_inputs(
    output_root: Path,
) -> dict[str, tuple[Path, ...]]:
    directories = stage_directories(output_root)
    benchmark_csvs = tuple(
        directories["benchmark"] / f"single_action_metrics_seed{seed}.csv"
        for seed in FIXED_SCORE_SEEDS
    )
    vector_files = tuple(
        directories["vectors"] / f"score_delta_vectors_seed{seed}.npz"
        for seed in FIXED_SCORE_SEEDS
    )
    return {
        "benchmark": (),
        "scalar_plans": benchmark_csvs,
        "vectors": benchmark_csvs,
        "vector_plans": (
            *benchmark_csvs,
            *vector_files,
        ),
        "confirmation_eval": (
            directories["scalar_plans"] / "planner_comparison_summary.csv",
            directories["vector_plans"] / "vector_planner_summary.csv",
        ),
        "test_eval": (
            directories["scalar_plans"] / "planner_comparison_summary.csv",
            directories["vector_plans"] / "vector_planner_summary.csv",
        ),
    }


def stage_expected_outputs(
    output_root: Path,
) -> dict[str, tuple[Path, ...]]:
    directories = stage_directories(output_root)
    return {
        "benchmark": (
            directories["benchmark"] / "split_indices.npz",
            directories["benchmark"] / "split_indices.json",
            *(
                directories["benchmark"]
                / f"single_action_metrics_seed{seed}.csv"
                for seed in FIXED_SCORE_SEEDS
            ),
        ),
        "scalar_plans": (
            directories["scalar_plans"] / "planner_comparison_summary.csv",
        ),
        "vectors": tuple(
            path
            for seed in FIXED_SCORE_SEEDS
            for path in (
                directories["vectors"]
                / f"score_delta_vectors_seed{seed}.npz",
                directories["vectors"]
                / f"score_delta_vectors_seed{seed}.json",
            )
        ),
        "vector_plans": (
            directories["vector_plans"] / "vector_planner_summary.csv",
        ),
        "confirmation_eval": (
            directories["confirmation_eval"]
            / "confirmation_primary_summary.csv",
        ),
        "test_eval": (
            directories["test_eval"] / "test_primary_summary.csv",
        ),
    }


def load_resume_manifest(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "pipeline_manifest.json"
    commands_path = output_root / "pipeline_commands.txt"
    require_file(manifest_path, "Resume pipeline manifest")
    require_file(commands_path, "Resume pipeline command history")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read the existing pipeline manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise TypeError("Existing pipeline manifest must contain a JSON object.")
    return manifest


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_resume_manifest(
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path_fields = {
        "checkpoint path": (
            manifest.get("checkpoint_path"),
            args.checkpoint,
        ),
        "canonical split path": (
            _nested_value(manifest, "canonical_split_paths", "npz"),
            args.canonical_split,
        ),
        "canonical metadata path": (
            _nested_value(manifest, "canonical_split_paths", "metadata"),
            args.canonical_metadata,
        ),
        "confirmation split path": (
            _nested_value(manifest, "confirmation_split_paths", "npz"),
            args.confirmation_split,
        ),
        "confirmation metadata path": (
            _nested_value(
                manifest, "confirmation_split_paths", "metadata"
            ),
            args.confirmation_metadata,
        ),
    }
    mismatches: list[str] = []
    for label, (actual, expected) in path_fields.items():
        if actual is None:
            mismatches.append(f"{label}: missing from existing manifest")
            continue
        if resolved(Path(str(actual))) != resolved(expected):
            mismatches.append(
                f"{label}: existing={resolved(Path(str(actual)))}, "
                f"requested={resolved(expected)}"
            )

    scalar_fields = {
        "run name": (manifest.get("run_name"), args.run_name),
        "batch size": (manifest.get("batch_size"), args.batch_size),
        "beam width": (
            _nested_value(manifest, "fixed_vector_settings", "beam_width"),
            args.beam_width,
        ),
        "max states per memory bin": (
            _nested_value(
                manifest,
                "fixed_vector_settings",
                "max_states_per_memory_bin",
            ),
            args.max_states_per_memory_bin,
        ),
        "memory quantum KB": (
            _nested_value(
                manifest, "fixed_vector_settings", "memory_quantum_kb"
            ),
            args.memory_quantum_kb,
        ),
    }
    for label, (actual, expected) in scalar_fields.items():
        if actual != expected:
            mismatches.append(
                f"{label}: existing={actual!r}, requested={expected!r}"
            )

    existing_seeds = manifest.get("fixed_score_seeds")
    if existing_seeds != list(args.score_seeds):
        mismatches.append(
            "score seeds: "
            f"existing={existing_seeds!r}, requested={list(args.score_seeds)!r}"
        )
    existing_ratios = manifest.get("fixed_memory_saving_ratios")
    requested_ratios = [float(value) for value in args.memory_saving_ratios]
    ratios_match = (
        isinstance(existing_ratios, list)
        and len(existing_ratios) == len(requested_ratios)
        and all(
            abs(float(actual) - expected) <= 1e-12
            for actual, expected in zip(existing_ratios, requested_ratios)
        )
    )
    if not ratios_match:
        mismatches.append(
            "memory-saving ratios: "
            f"existing={existing_ratios!r}, requested={requested_ratios!r}"
        )

    if mismatches:
        raise ValueError(
            "Resume manifest is incompatible with this invocation:\n"
            + "\n".join(f"- {mismatch}" for mismatch in mismatches)
        )


def completed_stages(manifest: dict[str, Any]) -> tuple[str, ...]:
    statuses = manifest.get("stage_completion_status")
    if not isinstance(statuses, dict):
        raise ValueError(
            "Existing manifest lacks a valid stage_completion_status object."
        )
    return tuple(
        stage
        for stage in STAGES
        if isinstance(statuses.get(stage), dict)
        and statuses[stage].get("status") == "completed"
    )


def preflight_stage_outputs(
    output_root: Path,
    selected_stages: Sequence[str],
    resume_manifest: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    directories = stage_directories(output_root)
    for stage in selected_stages:
        if directories[stage].exists():
            raise FileExistsError(
                "Refusing to run requested stage because its output directory "
                f"already exists: stage={stage}, path={directories[stage]}"
            )

    if resume_manifest is None:
        for path in (
            output_root / "pipeline_manifest.json",
            output_root / "pipeline_commands.txt",
        ):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing pipeline output: {path}"
                )
        return ()

    prior_completed = completed_stages(resume_manifest)
    completed_set = set(prior_completed)
    requested_set = set(selected_stages)
    for stage, directory in directories.items():
        if not directory.exists() or stage in requested_set:
            continue
        if stage not in completed_set:
            raise FileExistsError(
                "Resume found an output directory for a stage that is not "
                f"recorded complete: stage={stage}, path={directory}"
            )
    for stage in prior_completed:
        verify_stage_outputs(stage, output_root)
    return prior_completed


def verify_available_dependencies(
    output_root: Path,
    selected_stages: Sequence[str],
) -> None:
    requirements = stage_required_inputs(output_root)
    expected = stage_expected_outputs(output_root)
    selected = set(selected_stages)
    produced_by_selected = {
        path
        for stage in selected_stages
        for path in expected[stage]
    }
    missing = []
    for stage in selected_stages:
        for required_path in requirements[stage]:
            if required_path in produced_by_selected:
                continue
            if not required_path.is_file():
                missing.append(f"{stage}: {required_path}")
    if missing:
        raise FileNotFoundError(
            "Selected stages require missing prior outputs:\n"
            + "\n".join(missing)
        )


def make_manifest(
    args: argparse.Namespace,
    selected_stages: Sequence[str],
    commands: dict[str, list[str]],
    confirmation_info: dict[str, Any],
    invocation_arguments: Sequence[str],
) -> dict[str, Any]:
    invocation_time = timestamp()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": invocation_time,
        "resume": args.resume,
        "checkpoint_path": str(resolved(args.checkpoint)),
        "run_name": args.run_name,
        "git_commit_hash": git_commit_hash(),
        "canonical_split_paths": {
            "npz": str(resolved(args.canonical_split)),
            "metadata": str(resolved(args.canonical_metadata)),
        },
        "confirmation_split_paths": {
            "npz": str(resolved(args.confirmation_split)),
            "metadata": str(resolved(args.confirmation_metadata)),
        },
        "confirmation_split_provenance": confirmation_info,
        "output_root": str(resolved(args.output_root)),
        "batch_size": args.batch_size,
        "fixed_score_seeds": list(FIXED_SCORE_SEEDS),
        "fixed_memory_saving_ratios": list(FIXED_MEMORY_SAVING_RATIOS),
        "fixed_vector_settings": {
            "beam_width": FIXED_BEAM_WIDTH,
            "max_states_per_memory_bin": FIXED_MAX_STATES_PER_MEMORY_BIN,
            "memory_quantum_kb": FIXED_MEMORY_QUANTUM_KB,
            "objectives": [
                "vector_signed_mean_risk",
                "vector_signed_p95_risk",
            ],
        },
        "scalar_primary_metrics": [
            "weight_rel_l2",
            "activation_rel_mse",
            "output_kl_mean",
            "abs_delta_score_mean",
            "decision_risk_mean",
            "decision_risk_p95",
        ],
        "benchmark_invocation": {
            "native_cli_accepts_canonical_split": False,
            "mode": "explicit canonical-only subprocess adapter",
            "split_generation": False,
            "computation_delegate": (
                "experiments.go_no_go.run_single_action_benchmark.main"
            ),
        },
        "selected_stages": list(selected_stages),
        "commands": [
            {
                "stage": stage,
                "argv": command,
                "command_text": command_text(command),
            }
            for stage, command in commands.items()
        ],
        "stage_completion_status": {
            stage: {
                "status": "pending" if stage in selected_stages else "not_requested"
            }
            for stage in STAGES
        },
        "invocations": [
            {
                "timestamp": invocation_time,
                "resume": args.resume,
                "arguments": list(invocation_arguments),
                "requested_stages": list(selected_stages),
            }
        ],
        "selection_policy": PROTOCOL_STATEMENT,
        "test_evaluation_is_terminal": True,
    }


def update_manifest_for_resume(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    selected_stages: Sequence[str],
    commands: dict[str, list[str]],
    invocation_arguments: Sequence[str],
) -> dict[str, Any]:
    invocation_time = timestamp()
    statuses = manifest.setdefault("stage_completion_status", {})
    for stage in STAGES:
        statuses.setdefault(stage, {"status": "not_requested"})
    for stage in selected_stages:
        previous_status = dict(statuses[stage])
        statuses[stage] = {
            "status": "pending",
            "resume_requested_at": invocation_time,
            "previous_status": previous_status,
        }

    command_records = manifest.setdefault("commands", [])
    command_records.extend(
        {
            "stage": stage,
            "argv": command,
            "command_text": command_text(command),
            "resume_invocation": True,
            "resolved_at": invocation_time,
        }
        for stage, command in commands.items()
    )
    manifest.setdefault("invocations", []).append(
        {
            "timestamp": invocation_time,
            "resume": True,
            "arguments": list(invocation_arguments),
            "requested_stages": list(selected_stages),
        }
    )
    manifest["resume"] = True
    manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    manifest["updated_at"] = invocation_time
    manifest["selected_stages"] = list(selected_stages)
    manifest["last_invocation_arguments"] = list(invocation_arguments)
    return manifest


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_commands(
    path: Path,
    commands: dict[str, list[str]],
    invocation_arguments: Sequence[str],
    append: bool,
) -> None:
    invocation_time = timestamp()
    invocation_text = command_text(
        [
            sys.executable,
            "-m",
            "experiments.go_no_go.run_frozen_repro_pipeline",
            *invocation_arguments,
        ]
    )
    entries = []
    for stage, command in commands.items():
        entries.append(
            "\n".join(
                (
                    f"timestamp: {invocation_time}",
                    f"invocation_arguments: {invocation_text}",
                    f"stage: {stage}",
                    f"command: {command_text(command)}",
                )
            )
        )
    mode = "a" if append else "x"
    with path.open(mode, encoding="utf-8") as file:
        if append:
            file.write("\n")
        file.write("\n\n".join(entries) + "\n")


def print_resume_summary(
    prior_completed: Sequence[str],
    requested_stages: Sequence[str],
) -> None:
    requested_set = set(requested_stages)
    skipped_completed = tuple(
        stage for stage in prior_completed if stage not in requested_set
    )
    print("Resume summary")
    print(f"  prior completed stages: {list(prior_completed)}")
    print(f"  requested stages: {list(requested_stages)}")
    print(f"  pending stages to execute: {list(requested_stages)}")
    print(
        "  stages skipped because already complete: "
        f"{list(skipped_completed)}"
    )
    print("  selected-stage outputs are absent: PASS")


def print_dry_run(
    args: argparse.Namespace,
    selected_stages: Sequence[str],
    commands: dict[str, list[str]],
) -> None:
    print("Frozen reproducibility pipeline dry run")
    print(f"Checkpoint: {resolved(args.checkpoint)}")
    print(f"Run name: {args.run_name}")
    print(f"Canonical split: {resolved(args.canonical_split)}")
    print(f"Canonical metadata: {resolved(args.canonical_metadata)}")
    print(f"Confirmation split: {resolved(args.confirmation_split)}")
    print(f"Confirmation metadata: {resolved(args.confirmation_metadata)}")
    print(f"Output root: {resolved(args.output_root)}")
    print(
        "Pipeline manifest: "
        f"{resolved(args.output_root / 'pipeline_manifest.json')}"
    )
    print(
        "Pipeline commands: "
        f"{resolved(args.output_root / 'pipeline_commands.txt')}"
    )
    print("Stage output directories:")
    for stage, directory in stage_directories(args.output_root).items():
        print(f"  {stage}: {resolved(directory)}")
    print("Output isolation from pilot result directories: PASS")
    print(
        "Benchmark compatibility: native CLI lacks a canonical-split option; "
        "the explicit canonical-only subprocess adapter will be used."
    )
    print(f"Fixed score seeds: {list(FIXED_SCORE_SEEDS)}")
    print(f"Fixed memory-saving ratios: {list(FIXED_MEMORY_SAVING_RATIOS)}")
    print(
        "Fixed vector settings: "
        f"beam_width={FIXED_BEAM_WIDTH}, "
        "max_states_per_memory_bin="
        f"{FIXED_MAX_STATES_PER_MEMORY_BIN}, "
        f"memory_quantum_kb={FIXED_MEMORY_QUANTUM_KB}"
    )
    print(f"Stages: {list(selected_stages)}")
    print("\nResolved subprocess commands")
    for index, (stage, command) in enumerate(commands.items(), start=1):
        print(f"{index}. [{stage}] {command_text(command)}")
    print("\nExpected summary paths")
    for label, path in summary_paths(args.output_root).items():
        print(f"  {label}: {resolved(path)}")
    print(f"\n{PROTOCOL_STATEMENT}")
    print("Dry run complete: no stage executed and no file written.")


def verify_stage_outputs(stage: str, output_root: Path) -> None:
    missing = [
        path
        for path in stage_expected_outputs(output_root)[stage]
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Stage {stage} completed without required outputs: {missing}"
        )


def execute_pipeline(
    args: argparse.Namespace,
    selected_stages: Sequence[str],
    commands: dict[str, list[str]],
    manifest: dict[str, Any],
    invocation_arguments: Sequence[str],
) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "pipeline_manifest.json"
    commands_path = args.output_root / "pipeline_commands.txt"
    write_commands(
        commands_path,
        commands,
        invocation_arguments=invocation_arguments,
        append=args.resume,
    )
    write_json_atomic(manifest_path, manifest)

    for stage in selected_stages:
        status = manifest["stage_completion_status"][stage]
        status.update({"status": "running", "started_at": timestamp()})
        write_json_atomic(manifest_path, manifest)
        print(f"\n=== Stage: {stage} ===", flush=True)
        print(command_text(commands[stage]), flush=True)
        try:
            completed = subprocess.run(commands[stage], check=False)
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode, commands[stage]
                )
            verify_stage_outputs(stage, args.output_root)
        except Exception as error:
            status.update(
                {
                    "status": "failed",
                    "completed_at": timestamp(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            write_json_atomic(manifest_path, manifest)
            raise
        status.update(
            {
                "status": "completed",
                "completed_at": timestamp(),
                "return_code": 0,
            }
        )
        write_json_atomic(manifest_path, manifest)

    print("\nPipeline summaries")
    for label, path in summary_paths(args.output_root).items():
        print(f"{label}: {resolved(path)}")
    print(PROTOCOL_STATEMENT)


def canonical_benchmark_worker(argv: Sequence[str]) -> None:
    args = parse_benchmark_worker_arguments(argv)
    score_seeds = tuple(args.score_seeds)
    arrays, metadata = load_canonical_split(
        args.canonical_split,
        args.canonical_metadata,
        score_seeds,
    )

    import experiments.go_no_go.run_single_action_benchmark as benchmark
    import experiments.go_no_go.splits as split_utilities

    oracle_indices = tuple(
        int(value) for value in arrays["oracle_dataset_indices"].tolist()
    )
    score_indices = {
        score_seed: tuple(
            int(value)
            for value in arrays[
                f"score_seed_{score_seed}_dataset_indices"
            ].tolist()
        )
        for score_seed in score_seeds
    }
    oracle_class_counts = {
        class_id: int(arrays["oracle_class_counts"][class_id])
        for class_id in (0, 1)
    }
    score_class_counts = {
        score_seed: {
            class_id: int(
                arrays[f"score_seed_{score_seed}_class_counts"][class_id]
            )
            for class_id in (0, 1)
        }
        for score_seed in score_seeds
    }

    def build_from_canonical(
        dataset: Any,
        score_size: int,
        oracle_size: int,
        score_seeds: Sequence[int],
        oracle_seed: int,
        batch_size: int,
        num_workers: int = 0,
    ) -> Any:
        if tuple(score_seeds) != tuple(args.score_seeds):
            raise ValueError("Benchmark score seeds differ from canonical seeds.")
        if score_size != len(score_indices[score_seeds[0]]):
            raise ValueError("Benchmark score size differs from canonical split.")
        if oracle_size != len(oracle_indices):
            raise ValueError("Benchmark oracle size differs from canonical split.")
        if oracle_seed != int(metadata.get("oracle_seed")):
            raise ValueError("Benchmark oracle seed differs from canonical metadata.")

        labels = split_utilities.extract_binary_labels(dataset)
        all_indices = [*oracle_indices]
        for indices in score_indices.values():
            all_indices.extend(indices)
        if not all_indices or min(all_indices) < 0 or max(all_indices) >= len(dataset):
            raise IndexError("Canonical dataset index is outside the dataset.")

        for class_id in (0, 1):
            observed = sum(labels[index] == class_id for index in oracle_indices)
            if observed != oracle_class_counts[class_id]:
                raise ValueError("Canonical oracle class counts are invalid.")
        for score_seed, indices in score_indices.items():
            for class_id in (0, 1):
                observed = sum(labels[index] == class_id for index in indices)
                if observed != score_class_counts[score_seed][class_id]:
                    raise ValueError(
                        f"Canonical class counts are invalid for seed {score_seed}."
                    )

        if hasattr(dataset, "source_indices"):
            observed_oracle_sources = [
                int(dataset.source_indices[index]) for index in oracle_indices
            ]
            if observed_oracle_sources != arrays[
                "oracle_source_indices"
            ].astype(np.int64).tolist():
                raise ValueError("Canonical oracle source indices are invalid.")
            for score_seed, indices in score_indices.items():
                observed_sources = [
                    int(dataset.source_indices[index]) for index in indices
                ]
                expected_sources = arrays[
                    f"score_seed_{score_seed}_source_indices"
                ].astype(np.int64).tolist()
                if observed_sources != expected_sources:
                    raise ValueError(
                        "Canonical score source indices are invalid for seed "
                        f"{score_seed}."
                    )

        return split_utilities.CalibrationSplits(
            score_loaders={
                score_seed: split_utilities._build_loader(
                    dataset=dataset,
                    indices=indices,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                for score_seed, indices in score_indices.items()
            },
            oracle_loader=split_utilities._build_loader(
                dataset=dataset,
                indices=oracle_indices,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            score_indices=score_indices,
            oracle_indices=oracle_indices,
            score_class_counts=score_class_counts,
            oracle_class_counts=oracle_class_counts,
        )

    def copy_canonical_split(
        splits: Any,
        dataset: Any,
        output_dir: Path,
        source_split: str,
        oracle_seed: int,
    ) -> tuple[Path, Path]:
        del dataset
        if source_split != "train":
            raise ValueError("Benchmark attempted a non-train split.")
        if oracle_seed != int(metadata.get("oracle_seed")):
            raise ValueError("Benchmark oracle seed changed before split save.")
        if tuple(splits.oracle_indices) != oracle_indices:
            raise ValueError("Benchmark oracle indices changed before split save.")
        for score_seed, indices in score_indices.items():
            if tuple(splits.score_indices[score_seed]) != indices:
                raise ValueError(
                    f"Benchmark score indices changed for seed {score_seed}."
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_npz = output_dir / "split_indices.npz"
        output_json = output_dir / "split_indices.json"
        for path in (output_npz, output_json):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite: {path}")
        shutil.copy2(args.canonical_split, output_npz)
        shutil.copy2(args.canonical_metadata, output_json)
        return output_npz, output_json

    original_builder = benchmark.build_class_balanced_calibration_splits
    original_saver = benchmark.save_split_indices
    original_argv = sys.argv
    benchmark.build_class_balanced_calibration_splits = build_from_canonical
    benchmark.save_split_indices = copy_canonical_split
    benchmark_arguments = [
        str(Path(benchmark.__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--score-size",
        str(len(score_indices[score_seeds[0]])),
        "--oracle-size",
        str(len(oracle_indices)),
        "--oracle-seed",
        str(metadata["oracle_seed"]),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(args.output_dir),
        "--score-seeds",
        *(str(seed) for seed in score_seeds),
    ]
    print(
        "Benchmark canonical-split adapter: using exact supplied indices; "
        "split generation is disabled."
    )
    try:
        sys.argv = benchmark_arguments
        benchmark.main()
    finally:
        sys.argv = original_argv
        benchmark.build_class_balanced_calibration_splits = original_builder
        benchmark.save_split_indices = original_saver


def pipeline_main(argv: Sequence[str] | None = None) -> None:
    invocation_arguments = list(
        argv if argv is not None else sys.argv[1:]
    )
    args = parse_arguments(invocation_arguments)
    selected_stages = validate_fixed_protocol(args)

    args.checkpoint = resolved(args.checkpoint)
    args.canonical_split = resolved(args.canonical_split)
    args.canonical_metadata = resolved(args.canonical_metadata)
    args.confirmation_split = resolved(args.confirmation_split)
    args.confirmation_metadata = resolved(args.confirmation_metadata)
    args.output_root = resolved(args.output_root)

    require_file(args.checkpoint, "Checkpoint")
    load_canonical_split(
        args.canonical_split,
        args.canonical_metadata,
        FIXED_SCORE_SEEDS,
    )
    confirmation_info = validate_confirmation_inputs(
        args.confirmation_split,
        args.confirmation_metadata,
    )
    validate_output_isolation(args.output_root)

    existing_manifest = None
    if args.resume:
        existing_manifest = load_resume_manifest(args.output_root)
        validate_resume_manifest(existing_manifest, args)
    prior_completed = preflight_stage_outputs(
        args.output_root,
        selected_stages,
        resume_manifest=existing_manifest,
    )
    verify_available_dependencies(args.output_root, selected_stages)

    commands = build_stage_commands(args, selected_stages)
    if existing_manifest is None:
        manifest = make_manifest(
            args=args,
            selected_stages=selected_stages,
            commands=commands,
            confirmation_info=confirmation_info,
            invocation_arguments=invocation_arguments,
        )
    else:
        manifest = update_manifest_for_resume(
            manifest=existing_manifest,
            args=args,
            selected_stages=selected_stages,
            commands=commands,
            invocation_arguments=invocation_arguments,
        )
        print_resume_summary(prior_completed, selected_stages)
    if args.dry_run:
        print_dry_run(args, selected_stages, commands)
        return
    execute_pipeline(
        args,
        selected_stages,
        commands,
        manifest,
        invocation_arguments=invocation_arguments,
    )


def main() -> None:
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--_canonical-benchmark-worker":
        canonical_benchmark_worker(arguments[1:])
        return
    pipeline_main(arguments)


if __name__ == "__main__":
    main()
