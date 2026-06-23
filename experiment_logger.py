import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def _to_jsonable(value: Any) -> Any:
    """
    Convert common Python objects into JSON-safe values.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _to_jsonable(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _run_git_command(arguments: list[str]) -> str:
    """
    Run a git command safely.
    Return 'unknown' if Git information is unavailable.
    """

    try:
        output = subprocess.check_output(
            ["git", *arguments],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )

        return output.strip()

    except Exception:
        return "unknown"


def get_git_info() -> dict[str, str]:
    """
    Record the exact code version used for this experiment.
    """

    return {
        "commit": _run_git_command(
            ["rev-parse", "--short", "HEAD"]
        ),
        "branch": _run_git_command(
            ["branch", "--show-current"]
        ),
        "dirty_status": _run_git_command(
            ["status", "--porcelain"]
        ) or "clean",
    }


def get_runtime_info() -> dict[str, Any]:
    """
    Record Python / PyTorch / CUDA environment.
    """

    cuda_available = torch.cuda.is_available()

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if cuda_available
            else "CPU"
        ),
    }


def _markdown_table(values: dict[str, Any]) -> str:
    """
    Convert a dictionary into a Markdown table.
    """

    if not values:
        return "_None_"

    lines = [
        "| Key | Value |",
        "|---|---|",
    ]

    for key, value in values.items():
        safe_value = str(value).replace("|", "\\|")
        lines.append(
            f"| {key} | {safe_value} |"
        )

    return "\n".join(lines)


def record_experiment(
    run_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
) -> None:
    """
    Save one completed experiment automatically.

    Outputs:
    - docs/experiment_log.md
    - results/experiment_history.jsonl
    """

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    record = {
        "timestamp": timestamp,
        "run_name": run_name,
        "command": " ".join(
            [sys.executable, *sys.argv]
        ),
        "git": get_git_info(),
        "runtime": get_runtime_info(),
        "config": _to_jsonable(config),
        "metrics": _to_jsonable(metrics),
        "artifacts": _to_jsonable(artifacts or {}),
    }

    results_dir = Path("results")
    docs_dir = Path("docs")

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    docs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonl_path = (
        results_dir / "experiment_history.jsonl"
    )

    with open(
        jsonl_path,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    markdown_path = docs_dir / "experiment_log.md"

    if not markdown_path.exists():
        markdown_path.write_text(
            "# Experiment Log\n\n"
            "Automatically generated experiment records.\n",
            encoding="utf-8",
        )

    markdown_entry = f"""

## {timestamp} — {run_name}

- Command: `{record["command"]}`
- Git branch: `{record["git"]["branch"]}`
- Git commit: `{record["git"]["commit"]}`
- Git working tree: `{record["git"]["dirty_status"]}`
- Python: `{record["runtime"]["python_version"]}`
- PyTorch: `{record["runtime"]["torch_version"]}`
- CUDA available: `{record["runtime"]["cuda_available"]}`
- CUDA runtime: `{record["runtime"]["cuda_runtime"]}`
- GPU: `{record["runtime"]["gpu_name"]}`

### Configuration

{_markdown_table(record["config"])}

### Metrics

{_markdown_table(record["metrics"])}

### Artifacts

{_markdown_table(record["artifacts"])}

"""

    with open(
        markdown_path,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(markdown_entry)

    print(
        f"Experiment log saved: {markdown_path}"
    )