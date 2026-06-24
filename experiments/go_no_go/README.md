# Single-action Go/No-Go benchmark

This package measures whether layer/action sensitivity rankings are stable on
held-out binary CIFAR-10 examples before those rankings are used by the
existing additive planner. It does not train, alter checkpoints, change the
planner, or introduce another quantizer. Run commands from the repository
root with `python -m experiments.go_no_go.<module>`.

## Repository adapter map

| Benchmark adapter | Existing repository function(s) called |
|---|---|
| `load_reference_model(checkpoint_path, device)` | `model.load_binary_resnet18_checkpoint`; that loader calls `model.build_binary_resnet18(pretrained=False)` and loads `checkpoint["model_state_dict"]` |
| `build_binary_eval_dataset()` | `data.build_dataloaders(ExperimentConfig())["test"].dataset`; the existing path uses `data.BinaryCIFAR10` and the evaluation transform returned by `data.build_transforms` |
| `apply_existing_quantization_inplace(module, action_name)` | `additive_planner.action_to_bits`, then `quantization.apply_fake_quantization_to_module`; the latter calls `quantization.fake_quantize_weight_per_output_channel` |
| `get_checkpoint_default_path()` | `config.ExperimentConfig().checkpoint_path`, currently `checkpoints/resnet18_binary_best.pt` |

The reused quantizer preserves the current semantics: Conv2d/Linear weights
only; FP16 cast-and-restore for `fp16`; symmetric per-output-channel fake
quantization for `int8` and `int4`; float weights after dequantization.

## Benchmark flow

`run_single_action_benchmark.py` deterministically partitions the existing
binary test dataset into disjoint `ranking` and `holdout` subsets. For every
requested layer/action it deep-copies the reference model, applies the adapter
to exactly one module, and delegates model comparison to the existing
`metrics.compare_binary_models` function.

`analyze_rankings.py` compares ranking and holdout layer orderings with
Spearman correlation and top-k overlap and emits an explicit `GO`/`NO_GO`
decision under configurable thresholds. `planner_eval.py` is an offline check
for an existing additive-planner allocation CSV: it compares saved
`risk_value` entries with held-out single-action measurements. It never invokes
or changes the planner.

Generated CSV files default to `results/go_no_go/`. The scripts refuse to
overwrite an existing output file, preserving prior results. The repository's
older scripts use `ExperimentConfig.result_dir` (`results/`) for flat CSV
artifacts, while `experiment_logger.record_experiment` appends to
`results/experiment_history.jsonl` and `docs/experiment_log.md`; this isolated
benchmark does not call that logger.

Example commands (not run during scaffold creation):

```powershell
python -m experiments.go_no_go.run_single_action_benchmark
python -m experiments.go_no_go.analyze_rankings
python -m experiments.go_no_go.planner_eval --allocations path/to/allocations.csv
```

