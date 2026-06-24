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
| `build_binary_eval_dataset(source_split)` | `data.BinaryCIFAR10` with the evaluation transform returned by `data.build_transforms`; `source_split="train"` builds the no-augmentation calibration pool and `source_split="test"` is reserved for final evaluation |
| `apply_existing_quantization_inplace(module, action_name)` | `additive_planner.action_to_bits`, then `quantization.apply_fake_quantization_to_module`; the latter calls `quantization.fake_quantize_weight_per_output_channel` |
| `get_checkpoint_default_path()` | `config.ExperimentConfig().checkpoint_path`, currently `checkpoints/resnet18_binary_best.pt` |

The reused quantizer preserves the current semantics: Conv2d/Linear weights
only; FP16 cast-and-restore for `fp16`; symmetric per-output-channel fake
quantization for `int8` and `int4`; float weights after dequantization.

## Benchmark flow

`run_single_action_benchmark.py` uses the full binary TRAIN pool with the
existing evaluation transform (resize, tensor conversion, and ImageNet
normalization; no random augmentation). It builds a fixed, class-balanced
oracle set with `oracle_seed=2026`, removes those samples, and builds each
class-balanced score set from the remainder with its own score seed. For every
Conv2d/Linear layer and `fp16`/`int8`/`int4` action it deep-copies the reference
model and applies the adapter to exactly one copied module.

The exact split indices are stored in `split_indices.npz` and
`split_indices.json`. Candidate rows are written to
`single_action_metrics_seed{seed}.csv`. `whole_model_saving` and
`quantizable_weight_saving` are theoretical storage-saving ratios relative to
the FP32 whole-model parameter storage and FP32 quantizable-weight storage,
respectively; they are not latency claims.

`analyze_rankings.py` and `planner_eval.py` remain offline follow-on utilities;
neither is invoked by the single-action benchmark, and neither changes the
planner.

Generated CSV files default to `results/go_no_go/`. The scripts refuse to
overwrite an existing output file, preserving prior results. The repository's
older scripts use `ExperimentConfig.result_dir` (`results/`) for flat CSV
artifacts, while `experiment_logger.record_experiment` appends to
`results/experiment_history.jsonl` and `docs/experiment_log.md`; this isolated
benchmark does not call that logger.

Example commands (not run during scaffold creation):

```powershell
python -m experiments.go_no_go.run_single_action_benchmark `
  --score-size 512 --oracle-size 2000 `
  --score-seeds 0 --oracle-seed 2026
python -m experiments.go_no_go.analyze_rankings
python -m experiments.go_no_go.planner_eval --allocations path/to/allocations.csv
```
