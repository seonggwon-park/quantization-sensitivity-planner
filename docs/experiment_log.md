# Experiment Log

Automatically generated experiment records.


## 2026-06-23T20:02:56+09:00 — train_binary_resnet18

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe train.py --epochs 10`
- Git branch: `main`
- Git commit: `8110751`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| data_dir | data |
| checkpoint_dir | checkpoints |
| result_dir | results |
| figure_dir | results\figures |
| class_ids | [0, 1] |
| class_names | ['airplane', 'automobile'] |
| image_size | 224 |
| batch_size | 64 |
| num_workers | 0 |
| validation_fraction | 0.1 |
| seed | 42 |
| epochs | 10 |
| learning_rate | 0.0001 |
| weight_decay | 0.0001 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| model | ImageNet-pretrained ResNet-18 |
| quantization_target | FP32 training baseline |

### Metrics

| Key | Value |
|---|---|
| best_validation_accuracy | 0.996 |
| test_loss | 0.01517276841099374 |
| test_accuracy | 0.996 |
| test_num_samples | 2000 |

### Artifacts

| Key | Value |
|---|---|
| checkpoint | checkpoints\resnet18_binary_best.pt |



## 2026-06-23T20:13:12+09:00 — uniform_fake_quantization_baseline

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_baselines.py --max-samples 256`
- Git branch: `main`
- Git commit: `be4ab74`
- Git working tree: `M run_baselines.py`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| max_samples | 256 |
| actions | ['fp32', 'fp16', 'int8', 'int4'] |
| quantization | weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| fp32_quantized_accuracy | 0.98828125 |
| fp32_flip_rate | 0.0 |
| fp32_mean_margin_risk | 0.0 |
| fp32_memory_saving_ratio | 0.0 |
| fp16_quantized_accuracy | 0.98828125 |
| fp16_flip_rate | 0.0 |
| fp16_mean_margin_risk | 0.0012563599739223719 |
| fp16_memory_saving_ratio | 0.4995704778637299 |
| int8_quantized_accuracy | 0.98828125 |
| int8_flip_rate | 0.0 |
| int8_mean_margin_risk | 0.03523615002632141 |
| int8_memory_saving_ratio | 0.7493557167955949 |
| int4_quantized_accuracy | 0.88671875 |
| int4_flip_rate | 0.1171875 |
| int4_mean_margin_risk | 0.648716151714325 |
| int4_memory_saving_ratio | 0.8742483362615274 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\uniform_baselines.csv |



## 2026-06-23T20:15:13+09:00 — uniform_fake_quantization_baseline

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_baselines.py --max-samples 2000`
- Git branch: `main`
- Git commit: `e21d77a`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| max_samples | 2000 |
| actions | ['fp32', 'fp16', 'int8', 'int4'] |
| quantization | weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| fp32_quantized_accuracy | 0.996 |
| fp32_flip_rate | 0.0 |
| fp32_mean_margin_risk | 0.0 |
| fp32_memory_saving_ratio | 0.0 |
| fp16_quantized_accuracy | 0.996 |
| fp16_flip_rate | 0.0 |
| fp16_mean_margin_risk | 0.0013704068260267377 |
| fp16_memory_saving_ratio | 0.4995704778637299 |
| int8_quantized_accuracy | 0.9955 |
| int8_flip_rate | 0.0005 |
| int8_mean_margin_risk | 0.03685024008154869 |
| int8_memory_saving_ratio | 0.7493557167955949 |
| int4_quantized_accuracy | 0.846 |
| int4_flip_rate | 0.153 |
| int4_mean_margin_risk | 0.7086520195007324 |
| int4_memory_saving_ratio | 0.8742483362615274 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\uniform_baselines.csv |



## 2026-06-23T20:19:29+09:00 — single_layer_quantization_sweep

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe sweep_single_layer.py --max-samples 128 --layers conv1 fc`
- Git branch: `main`
- Git commit: `24076f3`
- Git working tree: `M sweep_single_layer.py`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| max_samples | 128 |
| requested_bits | [16, 8, 4] |
| num_layers | 2 |
| quantization | single-layer weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| num_experiments | 6 |
| highest_p95_risk_layer | conv1 |
| highest_p95_risk_action | int4 |
| highest_p95_margin_risk | 1.2771583795547485 |
| highest_flip_rate_layer | conv1 |
| highest_flip_rate_action | int4 |
| highest_flip_rate | 0.078125 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\single_layer_sweep.csv |



## 2026-06-23T20:22:02+09:00 — single_layer_quantization_sweep

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe sweep_single_layer.py --max-samples 500`
- Git branch: `main`
- Git commit: `0b77cee`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| max_samples | 500 |
| requested_bits | [16, 8, 4] |
| num_layers | 21 |
| quantization | single-layer weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| num_experiments | 63 |
| highest_p95_risk_layer | conv1 |
| highest_p95_risk_action | int4 |
| highest_p95_margin_risk | 1.214812159538269 |
| highest_flip_rate_layer | conv1 |
| highest_flip_rate_action | int4 |
| highest_flip_rate | 0.094 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\single_layer_sweep.csv |



## 2026-06-23T20:30:09+09:00 — single_layer_quantization_sweep

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe sweep_single_layer.py --max-samples 2000`
- Git branch: `main`
- Git commit: `777f7dd`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| max_samples | 2000 |
| requested_bits | [16, 8, 4] |
| num_layers | 21 |
| quantization | single-layer weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| num_experiments | 63 |
| highest_p95_risk_layer | conv1 |
| highest_p95_risk_action | int4 |
| highest_p95_margin_risk | 1.2427469491958618 |
| highest_flip_rate_layer | conv1 |
| highest_flip_rate_action | int4 |
| highest_flip_rate | 0.098 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\single_layer_sweep.csv |



## 2026-06-23T20:50:34+09:00 — single_layer_quantization_sweep

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe sweep_single_layer.py --split validation --max-samples 128 --layers conv1 fc --output results/validation_sweep_smoke.csv`
- Git branch: `main`
- Git commit: `8d16f04`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| data_split | validation |
| max_samples | 128 |
| requested_bits | [16, 8, 4] |
| num_layers | 2 |
| quantization | single-layer weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| num_experiments | 6 |
| highest_p95_risk_layer | conv1 |
| highest_p95_risk_action | int4 |
| highest_p95_margin_risk | 1.4809699058532715 |
| highest_flip_rate_layer | conv1 |
| highest_flip_rate_action | int4 |
| highest_flip_rate | 0.0859375 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\validation_sweep_smoke.csv |



## 2026-06-23T20:53:14+09:00 — single_layer_quantization_sweep

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe sweep_single_layer.py --split validation --max-samples 1000 --output results/validation_single_layer_sweep.csv`
- Git branch: `main`
- Git commit: `8d16f04`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| data_split | validation |
| max_samples | 1000 |
| requested_bits | [16, 8, 4] |
| num_layers | 21 |
| quantization | single-layer weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| num_experiments | 63 |
| highest_p95_risk_layer | conv1 |
| highest_p95_risk_action | int4 |
| highest_p95_margin_risk | 1.1971958875656128 |
| highest_flip_rate_layer | conv1 |
| highest_flip_rate_action | int4 |
| highest_flip_rate | 0.082 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\validation_single_layer_sweep.csv |



## 2026-06-23T20:53:45+09:00 — oracle_guided_mixed_precision_controls

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_oracle_guided_controls.py --ranking-csv results/validation_single_layer_sweep.csv --evaluation-split test --max-samples 128 --include-bottom-controls`
- Git branch: `main`
- Git commit: `1ffab6c`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| ranking_csv | results\validation_single_layer_sweep.csv |
| ranking_action | int4 |
| risk_metric | p95_margin_risk |
| evaluation_split | test |
| max_samples | 128 |
| protect_counts | [0, 1, 2, 4] |
| default_bits | 4 |
| protected_bits | 32 |
| include_bottom_controls | True |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| uniform_int4_flip_rate | 0.09375 |
| uniform_int4_quantized_accuracy | 0.8984375 |
| uniform_int4_memory_saving_ratio | 0.8742483362615274 |
| uniform_int4_mean_margin_risk | 0.6231024265289307 |
| oracle_top1_fp32_flip_rate | 0.0234375 |
| oracle_top1_fp32_quantized_accuracy | 0.96875 |
| oracle_top1_fp32_memory_saving_ratio | 0.873511859230539 |
| oracle_top1_fp32_mean_margin_risk | 0.26023995876312256 |
| oracle_bottom1_fp32_flip_rate | 0.09375 |
| oracle_bottom1_fp32_quantized_accuracy | 0.8984375 |
| oracle_bottom1_fp32_memory_saving_ratio | 0.8741681754962497 |
| oracle_bottom1_fp32_mean_margin_risk | 0.620140016078949 |
| oracle_top2_fp32_flip_rate | 0.015625 |
| oracle_top2_fp32_quantized_accuracy | 0.9765625 |
| oracle_top2_fp32_memory_saving_ratio | 0.8706260716805436 |
| oracle_top2_fp32_mean_margin_risk | 0.23297438025474548 |
| oracle_bottom2_fp32_flip_rate | 0.09375 |
| oracle_bottom2_fp32_quantized_accuracy | 0.8984375 |
| oracle_bottom2_fp32_memory_saving_ratio | 0.6894777722965468 |
| oracle_bottom2_fp32_mean_margin_risk | 0.6147286891937256 |
| oracle_top4_fp32_flip_rate | 0.0078125 |
| oracle_top4_fp32_quantized_accuracy | 0.984375 |
| oracle_top4_fp32_memory_saving_ratio | 0.867098998008327 |
| oracle_top4_fp32_mean_margin_risk | 0.14160706102848053 |
| oracle_bottom4_fp32_flip_rate | 0.09375 |
| oracle_bottom4_fp32_quantized_accuracy | 0.8984375 |
| oracle_bottom4_fp32_memory_saving_ratio | 0.5022222246079593 |
| oracle_bottom4_fp32_mean_margin_risk | 0.6146207451820374 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\oracle_guided_mixed_controls.csv |



## 2026-06-23T20:54:19+09:00 — oracle_guided_mixed_precision_controls

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_oracle_guided_controls.py --ranking-csv results/validation_single_layer_sweep.csv --evaluation-split test --max-samples 2000 --include-bottom-controls`
- Git branch: `main`
- Git commit: `1ffab6c`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| ranking_csv | results\validation_single_layer_sweep.csv |
| ranking_action | int4 |
| risk_metric | p95_margin_risk |
| evaluation_split | test |
| max_samples | 2000 |
| protect_counts | [0, 1, 2, 4] |
| default_bits | 4 |
| protected_bits | 32 |
| include_bottom_controls | True |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| uniform_int4_flip_rate | 0.153 |
| uniform_int4_quantized_accuracy | 0.846 |
| uniform_int4_memory_saving_ratio | 0.8742483362615274 |
| uniform_int4_mean_margin_risk | 0.7086520195007324 |
| oracle_top1_fp32_flip_rate | 0.0125 |
| oracle_top1_fp32_quantized_accuracy | 0.9845 |
| oracle_top1_fp32_memory_saving_ratio | 0.873511859230539 |
| oracle_top1_fp32_mean_margin_risk | 0.24971544742584229 |
| oracle_bottom1_fp32_flip_rate | 0.1505 |
| oracle_bottom1_fp32_quantized_accuracy | 0.8485 |
| oracle_bottom1_fp32_memory_saving_ratio | 0.8741681754962497 |
| oracle_bottom1_fp32_mean_margin_risk | 0.7063938975334167 |
| oracle_top2_fp32_flip_rate | 0.007 |
| oracle_top2_fp32_quantized_accuracy | 0.99 |
| oracle_top2_fp32_memory_saving_ratio | 0.8706260716805436 |
| oracle_top2_fp32_mean_margin_risk | 0.24099984765052795 |
| oracle_bottom2_fp32_flip_rate | 0.144 |
| oracle_bottom2_fp32_quantized_accuracy | 0.855 |
| oracle_bottom2_fp32_memory_saving_ratio | 0.6894777722965468 |
| oracle_bottom2_fp32_mean_margin_risk | 0.701242983341217 |
| oracle_top4_fp32_flip_rate | 0.0055 |
| oracle_top4_fp32_quantized_accuracy | 0.9915 |
| oracle_top4_fp32_memory_saving_ratio | 0.867098998008327 |
| oracle_top4_fp32_mean_margin_risk | 0.15953198075294495 |
| oracle_bottom4_fp32_flip_rate | 0.1445 |
| oracle_bottom4_fp32_quantized_accuracy | 0.8545 |
| oracle_bottom4_fp32_memory_saving_ratio | 0.5022222246079593 |
| oracle_bottom4_fp32_mean_margin_risk | 0.7014716863632202 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\oracle_guided_mixed_controls.csv |



## 2026-06-23T21:29:53+09:00 — weight_l2_ranked_mixed_precision_controls

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_ranked_controls.py --ranking-csv results/validation_single_layer_sweep.csv --ranking-action int4 --risk-metric relative_l2_weight_error --ranking-label weight_l2 --evaluation-split test --max-samples 128 --include-bottom-controls`
- Git branch: `main`
- Git commit: `c90b3ba`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| ranking_label | weight_l2 |
| ranking_csv | results\validation_single_layer_sweep.csv |
| ranking_action | int4 |
| risk_metric | relative_l2_weight_error |
| evaluation_split | test |
| max_samples | 128 |
| protect_counts | [0, 1, 2, 4] |
| default_bits | 4 |
| protected_bits | 32 |
| include_bottom_controls | True |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| uniform_int4_flip_rate | 0.09375 |
| uniform_int4_quantized_accuracy | 0.8984375 |
| uniform_int4_memory_saving_ratio | 0.8742483362615274 |
| uniform_int4_mean_margin_risk | 0.6231024265289307 |
| weight_l2_top1_fp32_flip_rate | 0.125 |
| weight_l2_top1_fp32_quantized_accuracy | 0.8671875 |
| weight_l2_top1_fp32_memory_saving_ratio | 0.871362548711532 |
| weight_l2_top1_fp32_mean_margin_risk | 0.6351816058158875 |
| weight_l2_bottom1_fp32_flip_rate | 0.09375 |
| weight_l2_bottom1_fp32_quantized_accuracy | 0.8984375 |
| weight_l2_bottom1_fp32_memory_saving_ratio | 0.8741681754962497 |
| weight_l2_bottom1_fp32_mean_margin_risk | 0.620140016078949 |
| weight_l2_top2_fp32_flip_rate | 0.078125 |
| weight_l2_top2_fp32_quantized_accuracy | 0.9140625 |
| weight_l2_top2_fp32_memory_saving_ratio | 0.8251899479116063 |
| weight_l2_top2_fp32_mean_margin_risk | 0.6134971380233765 |
| weight_l2_bottom2_fp32_flip_rate | 0.015625 |
| weight_l2_bottom2_fp32_quantized_accuracy | 0.9765625 |
| weight_l2_bottom2_fp32_memory_saving_ratio | 0.8734316984652613 |
| weight_l2_bottom2_fp32_mean_margin_risk | 0.2616928517818451 |
| weight_l2_top4_fp32_flip_rate | 0.0859375 |
| weight_l2_top4_fp32_quantized_accuracy | 0.90625 |
| weight_l2_top4_fp32_memory_saving_ratio | 0.6289563945119221 |
| weight_l2_top4_fp32_mean_margin_risk | 0.5865545868873596 |
| weight_l2_bottom4_fp32_flip_rate | 0.015625 |
| weight_l2_bottom4_fp32_quantized_accuracy | 0.9765625 |
| weight_l2_bottom4_fp32_memory_saving_ratio | 0.8702252678541553 |
| weight_l2_bottom4_fp32_mean_margin_risk | 0.27372318506240845 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\weight_l2_mixed_controls.csv |



## 2026-06-23T21:30:24+09:00 — weight_l2_ranked_mixed_precision_controls

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_ranked_controls.py --ranking-csv results/validation_single_layer_sweep.csv --ranking-action int4 --risk-metric relative_l2_weight_error --ranking-label weight_l2 --evaluation-split test --max-samples 2000 --include-bottom-controls`
- Git branch: `main`
- Git commit: `c90b3ba`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| ranking_label | weight_l2 |
| ranking_csv | results\validation_single_layer_sweep.csv |
| ranking_action | int4 |
| risk_metric | relative_l2_weight_error |
| evaluation_split | test |
| max_samples | 2000 |
| protect_counts | [0, 1, 2, 4] |
| default_bits | 4 |
| protected_bits | 32 |
| include_bottom_controls | True |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| uniform_int4_flip_rate | 0.153 |
| uniform_int4_quantized_accuracy | 0.846 |
| uniform_int4_memory_saving_ratio | 0.8742483362615274 |
| uniform_int4_mean_margin_risk | 0.7086520195007324 |
| weight_l2_top1_fp32_flip_rate | 0.134 |
| weight_l2_top1_fp32_quantized_accuracy | 0.865 |
| weight_l2_top1_fp32_memory_saving_ratio | 0.871362548711532 |
| weight_l2_top1_fp32_mean_margin_risk | 0.6763155460357666 |
| weight_l2_bottom1_fp32_flip_rate | 0.1505 |
| weight_l2_bottom1_fp32_quantized_accuracy | 0.8485 |
| weight_l2_bottom1_fp32_memory_saving_ratio | 0.8741681754962497 |
| weight_l2_bottom1_fp32_mean_margin_risk | 0.7063938975334167 |
| weight_l2_top2_fp32_flip_rate | 0.1195 |
| weight_l2_top2_fp32_quantized_accuracy | 0.8795 |
| weight_l2_top2_fp32_memory_saving_ratio | 0.8251899479116063 |
| weight_l2_top2_fp32_mean_margin_risk | 0.6574708819389343 |
| weight_l2_bottom2_fp32_flip_rate | 0.012 |
| weight_l2_bottom2_fp32_quantized_accuracy | 0.985 |
| weight_l2_bottom2_fp32_memory_saving_ratio | 0.8734316984652613 |
| weight_l2_bottom2_fp32_mean_margin_risk | 0.25149592757225037 |
| weight_l2_top4_fp32_flip_rate | 0.1075 |
| weight_l2_top4_fp32_quantized_accuracy | 0.8915 |
| weight_l2_top4_fp32_memory_saving_ratio | 0.6289563945119221 |
| weight_l2_top4_fp32_mean_margin_risk | 0.6378053426742554 |
| weight_l2_bottom4_fp32_flip_rate | 0.008 |
| weight_l2_bottom4_fp32_quantized_accuracy | 0.988 |
| weight_l2_bottom4_fp32_memory_saving_ratio | 0.8702252678541553 |
| weight_l2_bottom4_fp32_mean_margin_risk | 0.2593787610530853 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\weight_l2_mixed_controls.csv |



## 2026-06-23T21:43:37+09:00 — local_activation_proxy

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_activation_proxy.py --split validation --max-samples 128 --layers conv1 fc --bits 32 4 --output results/activation_proxy_smoke.csv`
- Git branch: `main`
- Git commit: `45e5b88`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| data_split | validation |
| max_samples | 128 |
| requested_bits | [32, 4] |
| num_layers | 2 |
| proxy | forward-only local module-output reconstruction error |
| quantization | weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| num_layer_action_pairs | 4 |
| proxy_runtime_seconds | 2.6807046998292208 |
| highest_p95_layer | conv1 |
| highest_p95_action | int4 |
| highest_p95_relative_activation_error | 0.23810240626335144 |
| highest_mean_layer | conv1 |
| highest_mean_action | int4 |
| highest_mean_relative_activation_error | 0.20688819885253906 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\activation_proxy_smoke.csv |



## 2026-06-23T21:44:20+09:00 — local_activation_proxy

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_activation_proxy.py --split validation --max-samples 1000 --bits 4 --output results/validation_local_activation_proxy.csv`
- Git branch: `main`
- Git commit: `b9a942e`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| data_split | validation |
| max_samples | 1000 |
| requested_bits | [4] |
| num_layers | 21 |
| proxy | forward-only local module-output reconstruction error |
| quantization | weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| num_layer_action_pairs | 21 |
| proxy_runtime_seconds | 3.4372811000794172 |
| highest_p95_layer | conv1 |
| highest_p95_action | int4 |
| highest_p95_relative_activation_error | 0.23875348269939423 |
| highest_mean_layer | conv1 |
| highest_mean_action | int4 |
| highest_mean_relative_activation_error | 0.20727916061878204 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\validation_local_activation_proxy.csv |



## 2026-06-23T21:47:46+09:00 — local_activation_p95_ranked_mixed_precision_controls

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_ranked_controls.py --ranking-csv results/validation_local_activation_proxy.csv --ranking-action int4 --risk-metric p95_relative_activation_error --ranking-label local_activation_p95 --evaluation-split test --max-samples 128`
- Git branch: `main`
- Git commit: `24b4911`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| ranking_label | local_activation_p95 |
| ranking_csv | results\validation_local_activation_proxy.csv |
| ranking_action | int4 |
| risk_metric | p95_relative_activation_error |
| evaluation_split | test |
| max_samples | 128 |
| protect_counts | [0, 1, 2, 4] |
| default_bits | 4 |
| protected_bits | 32 |
| include_bottom_controls | False |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| uniform_int4_flip_rate | 0.09375 |
| uniform_int4_quantized_accuracy | 0.8984375 |
| uniform_int4_memory_saving_ratio | 0.8742483362615274 |
| uniform_int4_mean_margin_risk | 0.6231024265289307 |
| local_activation_p95_top1_fp32_flip_rate | 0.0234375 |
| local_activation_p95_top1_fp32_quantized_accuracy | 0.96875 |
| local_activation_p95_top1_fp32_memory_saving_ratio | 0.873511859230539 |
| local_activation_p95_top1_fp32_mean_margin_risk | 0.26023995876312256 |
| local_activation_p95_top2_fp32_flip_rate | 0.015625 |
| local_activation_p95_top2_fp32_quantized_accuracy | 0.9765625 |
| local_activation_p95_top2_fp32_memory_saving_ratio | 0.8677402841305483 |
| local_activation_p95_top2_fp32_mean_margin_risk | 0.2503303289413452 |
| local_activation_p95_top4_fp32_flip_rate | 0.0078125 |
| local_activation_p95_top4_fp32_quantized_accuracy | 0.984375 |
| local_activation_p95_top4_fp32_memory_saving_ratio | 0.8642132104583317 |
| local_activation_p95_top4_fp32_mean_margin_risk | 0.19005124270915985 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\local_activation_p95_mixed_controls.csv |



## 2026-06-23T21:48:09+09:00 — local_activation_p95_ranked_mixed_precision_controls

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_ranked_controls.py --ranking-csv results/validation_local_activation_proxy.csv --ranking-action int4 --risk-metric p95_relative_activation_error --ranking-label local_activation_p95 --evaluation-split test --max-samples 2000`
- Git branch: `main`
- Git commit: `24b4911`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| ranking_label | local_activation_p95 |
| ranking_csv | results\validation_local_activation_proxy.csv |
| ranking_action | int4 |
| risk_metric | p95_relative_activation_error |
| evaluation_split | test |
| max_samples | 2000 |
| protect_counts | [0, 1, 2, 4] |
| default_bits | 4 |
| protected_bits | 32 |
| include_bottom_controls | False |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| uniform_int4_flip_rate | 0.153 |
| uniform_int4_quantized_accuracy | 0.846 |
| uniform_int4_memory_saving_ratio | 0.8742483362615274 |
| uniform_int4_mean_margin_risk | 0.7086520195007324 |
| local_activation_p95_top1_fp32_flip_rate | 0.0125 |
| local_activation_p95_top1_fp32_quantized_accuracy | 0.9845 |
| local_activation_p95_top1_fp32_memory_saving_ratio | 0.873511859230539 |
| local_activation_p95_top1_fp32_mean_margin_risk | 0.24971544742584229 |
| local_activation_p95_top2_fp32_flip_rate | 0.01 |
| local_activation_p95_top2_fp32_quantized_accuracy | 0.988 |
| local_activation_p95_top2_fp32_memory_saving_ratio | 0.8677402841305483 |
| local_activation_p95_top2_fp32_mean_margin_risk | 0.22679638862609863 |
| local_activation_p95_top4_fp32_flip_rate | 0.0065 |
| local_activation_p95_top4_fp32_quantized_accuracy | 0.9905 |
| local_activation_p95_top4_fp32_memory_saving_ratio | 0.8642132104583317 |
| local_activation_p95_top4_fp32_mean_margin_risk | 0.18606945872306824 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\local_activation_p95_mixed_controls.csv |



## 2026-06-23T21:50:36+09:00 — local_activation_proxy

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_activation_proxy.py --split validation --max-samples 1000 --bits 16 8 4 --output results/validation_local_activation_proxy_all_bits.csv`
- Git branch: `main`
- Git commit: `5af2a23`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| data_split | validation |
| max_samples | 1000 |
| requested_bits | [16, 8, 4] |
| num_layers | 21 |
| proxy | forward-only local module-output reconstruction error |
| quantization | weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| num_layer_action_pairs | 63 |
| proxy_runtime_seconds | 3.5649654995650053 |
| highest_p95_layer | conv1 |
| highest_p95_action | int4 |
| highest_p95_relative_activation_error | 0.23875348269939423 |
| highest_mean_layer | conv1 |
| highest_mean_action | int4 |
| highest_mean_relative_activation_error | 0.20727916061878204 |

### Artifacts

| Key | Value |
|---|---|
| csv | results\validation_local_activation_proxy_all_bits.csv |



## 2026-06-23T21:56:30+09:00 — local_activation_p95_additive_mixed_precision_planner

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_additive_planner.py --risk-csv results/validation_local_activation_proxy_all_bits.csv --risk-metric p95_relative_activation_error --risk-label local_activation_p95 --memory-saving-ratios 0.873 0.850 --evaluation-split test --max-samples 128`
- Git branch: `main`
- Git commit: `75b73aa`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| risk_label | local_activation_p95 |
| risk_csv | results\validation_local_activation_proxy_all_bits.csv |
| risk_metric | p95_relative_activation_error |
| evaluation_split | test |
| max_samples | 128 |
| requested_memory_saving_ratios | [0.873, 0.85] |
| memory_quantum_kb | 1 |
| planner | multiple-choice knapsack dynamic programming with additive layer-action risk |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| save_0_873_flip_rate | 0.0078125 |
| save_0_873_quantized_accuracy | 0.984375 |
| save_0_873_actual_memory_saving_ratio | 0.8730323260811101 |
| save_0_873_planner_objective | 2.0864445240586056 |
| save_0_850_flip_rate | 0.0 |
| save_0_850_quantized_accuracy | 0.9921875 |
| save_0_850_actual_memory_saving_ratio | 0.8500247550041878 |
| save_0_850_planner_objective | 0.6974130095127287 |

### Artifacts

| Key | Value |
|---|---|
| results_csv | results\local_activation_p95_additive_planner_results.csv |
| allocations_csv | results\local_activation_p95_additive_planner_allocations.csv |



## 2026-06-23T21:57:55+09:00 — local_activation_p95_additive_mixed_precision_planner

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_additive_planner.py --risk-csv results/validation_local_activation_proxy_all_bits.csv --risk-metric p95_relative_activation_error --risk-label local_activation_p95 --memory-saving-ratios 0.873 0.870 0.850 0.800 --evaluation-split test --max-samples 2000`
- Git branch: `main`
- Git commit: `0ad3c77`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| risk_label | local_activation_p95 |
| risk_csv | results\validation_local_activation_proxy_all_bits.csv |
| risk_metric | p95_relative_activation_error |
| evaluation_split | test |
| max_samples | 2000 |
| requested_memory_saving_ratios | [0.873, 0.87, 0.85, 0.8] |
| memory_quantum_kb | 1 |
| planner | multiple-choice knapsack dynamic programming with additive layer-action risk |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| save_0_873_flip_rate | 0.0055 |
| save_0_873_quantized_accuracy | 0.9915 |
| save_0_873_actual_memory_saving_ratio | 0.8730323260811101 |
| save_0_873_planner_objective | 2.0864445240586056 |
| save_0_870_flip_rate | 0.0045 |
| save_0_870_quantized_accuracy | 0.9935 |
| save_0_870_actual_memory_saving_ratio | 0.870077829303734 |
| save_0_870_planner_objective | 1.513387110549956 |
| save_0_850_flip_rate | 0.0025 |
| save_0_850_quantized_accuracy | 0.9945 |
| save_0_850_actual_memory_saving_ratio | 0.8500247550041878 |
| save_0_850_planner_objective | 0.6974130095127287 |
| save_0_800_flip_rate | 0.0005 |
| save_0_800_quantized_accuracy | 0.9955 |
| save_0_800_actual_memory_saving_ratio | 0.8000588322759449 |
| save_0_800_planner_objective | 0.2951178804107705 |

### Artifacts

| Key | Value |
|---|---|
| results_csv | results\local_activation_p95_additive_planner_results.csv |
| allocations_csv | results\local_activation_p95_additive_planner_allocations.csv |



## 2026-06-23T22:00:29+09:00 — weight_l2_additive_mixed_precision_planner

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_additive_planner.py --risk-csv results/validation_single_layer_sweep.csv --risk-metric relative_l2_weight_error --risk-label weight_l2 --memory-saving-ratios 0.873 0.870 0.850 0.800 --evaluation-split test --max-samples 2000`
- Git branch: `main`
- Git commit: `6c5dfba`
- Git working tree: `clean`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| risk_label | weight_l2 |
| risk_csv | results\validation_single_layer_sweep.csv |
| risk_metric | relative_l2_weight_error |
| evaluation_split | test |
| max_samples | 2000 |
| requested_memory_saving_ratios | [0.873, 0.87, 0.85, 0.8] |
| memory_quantum_kb | 1 |
| planner | multiple-choice knapsack dynamic programming with additive layer-action risk |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| save_0_873_flip_rate | 0.0055 |
| save_0_873_quantized_accuracy | 0.9915 |
| save_0_873_actual_memory_saving_ratio | 0.8730323260811101 |
| save_0_873_planner_objective | 3.741347243703785 |
| save_0_870_flip_rate | 0.0045 |
| save_0_870_quantized_accuracy | 0.9925 |
| save_0_870_actual_memory_saving_ratio | 0.8701236354553212 |
| save_0_870_planner_objective | 2.926059558216365 |
| save_0_850_flip_rate | 0.0025 |
| save_0_850_quantized_accuracy | 0.9945 |
| save_0_850_actual_memory_saving_ratio | 0.8500333436576104 |
| save_0_850_planner_objective | 1.5434066644374973 |
| save_0_800_flip_rate | 0.0005 |
| save_0_800_quantized_accuracy | 0.9955 |
| save_0_800_actual_memory_saving_ratio | 0.8000588322759449 |
| save_0_800_planner_objective | 0.6551589915616195 |

### Artifacts

| Key | Value |
|---|---|
| results_csv | results\weight_l2_additive_planner_results.csv |
| allocations_csv | results\weight_l2_additive_planner_allocations.csv |



## 2026-06-23T22:00:52+09:00 — empirical_margin_p95_additive_mixed_precision_planner

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_additive_planner.py --risk-csv results/validation_single_layer_sweep.csv --risk-metric p95_margin_risk --risk-label empirical_margin_p95 --memory-saving-ratios 0.873 0.870 0.850 0.800 --evaluation-split test --max-samples 2000`
- Git branch: `main`
- Git commit: `6c5dfba`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| risk_label | empirical_margin_p95 |
| risk_csv | results\validation_single_layer_sweep.csv |
| risk_metric | p95_margin_risk |
| evaluation_split | test |
| max_samples | 2000 |
| requested_memory_saving_ratios | [0.873, 0.87, 0.85, 0.8] |
| memory_quantum_kb | 1 |
| planner | multiple-choice knapsack dynamic programming with additive layer-action risk |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| save_0_873_flip_rate | 0.0055 |
| save_0_873_quantized_accuracy | 0.9915 |
| save_0_873_actual_memory_saving_ratio | 0.8730323260811101 |
| save_0_873_planner_objective | 1.337395449329051 |
| save_0_870_flip_rate | 0.003 |
| save_0_870_quantized_accuracy | 0.995 |
| save_0_870_actual_memory_saving_ratio | 0.8700964380528162 |
| save_0_870_planner_objective | 0.8315534689318145 |
| save_0_850_flip_rate | 0.0025 |
| save_0_850_quantized_accuracy | 0.9955 |
| save_0_850_actual_memory_saving_ratio | 0.8500247550041878 |
| save_0_850_planner_objective | 0.36037927741563086 |
| save_0_800_flip_rate | 0.0005 |
| save_0_800_quantized_accuracy | 0.9955 |
| save_0_800_actual_memory_saving_ratio | 0.8000588322759449 |
| save_0_800_planner_objective | 0.11367605144914612 |

### Artifacts

| Key | Value |
|---|---|
| results_csv | results\empirical_margin_p95_additive_planner_results.csv |
| allocations_csv | results\empirical_margin_p95_additive_planner_allocations.csv |



## 2026-06-23T22:05:47+09:00 — weight_l2_additive_mixed_precision_planner

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_additive_planner.py --risk-csv results/validation_single_layer_sweep.csv --risk-metric relative_l2_weight_error --risk-label weight_l2 --memory-saving-ratios 0.874 0.8735 0.873 0.8725 0.872 0.8715 0.871 0.8705 0.870 --evaluation-split test --max-samples 2000`
- Git branch: `main`
- Git commit: `6c5dfba`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| risk_label | weight_l2 |
| risk_csv | results\validation_single_layer_sweep.csv |
| risk_metric | relative_l2_weight_error |
| evaluation_split | test |
| max_samples | 2000 |
| requested_memory_saving_ratios | [0.874, 0.8735, 0.873, 0.8725, 0.872, 0.8715, 0.871, 0.8705, 0.87] |
| memory_quantum_kb | 1 |
| planner | multiple-choice knapsack dynamic programming with additive layer-action risk |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| save_0_874_flip_rate | 0.0055 |
| save_0_874_quantized_accuracy | 0.9925 |
| save_0_874_actual_memory_saving_ratio | 0.8735590968243633 |
| save_0_874_planner_objective | 3.972181596793234 |
| save_0_873_flip_rate | 0.1435 |
| save_0_873_quantized_accuracy | 0.8555 |
| save_0_873_actual_memory_saving_ratio | 0.8725191540391095 |
| save_0_873_planner_objective | 3.520087460405193 |
| save_0_872_flip_rate | 0.004 |
| save_0_872_quantized_accuracy | 0.994 |
| save_0_872_actual_memory_saving_ratio | 0.8715436261545253 |
| save_0_872_planner_objective | 3.1427667237585406 |
| save_0_871_flip_rate | 0.004 |
| save_0_871_quantized_accuracy | 0.994 |
| save_0_871_actual_memory_saving_ratio | 0.8707377241750375 |
| save_0_871_planner_objective | 2.965286029517301 |
| save_0_870_flip_rate | 0.0045 |
| save_0_870_quantized_accuracy | 0.9925 |
| save_0_870_actual_memory_saving_ratio | 0.8701236354553212 |
| save_0_870_planner_objective | 2.926059558216365 |

### Artifacts

| Key | Value |
|---|---|
| results_csv | results\weight_l2_additive_planner_results.csv |
| allocations_csv | results\weight_l2_additive_planner_allocations.csv |



## 2026-06-23T22:06:26+09:00 — local_activation_p95_additive_mixed_precision_planner

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_additive_planner.py --risk-csv results/validation_local_activation_proxy_all_bits.csv --risk-metric p95_relative_activation_error --risk-label local_activation_p95 --memory-saving-ratios 0.874 0.8735 0.873 0.8725 0.872 0.8715 0.871 0.8705 0.870 --evaluation-split test --max-samples 2000`
- Git branch: `main`
- Git commit: `6c5dfba`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| risk_label | local_activation_p95 |
| risk_csv | results\validation_local_activation_proxy_all_bits.csv |
| risk_metric | p95_relative_activation_error |
| evaluation_split | test |
| max_samples | 2000 |
| requested_memory_saving_ratios | [0.874, 0.8735, 0.873, 0.8725, 0.872, 0.8715, 0.871, 0.8705, 0.87] |
| memory_quantum_kb | 1 |
| planner | multiple-choice knapsack dynamic programming with additive layer-action risk |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| save_0_874_flip_rate | 0.009 |
| save_0_874_quantized_accuracy | 0.988 |
| save_0_874_actual_memory_saving_ratio | 0.8735590968243633 |
| save_0_874_planner_objective | 2.2413226412609215 |
| save_0_873_flip_rate | 0.0055 |
| save_0_873_quantized_accuracy | 0.9915 |
| save_0_873_actual_memory_saving_ratio | 0.8725699702385266 |
| save_0_873_planner_objective | 1.9521693855349433 |
| save_0_872_flip_rate | 0.0045 |
| save_0_872_quantized_accuracy | 0.9935 |
| save_0_872_actual_memory_saving_ratio | 0.8715436261545253 |
| save_0_872_planner_objective | 1.6666676872409873 |
| save_0_871_flip_rate | 0.004 |
| save_0_871_quantized_accuracy | 0.994 |
| save_0_871_actual_memory_saving_ratio | 0.8707377241750375 |
| save_0_871_planner_objective | 1.5369682900200121 |
| save_0_870_flip_rate | 0.0045 |
| save_0_870_quantized_accuracy | 0.9935 |
| save_0_870_actual_memory_saving_ratio | 0.870077829303734 |
| save_0_870_planner_objective | 1.513387110549956 |

### Artifacts

| Key | Value |
|---|---|
| results_csv | results\local_activation_p95_additive_planner_results.csv |
| allocations_csv | results\local_activation_p95_additive_planner_allocations.csv |



## 2026-06-23T22:12:17+09:00 — weight_l2_dense_additive_mixed_precision_planner

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_additive_planner.py --risk-csv results/validation_single_layer_sweep.csv --risk-metric relative_l2_weight_error --risk-label weight_l2_dense --memory-saving-ratios 0.874 0.8735 0.873 0.8725 0.872 0.8715 0.871 0.8705 0.870 --evaluation-split test --max-samples 2000 --results-output results/weight_l2_dense_results.csv --allocations-output results/weight_l2_dense_allocations.csv`
- Git branch: `main`
- Git commit: `ed521ba`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| risk_label | weight_l2_dense |
| risk_csv | results\validation_single_layer_sweep.csv |
| risk_metric | relative_l2_weight_error |
| evaluation_split | test |
| max_samples | 2000 |
| requested_memory_saving_ratios | [0.874, 0.8735, 0.873, 0.8725, 0.872, 0.8715, 0.871, 0.8705, 0.87] |
| memory_quantum_kb | 1 |
| planner | multiple-choice knapsack dynamic programming with additive layer-action risk |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| save_8740bp_flip_rate | 0.0075 |
| save_8740bp_quantized_accuracy | 0.9895 |
| save_8740bp_actual_memory_saving_ratio | 0.8740400614160292 |
| save_8740bp_planner_objective | 4.2363534714095294 |
| save_8735bp_flip_rate | 0.0055 |
| save_8735bp_quantized_accuracy | 0.9925 |
| save_8735bp_actual_memory_saving_ratio | 0.8735590968243633 |
| save_8735bp_planner_objective | 3.972181596793234 |
| save_8730bp_flip_rate | 0.0055 |
| save_8730bp_quantized_accuracy | 0.9915 |
| save_8730bp_actual_memory_saving_ratio | 0.8730323260811101 |
| save_8730bp_planner_objective | 3.741347243703785 |
| save_8725bp_flip_rate | 0.1435 |
| save_8725bp_quantized_accuracy | 0.8555 |
| save_8725bp_actual_memory_saving_ratio | 0.8725191540391095 |
| save_8725bp_planner_objective | 3.520087460405193 |
| save_8720bp_flip_rate | 0.0045 |
| save_8720bp_quantized_accuracy | 0.9925 |
| save_8720bp_actual_memory_saving_ratio | 0.8720245907461912 |
| save_8720bp_planner_objective | 3.190388258080929 |
| save_8715bp_flip_rate | 0.004 |
| save_8715bp_quantized_accuracy | 0.994 |
| save_8715bp_actual_memory_saving_ratio | 0.8715436261545253 |
| save_8715bp_planner_objective | 3.1427667237585406 |
| save_8710bp_flip_rate | 0.004 |
| save_8710bp_quantized_accuracy | 0.994 |
| save_8710bp_actual_memory_saving_ratio | 0.8711313707902402 |
| save_8710bp_planner_objective | 2.9828989226371045 |
| save_8705bp_flip_rate | 0.004 |
| save_8705bp_quantized_accuracy | 0.994 |
| save_8705bp_actual_memory_saving_ratio | 0.8707377241750375 |
| save_8705bp_planner_objective | 2.965286029517301 |
| save_8700bp_flip_rate | 0.0045 |
| save_8700bp_quantized_accuracy | 0.9925 |
| save_8700bp_actual_memory_saving_ratio | 0.8701236354553212 |
| save_8700bp_planner_objective | 2.926059558216365 |

### Artifacts

| Key | Value |
|---|---|
| results_csv | results\weight_l2_dense_results.csv |
| allocations_csv | results\weight_l2_dense_allocations.csv |



## 2026-06-23T22:12:52+09:00 — local_activation_dense_additive_mixed_precision_planner

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_additive_planner.py --risk-csv results/validation_local_activation_proxy_all_bits.csv --risk-metric p95_relative_activation_error --risk-label local_activation_dense --memory-saving-ratios 0.874 0.8735 0.873 0.8725 0.872 0.8715 0.871 0.8705 0.870 --evaluation-split test --max-samples 2000 --results-output results/local_activation_dense_results.csv --allocations-output results/local_activation_dense_allocations.csv`
- Git branch: `main`
- Git commit: `ed521ba`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| risk_label | local_activation_dense |
| risk_csv | results\validation_local_activation_proxy_all_bits.csv |
| risk_metric | p95_relative_activation_error |
| evaluation_split | test |
| max_samples | 2000 |
| requested_memory_saving_ratios | [0.874, 0.8735, 0.873, 0.8725, 0.872, 0.8715, 0.871, 0.8705, 0.87] |
| memory_quantum_kb | 1 |
| planner | multiple-choice knapsack dynamic programming with additive layer-action risk |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| save_8740bp_flip_rate | 0.0075 |
| save_8740bp_quantized_accuracy | 0.9895 |
| save_8740bp_actual_memory_saving_ratio | 0.8740400614160292 |
| save_8740bp_planner_objective | 2.396156709233764 |
| save_8735bp_flip_rate | 0.009 |
| save_8735bp_quantized_accuracy | 0.988 |
| save_8735bp_actual_memory_saving_ratio | 0.8735590968243633 |
| save_8735bp_planner_objective | 2.2413226412609215 |
| save_8730bp_flip_rate | 0.0055 |
| save_8730bp_quantized_accuracy | 0.9915 |
| save_8730bp_actual_memory_saving_ratio | 0.8730323260811101 |
| save_8730bp_planner_objective | 2.0864445240586056 |
| save_8725bp_flip_rate | 0.0055 |
| save_8725bp_quantized_accuracy | 0.9915 |
| save_8725bp_actual_memory_saving_ratio | 0.8725699702385266 |
| save_8725bp_planner_objective | 1.9521693855349433 |
| save_8720bp_flip_rate | 0.0045 |
| save_8720bp_quantized_accuracy | 0.9925 |
| save_8720bp_actual_memory_saving_ratio | 0.8720245907461912 |
| save_8720bp_planner_objective | 1.728305580618325 |
| save_8715bp_flip_rate | 0.0045 |
| save_8715bp_quantized_accuracy | 0.9935 |
| save_8715bp_actual_memory_saving_ratio | 0.8715436261545253 |
| save_8715bp_planner_objective | 1.6666676872409873 |
| save_8710bp_flip_rate | 0.004 |
| save_8710bp_quantized_accuracy | 0.994 |
| save_8710bp_actual_memory_saving_ratio | 0.8711313707902402 |
| save_8710bp_planner_objective | 1.5576319852843872 |
| save_8705bp_flip_rate | 0.004 |
| save_8705bp_quantized_accuracy | 0.994 |
| save_8705bp_actual_memory_saving_ratio | 0.8707377241750375 |
| save_8705bp_planner_objective | 1.5369682900200121 |
| save_8700bp_flip_rate | 0.0045 |
| save_8700bp_quantized_accuracy | 0.9935 |
| save_8700bp_actual_memory_saving_ratio | 0.870077829303734 |
| save_8700bp_planner_objective | 1.513387110549956 |

### Artifacts

| Key | Value |
|---|---|
| results_csv | results\local_activation_dense_results.csv |
| allocations_csv | results\local_activation_dense_allocations.csv |



## 2026-06-23T22:13:33+09:00 — empirical_margin_dense_additive_mixed_precision_planner

- Command: `C:\Users\coin\anaconda3\envs\quant-planner\python.exe run_additive_planner.py --risk-csv results/validation_single_layer_sweep.csv --risk-metric p95_margin_risk --risk-label empirical_margin_dense --memory-saving-ratios 0.874 0.8735 0.873 0.8725 0.872 0.8715 0.871 0.8705 0.870 --evaluation-split test --max-samples 2000 --results-output results/empirical_margin_dense_results.csv --allocations-output results/empirical_margin_dense_allocations.csv`
- Git branch: `main`
- Git commit: `ed521ba`
- Git working tree: `M docs/experiment_log.md`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- CUDA runtime: `11.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

### Configuration

| Key | Value |
|---|---|
| task | ['airplane', 'automobile'] |
| model | binary ResNet-18 |
| risk_label | empirical_margin_dense |
| risk_csv | results\validation_single_layer_sweep.csv |
| risk_metric | p95_margin_risk |
| evaluation_split | test |
| max_samples | 2000 |
| requested_memory_saving_ratios | [0.874, 0.8735, 0.873, 0.8725, 0.872, 0.8715, 0.871, 0.8705, 0.87] |
| memory_quantum_kb | 1 |
| planner | multiple-choice knapsack dynamic programming with additive layer-action risk |
| quantization | mixed weight-only fake quantization, per-output-channel symmetric |

### Metrics

| Key | Value |
|---|---|
| save_8740bp_flip_rate | 0.0075 |
| save_8740bp_quantized_accuracy | 0.9895 |
| save_8740bp_actual_memory_saving_ratio | 0.8740400614160292 |
| save_8740bp_planner_objective | 1.8911936713557223 |
| save_8735bp_flip_rate | 0.0055 |
| save_8735bp_quantized_accuracy | 0.9925 |
| save_8735bp_actual_memory_saving_ratio | 0.8735590968243633 |
| save_8735bp_planner_objective | 1.5929703693836923 |
| save_8730bp_flip_rate | 0.0055 |
| save_8730bp_quantized_accuracy | 0.9915 |
| save_8730bp_actual_memory_saving_ratio | 0.8730323260811101 |
| save_8730bp_planner_objective | 1.337395449329051 |
| save_8725bp_flip_rate | 0.004 |
| save_8725bp_quantized_accuracy | 0.993 |
| save_8725bp_actual_memory_saving_ratio | 0.8725241640869393 |
| save_8725bp_planner_objective | 1.1514430558308952 |
| save_8720bp_flip_rate | 0.005 |
| save_8720bp_quantized_accuracy | 0.992 |
| save_8720bp_actual_memory_saving_ratio | 0.8721119087226543 |
| save_8720bp_planner_objective | 1.0538677768781772 |
| save_8715bp_flip_rate | 0.004 |
| save_8715bp_quantized_accuracy | 0.994 |
| save_8715bp_actual_memory_saving_ratio | 0.8715436261545253 |
| save_8715bp_planner_objective | 0.9555634488933709 |
| save_8710bp_flip_rate | 0.004 |
| save_8710bp_quantized_accuracy | 0.994 |
| save_8710bp_actual_memory_saving_ratio | 0.8711041733877353 |
| save_8710bp_planner_objective | 0.8749890224571563 |
| save_8705bp_flip_rate | 0.003 |
| save_8705bp_quantized_accuracy | 0.995 |
| save_8705bp_actual_memory_saving_ratio | 0.8705315964928949 |
| save_8705bp_planner_objective | 0.8502979230397602 |
| save_8700bp_flip_rate | 0.003 |
| save_8700bp_quantized_accuracy | 0.995 |
| save_8700bp_actual_memory_saving_ratio | 0.8700964380528162 |
| save_8700bp_planner_objective | 0.8315534689318145 |

### Artifacts

| Key | Value |
|---|---|
| results_csv | results\empirical_margin_dense_results.csv |
| allocations_csv | results\empirical_margin_dense_allocations.csv |

