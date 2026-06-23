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

