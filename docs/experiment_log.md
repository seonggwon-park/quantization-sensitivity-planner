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

