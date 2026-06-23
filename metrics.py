import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader


def binary_score(logits: torch.Tensor) -> torch.Tensor:
    """
    S(x) = z_1(x) - z_0(x)
    """

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            "Expected binary logits with shape [batch_size, 2]."
        )

    return logits[:, 1] - logits[:, 0]


def binary_prediction(logits: torch.Tensor) -> torch.Tensor:
    """
    Class 1 if S(x) > 0, otherwise class 0.
    """

    return (binary_score(logits) > 0).long()


@torch.inference_mode()
def evaluate_classifier(
    model,
    dataloader: DataLoader,
    device: torch.device,
):
    """
    Standard classification evaluation for one model.
    """

    model.eval()

    total_samples = 0
    total_correct = 0
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = functional.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )

        predictions = binary_prediction(logits)

        total_samples += labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_loss += loss.item()

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_samples": total_samples,
    }


@torch.inference_mode()
def compare_binary_models(
    fp32_model,
    quantized_model,
    dataloader: DataLoader,
    device: torch.device,
    epsilon: float = 1e-8,
):
    """
    Compare FP32 reference model and quantized model.

    Important:
    - flip_rate compares quantized predictions with FP32 predictions.
    - It is not the same as classification error.
    """

    fp32_model.eval()
    quantized_model.eval()

    total_samples = 0

    fp32_correct = 0
    quantized_correct = 0

    fp32_loss_sum = 0.0
    quantized_loss_sum = 0.0

    flip_count = 0

    all_logit_errors = []
    all_margin_risks = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        fp32_logits = fp32_model(images)
        quantized_logits = quantized_model(images)

        fp32_predictions = binary_prediction(fp32_logits)
        quantized_predictions = binary_prediction(
            quantized_logits
        )

        fp32_score = binary_score(fp32_logits)
        quantized_score = binary_score(quantized_logits)

        logit_error = (
            quantized_score - fp32_score
        ).abs()

        fp32_margin = fp32_score.abs()

        margin_normalized_risk = (
            logit_error / (fp32_margin + epsilon)
        )

        fp32_loss_sum += functional.cross_entropy(
            fp32_logits,
            labels,
            reduction="sum",
        ).item()

        quantized_loss_sum += functional.cross_entropy(
            quantized_logits,
            labels,
            reduction="sum",
        ).item()

        fp32_correct += (
            fp32_predictions == labels
        ).sum().item()

        quantized_correct += (
            quantized_predictions == labels
        ).sum().item()

        flip_count += (
            fp32_predictions != quantized_predictions
        ).sum().item()

        total_samples += labels.size(0)

        all_logit_errors.append(
            logit_error.detach().cpu()
        )

        all_margin_risks.append(
            margin_normalized_risk.detach().cpu()
        )

    all_logit_errors = torch.cat(all_logit_errors)
    all_margin_risks = torch.cat(all_margin_risks)

    fp32_accuracy = fp32_correct / total_samples
    quantized_accuracy = quantized_correct / total_samples

    return {
        "num_samples": total_samples,

        "fp32_loss": fp32_loss_sum / total_samples,
        "quantized_loss": quantized_loss_sum / total_samples,

        "fp32_accuracy": fp32_accuracy,
        "quantized_accuracy": quantized_accuracy,
        "accuracy_drop": fp32_accuracy - quantized_accuracy,

        "flip_rate": flip_count / total_samples,

        "mean_logit_error": all_logit_errors.mean().item(),

        "mean_margin_risk": all_margin_risks.mean().item(),
        "p95_margin_risk": torch.quantile(
            all_margin_risks,
            0.95,
        ).item(),

        "max_margin_risk": all_margin_risks.max().item(),
    }