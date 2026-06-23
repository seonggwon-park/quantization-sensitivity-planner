import copy

import torch
import torch.nn as nn


SUPPORTED_BITS = {32, 16, 8, 4}


def fake_quantize_weight_per_output_channel(
    weight: torch.Tensor,
    bits: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetric per-output-channel fake quantization.

    FP32:
        Return original weights.

    FP16:
        Round weights to FP16 values, then cast back to FP32.

    INT8 / INT4:
        Quantize-dequantize with symmetric per-output-channel scaling.

    Returned tensor is still float32.
    This is intentional for perturbation analysis.
    """

    if bits not in SUPPORTED_BITS:
        raise ValueError(
            f"Unsupported bit-width: {bits}"
        )

    original_weight = weight.detach()

    if bits == 32:
        return original_weight.clone()

    if bits == 16:
        return original_weight.to(
            torch.float16
        ).to(original_weight.dtype)

    qmax = (2 ** (bits - 1)) - 1

    # Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
    # Linear: [out_features, in_features]
    reduction_dimensions = tuple(
        range(1, original_weight.ndim)
    )

    max_abs_per_output_channel = original_weight.abs().amax(
        dim=reduction_dimensions,
        keepdim=True,
    )

    scale = (
        max_abs_per_output_channel / qmax
    ).clamp_min(epsilon)

    quantized_integer = torch.round(
        original_weight / scale
    )

    quantized_integer = torch.clamp(
        quantized_integer,
        min=-qmax,
        max=qmax,
    )

    dequantized_weight = quantized_integer * scale

    return dequantized_weight.to(
        original_weight.dtype
    )


def list_quantizable_layers(model: nn.Module) -> list[str]:
    """
    Only Conv2d and Linear layers are quantization targets.

    BatchNorm / bias / activation remain FP32 in this first experiment.
    """

    layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)

    return layer_names


def apply_fake_quantization_to_module(
    module: nn.Module,
    bits: int,
) -> None:
    """
    Replace one module's weight with fake-quantized values.
    """

    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        raise TypeError(
            "Only Conv2d and Linear are supported."
        )

    with torch.no_grad():
        fake_quantized_weight = (
            fake_quantize_weight_per_output_channel(
                module.weight,
                bits=bits,
            )
        )

        module.weight.copy_(fake_quantized_weight)


def build_single_layer_quantized_model(
    fp32_model: nn.Module,
    layer_name: str,
    bits: int,
    device: torch.device,
) -> nn.Module:
    """
    Copy FP32 model and fake-quantize exactly one layer.
    """

    quantized_model = copy.deepcopy(fp32_model)
    quantized_model = quantized_model.cpu().eval()

    target_module = quantized_model.get_submodule(
        layer_name
    )

    apply_fake_quantization_to_module(
        target_module,
        bits=bits,
    )

    return quantized_model.to(device).eval()


def build_uniform_quantized_model(
    fp32_model: nn.Module,
    bits: int,
    device: torch.device,
) -> nn.Module:
    """
    Fake-quantize every Conv2d and Linear layer.
    """

    quantized_model = copy.deepcopy(fp32_model)
    quantized_model = quantized_model.cpu().eval()

    for layer_name in list_quantizable_layers(
        quantized_model
    ):
        module = quantized_model.get_submodule(
            layer_name
        )

        apply_fake_quantization_to_module(
            module,
            bits=bits,
        )

    return quantized_model.to(device).eval()


def relative_l2_weight_error(
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    epsilon: float = 1e-8,
) -> float:
    """
    Simple baseline proxy:
    ||W_q - W||_2 / ||W||_2
    """

    numerator = (
        quantized_weight - original_weight
    ).norm(p=2)

    denominator = original_weight.norm(p=2).clamp_min(
        epsilon
    )

    return (numerator / denominator).item()


def estimate_parameter_memory_mb(
    model: nn.Module,
    layer_bits: dict[str, int],
) -> float:
    """
    Estimate parameter memory only.

    Quantized Conv/Linear weights use chosen bits.
    Everything else remains FP32.

    This does not include:
    - runtime activations
    - CUDA kernel workspace
    - optimizer states
    - real hardware packing overhead
    """

    quantizable_layers = set(
        list_quantizable_layers(model)
    )

    total_bits = 0

    for parameter_name, parameter in model.named_parameters():
        module_name, parameter_type = parameter_name.rsplit(
            ".",
            maxsplit=1,
        )

        if (
            parameter_type == "weight"
            and module_name in quantizable_layers
        ):
            bits = layer_bits.get(module_name, 32)
        else:
            bits = 32

        total_bits += parameter.numel() * bits

    return total_bits / (8 * 1024 * 1024)