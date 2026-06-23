import copy
from collections.abc import Mapping

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

    Returned tensor is still float32.
    This is intentional: this project currently measures numerical
    perturbation, not real packed INT4 hardware inference.
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
    Quantization targets in the first study:
    Conv2d and Linear weights only.
    """

    layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)

    return layer_names


def validate_layer_bits(
    model: nn.Module,
    layer_bits: Mapping[str, int],
) -> None:
    """
    Validate a partial or complete layer -> bit-width mapping.
    """

    known_layers = set(
        list_quantizable_layers(model)
    )

    unknown_layers = set(layer_bits) - known_layers

    if unknown_layers:
        raise ValueError(
            f"Unknown quantizable layers: {unknown_layers}"
        )

    unsupported = {
        layer_name: bits
        for layer_name, bits in layer_bits.items()
        if int(bits) not in SUPPORTED_BITS
    }

    if unsupported:
        raise ValueError(
            f"Unsupported bit assignments: {unsupported}"
        )


def resolve_layer_bits(
    model: nn.Module,
    layer_bits: Mapping[str, int] | None = None,
    default_bits: int = 32,
) -> dict[str, int]:
    """
    Convert a partial assignment into a complete assignment.

    Example:
        default_bits = 4
        layer_bits = {"conv1": 32}

    means:
        conv1 -> FP32
        every other Conv2d / Linear -> INT4
    """

    if default_bits not in SUPPORTED_BITS:
        raise ValueError(
            f"Unsupported default bit-width: {default_bits}"
        )

    layer_bits = dict(layer_bits or {})

    validate_layer_bits(model, layer_bits)

    resolved = {
        layer_name: default_bits
        for layer_name in list_quantizable_layers(model)
    }

    resolved.update(
        {
            layer_name: int(bits)
            for layer_name, bits in layer_bits.items()
        }
    )

    return resolved


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


def build_mixed_quantized_model(
    fp32_model: nn.Module,
    layer_bits: Mapping[str, int] | None,
    device: torch.device,
    default_bits: int = 32,
) -> nn.Module:
    """
    Build a mixed-precision fake-quantized copy of the FP32 model.

    Important:
    - fp32_model is never modified.
    - every experiment starts from a fresh FP32 copy.
    - layer_bits can be partial when default_bits is provided.
    """

    resolved_bits = resolve_layer_bits(
        model=fp32_model,
        layer_bits=layer_bits,
        default_bits=default_bits,
    )

    quantized_model = copy.deepcopy(fp32_model)
    quantized_model = quantized_model.cpu().eval()

    for layer_name, bits in resolved_bits.items():
        if bits == 32:
            continue

        target_module = quantized_model.get_submodule(
            layer_name
        )

        apply_fake_quantization_to_module(
            module=target_module,
            bits=bits,
        )

    return quantized_model.to(device).eval()


def build_single_layer_quantized_model(
    fp32_model: nn.Module,
    layer_name: str,
    bits: int,
    device: torch.device,
) -> nn.Module:
    """
    FP32 everywhere except one selected layer.
    """

    return build_mixed_quantized_model(
        fp32_model=fp32_model,
        layer_bits={
            layer_name: bits,
        },
        default_bits=32,
        device=device,
    )


def build_uniform_quantized_model(
    fp32_model: nn.Module,
    bits: int,
    device: torch.device,
) -> nn.Module:
    """
    Same bit-width for every Conv2d and Linear layer.
    """

    return build_mixed_quantized_model(
        fp32_model=fp32_model,
        layer_bits={},
        default_bits=bits,
        device=device,
    )


def relative_l2_weight_error(
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    epsilon: float = 1e-8,
) -> float:
    """
    Simple weight-space proxy baseline:
    ||W_q - W||_2 / ||W||_2
    """

    numerator = (
        quantized_weight - original_weight
    ).norm(p=2)

    denominator = original_weight.norm(
        p=2
    ).clamp_min(epsilon)

    return (numerator / denominator).item()


def estimate_parameter_memory_mb(
    model: nn.Module,
    layer_bits: Mapping[str, int] | None = None,
    default_bits: int = 32,
) -> float:
    """
    Estimate parameter storage only.

    Conv2d / Linear weights use assigned bit-widths.
    Bias, BatchNorm and all other parameters remain FP32.
    """

    resolved_bits = resolve_layer_bits(
        model=model,
        layer_bits=layer_bits,
        default_bits=default_bits,
    )

    quantizable_layers = set(resolved_bits)

    total_bits = 0

    for parameter_name, parameter in model.named_parameters():
        if "." not in parameter_name:
            total_bits += parameter.numel() * 32
            continue

        module_name, parameter_type = parameter_name.rsplit(
            ".",
            maxsplit=1,
        )

        if (
            parameter_type == "weight"
            and module_name in quantizable_layers
        ):
            bits = resolved_bits[module_name]
        else:
            bits = 32

        total_bits += parameter.numel() * bits

    return total_bits / (8 * 1024 * 1024)