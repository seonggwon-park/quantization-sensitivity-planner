from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as functional

from quantization import fake_quantize_weight_per_output_channel


def bits_to_action_name(bits: int) -> str:
    mapping = {
        32: "fp32",
        16: "fp16",
        8: "int8",
        4: "int4",
    }

    if bits not in mapping:
        raise ValueError(
            f"Unsupported bit-width: {bits}"
        )

    return mapping[bits]


def per_sample_relative_l2_error(
    reference_output: torch.Tensor,
    perturbed_output: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute one relative L2 reconstruction error per sample.

    reference_output: FP32 module output
    perturbed_output: output from the same module input,
                      but using fake-quantized weight
    """

    if reference_output.shape != perturbed_output.shape:
        raise ValueError(
            "Reference and perturbed outputs must have "
            "the same shape."
        )

    reference_flat = reference_output.flatten(
        start_dim=1
    )

    difference_flat = (
        perturbed_output - reference_output
    ).flatten(start_dim=1)

    numerator = torch.linalg.vector_norm(
        difference_flat,
        ord=2,
        dim=1,
    )

    denominator = torch.linalg.vector_norm(
        reference_flat,
        ord=2,
        dim=1,
    ).clamp_min(epsilon)

    return numerator / denominator


def forward_module_with_weight(
    module: nn.Module,
    module_input: torch.Tensor,
    replacement_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Run one Conv2d or Linear module with a replacement weight.

    This does not modify the original model.
    """

    if isinstance(module, nn.Conv2d):
        if module.padding_mode != "zeros":
            raise NotImplementedError(
                "Only zero-padding Conv2d is supported."
            )

        return functional.conv2d(
            input=module_input,
            weight=replacement_weight,
            bias=module.bias,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    if isinstance(module, nn.Linear):
        return functional.linear(
            input=module_input,
            weight=replacement_weight,
            bias=module.bias,
        )

    raise TypeError(
        "Only Conv2d and Linear are supported."
    )


class LocalActivationProxyCollector:
    """
    Collect local feature reconstruction errors through forward hooks.

    For each target layer:
    1. Run the original FP32 model once.
    2. Receive the layer's FP32 input and FP32 output.
    3. Re-run only that local module with fake-quantized weight.
    4. Store per-sample relative output error.

    No gradients, backward passes, or per-layer full-model
    forwards are used.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Iterable[str],
        bits: Iterable[int],
        epsilon: float = 1e-8,
    ) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.bits = [int(bit) for bit in bits]
        self.epsilon = epsilon

        self.modules = {
            layer_name: model.get_submodule(layer_name)
            for layer_name in self.layer_names
        }

        for layer_name, module in self.modules.items():
            if not isinstance(
                module,
                (nn.Conv2d, nn.Linear),
            ):
                raise TypeError(
                    f"{layer_name} is not Conv2d or Linear."
                )

        self.quantized_weights = {
            layer_name: {
                bit: fake_quantize_weight_per_output_channel(
                    module.weight.detach(),
                    bits=bit,
                ).detach()
                for bit in self.bits
            }
            for layer_name, module in self.modules.items()
        }

        self.error_chunks = {
            layer_name: {
                bit: []
                for bit in self.bits
            }
            for layer_name in self.layer_names
        }

        self.handles: list[
            torch.utils.hooks.RemovableHandle
        ] = []

    def _make_hook(
        self,
        layer_name: str,
    ):
        def hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if len(inputs) == 0:
                raise RuntimeError(
                    f"{layer_name} received no input."
                )

            module_input = inputs[0]

            if not isinstance(module_input, torch.Tensor):
                raise TypeError(
                    f"{layer_name} input is not a tensor."
                )

            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"{layer_name} output is not a tensor."
                )

            reference_output = output.detach()

            for bit in self.bits:
                quantized_output = (
                    forward_module_with_weight(
                        module=module,
                        module_input=module_input,
                        replacement_weight=(
                            self.quantized_weights[
                                layer_name
                            ][bit]
                        ),
                    )
                )

                relative_error = (
                    per_sample_relative_l2_error(
                        reference_output=reference_output,
                        perturbed_output=quantized_output,
                        epsilon=self.epsilon,
                    )
                )

                # Keep tensors on GPU during calibration.
                # They are very small: one scalar per sample.
                self.error_chunks[layer_name][bit].append(
                    relative_error.detach()
                )

        return hook

    def register(self) -> None:
        if self.handles:
            raise RuntimeError(
                "Hooks are already registered."
            )

        for layer_name, module in self.modules.items():
            handle = module.register_forward_hook(
                self._make_hook(layer_name)
            )

            self.handles.append(handle)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    def to_rows(
        self,
        data_split: str,
        num_samples: int,
    ) -> list[dict]:
        rows = []

        for layer_name in self.layer_names:
            for bit in self.bits:
                chunks = self.error_chunks[layer_name][bit]

                if not chunks:
                    raise RuntimeError(
                        f"No proxy values collected for "
                        f"{layer_name}, bit={bit}."
                    )

                values = torch.cat(
                    chunks,
                    dim=0,
                ).float().cpu()

                row = {
                    "data_split": data_split,
                    "layer": layer_name,
                    "action": bits_to_action_name(bit),
                    "bits": bit,
                    "num_samples": num_samples,
                    "mean_relative_activation_error": (
                        values.mean().item()
                    ),
                    "p50_relative_activation_error": (
                        torch.quantile(
                            values,
                            0.50,
                        ).item()
                    ),
                    "p95_relative_activation_error": (
                        torch.quantile(
                            values,
                            0.95,
                        ).item()
                    ),
                    "max_relative_activation_error": (
                        values.max().item()
                    ),
                }

                rows.append(row)

        return rows


def collect_local_activation_proxy(
    model: nn.Module,
    dataloader,
    layer_names: Iterable[str],
    bits: Iterable[int],
    device: torch.device,
    data_split: str,
) -> list[dict]:
    """
    Run one FP32 model forward per calibration batch and collect
    local activation reconstruction proxy scores.
    """

    collector = LocalActivationProxyCollector(
        model=model,
        layer_names=layer_names,
        bits=bits,
    )

    original_training_state = model.training
    model.eval()

    collector.register()

    num_samples = 0

    try:
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(
                    device,
                    non_blocking=True,
                )

                model(inputs)

                num_samples += inputs.shape[0]

    finally:
        collector.close()

        if original_training_state:
            model.train()

    return collector.to_rows(
        data_split=data_split,
        num_samples=num_samples,
    )