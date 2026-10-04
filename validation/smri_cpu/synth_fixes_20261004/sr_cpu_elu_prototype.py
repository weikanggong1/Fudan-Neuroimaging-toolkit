"""Rejected CPU-only SynthSR ELU prototype for reproducible stage diagnosis.

Its whole first ELU is bitwise equal to the fixed TensorFlow 2.13 reference.
Whole inference still fails the existing floating gate and is slower, so this
prototype is never called by production FNIT. No TensorFlow is imported here.
"""

import torch
from torch.nn import functional as F


def _cpu_reference_elu(value):
    """FP32 ELU with the separate-rounding Eigen exp polynomial of TF 2.13.

    This independently expressed polynomial matches the CPU reference's
    exp(x)-1 rounding. It is used only for contiguous FP32 CPU inference;
    training, autograd, other dtypes and CUDA retain PyTorch's native ELU.
    Blocks bound temporary storage without changing elementwise arithmetic.
    """
    result = torch.empty_like(value)
    source, target = value.view(-1), result.view(-1)
    for start in range(0, source.numel(), 16 * 1024 ** 2):
        original = source[start:start + 16 * 1024 ** 2]
        # Positive values are selected unchanged. Below -104 the reference
        # exponential is zero and ELU is exactly -1 in FP32.
        negative = original.clamp(min=-104, max=0)
        power = torch.floor(negative * 1.44269504088896341 + 0.5)
        remainder = power * -0.693359375 + negative
        remainder = power * 2.12194440e-4 + remainder
        square = remainder * remainder
        even = square * 1.37449637986719608306884765625e-3 + 4.166965186595916748046875e-2
        odd = square * 8.36894474923610687255859375e-3 + 0.16666518151760101318359375
        even = square * even + 0.49999988079071044921875
        exponential = square * (remainder * odd + even) + (remainder + 1)
        exponential = torch.ldexp(exponential, power.to(torch.int32))
        target[start:start + original.numel()].copy_(
            torch.where(original < 0, exponential - 1, original))
    return result


def cpu_inference(model, image):
    """Keep the established CUDA forward while matching CPU ELU rounding."""
    skips = []
    value = image
    for level, (convs, norm) in enumerate(zip(model.down, model.down_bn)):
        value = _cpu_reference_elu(convs[0](value))
        value = _cpu_reference_elu(convs[1](value))
        skips.append(value)
        value = norm(value)
        if level < len(model.down) - 1:
            value = F.max_pool3d(value, 2)
    for convs, norm, skip in zip(model.up, model.up_bn, reversed(skips[:-1])):
        value = F.interpolate(value, scale_factor=2, mode='nearest')
        value = torch.cat((skip, value), dim=1)
        value = _cpu_reference_elu(convs[0](value))
        value = _cpu_reference_elu(convs[1](value))
        value = norm(value)
    return model.likelihood(value)
