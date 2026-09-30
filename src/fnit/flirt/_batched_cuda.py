"""Optional fused float32 reductions for affine candidate execution."""

import triton
import triton.language as tl


@triton.jit
def _compact_sum_kernel(packed, lengths, output, BINS: tl.constexpr,
                        ROWS: tl.constexpr, LANES: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(lengths + row // ROWS)
    lanes = tl.arange(0, LANES)
    vectorized = length > 128
    input_width = tl.maximum(tl.where(vectorized, length // 4, length), 1)
    width = 1
    for exponent in tl.static_range(1, 10):
        width = tl.where(input_width >= (1 << exponent), 1 << exponent, width)
    if vectorized:
        value0 = tl.full((LANES,), 0, tl.float32)
        value1 = tl.full((LANES,), 0, tl.float32)
        value2 = tl.full((LANES,), 0, tl.float32)
        value3 = tl.full((LANES,), 0, tl.float32)
        iteration = 0
        while iteration * width * 4 < length:
            base = (lanes + iteration * width) * 4
            valid = (lanes < width) & (base + 3 < length)
            value0 = value0 + tl.load(packed + row * BINS + base, valid, 0)
            value1 = value1 + tl.load(packed + row * BINS + base + 1, valid, 0)
            value2 = value2 + tl.load(packed + row * BINS + base + 2, valid, 0)
            value3 = value3 + tl.load(packed + row * BINS + base + 3, valid, 0)
            iteration += 1
        tail = length - length % 4 + lanes
        value0 = value0 + tl.load(packed + row * BINS + tail,
                                 (lanes < 4) & (tail < length), 0)
        value = ((value0 + value1) + value2) + value3
    else:
        value = tl.load(packed + row * BINS + lanes, lanes < length, 0)
        value = value + tl.load(packed + row * BINS + lanes + width,
                               lanes + width < length, 0)
    value = tl.where(lanes < width, value, 0)
    if LANES > 256:
        low, high = tl.split(tl.trans(tl.reshape(value, (2, 256))))
        value = tl.where(width > 256, low + high, low)
    if LANES > 128:
        low, high = tl.split(tl.trans(tl.reshape(value, (2, 128))))
        value = tl.where(width > 128, low + high, low)
    low, high = tl.split(tl.trans(tl.reshape(value, (2, 64))))
    value = tl.where(width > 64, low + high, low)
    low, high = tl.split(tl.trans(tl.reshape(value, (2, 32))))
    value = tl.where(width > 32, low + high, low)
    low, high = tl.split(tl.reshape(value, (16, 2)))
    value = low + high
    low, high = tl.split(tl.reshape(value, (8, 2)))
    value = low + high
    low, high = tl.split(tl.reshape(value, (4, 2)))
    value = low + high
    low, high = tl.split(tl.reshape(value, (2, 2)))
    value = low + high
    low, high = tl.split(tl.reshape(value, (1, 2)))
    tl.store(output + row, tl.sum(low + high))
