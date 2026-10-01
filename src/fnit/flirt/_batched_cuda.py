"""Optional fused float32 sampling and reductions for affine candidates."""

import triton
import triton.language as tl


@triton.jit
def _add_f32(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f",
                                     [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sub_f32(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=f,f,f",
                                     [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul_f32(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f",
                                     [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _div_f32(a, b):
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f",
                                     [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sample_kernel(grid, coefficients, moving, upper, smooth, values, weights,
                   N: tl.constexpr, SIZE_X: tl.constexpr, SIZE_Y: tl.constexpr,
                   SIZE_Z: tl.constexpr, TAPER: tl.constexpr, BLOCK: tl.constexpr):
    candidate = tl.program_id(1)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    present = index < N
    gx = tl.load(grid + index, present, 0)
    gy = tl.load(grid + N + index, present, 0)
    gz = tl.load(grid + 2 * N + index, present, 0)
    cx = coefficients + candidate * 12
    cy = cx + 4
    cz = cx + 8
    # Match the tensor path's assigned float32 intermediates. The launch
    # disables FMA; explicit rounded PTX arithmetic retains subnormal FOV
    # boundaries. No dot product or TF32 multiplication is involved.
    x = _add_f32(_add_f32(_add_f32(_mul_f32(gy, tl.load(cx + 1)),
                                   _mul_f32(gz, tl.load(cx + 2))), tl.load(cx + 3)),
                   _mul_f32(gx, tl.load(cx)))
    y = _add_f32(_add_f32(_add_f32(_mul_f32(gy, tl.load(cy + 1)),
                                   _mul_f32(gz, tl.load(cy + 2))), tl.load(cy + 3)),
                   _mul_f32(gx, tl.load(cy)))
    z = _add_f32(_add_f32(_add_f32(_mul_f32(gy, tl.load(cz + 1)),
                                   _mul_f32(gz, tl.load(cz + 2))), tl.load(cz + 3)),
                   _mul_f32(gx, tl.load(cz)))
    ux, uy, uz = tl.load(upper), tl.load(upper + 1), tl.load(upper + 2)
    valid = (x >= 0) & (x <= ux) & (y >= 0) & (y <= uy) & (z >= 0) & (z <= uz)
    # PyTorch minimum/maximum propagate NaN. Keep that property rather than
    # CUDA minNum's replacement of a NaN coordinate by a finite endpoint.
    sx = tl.where(x != x, x, tl.minimum(tl.maximum(x, 0), ux))
    sy = tl.where(y != y, y, tl.minimum(tl.maximum(y, 0), uy))
    sz = tl.where(z != z, z, tl.minimum(tl.maximum(z, 0), uz))
    ix = tl.maximum(tl.minimum(tl.floor(sx).to(tl.int32), SIZE_X - 2), 0)
    iy = tl.maximum(tl.minimum(tl.floor(sy).to(tl.int32), SIZE_Y - 2), 0)
    iz = tl.maximum(tl.minimum(tl.floor(sz).to(tl.int32), SIZE_Z - 2), 0)
    dx = _sub_f32(sx, ix.to(tl.float32))
    dy = _sub_f32(sy, iy.to(tl.float32))
    dz = _sub_f32(sz, iz.to(tl.float32))
    stride_x, stride_y = SIZE_Y * SIZE_Z, SIZE_Z
    offset = ix * stride_x + iy * stride_y + iz
    v000 = tl.load(moving + offset, present, 0)
    v001 = tl.load(moving + offset + 1, present, 0)
    v010 = tl.load(moving + offset + stride_y, present, 0)
    v011 = tl.load(moving + offset + stride_y + 1, present, 0)
    v100 = tl.load(moving + offset + stride_x, present, 0)
    v101 = tl.load(moving + offset + stride_x + 1, present, 0)
    v110 = tl.load(moving + offset + stride_x + stride_y, present, 0)
    v111 = tl.load(moving + offset + stride_x + stride_y + 1, present, 0)
    t1 = _add_f32(_mul_f32(_sub_f32(v100, v000), dx), v000)
    t2 = _add_f32(_mul_f32(_sub_f32(v101, v001), dx), v001)
    t3 = _add_f32(_mul_f32(_sub_f32(v110, v010), dx), v010)
    t4 = _add_f32(_mul_f32(_sub_f32(v111, v011), dx), v011)
    t5 = _add_f32(_mul_f32(_sub_f32(t3, t1), dy), t1)
    t6 = _add_f32(_mul_f32(_sub_f32(t4, t2), dy), t2)
    value = _add_f32(_mul_f32(_sub_f32(t6, t5), dz), t5)
    if TAPER:
        mx, my, mz = tl.load(smooth), tl.load(smooth + 1), tl.load(smooth + 2)
        fx, fy, fz = _sub_f32(ux, x), _sub_f32(uy, y), _sub_f32(uz, z)
        wx = tl.where(x < mx, _div_f32(x, mx), tl.where(fx < mx, _div_f32(fx, mx), 1.0))
        wy = tl.where(y < my, _div_f32(y, my), tl.where(fy < my, _div_f32(fy, my), 1.0))
        wz = tl.where(z < mz, _div_f32(z, mz), tl.where(fz < mz, _div_f32(fz, mz), 1.0))
        weight = _mul_f32(_mul_f32(wx, wy), wz)
        # clamp_min(0) replaces negative zero as well as negative values.
        weight = tl.where(weight <= 0, 0.0, weight)
    else:
        weight = tl.full((BLOCK,), 1.0, tl.float32)
    tl.store(values + candidate * N + index, value, present)
    tl.store(weights + candidate * N + index, _mul_f32(weight, valid.to(tl.float32)), present)


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
