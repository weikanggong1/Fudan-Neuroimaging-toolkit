"""Optional BBR sampling fusion; FP64 geometry/contrast and RN FP32 interpolation."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ..flirt._batched_cuda import _add_f32, _sub_f32, _mul_f32


@triton.jit
def _corner(image, ix, iy, iz, present, SX: tl.constexpr, SY: tl.constexpr, SZ: tl.constexpr):
    valid = present & (ix >= 0) & (ix < SX) & (iy >= 0) & (iy < SY) & (iz >= 0) & (iz < SZ)
    return tl.load(image + ix * SY * SZ + iy * SZ + iz, valid, 0.)


@triton.jit
def _interpolate(image, x, y, z, present, SX: tl.constexpr, SY: tl.constexpr, SZ: tl.constexpr):
    x, y, z = x.to(tl.float32), y.to(tl.float32), z.to(tl.float32)
    ix, iy, iz = tl.floor(x).to(tl.int32), tl.floor(y).to(tl.int32), tl.floor(z).to(tl.int32)
    dx, dy, dz = _sub_f32(x, ix.to(tl.float32)), _sub_f32(y, iy.to(tl.float32)), _sub_f32(z, iz.to(tl.float32))
    v000 = _corner(image, ix, iy, iz, present, SX, SY, SZ)
    v001 = _corner(image, ix, iy, iz + 1, present, SX, SY, SZ)
    v010 = _corner(image, ix, iy + 1, iz, present, SX, SY, SZ)
    v011 = _corner(image, ix, iy + 1, iz + 1, present, SX, SY, SZ)
    v100 = _corner(image, ix + 1, iy, iz, present, SX, SY, SZ)
    v101 = _corner(image, ix + 1, iy, iz + 1, present, SX, SY, SZ)
    v110 = _corner(image, ix + 1, iy + 1, iz, present, SX, SY, SZ)
    v111 = _corner(image, ix + 1, iy + 1, iz + 1, present, SX, SY, SZ)
    a = _add_f32(_mul_f32(_sub_f32(v100, v000), dx), v000)
    b = _add_f32(_mul_f32(_sub_f32(v101, v001), dx), v001)
    c = _add_f32(_mul_f32(_sub_f32(v110, v010), dx), v010)
    d = _add_f32(_mul_f32(_sub_f32(v111, v011), dx), v011)
    e = _add_f32(_mul_f32(_sub_f32(c, a), dy), a)
    f = _add_f32(_mul_f32(_sub_f32(d, b), dy), b)
    return _add_f32(_mul_f32(_sub_f32(f, e), dz), e)


@triton.jit
def _cost_kernel(image, points, transforms, output, N: tl.constexpr,
                 SX: tl.constexpr, SY: tl.constexpr, SZ: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    present = index < N
    for kind in tl.static_range(2):
        p0 = tl.load(points + (kind * N + index) * 3, present, 0.)
        p1 = tl.load(points + (kind * N + index) * 3 + 1, present, 0.)
        p2 = tl.load(points + (kind * N + index) * 3 + 2, present, 0.)
        x = tl.load(transforms + row * 16) * p0 + tl.load(transforms + row * 16 + 1) * p1
        x = x + tl.load(transforms + row * 16 + 2) * p2
        x = x + tl.load(transforms + row * 16 + 3)
        y = tl.load(transforms + row * 16 + 4) * p0 + tl.load(transforms + row * 16 + 5) * p1
        y = y + tl.load(transforms + row * 16 + 6) * p2
        y = y + tl.load(transforms + row * 16 + 7)
        z = tl.load(transforms + row * 16 + 8) * p0 + tl.load(transforms + row * 16 + 9) * p1
        z = z + tl.load(transforms + row * 16 + 10) * p2
        z = z + tl.load(transforms + row * 16 + 11)
        value = _interpolate(image, x, y, z, present, SX, SY, SZ).to(tl.float64)
        if kind == 0:
            grey = value
        else:
            white = value
    total = grey + white
    valid = tl.abs(total) > 1.e-6
    denominator = tl.where(valid, total, 1.)
    difference = tl.where(valid, 200. * (grey - white) / denominator, 0.)
    update = 1. + libdevice.tanh(-.5 * difference)
    tl.store(output + row * N + index, update, present)
