"""Ordered FP32 GCAM neighborhood kernels (Triton bundled with CUDA PyTorch).

Each output voxel accumulates neighbors in native dz/dy/dx order. No atomics,
TF32 GEMM, fast-math reassociation, or reduced precision are used.
"""
import triton
import triton.language as tl


@triton.jit
def _expand(F, M, OUT, W: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n: tl.constexpr = W * H * D
    ok = i < n
    x, y, z = i // (H * D), (i // D) % H, i % D
    marked = tl.load(M + i, ok, other=1) != 0
    sx = tl.full((BLOCK,), 0, tl.float32)
    sy = tl.full((BLOCK,), 0, tl.float32)
    sz = tl.full((BLOCK,), 0, tl.float32)
    count = tl.full((BLOCK,), 0, tl.int32)
    for dz in tl.static_range(-1, 2):
        for dy in tl.static_range(-1, 2):
            for dx in tl.static_range(-1, 2):
                j = ((tl.minimum(tl.maximum(x + dx, 0), W - 1) * H +
                      tl.minimum(tl.maximum(y + dy, 0), H - 1)) * D +
                     tl.minimum(tl.maximum(z + dz, 0), D - 1))
                use = ok & ~marked & (tl.load(M + j, ok, other=0) != 0)
                count += use.to(tl.int32)
                sx += tl.load(F + j, use, other=0)
                sy += tl.load(F + n + j, use, other=0)
                sz += tl.load(F + 2 * n + j, use, other=0)
    new = ok & ~marked & (count > 0)
    denom = tl.maximum(count, 1).to(tl.float32)
    tl.store(F + i, tl.div_rn(sx, denom), new)
    tl.store(F + n + i, tl.div_rn(sy, denom), new)
    tl.store(F + 2 * n + i, tl.div_rn(sz, denom), new)
    tl.store(OUT + i, marked | new, ok)


@triton.jit
def _soap(F, T, C, CHANGE, W: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
          X1, X2, Y1, Y2, Z1, Z2, SEED: tl.constexpr,
          BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < W * H * D
    x, y, z = i // (H * D), (i // D) % H, i % D
    active = (ok & (x >= X1) & (x <= X2) & (y >= Y1) & (y <= Y2) &
              (z >= Z1) & (z <= Z2) & (tl.load(C + i, ok, other=1) == 0))
    total = tl.full((BLOCK,), 0, tl.float32)
    count = tl.full((BLOCK,), 0, tl.int32)
    radius: tl.constexpr = 2 if SEED else 1
    for dz in tl.static_range(-radius, radius + 1):
        for dy in tl.static_range(-radius, radius + 1):
            for dx in tl.static_range(-radius, radius + 1):
                j = ((tl.minimum(tl.maximum(x + dx, 0), W - 1) * H +
                      tl.minimum(tl.maximum(y + dy, 0), H - 1)) * D +
                     tl.minimum(tl.maximum(z + dz, 0), D - 1))
                use = active
                if SEED:
                    use = active & (tl.load(C + j, active, other=0) != 0)
                count += use.to(tl.int32)
                total += tl.load(F + j, use, other=0)
    active &= count > 0
    new = tl.div_rn(total, tl.maximum(count, 1).to(tl.float32))
    old = tl.load(T + i, active, other=0)
    if SEED:
        change = tl.abs(new.to(tl.float64) - old.to(tl.float64))
    else:
        change = tl.abs(total.to(tl.float64) / 27.0 - old.to(tl.float64))
    tl.store(T + i, new, active)
    tl.store(CHANGE + i, tl.where(active, change, 0), ok)
