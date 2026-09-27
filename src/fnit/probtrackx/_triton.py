"""Fused float32 CUDA walk for FSL-style volume tractography."""

import torch
import triton
import triton.language as tl


@triton.jit
def _walk_kernel(Starts, Reverse, Mask, Theta, Phi, Fraction, History, First,
                Seed, COUNT: tl.constexpr, HALF: tl.constexpr,
                SX: tl.constexpr, SY: tl.constexpr, SZ: tl.constexpr,
                NTIME: tl.constexpr, NFIB: tl.constexpr,
                STEP_X: tl.constexpr, STEP_Y: tl.constexpr, STEP_Z: tl.constexpr,
                CTHR: tl.constexpr, FTHR: tl.constexpr,
                IS_REVERSE: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lanes = idx < COUNT
    px = tl.load(Starts + idx * 3, mask=lanes, other=0)
    py = tl.load(Starts + idx * 3 + 1, mask=lanes, other=0)
    pz = tl.load(Starts + idx * 3 + 2, mask=lanes, other=0)
    if IS_REVERSE:
        vx = -tl.load(Reverse + idx * 3, mask=lanes, other=0)
        vy = -tl.load(Reverse + idx * 3 + 1, mask=lanes, other=0)
        vz = -tl.load(Reverse + idx * 3 + 2, mask=lanes, other=0)
        jumped = vx * vx + vy * vy + vz * vz > 0
    else:
        vx = tl.full((BLOCK,), 0, tl.float32)
        vy = tl.full((BLOCK,), 0, tl.float32)
        vz = tl.full((BLOCK,), 0, tl.float32)
        jumped = tl.full((BLOCK,), False, tl.int1)
    active = lanes
    seed = tl.load(Seed)
    for step in range(HALF):
        ix = tl.floor(px + 0.5).to(tl.int32)
        iy = tl.floor(py + 0.5).to(tl.int32)
        iz = tl.floor(pz + 0.5).to(tl.int32)
        inside = (ix >= 0) & (ix < SX) & (iy >= 0) & (iy < SY) & (iz >= 0) & (iz < SZ)
        index = (ix * SY + iy) * SZ + iz
        inbrain = tl.load(Mask + index, mask=lanes & inside, other=0) != 0
        active = active & inside & inbrain
        tl.store(History + idx * HALF + step, index, mask=active)
        lx = tl.floor(px)
        ly = tl.floor(py)
        lz = tl.floor(pz)
        random_offset = idx * HALF + step
        r0, r1, r2, r3 = tl.rand4x(seed, random_offset)
        sx = lx.to(tl.int32) + (r0 < px - lx).to(tl.int32)
        sy = ly.to(tl.int32) + (r1 < py - ly).to(tl.int32)
        sz = lz.to(tl.int32) + (r2 < pz - lz).to(tl.int32)
        valid = (sx >= 0) & (sx < SX) & (sy >= 0) & (sy < SY) & (sz >= 0) & (sz < SZ)
        sample_index = (sx * SY + sy) * SZ + sz
        sample_mask = tl.load(Mask + sample_index, mask=lanes & valid, other=0) != 0
        posterior = tl.floor(r3 * (NTIME - 1) + 0.5).to(tl.int32)
        for fibre in tl.static_range(NFIB):
            data_index = (fibre * SX * SY * SZ + sample_index) * NTIME + posterior
            th = tl.load(Theta + data_index, mask=lanes & valid & sample_mask, other=0)
            ph = tl.load(Phi + data_index, mask=lanes & valid & sample_mask, other=0)
            fraction = tl.load(Fraction + data_index, mask=lanes & valid & sample_mask, other=0)
            dx = tl.sin(th) * tl.cos(ph)
            dy = tl.sin(th) * tl.sin(ph)
            dz = tl.cos(th)
            align = tl.abs(dx * vx + dy * vy + dz * vz)
            if fibre == 0:
                best_x, best_y, best_z = dx, dy, dz
                best_th, best_ph = th, ph
                best_align = tl.where(fraction > FTHR, align, -1.0)
            else:
                better = (step > 0) & (fraction > FTHR) & (align > best_align)
                best_x = tl.where(better, dx, best_x)
                best_y = tl.where(better, dy, best_y)
                best_z = tl.where(better, dz, best_z)
                best_th = tl.where(better, th, best_th)
                best_ph = tl.where(better, ph, best_ph)
                best_align = tl.where(better, align, best_align)
        cosine = best_x * vx + best_y * vy + best_z * vz
        active = active & valid & sample_mask & (best_th != 0) & (best_ph != 0)
        active = active & ((~jumped) | (tl.abs(cosine) > CTHR))
        random_sign = tl.where(tl.rand(seed + 1, random_offset) > 0.5, 1.0, -1.0)
        sign = tl.where(jumped, tl.where(cosine > 0, 1.0, -1.0), random_sign)
        dx, dy, dz = best_x * sign, best_y * sign, best_z * sign
        if step == 0:
            tl.store(First + idx * 3, dx, mask=active)
            tl.store(First + idx * 3 + 1, dy, mask=active)
            tl.store(First + idx * 3 + 2, dz, mask=active)
        px = tl.where(active, px + dx * STEP_X, px)
        py = tl.where(active, py + dy * STEP_Y, py)
        pz = tl.where(active, pz + dz * STEP_Z, pz)
        vx, vy, vz = tl.where(active, dx, vx), tl.where(active, dy, vy), tl.where(active, dz, vz)
        jumped = jumped | active


def walk(tracker, starts, generator, reverse_direction=None):
    count = starts.shape[0]
    history = torch.full((count, tracker.nsteps // 2), -1, dtype=torch.int32, device=starts.device)
    first = torch.zeros_like(starts)
    random_seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32,
                                device=starts.device, generator=generator)
    _walk_kernel[(triton.cdiv(count, 128),)](
        starts, reverse_direction if reverse_direction is not None else starts,
        tracker._mask, tracker._theta, tracker._phi, tracker._fraction,
        history, first, random_seed,
        count, tracker.nsteps // 2, *tracker._shape, tracker._ntime, tracker._theta.shape[0],
        *(tracker.steplength / tracker._voxel_size).tolist(),
        tracker.cthr, tracker.fibthresh, reverse_direction is not None, 128,
        num_warps=4)
    return history.cpu().numpy(), first
