"""MRtrix SIFT2 precise streamline-to-fixel mapping in PyTorch.

This ports the mapping stage of MRtrix3 3.0.3-103-g026e850d. The sparse
FMLS direction lookup table is an input, so fixel segmentation can be checked
separately. Source: ``src/dwi/tractography/mapping/mapper.h``,
``src/dwi/tractography/SIFT/model.h`` and ``track_contribution.h`` at
https://github.com/MRtrix3/mrtrix3/tree/026e850d (MPL-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class SIFT2FixelMapping:
    """Sparse contributions and density: track/fixel/length [K], TDI [F].

    ``track_index`` is zero-based; ``fixel_index`` uses MRtrix's global dummy
    fixel 0. ``length_mm`` is float32 after 8-bit MRtrix quantization, and
    ``tdi_mm`` is the unweighted sum of those lengths. All tensors remain on
    the input device. Duplicate track/fixel pairs can occur after 8-bit
    storage saturation and are valid SIFT2 optimizer input.
    """

    track_index: torch.Tensor
    fixel_index: torch.Tensor
    length_mm: torch.Tensor
    tdi_mm: torch.Tensor


def _hermite_weights(t: torch.Tensor) -> torch.Tensor:
    """Return MRtrix tension-0.1 Hermite coefficients [N,4] for positions [N]."""
    t2 = t * t
    t3 = t2 * t
    return torch.stack((
        .45 * (2 * t2 - t3 - t),
        1 + 1.55 * t3 - 2.55 * t2,
        2.1 * t2 + .45 * t - 1.55 * t3,
        .45 * (t3 - t2),
    ), dim=-1)


def _round_voxel(point: torch.Tensor) -> torch.Tensor:
    """Round voxel coordinates [...,3] to int64, with C++ half-away ties."""
    return (point.sign() * (point.abs() + .5).floor()).long()


def _upsample_tracks(paths: Sequence[torch.Tensor], ratio: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hermite upsample world-mm paths to points [P,3], track IDs [P], starts [T].

    Each input path is float32 [Ni,3] on one device. ``ratio`` is the integer
    MRtrix step-size upsample factor; output preserves path and point order.
    Equivalent original operation: ``Resampling::Upsampler`` inside
    ``tcksift2 tracks.tck wm_fod.mif weights.txt -act 5tt.mif``.
    """
    device = paths[0].device
    lengths = torch.tensor([len(path) for path in paths], device=device, dtype=torch.long)
    old_starts = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)[:-1]))
    points = torch.cat(paths)
    point_track = torch.repeat_interleave(torch.arange(len(paths), device=device), lengths)
    if ratio == 1:
        # Precise FA sampling uses the original points. Avoid constructing
        # Hermite neighbours and another copy when no upsampling is requested.
        return points, point_track, old_starts
    new_lengths = (lengths - 1) * ratio + 1
    new_starts = torch.cat((lengths.new_zeros(1), new_lengths.cumsum(0)[:-1]))
    seg = torch.arange(len(points) - 1, device=device)
    seg = seg[point_track[:-1] == point_track[1:]]
    seg_track = point_track[seg]
    first = seg == old_starts[seg_track]
    last = seg + 1 == old_starts[seg_track] + lengths[seg_track] - 1
    b, c = points[seg], points[seg + 1]
    a = points[(seg - 1).clamp_min(0)]
    d = points[(seg + 2).clamp_max(len(points) - 1)]
    a = torch.where(first[:, None], 2 * b - c, a)
    d = torch.where(last[:, None], 2 * c - b, d)
    out = torch.empty((int(new_lengths.sum()), 3), device=device, dtype=torch.float32)
    out_index = new_starts[seg_track] + (seg - old_starts[seg_track]) * ratio
    out[out_index] = b
    for j in range(1, ratio):
        w = _hermite_weights(torch.full((len(seg),), j / ratio, device=device, dtype=torch.float32))
        out[out_index + j] = w[:, :1] * a + w[:, 1:2] * b + w[:, 2:3] * c + w[:, 3:] * d
    out[new_starts + new_lengths - 1] = points[old_starts + lengths - 1]
    return out, torch.repeat_interleave(torch.arange(len(paths), device=device), new_lengths), new_starts


def _pack_fixel_bytes(
    track: torch.Tensor, fixel: torch.Tensor, encoded: torch.Tensor, n_fixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ordered dixel bytes [D] into MRtrix first-fit track/fixel records [K].

    Inputs are int tensors on one device, in MRtrix z/y/x/bin dixel order per
    track. Existing record bytes are incremented when their sum stays <=255;
    otherwise a new record is created. Outputs are track IDs, fixel IDs and
    uint8-equivalent positive byte counts, each [K]. This is the storage step
    of ``tcksift2 tracks.tck wm_fod.mif weights.txt -act 5tt.mif``.
    """
    if not len(track):
        return track, fixel, encoded
    key = track * n_fixels + fixel
    order = torch.argsort(key, stable=True)
    key, encoded = key[order], encoded[order].to(torch.int32)
    pairs, group, counts = torch.unique_consecutive(key, return_inverse=True, return_counts=True)
    total = torch.zeros(len(pairs), device=track.device, dtype=torch.int32).index_add_(0, group, encoded)
    simple = total <= 255
    out_key = pairs[simple]
    out_bytes = total[simple]
    overflow_groups = (~simple).nonzero().flatten()
    if len(overflow_groups):
        starts = counts.cumsum(0) - counts
        local = torch.arange(len(key), device=track.device) - starts[group]
        member = torch.isin(group, overflow_groups)
        overflow_id = torch.searchsorted(overflow_groups, group[member])
        local, incoming = local[member], encoded[member]
        slots = torch.full(
            (len(overflow_groups), int(counts[overflow_groups].max())), -1,
            device=track.device, dtype=torch.int32,
        )
        for position in range(slots.shape[1]):
            active = local == position
            if not bool(active.any()):
                continue
            row, value = overflow_id[active], incoming[active]
            prior = slots[row]
            fits = (prior >= 0) & (prior + value[:, None] <= 255)
            destination = torch.where(fits.any(1), fits.int().argmax(1), (prior < 0).int().argmax(1))
            old = slots[row, destination]
            slots[row, destination] = torch.where(old >= 0, old + value, value)
        valid = slots > 0
        out_key = torch.cat((out_key, pairs[overflow_groups][:, None].expand_as(slots)[valid]))
        out_bytes = torch.cat((out_bytes, slots[valid]))
    return out_key // n_fixels, out_key % n_fixels, out_bytes


def _scanner_to_voxel(point: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """Map world-mm points [N,3] through float32 inverse affine [4,4] to voxel [N,3].

    Scalar products reproduce MRtrix ``scanner2voxel`` without TF32 matrix
    multiplication rounding at voxel boundaries. TF32 remains enabled for
    other CUDA matrix operations; output is float32 on the input device.
    """
    x, y, z = point.unbind(-1)
    return torch.stack((
        x * inverse[0, 0] + y * inverse[0, 1] + z * inverse[0, 2] + inverse[0, 3],
        x * inverse[1, 0] + y * inverse[1, 1] + z * inverse[1, 2] + inverse[1, 3],
        x * inverse[2, 0] + y * inverse[2, 1] + z * inverse[2, 2] + inverse[2, 3],
    ), dim=-1)


def _voxelise_precise(
    points: torch.Tensor, point_track: torch.Tensor, starts: torch.Tensor,
    affine: torch.Tensor, volume_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MRtrix precise voxel visits: voxel [R,3], track/length [R], tangent [R,3].

    ``points`` are upsampled world-mm float32 [P,3], ``point_track`` [P]
    names their tracks, and ``starts`` [T] gives first point indices. The
    float32 affine [4,4] and shape [3] describe voxel centers. Each visit's
    length is double mm; tangent is a double unit vector. A single upsampled
    edge may visit several corner voxels, exactly as ``voxelise_precise`` in
    ``tcksift2 tracks.tck wm_fod.mif weights.txt -act 5tt.mif``.
    """
    device = points.device
    spacing = torch.linalg.vector_norm(affine[:3, :3].double(), dim=0)
    accuracy = (.005 * float(spacing.min())) ** 2
    inv = torch.linalg.inv(affine.double()).float()
    vox = _round_voxel(_scanner_to_voxel(points, inv))
    crossing = ((point_track[1:] == point_track[:-1]) & (vox[1:] != vox[:-1]).any(1)).nonzero().flatten() + 1
    event_points, event_voxels, event_edges, event_ordinals = [], [], [], []
    if len(crossing):
        prev, nxt = points[crossing - 1], points[crossing]
        is_first = crossing == starts[point_track[crossing]] + 1
        ends = torch.cat((starts[1:] - 1, starts.new_tensor([len(points) - 1])))
        is_last = crossing == ends[point_track[crossing]]
        a = torch.where(is_first[:, None], 2 * prev - nxt, points[(crossing - 2).clamp_min(0)])
        d = torch.where(is_last[:, None], 2 * nxt - prev, points[(crossing + 1).clamp_max(len(points) - 1)])
        current = vox[crossing - 1].clone()
        endpoint = vox[crossing]
        entry = prev.clone()
        mu_previous = torch.zeros(len(crossing), device=device, dtype=torch.float32)
        for ordinal in range(1, 9):
            active_index = (current != endpoint).any(1).nonzero().flatten()
            if not len(active_index):
                break
            b0, c0 = prev[active_index], nxt[active_index]
            aa, dd = a[active_index], d[active_index]
            this_voxel = current[active_index]
            low = mu_previous[active_index].clone()
            high = torch.ones_like(low)
            p_low, p_high = entry[active_index].clone(), c0.clone()
            new_voxel = endpoint[active_index].clone()
            mu = low.clone()
            for _ in range(32):
                moving = (p_low - p_high).square().sum(1) > accuracy
                if not bool(moving.any()):
                    break
                midpoint = (low + high) * .5
                w = _hermite_weights(midpoint)
                p_mid = w[:, :1] * aa + w[:, 1:2] * b0 + w[:, 2:3] * c0 + w[:, 3:] * dd
                mid_voxel = _round_voxel(_scanner_to_voxel(p_mid, inv))
                same = (mid_voxel == this_voxel).all(1)
                left = moving & same
                right = moving & ~same
                low = torch.where(left, midpoint, low)
                high = torch.where(right, midpoint, high)
                p_low = torch.where(left[:, None], p_mid, p_low)
                p_high = torch.where(right[:, None], p_mid, p_high)
                new_voxel = torch.where(right[:, None], mid_voxel, new_voxel)
                mu = torch.where(moving, midpoint, mu)
            event_points.append(p_high)
            event_voxels.append(new_voxel)
            event_edges.append(crossing[active_index] - 1)
            event_ordinals.append(torch.full_like(active_index, ordinal))
            current[active_index] = new_voxel
            entry[active_index] = p_high
            mu_previous[active_index] = mu
        if bool((current != endpoint).any()):
            raise RuntimeError("more than eight voxel crossings within one upsampled step")
    if event_points:
        event_point = torch.cat(event_points)
        event_voxel = torch.cat(event_voxels)
        event_edge = torch.cat(event_edges)
        event_ordinal = torch.cat(event_ordinals)
        keys = torch.cat((torch.arange(len(points), device=device) * 16,
                          event_edge * 16 + event_ordinal))
        order = torch.argsort(keys)
        augmented = torch.cat((points, event_point))[order]
        labels = torch.cat((vox, event_voxel))[order]
        track = torch.cat((point_track, point_track[event_edge]))[order]
    else:
        augmented, labels, track = points, vox, point_track
    same_track = track[1:] == track[:-1]
    left, right = augmented[:-1][same_track], augmented[1:][same_track]
    interval_voxel, interval_track = labels[:-1][same_track], track[:-1][same_track]
    interval_length = torch.linalg.vector_norm(right - left, dim=1).double()
    new_run = torch.ones(len(left), device=device, dtype=torch.bool)
    new_run[1:] = (interval_track[1:] != interval_track[:-1]) | (interval_voxel[1:] != interval_voxel[:-1]).any(1)
    run_start = new_run.nonzero().flatten()
    run_end = torch.cat((run_start[1:] - 1, run_start.new_tensor([len(left) - 1])))
    cumulative = torch.cat((interval_length.new_zeros(1), interval_length.cumsum(0)))
    length = cumulative[run_end + 1] - cumulative[run_start]
    traversal = right[run_end] - left[run_start]
    norm = torch.linalg.vector_norm(traversal.double(), dim=1)
    valid = (norm > 0) & torch.isfinite(norm) & (length > 0)
    valid &= ((interval_voxel[run_start] >= 0) &
              (interval_voxel[run_start] < interval_voxel.new_tensor(volume_shape))).all(1)
    return (interval_voxel[run_start[valid]], interval_track[run_start[valid]],
            length[valid], traversal[valid].double() / norm[valid, None])


@torch.inference_mode()
def map_streamlines_to_fixels(
    streamlines: Sequence[torch.Tensor],
    affine: torch.Tensor,
    volume_shape: tuple[int, int, int],
    voxel_ids: torch.Tensor,
    first_fixel_index: torch.Tensor,
    count: torch.Tensor,
    lookup_table: torch.Tensor,
    directions: torch.Tensor,
    *,
    step_size_mm: float,
    n_fixels: int,
) -> SIFT2FixelMapping:
    """Map fixed streamlines into MRtrix FMLS fixels and 8-bit lengths.

    ``streamlines`` are nonempty float32 world-mm tensors [Ni,3], Ni>=2, on
    one device in TCK order. ``affine`` [4,4] maps voxel centers (x,y,z) to
    world mm; ``volume_shape`` [3] is its image shape. ``voxel_ids`` [V] are
    sorted C-order IDs ``(x*Y+y)*Z+z`` for occupied voxels. Aligned arrays
    ``first_fixel_index`` [V] and ``count`` [V] give each voxel's contiguous
    global fixel span, including dummy fixel 0. ``lookup_table`` uint8
    [V,1281] maps MRtrix direction bin to local fixel offset; ``count`` means
    no fixel. ``directions`` float64 [1281,3] is the official dixel sphere.
    ``step_size_mm`` comes from the TCK header; ``n_fixels`` includes dummy 0.
    Integer arrays and all float tensors must reside on the path device.

    Returns ``SIFT2FixelMapping``: zero-based track IDs [K], one-based global
    fixel IDs [K], quantized float32 lengths in mm [K], and raw TDI [F]. The
    original command is ``tcksift2 tracks.tck wm_fod.mif weights.txt -act
    5tt.mif -nthreads 8 -debug``; compare ``before_tdi_fixel.msf`` divided
    by the reported SIFT2 proportionality coefficient ``mu`` to raw TDI.
    """
    if not streamlines or any(path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 for path in streamlines):
        raise ValueError("streamlines must be [Ni,3] with Ni >= 2")
    device = streamlines[0].device
    if any(path.device != device or path.dtype != torch.float32 for path in streamlines):
        raise ValueError("streamlines must share one device and float32 dtype")
    if affine.shape != (4, 4) or len(volume_shape) != 3 or min(volume_shape) < 1:
        raise ValueError("invalid voxel affine or image shape")
    if any(x.device != device for x in (affine, voxel_ids, first_fixel_index, count, lookup_table, directions)):
        raise ValueError("all inputs must be on one device")
    if (voxel_ids.ndim != 1 or first_fixel_index.shape != voxel_ids.shape or count.shape != voxel_ids.shape
            or lookup_table.shape != (len(voxel_ids), 1281) or directions.shape != (1281, 3)):
        raise ValueError("invalid sparse FMLS lookup dimensions")
    if step_size_mm <= 0 or n_fixels < 2:
        raise ValueError("invalid step size or fixel count")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    spacing = torch.linalg.vector_norm(affine[:3, :3].to(torch.float64), dim=0)
    min_spacing = float(spacing.min())
    ratio = max(1, int(torch.ceil(torch.tensor(step_size_mm / (min_spacing * .1))).item()))
    points, point_track, starts = _upsample_tracks(streamlines, ratio)
    voxel, run_track, length, tangent = _voxelise_precise(
        points, point_track, starts, affine, volume_shape,
    )
    voxel_id = (voxel[:, 0] * volume_shape[1] + voxel[:, 1]) * volume_shape[2] + voxel[:, 2]
    bins = torch.empty(len(tangent), device=device, dtype=torch.long)
    sphere = directions.to(torch.float64)
    for start in range(0, len(tangent), 8192):
        end = min(start + 8192, len(tangent))
        bins[start:end] = (tangent[start:end] @ sphere.T).abs().argmax(1)

    # SetDixel first sums all passages of one track through a voxel/bin.
    x, y, z = voxel.unbind(1)
    mrtrix_voxel_id = (z * volume_shape[1] + y) * volume_shape[0] + x
    key = ((run_track * (volume_shape[0] * volume_shape[1] * volume_shape[2])
            + mrtrix_voxel_id) * 1281 + bins)
    unique, inverse = torch.unique(key, sorted=True, return_inverse=True)
    dixel_length = torch.zeros(len(unique), device=device, dtype=torch.float64).index_add_(0, inverse, length)
    dixel_bin = unique % 1281
    pair = unique // 1281
    dixel_mrtrix = pair % (volume_shape[0] * volume_shape[1] * volume_shape[2])
    dx = dixel_mrtrix % volume_shape[0]
    dy = (dixel_mrtrix // volume_shape[0]) % volume_shape[1]
    dz = dixel_mrtrix // (volume_shape[0] * volume_shape[1])
    dixel_voxel = (dx * volume_shape[1] + dy) * volume_shape[2] + dz
    track_index = pair // (volume_shape[0] * volume_shape[1] * volume_shape[2])
    row = torch.searchsorted(voxel_ids.long(), dixel_voxel)
    in_table = row < len(voxel_ids)
    row = row.clamp_max(len(voxel_ids) - 1)
    in_table &= voxel_ids[row] == dixel_voxel
    local = lookup_table[row, dixel_bin].long()
    in_table &= local < count[row]
    maximum = float(torch.linalg.vector_norm(spacing).to(torch.float32))
    scale = float(torch.tensor(255. / maximum, dtype=torch.float32))
    in_table &= dixel_length > .5 / scale
    track_index = track_index[in_table]
    fixel_index = first_fixel_index[row[in_table]].long() + local[in_table]
    encoded = (dixel_length[in_table].float() * scale + .5).floor().clamp_max(255)
    track_index, fixel_index, packed_bytes = _pack_fixel_bytes(
        track_index, fixel_index, encoded.to(torch.int32), n_fixels,
    )
    quantized = packed_bytes.float() * float(torch.tensor(maximum / 255., dtype=torch.float32))
    tdi = torch.zeros(n_fixels, device=device, dtype=torch.float32).index_add_(0, fixel_index, quantized)
    return SIFT2FixelMapping(track_index, fixel_index, quantized, tdi)
