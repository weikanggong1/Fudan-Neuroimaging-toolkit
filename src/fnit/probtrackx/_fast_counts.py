"""无额外路径约束时，按轨迹累计密度和区域连接的 CPU 快速路径。"""

import numpy as np
from numba import njit


@njit(cache=True)
def accumulate_paths(forward, backward, roi_row, roi_lookup, nregions,
                     steplength, distthresh, pathdist, mean_path_length,
                     density, length_sum, visit_count, network,
                     network_length_sum, network_count, totals):
    """输入两段轨迹的体素索引；就地更新路径图、长度图、区域矩阵及有效轨迹数。"""
    seen = np.zeros(density.size, dtype=np.int32)
    seen_forward = np.zeros(density.size, dtype=np.int32)
    lengths = np.zeros(density.size, dtype=np.float32)
    visited = np.empty(forward.shape[1] + backward.shape[1], dtype=np.int32)
    hit_forward = np.zeros(nregions, dtype=np.bool_)
    hit_backward = np.zeros(nregions, dtype=np.bool_)
    first_forward = np.zeros(nregions, dtype=np.float32)
    first_backward = np.zeros(nregions, dtype=np.float32)

    for stream in range(forward.shape[0]):
        count_forward = 0
        count_backward = 0
        while count_forward < forward.shape[1] and forward[stream, count_forward] >= 0:
            count_forward += 1
        while count_backward < backward.shape[1] and backward[stream, count_backward] >= 0:
            count_backward += 1
        if max(0, count_forward - 1) * steplength < distthresh:
            count_forward = 0
        if max(0, count_backward - 1) * steplength < distthresh:
            count_backward = 0

        keep_forward = nregions == 0
        keep_backward = nregions == 0
        if nregions:
            for target in range(nregions):
                hit_forward[target] = False
                hit_backward[target] = False
            for step in range(1, count_forward):
                target = roi_lookup[forward[stream, step]]
                if target >= 0 and target != roi_row and not hit_forward[target]:
                    hit_forward[target] = True
                    first_forward[target] = step * steplength
                    keep_forward = True
            for step in range(1, count_backward):
                target = roi_lookup[backward[stream, step]]
                if target >= 0 and target != roi_row and not hit_backward[target]:
                    hit_backward[target] = True
                    first_backward[target] = step * steplength
                    keep_backward = True
            if not (keep_forward or keep_backward):
                continue
            for target in range(nregions):
                if hit_forward[target] or hit_backward[target]:
                    distance = (first_forward[target] + first_backward[target]) / 2 \
                        if hit_forward[target] and hit_backward[target] else \
                        (first_forward[target] if hit_forward[target] else first_backward[target])
                    if pathdist:
                        network[roi_row, target] += distance
                    else:
                        network[roi_row, target] += 1
                    if mean_path_length:
                        network_length_sum[roi_row, target] += distance
                        network_count[roi_row, target] += 1

        tag = stream + 1
        count_visited = 0
        if keep_backward:
            for step in range(count_backward):
                voxel = backward[stream, step]
                if seen[voxel] != tag:
                    seen[voxel] = tag
                    visited[count_visited] = voxel
                    count_visited += 1
                    lengths[voxel] = step * steplength
        if keep_forward:
            for step in range(count_forward):
                voxel = forward[stream, step]
                if seen[voxel] != tag:
                    seen[voxel] = tag
                    visited[count_visited] = voxel
                    count_visited += 1
                if seen_forward[voxel] != tag:
                    seen_forward[voxel] = tag
                    lengths[voxel] = step * steplength
        if count_visited:
            totals[roi_row] += 1
            for index in range(count_visited):
                voxel = visited[index]
                density[voxel] += lengths[voxel] if pathdist else 1
                if mean_path_length:
                    length_sum[voxel] += lengths[voxel]
                    visit_count[voxel] += 1


@njit(cache=True)
def accumulate_single_waypoint_paths(forward, backward, avoid, waypoint,
                                     density, totals, roi_row):
    """单 waypoint 和 avoid 的计数路径；每条有效轨迹对体素只计一次。"""
    seen = np.zeros(density.size, dtype=np.int32)
    for stream in range(forward.shape[0]):
        count_forward = 0
        hit_forward = False
        rejected = False
        for step in range(forward.shape[1]):
            voxel = forward[stream, step]
            if voxel < 0:
                break
            count_forward += 1
            rejected |= avoid[voxel]
            if step > 0:
                hit_forward |= waypoint[voxel]
        if rejected:
            count_forward = 0
            hit_forward = False

        count_backward = 0
        hit_backward = False
        rejected = False
        for step in range(backward.shape[1]):
            voxel = backward[stream, step]
            if voxel < 0:
                break
            count_backward += 1
            rejected |= avoid[voxel]
            if step > 0:
                hit_backward |= waypoint[voxel]
        if rejected:
            count_backward = 0
            hit_backward = False

        tag = stream + 1
        visited = False
        if hit_forward:
            for step in range(count_forward):
                voxel = forward[stream, step]
                if seen[voxel] != tag:
                    seen[voxel] = tag
                    density[voxel] += 1
                    visited = True
        if hit_forward or hit_backward:
            for step in range(count_backward):
                voxel = backward[stream, step]
                if seen[voxel] != tag:
                    seen[voxel] = tag
                    density[voxel] += 1
                    visited = True
        if visited:
            totals[roi_row] += 1
