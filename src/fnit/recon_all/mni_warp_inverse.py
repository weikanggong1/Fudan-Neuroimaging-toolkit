"""Complete GCAM inverse: native ordered splat, GPU Voronoi and soap bubble."""
from __future__ import annotations
import time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from .ca_register_inverse import (read_warp_geometries, warp_to_source_voxels,
                                 splat_inverse_counts, splat_inverse_coordinate_sums)
from .ca_register_inverse_output import (inverse_coordinates_to_displacement_ras,
                                        write_inverse_warp_nifti)


@torch.inference_mode()
def fill_inverse_fields(averages, control, *, device="cuda:0"):
    """填充三通道源体素网格；保留0.1控制点、完整壳层及50轮/1体素停止规则。

    averages: FP32 (3,X,Y,Z)目标体素绝对坐标；control: bool (X,Y,Z)。
    device必须为可用CUDA；返回CPU FP32同形坐标及各轴迭代统计。
    不减少迭代、不以负位移近似；没有控制点或CUDA时明确报错。
    CUDA 分配、Triton launch 和下载均绑定所选设备的局部作用域；
    正常返回或异常均恢复调用者当前设备，不更改 TF32/allocator。
    """
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GCAM GPU fill requires an available explicit CUDA device")
    from .mni_warp_kernels import _expand, _soap
    import triton
    ctrl = np.asarray(control, dtype=bool)
    values = np.asarray(averages, dtype=np.float32)
    if values.shape != (3, *ctrl.shape) or ctrl.ndim != 3 or not ctrl.any():
        raise ValueError("expected (3,X,Y,Z) coordinates and a nonempty control mask")
    if not np.isfinite(values).all():
        raise ValueError("inverse seeds must be finite")
    # 当前 CUDA 设备也决定 Triton launch 的 stream；as_tensor(device=...)
    # 只选择存储设备。局部作用域覆盖全部 GPU 操作并在异常时恢复调用者。
    with torch.cuda.device(selected):
        w, h, d = ctrl.shape
        n = ctrl.size
        fields = torch.as_tensor(np.ascontiguousarray(values), device=selected).clone()
        fixed = torch.as_tensor(np.ascontiguousarray(ctrl), device=selected)
        marked = fixed.clone()
        next_marked = torch.empty_like(marked)
        grid = (triton.cdiv(n, 128),)
        expansions = 0
        while not bool(marked.all()):
            _expand[grid](fields, marked, next_marked, w, h, d, 128,
                          enable_fp_fusion=False)
            if torch.equal(marked, next_marked):
                raise RuntimeError("GCAM Voronoi frontier failed to reach the full grid")
            marked, next_marked = next_marked, marked
            expansions += 1
        coords = np.nonzero(ctrl)
        initial = tuple(v for axis in coords for v in (int(axis.min()), int(axis.max())))
        change = torch.empty(ctrl.shape, dtype=torch.float64, device=selected)
        iterations = []
        for axis in range(3):
            field = fields[axis].contiguous()
            target = field.clone()  # native target is copied BEFORE control seeding
            bounds = initial
            seed_bounds = tuple(max(lo - 2, 0) if j % 2 == 0 else min(lo + 2, ctrl.shape[j//2]-1)
                                for j, lo in enumerate(bounds))
            _soap[grid](field, field, fixed, change, w, h, d, *seed_bounds, True, 128,
                        enable_fp_fusion=False)
            maximum = float(change.max())
            for iteration in range(50):
                _soap[grid](field, target, fixed, change, w, h, d, *bounds, False, 128,
                            enable_fp_fusion=False)
                maximum = max(maximum, float(change.max()))
                field.copy_(target)  # keep native buffer lifecycle and expanding bounds
                bounds = tuple(max(lo-1,0) if j % 2 == 0 else min(lo+1,ctrl.shape[j//2]-1)
                               for j,lo in enumerate(bounds))
                if maximum < 1.0:
                    break
                maximum = 0.0
            iterations.append(iteration + 1)
        return fields.cpu().numpy(), {"voronoi_iterations": expansions,
                                      "soap_iterations_xyz": iterations,
                                      "splat": "ordered CPU Numba FP32 (no atomics)",
                                      "fill": "CUDA Triton ordered FP32; FP64 stopping test"}


def invert_mni_warp(forward, output, *, device="cuda:0"):
    """从FS NIfTI前向位移生成完整逆向位移，返回含读写的分步秒数。

    forward/output为.nii.gz路径，需原生source/target扩展、无shear、256³源网格。
    输出(X,Y,Z,1,3) scanner RAS毫米，source/target互换；device指定CUDA。
    保留原定义的CPU有序散射，GPU执行全域填充/平滑；任何失败不回退。
    """
    started = time.perf_counter()
    image = nib.load(str(forward))
    source, atlas, source_shape = read_warp_geometries(image)
    if source_shape != (256, 256, 256):
        raise ValueError("fixed MNI inverse writer requires a 256-cubed source grid")
    displacement = np.asarray(image.dataobj, dtype=np.float32)[:, :, :, 0, :]
    coordinates = warp_to_source_voxels(displacement, atlas, source)
    if not np.isfinite(coordinates).all():
        raise ValueError("forward coordinates contain nonfinite values")
    counts = splat_inverse_counts(coordinates, source_shape)
    sums = splat_inverse_coordinate_sums(coordinates, source_shape)
    control = counts >= np.float32(0.1)
    averages = np.zeros((3, *source_shape), dtype=np.float32)
    for axis in range(3):
        np.divide(sums[axis], counts, out=averages[axis], where=control)
    timings = {"load_convert_ordered_splat": time.perf_counter() - started}
    del displacement, coordinates, counts, sums
    tick = time.perf_counter()
    fields, report = fill_inverse_fields(averages, control, device=device)
    timings["transfer_fill_soap_download"] = time.perf_counter() - tick
    tick = time.perf_counter()
    result = inverse_coordinates_to_displacement_ras(fields.transpose(1,2,3,0), atlas, source)
    timings["displacement_conversion"] = time.perf_counter() - tick
    tick = time.perf_counter()
    write_inverse_warp_nifti(Path(forward), result, Path(output))
    timings["save"] = time.perf_counter() - tick
    report.update(timings_seconds=timings, total_seconds=time.perf_counter()-started,
                  device=str(device), output=str(output), full_grid=True)
    return report
