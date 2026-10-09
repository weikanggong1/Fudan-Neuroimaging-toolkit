"""WM/aseg 的确定性 PyTorch 子阶段，保持体素网格和整数标签语义。

只替换输入决定、无需有序反馈的 MTL 补白质和后编辑；有序核心及
remove_paths_to_cortex 尚未迁移。默认 CUDA，可用 CPU 做同输入诊断。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .mgh_compat import save_same_dtype_mgh
from .wm_edits_gpu import _amygdala_cortex_junction_torch


def _validate(wm: torch.Tensor, aseg: torch.Tensor) -> None:
    if (wm.ndim != 3 or wm.shape != aseg.shape or wm.dtype != torch.uint8
            or aseg.dtype.is_floating_point or aseg.dtype == torch.bool
            or wm.device != aseg.device):
        raise ValueError("expected same-device 3D uint8 WM and integer aseg on one grid")


def _labels(seg: torch.Tensor, values: tuple[int, ...]) -> torch.Tensor:
    result = torch.zeros_like(seg, dtype=torch.bool)
    for value in values:
        result |= seg == value
    return result


def _integer_label_array(image: nib.MGHImage) -> np.ndarray:
    """MGH may encode exact integer labels as float32; reject fractional labels."""
    source = np.asarray(image.dataobj)
    if (not np.isfinite(source).all() or source.min() < np.iinfo(np.int32).min
            or source.max() > np.iinfo(np.int32).max):
        raise ValueError("label volume is not finite int32-compatible data")
    result = np.asarray(source, dtype=np.int32, order="C")
    if not np.array_equal(source, result):
        raise ValueError("label volume contains noninteger values")
    return result


def _validate_hybrid_aseg_storage(image: nib.MGHImage) -> None:
    """完整混合接口的byte-access证明只覆盖原始4字节int32/float32存储。

    image为nibabel读取的aseg影像；float32还需随后精确整数检查。
    拒绝int16/uint8等其他存储；转换到int32不能替代原存储几何证明。
    """
    storage = image.get_data_dtype().newbyteorder("=")
    if storage not in (np.dtype(np.int32), np.dtype(np.float32)):
        raise ValueError("complete hybrid byte-access proof requires original int32 or float32 aseg storage")


def _dilate(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    # Binary 0/1 max pooling is exact in float32; no TF32/half arithmetic.
    if not radius:
        return mask
    return F.max_pool3d(mask.float()[None, None], 2 * radius + 1,
                        stride=1, padding=radius)[0, 0] > 0


@torch.no_grad()
def fix_subcortical_mass_ha_torch(wm: torch.Tensor, aseg: torch.Tensor,
                                 *, ndilate: int = 1) -> torch.Tensor:
    """按 SCM-HA 规则清除 ILV/杏仁核及远离皮层的海马 WM。

    输入为同网格 3D uint8 WM 和整数 aseg，坐标为 x/y/z 体素，无 mm
    插值；ndilate 默认1且非负。返回同 dtype/device 的新 WM，不修改输入。
    网格、dtype、device 或半径错误抛 ValueError，CUDA 失败原样传播。
    """
    _validate(wm, aseg)
    if not isinstance(ndilate, int) or ndilate < 0:
        raise ValueError("ndilate must be a nonnegative integer")
    result = wm.clone()
    mask = _dilate(_labels(aseg, (0, 2, 3, 41, 42)), ndilate)
    result[_labels(aseg, (18, 54, 5, 44))] = 0
    result[_labels(aseg, (17, 53)) & ~mask] = 0
    return result


@torch.no_grad()
def fill_seg_wm_seed_torch(aseg: torch.Tensor) -> torch.Tensor:
    """产生原生第一遍WM分支用的静态fill候选，返回同网格bool张量。

    aseg为3D整数标签，坐标为x/y/z体素；排除原扫描不访问的x/y边缘，
    不排除z边缘。该掩膜必须在每个有序WM分支访问时消费，不能预先
    修改整卷WM。输入错误抛ValueError，不改变输入。
    """
    if aseg.ndim != 3 or aseg.dtype.is_floating_point or aseg.dtype == torch.bool:
        raise ValueError("expected a 3D integer aseg volume")
    cortex = _dilate(_labels(aseg, (3, 42)))
    seed = _labels(aseg, (2, 41)) & ~cortex
    seed[[0, -1], :, :] = False
    seed[:, [0, -1], :] = False
    return seed


@torch.no_grad()
def propagate_wm_from_filled_torch(wm: torch.Tensor, aseg: torch.Tensor,
                                   filled: torch.Tensor) -> torch.Tensor:
    """按固定filled种子的26邻域补IS_WM标签，包含temporal WM 186/187。

    wm uint8、aseg整数、filled bool/uint8，同3D网格/device。输出新WM，
    filled不增长；完整原生do/while只重复无变化扫描，因此一次邻域
    汇总足够。输入错误抛ValueError，不修改输入。
    """
    _validate(wm, aseg)
    if (filled.shape != wm.shape or filled.device != wm.device
            or filled.dtype not in (torch.bool, torch.uint8)):
        raise ValueError("filled flags must share the WM grid/device and be bool/uint8")
    result = wm.clone()
    neighbors = _dilate(filled != 0)
    target = _labels(aseg, (2, 41, 186, 187, 28, 60, 7, 46,
                            251, 252, 253, 254, 255))
    result[neighbors & target & (wm < 5)] = 250
    return result


@torch.no_grad()
def fill_seg_wm_from_core_torch(core: torch.Tensor,
                                aseg: torch.Tensor) -> torch.Tensor:
    """在已生成 core 上应用既有 -fill-seg-wm 后置表达式。

    完整命令中该选项位于有序核心内部；此函数只与仓库已有后置 CPU
    函数比较，尚不能证明任意输入与完整原生命令等价。输入/输出为
    同网格、同设备3D体积，WM uint8，aseg整数；不修改输入。
    """
    _validate(core, aseg)
    seed = fill_seg_wm_seed_torch(aseg)
    result = core.clone()
    result[seed] = 250
    neighbor = _dilate(seed)
    wm_label = _labels(aseg, (2, 41, 186, 187, 28, 60, 7, 46,
                             251, 252, 253, 254, 255))
    result[neighbor & wm_label & (result < 5)] = 250
    return result


@torch.no_grad()
def apply_late_wm_edits_torch(core: torch.Tensor, aseg: torch.Tensor,
                             entowm: torch.Tensor, original_wm: torch.Tensor,
                             *, fill_seg_wm: bool = False) -> torch.Tensor:
    """按 fill → SCM → keep-in → EntoWM → ACJ 顺序执行后编辑。

    四个输入均同网格、同设备；core/original_wm为uint8，aseg/entowm
    为整数。fill_seg_wm默认False，True的完整命令等价边界同上述函数。
    输出为新uint8 WM。标签含义、边缘种子排除和ACJ最后邻居规则复用
    既有实现；输入错误抛ValueError，不调用外部程序。
    """
    _validate(core, aseg)
    _validate(original_wm, entowm)
    if core.shape != original_wm.shape or core.device != original_wm.device:
        raise ValueError("all inputs must share one grid and device")
    result = (fill_seg_wm_from_core_torch(core, aseg) if fill_seg_wm
              else core.clone())
    result = fix_subcortical_mass_ha_torch(result, aseg)
    edited = (original_wm == 255) | (original_wm == 1)
    result[edited] = original_wm[edited]
    result[_labels(entowm, (3006, 3201, 4006, 4201))] = 255
    junction = _amygdala_cortex_junction_torch(aseg)
    result[(junction == 7030) | (junction == 7031)] = 255
    return result


def _edge_distances(aseg: torch.Tensor, points: torch.Tensor, labels: torch.Tensor,
                    *, axis: int, direction: int, point_chunk: int) -> torch.Tensor:
    """Exact first gap of nine label-free voxels; absent gap returns -1."""
    distances = torch.arange(aseg.shape[1], device=aseg.device)
    steps = torch.arange(1, 10, device=aseg.device)
    results = []
    for start in range(0, points.shape[0], point_chunk):
        block = points[start:start + point_chunk]
        target = labels[start:start + point_chunk]
        coordinate = (block[:, axis, None, None] + direction *
                      (distances[None, :, None] + steps[None, None, :]))
        coordinate = coordinate.clamp(0, aseg.shape[axis] - 1)
        if axis == 1:
            samples = aseg[block[:, 0, None, None], coordinate,
                           block[:, 2, None, None]]
        else:
            samples = aseg[block[:, 0, None, None],
                           block[:, 1, None, None], coordinate]
        gap = ~(samples == target[:, None, None]).any(dim=2)
        first = torch.where(gap, distances[None, :], aseg.shape[1]).amin(dim=1)
        results.append(torch.where(first == aseg.shape[1], -1, first))
    return torch.cat(results) if results else torch.empty(0, dtype=torch.long,
                                                         device=aseg.device)


@torch.no_grad()
def spackle_wm_superior_to_mtl_torch(wm: torch.Tensor, aseg: torch.Tensor,
                                    *, point_chunk: int = 512) -> torch.Tensor:
    """在 MTL 上缘补 WM=250；完整距离搜索、标签和写入目标保持原规则。

    输入同网格 uint8 WM/integer aseg，point_chunk 默认512控制临时矩阵，
    不改变候选集。返回新WM。原CPU候选若到达无定义的 lateral/y+1
    越界位置则显式拒绝，不夹取成另一算法；其它输入错误抛ValueError。
    """
    _validate(wm, aseg)
    if not isinstance(point_chunk, int) or point_chunk <= 0:
        raise ValueError("point_chunk must be a positive integer")
    candidate = _labels(aseg, (17, 53, 18, 54))
    candidate[:, 0, :] = False
    points = candidate.nonzero()
    result = wm.clone()
    if not points.shape[0]:
        return result
    labels = aseg[points[:, 0], points[:, 1], points[:, 2]]
    invalid = points[:, 1] == aseg.shape[1] - 1
    invalid |= ((labels == 18) & (points[:, 0] == aseg.shape[0] - 1))
    invalid |= ((labels == 54) & (points[:, 0] == 0))
    if bool(invalid.any()):
        raise ValueError("MTL candidate reaches an undefined original CPU boundary")
    superior = _edge_distances(aseg, points, labels, axis=1, direction=-1,
                               point_chunk=point_chunk)
    anterior = _edge_distances(aseg, points, labels, axis=2, direction=1,
                               point_chunk=point_chunk)
    below = points.clone()
    below[:, 1] += 1
    anterior_below = _edge_distances(aseg, below, labels, axis=2, direction=1,
                                     point_chunk=point_chunk)
    active = (superior <= 1) & ~((anterior < 2) & (anterior_below < 2))
    points = points[active]
    labels = labels[active]
    above = points.clone()
    above[:, 1] -= 1
    targets = [above]
    amygdala = (labels == 18) | (labels == 54)
    lateral = points[amygdala].clone()
    lateral[:, 0] += torch.where(labels[amygdala] == 18, 1, -1)
    diagonal = lateral.clone()
    diagonal[:, 1] -= 1
    targets += [lateral, diagonal]
    for target in targets:
        x, y, z = target.unbind(dim=1)
        keep = ((aseg[x, y, z] == 3) | (aseg[x, y, z] == 42)) & (result[x, y, z] < 5)
        result[x[keep], y[keep], z[keep]] = 250
    return result


@torch.no_grad()
def add_aseg_wm_below_hippocampus_torch(wm: torch.Tensor,
                                      aseg: torch.Tensor) -> torch.Tensor:
    """依据 y-1 平面3×3海马邻域补白质；标签固定，输出无更新反馈。

    WM uint8、aseg整数、同设备同网格。保持原x/y/z扫描范围。
    z最后一层存在可执行候选时拒绝原CPU未定义的z+1越界。
    """
    _validate(wm, aseg)
    candidate = _labels(aseg, (2, 41)) & (wm < 5)
    candidate[:2] = False
    candidate[-2:] = False
    candidate[:, [0, -1], :] = False
    candidate[:, :, 0] = False
    if bool(candidate[:, :, -1].any()):
        raise ValueError("WM candidate reaches an undefined original CPU z boundary")
    result = wm.clone()
    for white, hippo in ((2, 17), (41, 53)):
        plane = F.max_pool3d((aseg == hippo).float()[None, None],
                             kernel_size=(3, 1, 3), stride=1,
                             padding=(1, 0, 1))[0, 0] > 0
        below = torch.zeros_like(plane)
        below[:, 1:, :] = plane[:, :-1, :]
        result[candidate & (aseg == white) & below] = 250
    return result


def write_late_wm_edits_torch(core_file: str | Path, aseg_file: str | Path,
                             entowm_file: str | Path, original_wm_file: str | Path,
                             output_file: str | Path, *, device: str = "cuda:0",
                             fill_seg_wm: bool = False) -> dict:
    """nibabel读写同网格MGH/MGZ，保留core头/尾并报告包含读写的墙钟。

    输入文件同3D网格、affine单位mm且绝对差<=1e-4；WM uint8，标签
    整数；MGH浮点存储的精确整数标签可转换，非整数标签拒绝。
    output_file为输出路径，device默认cuda:0且不回退CPU。
    返回实现、device、输出、总时间和fill限制；失败抛原异常。
    """
    started = time.perf_counter()
    images = [nib.load(str(path)) for path in
              (core_file, aseg_file, entowm_file, original_wm_file)]
    first = images[0]
    if any(image.shape != first.shape or len(image.shape) != 3 or
           not np.allclose(image.affine, first.affine, rtol=0, atol=1e-4)
           for image in images):
        raise ValueError("all inputs must share a 3D grid and affine")
    arrays = [np.array(image.dataobj,
                dtype=np.asarray(image.dataobj).dtype.newbyteorder("="),
                order="C", copy=True) if index in (0, 3)
              else _integer_label_array(image) for index, image in enumerate(images)]
    volumes = [torch.as_tensor(array, device=device) for array in arrays]
    result = apply_late_wm_edits_torch(*volumes, fill_seg_wm=fill_seg_wm)
    save_same_dtype_mgh(core_file, output_file, result.cpu().numpy())
    return {"implementation": "FNIT PyTorch deterministic late WM edits",
            "device": str(device), "output": str(output_file),
            "total_seconds": time.perf_counter() - started,
            "fill_seg_wm": fill_seg_wm, "full_native_core_replaced": False}


def edit_wm_aseg_hybrid_diagnostic(wm: np.ndarray, brain: np.ndarray,
                                  aseg: np.ndarray, entowm: np.ndarray, *,
                                  device: str = "cuda:0",
                                  fill_seg_wm: bool = False) -> tuple[np.ndarray, dict]:
    """同输入完整 WM 编辑诊断：有序 Numba 前段 + Torch 确定性子阶段。

    输入为同网格3D数组，WM/brain uint8、aseg/entowm int32；device
    默认cuda:0，fill_seg_wm默认False。保留remove_paths的几何no-op证明，
    未覆盖的byte-access路径抛NotImplementedError。原固定被试公共入口
    保持不变；fill静态候选在首遍WM分支消费，固定filled传播移至GPU。
    返回新uint8 WM和分步时间/限制字典，不调用原生程序或读取参考。
    """
    from .edit_wm_aseg_core_python import (
        remove_paths_to_cortex, _edit_until_propagation,
        _post_spackle_early, _post_spackle_late,
    )
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("hybrid diagnostic requires an explicit CUDA device")
    if (wm.ndim != 3 or wm.shape != brain.shape or wm.shape != aseg.shape
            or wm.shape != entowm.shape or wm.dtype != np.uint8
            or brain.dtype != np.uint8 or aseg.dtype.newbyteorder("=") != np.dtype("int32")
            or entowm.dtype.newbyteorder("=") != np.dtype("int32")):
        raise ValueError("expected same-grid uint8 WM/brain and int32 aseg/entowm")
    started = time.perf_counter()
    tick = started
    ordered, path_changed = remove_paths_to_cortex(wm, brain, aseg)
    path_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    labels = torch.as_tensor(aseg, device=target)
    seed = fill_seg_wm_seed_torch(labels).cpu().numpy() if fill_seg_wm else None
    torch.cuda.synchronize(target)
    seed_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    filled = _edit_until_propagation(ordered, aseg, fill_seg_wm=fill_seg_wm,
                                     fill_seed=seed, propagate_wm=False)
    first_scan_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    propagated = propagate_wm_from_filled_torch(
        torch.as_tensor(ordered, device=target), labels,
        torch.as_tensor(filled, device=target))
    ordered = propagated.cpu().numpy()
    propagation_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    _post_spackle_early(ordered, brain, aseg)
    _post_spackle_late(ordered, aseg)
    post_scan_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    original = torch.as_tensor(wm, device=target)
    result = add_aseg_wm_below_hippocampus_torch(
        torch.as_tensor(ordered, device=target), labels)
    result = spackle_wm_superior_to_mtl_torch(result, labels)
    result = apply_late_wm_edits_torch(result, labels,
        torch.as_tensor(entowm, device=target), original, fill_seg_wm=False)
    output = result.cpu().numpy()
    cuda_seconds = time.perf_counter() - tick
    return output, {"implementation": "FNIT ordered Numba + PyTorch CUDA diagnostic",
        "device": str(target), "used_native_programs": False,
        "ordered_cpu_seconds": path_seconds + first_scan_seconds + post_scan_seconds,
        "torch_seconds_including_transfers": seed_seconds + propagation_seconds + cuda_seconds,
        "timing_parts_seconds": {"remove_paths_cpu": path_seconds,
            "label_upload_and_fill_seed": seed_seconds, "first_ordered_scans_cpu": first_scan_seconds,
            "fixed_filled_propagation_and_transfers": propagation_seconds,
            "post_ordered_scans_cpu": post_scan_seconds, "mtl_late_and_transfers": cuda_seconds},
        "total_seconds": time.perf_counter() - started,
        "path_proof_changed": int(path_changed), "fill_seg_wm": fill_seg_wm,
        "fill_insertion": "native first ordered WM-case scan; fixed filled flags GPU propagation",
        "production_default": False}


def write_wm_asegedit_hybrid_diagnostic(wm_file: str | Path,
                                       brain_file: str | Path,
                                       aseg_file: str | Path,
                                       entowm_file: str | Path,
                                       output_file: str | Path, *,
                                       device: str = "cuda:0",
                                       fill_seg_wm: bool = False) -> dict:
    """独立完整混合诊断文件API；不是默认生产入口。

    wm_file/brain_file为同网格3D uint8 MGH/MGZ，aseg_file/entowm_file
    为同网格整数语义标签。aseg原存储须int32或float32精确整数，
    byte-access证明不能用int16/uint8先转换到int32来替代。
    output_file保存新WM并保留wm_file头/尾。
    device默认cuda:0、fill_seg_wm默认False，限制同array诊断接口。
    返回分段时间及包含加载/搬运/压缩/写出的total_seconds。
    未支持的路径/错误网格/非整数标签/CUDA失败分别抛异常，不回退。
    """
    started = time.perf_counter()
    images = [nib.load(str(path)) for path in
              (wm_file, brain_file, aseg_file, entowm_file)]
    first = images[0]
    if any(image.shape != first.shape or len(image.shape) != 3 or
           not np.allclose(image.affine, first.affine, rtol=0, atol=1e-4)
           for image in images):
        raise ValueError("all inputs must share a 3D grid and affine")
    if any(image.get_data_dtype() != np.dtype(np.uint8) for image in images[:2]):
        raise ValueError("WM and brain input storage must be uint8")
    _validate_hybrid_aseg_storage(images[2])
    wm, brain = [np.array(image.dataobj, dtype=np.uint8, order="C", copy=True)
                 for image in images[:2]]
    aseg, entowm = [_integer_label_array(image) for image in images[2:]]
    output, report = edit_wm_aseg_hybrid_diagnostic(wm, brain, aseg, entowm,
                               device=device, fill_seg_wm=fill_seg_wm)
    save_same_dtype_mgh(wm_file, output_file, output)
    report["array_stage_seconds"] = report["total_seconds"]
    report["total_seconds"] = time.perf_counter() - started
    report["output"] = str(output_file)
    report["file_io_included"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core_file", "aseg_file", "entowm_file", "original_wm_file",
                 "output_file"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fill-seg-wm", action="store_true")
    args = parser.parse_args()
    print(write_late_wm_edits_torch(**vars(args)))


if __name__ == "__main__":
    main()
