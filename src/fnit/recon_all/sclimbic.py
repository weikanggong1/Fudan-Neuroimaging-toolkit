"""PyTorch inference for FreeSurfer's auxiliary sclimbic/EntoWM model.

Network tensors use (batch, channels, x, y, z). H5 weights are read in place.
The image path handles 1 mm recon-all inputs without a FreeSurfer runtime.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .estimated_tiv import estimate_tiv


class LimbicUNet(nn.Module):
    """The three-level U-Net used by entowm, MCA/dura, and vsinus models."""

    def __init__(self, classes: int):
        super().__init__()
        channels = (24, 48, 96)
        self.down = nn.ModuleList()
        self.down_bn = nn.ModuleList()
        previous = 1
        for width in channels:
            self.down.append(nn.ModuleList((
                nn.Conv3d(previous, width, 3, padding=1),
                nn.Conv3d(width, width, 3, padding=1),
            )))
            self.down_bn.append(nn.BatchNorm3d(width, eps=1e-3))
            previous = width

        self.up = nn.ModuleList()
        self.up_bn = nn.ModuleList()
        for width, skip in ((48, 48), (24, 24)):
            self.up.append(nn.ModuleList((
                nn.Conv3d(previous + skip, width, 3, padding=1),
                nn.Conv3d(width, width, 3, padding=1),
            )))
            self.up_bn.append(nn.BatchNorm3d(width, eps=1e-3))
            previous = width
        self.likelihood = nn.Conv3d(24, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips = []
        x = image
        for level, (block, norm) in enumerate(zip(self.down, self.down_bn)):
            x = F.elu(block[1](F.elu(block[0](x))))
            if level < 2:
                # The reference concatenates the second convolution's output,
                # before batch normalization, into the decoder.
                skips.append(x)
            x = norm(x)
            if level < 2:
                x = F.max_pool3d(x, 2)
        for block, norm in zip(self.up, self.up_bn):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat((skips.pop(), x), dim=1)
            x = norm(F.elu(block[1](F.elu(block[0](x)))))
        return torch.softmax(self.likelihood(x), dim=1)

    @classmethod
    def from_h5(cls, path: str | Path) -> "LimbicUNet":
        """Load convolution and inference batch-normalization parameters."""
        with h5py.File(path, "r") as source:
            classes = int(source["unet_likelihood"]["unet_likelihood"]["bias:0"].shape[0])
            model = cls(classes)

            def conv(name: str, layer: nn.Conv3d) -> None:
                group = source[name][name]
                weight = torch.from_numpy(group["kernel:0"][()]).permute(4, 3, 0, 1, 2)
                layer.weight.copy_(weight)
                layer.bias.copy_(torch.from_numpy(group["bias:0"][()]))

            def batch_norm(name: str, layer: nn.BatchNorm3d) -> None:
                group = source[name][name]
                layer.weight.copy_(torch.from_numpy(group["gamma:0"][()]))
                layer.bias.copy_(torch.from_numpy(group["beta:0"][()]))
                layer.running_mean.copy_(torch.from_numpy(group["moving_mean:0"][()]))
                layer.running_var.copy_(torch.from_numpy(group["moving_variance:0"][()]))

            with torch.no_grad():
                for level, (block, norm) in enumerate(zip(model.down, model.down_bn)):
                    for index, layer in enumerate(block):
                        conv(f"unet_conv_downarm_{level}_{index}", layer)
                    batch_norm(f"unet_bn_down_{level}", norm)
                for index, (block, norm) in enumerate(zip(model.up, model.up_bn)):
                    for conv_index, layer in enumerate(block):
                        conv(f"unet_conv_uparm_{index + 3}_{conv_index}", layer)
                    batch_norm(f"unet_bn_up_{index}", norm)
                conv("unet_likelihood", model.likelihood)
        return model.eval()


def _orient(data: torch.Tensor, transform: np.ndarray) -> torch.Tensor:
    flips = [axis for axis in range(3) if transform[axis, 1] < 0]
    return data.flip(flips).permute(*np.argsort(transform[:, 0]).tolist())


def _fit_shape(data: torch.Tensor, fov: int) -> tuple[torch.Tensor, np.ndarray]:
    shape = np.asarray(data.shape)
    delta = (fov - shape) / 2
    low = np.floor(delta).astype(int)
    high = np.ceil(delta).astype(int)
    pad_low, pad_high = np.maximum(low, 0), np.maximum(high, 0)
    data = F.pad(data, (int(pad_low[2]), int(pad_high[2]),
                        int(pad_low[1]), int(pad_high[1]),
                        int(pad_low[0]), int(pad_high[0])))
    crop_low, crop_high = np.maximum(-high, 0), np.maximum(-low, 0)
    stop = np.asarray(data.shape) - crop_high
    fitted = data[crop_low[0]:stop[0], crop_low[1]:stop[1], crop_low[2]:stop[2]]
    return fitted, crop_low - pad_low


def _ctab_rows(path: str | Path) -> list[tuple[int, str]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            fields = line.split()
            rows.append((int(fields[0]), fields[1]))
    return rows


def _etiv_from_lta(path: str | Path) -> float:
    lines = Path(path).read_text().splitlines()
    first = next(i + 1 for i, line in enumerate(lines) if line.strip() == "1 4 4")
    matrix = np.array([[float(value) for value in line.split()]
                       for line in lines[first:first + 4]])
    return 1_948_106.0 / np.linalg.det(matrix[:3, :3])


def _write_stats(path: str | Path, rows: list[tuple[int, str]],
                 counts: torch.Tensor, volumes: torch.Tensor,
                 etiv: float | None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        file.write("# Subcortical Limbic Volumetric Stats\n")
        file.write("# Created by mri_sclimbic_seg\n")
        if etiv is not None:
            file.write("# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated "
                       f"Total Intracranial Volume, {etiv:.6f}, mm^3\n")
        file.write(f"# NRows {len(rows) - 1}\n")
        file.write("# NTableCols 5\n")
        file.write("# ColHeaders Index SegId NVoxels Volume_mm3 StructName\n")
        for channel, (label, name) in enumerate(rows[1:], 1):
            file.write(f"{channel:<4} {label:>6}{int(counts[channel]):>6}"
                       f"{float(volumes[channel]):>12.4f}    {name}\n")


def _cleanup(posterior: torch.Tensor) -> tuple[torch.Tensor, tuple[slice, ...]]:
    """Match mri_sclimbic_seg's six-neighbor per-label dilation cleanup."""
    hard = posterior.argmax(0)
    locations = (hard > 0).nonzero()
    if locations.numel():
        lo = (locations.min(0).values - 2).clamp(min=0)
        hi = (locations.max(0).values + 3).minimum(torch.tensor(hard.shape, device=hard.device))
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    else:
        box = tuple(slice(0, side) for side in hard.shape)
    posterior = posterior[(slice(None),) + box].clone()
    hard = hard[box]
    kernel = torch.zeros((1, 1, 3, 3, 3), device=posterior.device)
    kernel[0, 0, 1, 1, 1] = 1
    for axis in range(3):
        index = [1, 1, 1]
        index[axis] = 0
        kernel[(0, 0, *index)] = 1
        index[axis] = 2
        kernel[(0, 0, *index)] = 1
    for label in range(1, posterior.shape[0]):
        near = F.conv3d((hard == label).float()[None, None], kernel, padding=1)[0, 0] > 0
        posterior[label] *= near
    posterior[0] = (1 - posterior[1:].sum(0)).clamp(0, 1)
    posterior.clamp_(0, 1)
    posterior /= posterior.sum(0, keepdim=True)
    return posterior, box


@torch.inference_mode()
def segment_sclimbic_image(source: nib.spatialimages.SpatialImage, *,
                           model_path: str | Path, rows,
                           fov: int = 160, device: str = "cpu",
                           model: torch.nn.Module | None = None,
                           stats_path: str | Path | None = None,
                           etiv: float | None = None,
                           precision_report: list | None = None) -> nib.MGHImage:
    """在内存中对1 mm图像执行既有sclimbic推理，返回同网格int32标签图。

    source为nibabel影像，rows为(label_id, 中文或原标签名)序列。
    model_path用于加载HDF5权重；model可传同权重、同device的已加载模型，
    默认None时加载一次。fov为RAS网络立方边长（体素），默认160；
    device为显式CPU/CUDA。stats_path可写体积统计，etiv单位mm³。
    precision_report可传列表记录实际前向dtype、TF32和autocast。
    cuDNN TF32继承调用者当前设置；本函数不强制开启或关闭，不改全局默认。
    返回MGHImage，原网格/affine、标签int32；不写影像临时文件。
    非1 mm输入、标签/模型通道不符或推理失败抛异常。
    算法对应mri_sclimbic_seg内部推理，没有独立原软件CLI。
    """
    voxel_size = np.linalg.norm(source.affine[:3, :3], axis=0)
    if not np.allclose(voxel_size, 1, atol=1e-2):
        raise ValueError("This prototype currently requires a 1 mm recon-all input")
    image = torch.as_tensor(np.asanyarray(source.dataobj).astype("float32"), device=device)
    dmin = image.min()
    dmax = torch.quantile(image[image != 0], 0.999)
    image = ((image - dmin) / (dmax - dmin)).clamp_(0, 1)

    native_orientation = nib.orientations.io_orientation(source.affine)
    ras_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    to_ras = nib.orientations.ornt_transform(native_orientation, ras_orientation)
    to_native = nib.orientations.ornt_transform(ras_orientation, native_orientation)
    ras = _orient(image, to_ras)
    conformed, offset = _fit_shape(ras, fov)

    if model is None:
        model = LimbicUNet.from_h5(model_path).to(device).eval()
    with torch.backends.cudnn.flags(enabled=True,
                                    allow_tf32=torch.backends.cudnn.allow_tf32):
        if precision_report is not None:
            from .profiling import record_network_forward
            record_network_forward(model, conformed, precision_report, model=str(model_path))
        posterior = model(conformed[None, None])[0]
    posterior, box = _cleanup(posterior)
    labels = torch.tensor([label for label, _ in rows], dtype=torch.int32, device=device)
    if len(labels) != posterior.shape[0]:
        raise ValueError("Color table and model channel count differ")
    hard = posterior.argmax(0)
    if stats_path is not None:
        counts = torch.bincount(hard.flatten(), minlength=len(rows)).cpu()
        volumes = posterior.flatten(1).sum(1).cpu()
        _write_stats(stats_path, rows, counts, volumes, etiv)
    conformed_seg = torch.zeros((fov, fov, fov), dtype=torch.int32, device=device)
    conformed_seg[box] = labels[hard]

    ras_seg = torch.zeros_like(ras, dtype=torch.int32)
    ras_low = np.maximum(offset, 0)
    ras_high = np.minimum(np.asarray(ras.shape), offset + fov)
    fitted_low = ras_low - offset
    fitted_high = ras_high - offset
    target = tuple(slice(int(a), int(b)) for a, b in zip(ras_low, ras_high))
    original = tuple(slice(int(a), int(b)) for a, b in zip(fitted_low, fitted_high))
    ras_seg[target] = conformed_seg[original]
    native = _orient(ras_seg, to_native).cpu().numpy()
    header = source.header.copy() if isinstance(source, nib.MGHImage) else None
    if header is not None:
        header.set_data_dtype(np.int32)
    return nib.MGHImage(np.ascontiguousarray(native),
                        source.affine if header is None else None, header=header)


@torch.inference_mode()
def mri_sclimbic_seg(input_path: str | Path, output_path: str | Path, *,
                     model_path: str | Path, ctab_path: str | Path, fov: int = 160,
                     device: str = "cpu", stats_path: str | Path | None = None,
                     etiv: float | None = None, precision_report: list | None = None) -> Path:
    """读取1 mm影像与ctab，调用同网格内存推理并写出MGZ标签。

    路径、fov、device、stats_path和etiv沿用原接口；precision_report为可选
    实际前向记录列表。输出Path指向原网格int32标签，失败抛异常。
    """
    result = segment_sclimbic_image(
        nib.load(str(input_path)), model_path=model_path, rows=_ctab_rows(ctab_path),
        fov=fov, device=device, stats_path=stats_path, etiv=etiv,
        precision_report=precision_report)
    nib.save(result, str(output_path))
    return Path(output_path)

ENTOWM_MODEL = "entowm.fsm31.t1.nstd00-30.nstd21-108.h5"
ENTOWM_CTAB = "entowm.ctab"


def mri_entowm_seg(input_path: str | Path, output_path: str | Path,
                   asset_dir: str | Path, *, device: str = "cpu",
                   stats_path: str | Path | None = None,
                   talairach_lta: str | Path | None = None,
                   talairach_xfm: str | Path | None = None,
                   precision_report: list | None = None) -> Path:
    """Segment entorhinal/ambiens white matter from a 1 mm recon-all T1."""
    assets = Path(asset_dir)
    etiv = (_etiv_from_lta(talairach_lta) if talairach_lta else
            estimate_tiv(talairach_xfm) if talairach_xfm else None)
    return mri_sclimbic_seg(input_path, output_path,
                            model_path=assets / ENTOWM_MODEL,
                            ctab_path=assets / ENTOWM_CTAB,
                            fov=160, device=device, stats_path=stats_path,
                            etiv=etiv, precision_report=precision_report)


def main(argv: list[str] | None = None) -> int:
    """CLI compatible with recon-all's `mri_entowm_seg --s ...` call."""
    parser = argparse.ArgumentParser(prog="mri_entowm_seg")
    parser.add_argument("--s", nargs="+", metavar="SUBJECT")
    parser.add_argument("--sd", default=os.environ.get("SUBJECTS_DIR"))
    parser.add_argument("--i")
    parser.add_argument("--o")
    parser.add_argument("--assets", default=os.environ.get("FNIT_WEIGHTS"))
    parser.add_argument("--conform", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if os.environ.get("FS_TORCH_MODEL_DIR"):
        args.assets = os.environ["FS_TORCH_MODEL_DIR"]
    if args.assets is None:
        parser.error("--assets or FNIT_WEIGHTS is required")
    if args.s is None and (args.i is None or args.o is None):
        parser.error("provide --s SUBJECT or both --i and --o")
    if args.s is not None and args.sd is None:
        parser.error("--sd or SUBJECTS_DIR is required with --s")
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    if args.s is None:
        mri_entowm_seg(args.i, args.o, args.assets, device=device)
        return 0

    summary = []
    names = [name for _, name in _ctab_rows(Path(args.assets) / ENTOWM_CTAB)][1:]
    for subject in args.s:
        subject_dir = Path(args.sd) / subject
        stats = subject_dir / "stats" / "entowm.stats"
        lta = subject_dir / "mri" / "transforms" / "talairach.xfm.lta"
        mri_entowm_seg(subject_dir / "mri" / "nu.mgz",
                       subject_dir / "mri" / "entowm.mgz", args.assets,
                       device=device, stats_path=stats,
                       talairach_lta=lta if lta.is_file() else None,
                       talairach_xfm=None if lta.is_file() else
                           subject_dir / "mri" / "transforms" / "talairach.xfm")
        values = [float(line.split()[3]) for line in stats.read_text().splitlines()
                  if line and not line.startswith("#")]
        summary.append((subject, values))
    with (Path(args.sd) / "entowm_volumes_all.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case", *names])
        for subject, values in summary:
            writer.writerow([subject, *[f"{value:.4f}" for value in values]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
