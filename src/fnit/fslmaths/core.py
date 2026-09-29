"""Common fslmaths operations on NIfTI images; no FSL executable is used."""

from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


_DTYPES = {"char": np.uint8, "short": np.int16, "int": np.int32,
           "float": np.float32, "double": np.float64}
_BINARY = {"-add", "-sub", "-mul", "-div", "-mas", "-max", "-min"}
_UNARY = {"-abs", "-sqr", "-sqrt", "-recip", "-exp", "-log", "-sin",
          "-cos", "-tan", "-asin", "-acos", "-atan", "-bin", "-binv",
          "-nan", "-nanm"}
_REDUCE = {"mean", "std", "max", "maxn", "min", "median"}


def _path(name):
    path = Path(name)
    return path if path.name.endswith((".nii", ".nii.gz")) else Path(f"{path}.nii.gz")


def _load(path, device, dtype):
    image = nib.load(str(path))
    if image.ndim not in (3, 4):
        raise ValueError(f"only 3D/4D NIfTI images are supported: {path}")
    data = np.asarray(image.dataobj, dtype=np.float64 if dtype == torch.float64 else np.float32)
    return image, torch.as_tensor(data.copy(), device=device, dtype=dtype)


def _save(data, source, path, output_dtype, display_range=False, modified=True):
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = data.cpu().numpy()
    if np.issubdtype(output_dtype, np.integer):
        array = np.copysign(np.floor(np.abs(array) + 0.5), array)
    array = array.astype(output_dtype)
    header = source.header.copy()
    header.set_data_dtype(output_dtype)
    header.set_slope_inter(1, 0)
    if display_range:
        header["cal_min"] = float(np.nanmin(array))
        header["cal_max"] = float(np.nanmax(array))
    elif modified:
        header["cal_min"] = 0
        header["cal_max"] = 0
    image = nib.Nifti1Image(array, source.affine, header)
    image.set_qform(source.get_qform(), int(source.header["qform_code"]))
    image.set_sform(source.get_sform(), int(source.header["sform_code"]))
    nib.save(image, str(path))
    return path


def _operand(token, data, source, device, dtype):
    try:
        return torch.as_tensor(float(token), device=device, dtype=dtype)
    except ValueError:
        image, other = _load(token, device, dtype)
        if not np.allclose(image.affine, source.affine, atol=1e-4):
            raise ValueError(f"image geometry differs from input: {token}")
        if other.shape[:3] != data.shape[:3]:
            raise ValueError(f"image grid differs from input: {token}")
        if data.ndim == 4 and other.ndim == 3:
            other = other[..., None]
        if other.ndim == 4 and data.ndim == 4 and other.shape[3] not in (1, data.shape[3]):
            raise ValueError(f"time dimensions differ: {token}")
        return other


def _kernel(kind, args, spacing, device, dtype):
    if kind in ("3D", "2D"):
        sizes = (3, 3, 3 if kind == "3D" else 1)
        kernel = torch.ones(sizes, device=device, dtype=dtype)
        return kernel, 0
    if kind in ("boxv", "boxv3"):
        count = 3 if kind == "boxv3" else 1
        if len(args) < count:
            raise ValueError(f"-kernel {kind} requires {count} size argument(s)")
        sizes = tuple(int(v) for v in args[:count])
        if count == 1:
            sizes *= 3
        if any(v < 1 or v % 2 == 0 for v in sizes):
            raise ValueError("boxv and boxv3 sizes must be positive odd integers")
        return torch.ones(sizes, device=device, dtype=dtype), count
    if kind not in ("gauss", "box", "sphere"):
        raise NotImplementedError(f"unsupported kernel: {kind}")
    if not args:
        raise ValueError(f"-kernel {kind} requires a size")
    size = float(args[0])
    if size <= 0:
        raise ValueError("kernel size must be positive")
    if kind == "gauss":
        radius = [max(1, math.ceil(4 * size / v)) for v in spacing]
    elif kind == "box":
        radius = [max(0, int(size / v / 2)) for v in spacing]
    else:
        radius = [int(math.floor(size / v + 0.5)) for v in spacing]
    grid = torch.meshgrid(*[torch.arange(-r, r + 1, device=device, dtype=dtype) * v
                            for r, v in zip(radius, spacing)], indexing="ij")
    distance2 = sum(axis.square() for axis in grid)
    if kind == "gauss":
        kernel = torch.exp(-0.5 * distance2 / (size * size))
    elif kind == "sphere":
        kernel = (distance2 <= size * size).to(dtype)
    else:
        kernel = torch.ones_like(distance2)
    return kernel, 1


def _spatial(data, kernel, mode):
    """Apply a 3D filter to each volume, keeping the time dimension separate."""
    weights = kernel.permute(2, 1, 0)[None, None]
    padding = tuple((v // 2) for v in weights.shape[-3:])
    frames = data.unbind(-1) if data.ndim == 4 else (data,)
    result = []
    for frame in frames:
        x = frame.permute(2, 1, 0)[None, None]
        if mode == "mean":
            numerator = F.conv3d(x, weights, padding=padding)
            denominator = F.conv3d(torch.ones_like(x), weights, padding=padding)
            y = numerator / denominator.clamp_min(1e-12)
        elif mode == "mean_unnormalized":
            y = F.conv3d(x, weights / weights.sum(), padding=padding)
        elif mode == "dilate_mean":
            if not torch.all(kernel == 1):
                raise NotImplementedError("-dilM currently requires a box kernel")
            nonzero = (x != 0).to(x.dtype)
            numerator = F.conv3d(x, weights, padding=padding)
            count = F.conv3d(nonzero, weights, padding=padding)
            y = torch.where(x != 0, x, numerator / count.clamp_min(1))
        elif mode == "erode_sparse":
            if not torch.all(kernel == 1):
                raise NotImplementedError("-ero currently requires a box kernel")
            count = F.conv3d((x == 0).to(x.dtype), weights, padding=padding)
            y = torch.where(count > 0, 0, x)
        elif mode in ("max", "min"):
            if not torch.all(kernel == 1):
                raise NotImplementedError("max/min filter currently requires a box kernel")
            value = -torch.inf if mode == "max" else torch.inf
            x = F.pad(x, (padding[2], padding[2], padding[1], padding[1],
                          padding[0], padding[0]), value=value)
            y = F.max_pool3d(x if mode == "max" else -x, weights.shape[-3:], stride=1)
            if mode == "min":
                y = -y
        else:
            raise NotImplementedError(mode)
        result.append(y[0, 0].permute(2, 1, 0))
    return torch.stack(result, dim=-1) if data.ndim == 4 else result[0]


def _median_filter(data, kernel):
    if not torch.all(kernel == 1):
        raise NotImplementedError("median filter currently requires a box kernel")
    sizes = kernel.shape
    radius = [s // 2 for s in sizes]
    frames = data.unbind(-1) if data.ndim == 4 else (data,)
    result = []
    for frame in frames:
        padded = F.pad(frame, (radius[2], radius[2], radius[1], radius[1],
                               radius[0], radius[0]), value=float("inf"))
        windows = padded.unfold(0, sizes[0], 1).unfold(1, sizes[1], 1).unfold(2, sizes[2], 1)
        values = windows.reshape(*frame.shape, -1).sort(dim=-1).values
        count = torch.isfinite(values).sum(dim=-1)
        result.append(values.gather(-1, (count // 2).unsqueeze(-1)).squeeze(-1))
    return torch.stack(result, dim=-1) if data.ndim == 4 else result[0]


def _reduce(data, axis, operation):
    keepdim = axis != 3
    if operation == "mean":
        return data.mean(dim=axis, keepdim=keepdim)
    if operation == "std":
        return data.std(dim=axis, unbiased=data.shape[axis] > 1, keepdim=keepdim)
    if operation == "max":
        return data.amax(dim=axis, keepdim=keepdim)
    if operation == "maxn":
        return data.argmax(dim=axis, keepdim=keepdim).to(data.dtype)
    if operation == "min":
        return data.amin(dim=axis, keepdim=keepdim)
    values = data.sort(dim=axis).values
    return values.select(axis, data.shape[axis] // 2).unsqueeze(axis) if keepdim else values.select(axis, data.shape[axis] // 2)


def run_fslmaths(input_path, operations, output_path, *, device="cpu",
                 input_dtype=None, output_dtype=None):
    """Run supported FSL-style operations in order and save a NIfTI image.

    ``input_path`` is a 3D/4D NIfTI, ``operations`` is a sequence such as
    ``["-thr", "100", "-bin"]``, and ``output_path`` is the output NIfTI path.
    The returned ``Path`` is the saved image. ``device`` accepts ``cpu`` or a
    CUDA device; ``input_dtype`` controls calculation precision, while
    ``output_dtype`` controls the stored NIfTI data type. Unimplemented options
    raise ``NotImplementedError``.
    """
    computing_device = torch.device(device)
    if computing_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    source = nib.load(str(input_path))
    if input_dtype is None:
        input_dtype = "double" if source.get_data_dtype() == np.dtype("float64") else "float"
    if input_dtype == "input":
        source_dtype = source.get_data_dtype()
        if np.issubdtype(source_dtype, np.integer):
            raise NotImplementedError("-dt input for integer images is not implemented")
        input_dtype = "double" if source_dtype == np.dtype("float64") else "float"
    if input_dtype not in ("float", "double"):
        raise NotImplementedError("-dt currently supports float and double")
    if output_dtype is not None and output_dtype not in (*_DTYPES, "input"):
        raise ValueError("-odt must be char, short, int, float, double or input")
    dtype = torch.float64 if input_dtype == "double" else torch.float32
    source, data = _load(input_path, computing_device, dtype)
    if output_dtype is None:
        output_dtype = "double" if source.get_data_dtype() == np.dtype("float64") else "float"
    result_dtype = source.get_data_dtype() if output_dtype == "input" else _DTYPES[output_dtype]
    kernel = torch.ones((3, 3, 3), device=computing_device, dtype=dtype)
    spacing = source.header.get_zooms()[:3]
    display_range = False
    i = 0
    with torch.no_grad():
        while i < len(operations):
            op = operations[i]
            if op in _BINARY:
                if i + 1 >= len(operations):
                    raise ValueError(f"missing operand for {op}")
                other = _operand(operations[i + 1], data, source, computing_device, dtype)
                if op == "-mas" and other.ndim == 0:
                    raise ValueError("-mas requires a mask image")
                if data.ndim == 3 and other.ndim == 4:
                    data = data[..., None]
                if op == "-add":
                    data = data + other
                elif op == "-sub":
                    data = data - other
                elif op == "-mul":
                    data = data * other
                elif op == "-div":
                    if other.numel() == 1 and other.item() == 0:
                        pass  # FSL leaves scalar division by zero unchanged.
                    else:
                        data = torch.where(other != 0, data / other, 0)
                elif op == "-mas":
                    data = data * (other > 0)
                elif op == "-max":
                    data = torch.maximum(data, other)
                else:
                    data = torch.minimum(data, other)
                i += 2
            elif op in ("-thr", "-uthr"):
                if i + 1 >= len(operations):
                    raise ValueError(f"missing threshold for {op}")
                value = float(operations[i + 1])
                data = torch.where(data >= value if op == "-thr" else data <= value, data, 0)
                i += 2
            elif op in ("-inm", "-ing"):
                if i + 1 >= len(operations):
                    raise ValueError(f"missing mean for {op}")
                target = float(operations[i + 1])
                if op == "-inm" and data.ndim == 4:
                    for t in range(data.shape[3]):
                        frame = data[..., t]
                        selected = frame[frame > 0]
                        if selected.numel() == 0:
                            raise ValueError("cannot normalise an empty positive image")
                        data[..., t] = frame * (target / selected.mean())
                else:
                    selected = data[data > 0]
                    if selected.numel() == 0:
                        raise ValueError("cannot normalise an empty positive image")
                    data = data * (target / selected.mean())
                i += 2
            elif op == "-pow":
                if i + 1 >= len(operations):
                    raise ValueError("-pow requires an exponent")
                data = data.pow(float(operations[i + 1]))
                i += 2
            elif op in _UNARY:
                if op == "-abs": data = data.abs()
                elif op == "-sqr": data = data.square()
                elif op == "-sqrt": data = data.clamp_min(0).sqrt()
                elif op == "-recip": data = torch.where(data != 0, data.reciprocal(), data)
                elif op == "-exp": data = data.exp()
                elif op == "-log": data = torch.where(data > 0, data.log(), data)
                elif op in ("-sin", "-cos", "-tan", "-asin", "-acos", "-atan"):
                    data = getattr(torch, op[1:])(data)
                elif op == "-bin": data = (data > 0).to(dtype)
                elif op == "-binv": data = (data <= 0).to(dtype)
                elif op == "-nan": data = torch.where(torch.isfinite(data), data, 0)
                else: data = (~torch.isfinite(data)).to(dtype)
                i += 1
            elif op == "-kernel":
                if i + 1 >= len(operations):
                    raise ValueError("missing kernel kind")
                kernel, consumed = _kernel(operations[i + 1], operations[i + 2:],
                                           spacing, computing_device, dtype)
                i += 2 + consumed
            elif op in ("-fmean", "-fmeanu", "-s", "-dilF", "-eroF", "-dilM", "-ero", "-fmedian"):
                if op == "-s":
                    if i + 1 >= len(operations):
                        raise ValueError("-s requires a sigma in mm")
                    kernel, _ = _kernel("gauss", operations[i + 1:], spacing,
                                        computing_device, dtype)
                if op == "-fmedian":
                    data = _median_filter(data, kernel)
                else:
                    mode = {"-fmean": "mean", "-fmeanu": "mean_unnormalized",
                            "-s": "mean", "-dilF": "max", "-eroF": "min",
                            "-dilM": "dilate_mean", "-ero": "erode_sparse"}[op]
                    data = _spatial(data, kernel, mode)
                i += 2 if op == "-s" else 1
            elif len(op) >= 3 and op[0] == "-" and op[1] in "TXYZ" and op[2:] in (*_REDUCE, "perc", "ar1"):
                axis = {"X": 0, "Y": 1, "Z": 2, "T": 3}[op[1]]
                if axis >= data.ndim:
                    raise ValueError(f"{op} requires a 4D image")
                if op.endswith("perc"):
                    if i + 1 >= len(operations):
                        raise ValueError(f"{op} requires a percentage")
                    percentage = float(operations[i + 1])
                    if not 0 <= percentage <= 100:
                        raise ValueError("percentage must be in [0, 100]")
                    values = data.sort(dim=axis).values
                    index = min(int(data.shape[axis] * percentage / 100), data.shape[axis] - 1)
                    selected = values.select(axis, index)
                    data = selected if axis == 3 else selected.unsqueeze(axis)
                    i += 2
                elif op.endswith("ar1"):
                    centered = data - data.mean(dim=axis, keepdim=True)
                    first = centered.narrow(axis, 0, data.shape[axis] - 1)
                    second = centered.narrow(axis, 1, data.shape[axis] - 1)
                    numerator = (first * second).sum(dim=axis, keepdim=axis != 3)
                    denominator = centered.square().sum(dim=axis, keepdim=axis != 3)
                    data = torch.where(denominator != 0, numerator / denominator, 0)
                    i += 1
                else:
                    data = _reduce(data, axis, op[2:])
                    i += 1
            elif op == "-roi":
                if i + 8 >= len(operations):
                    raise ValueError("-roi requires 8 integers")
                values = [int(v) for v in operations[i + 1:i + 9]]
                mask = torch.zeros_like(data)
                spans = []
                for axis, (start, size) in enumerate(zip(values[::2], values[1::2])):
                    extent = data.shape[axis] if axis < data.ndim else 1
                    if size == -1:
                        size = extent - start
                    if size < 0:
                        raise ValueError("ROI sizes must be positive or -1")
                    spans.append(slice(max(0, start), min(extent, start + size)))
                if data.ndim == 4 or spans[3].start == 0 and spans[3].stop > 0:
                    mask[tuple(spans[:data.ndim])] = data[tuple(spans[:data.ndim])]
                data = mask
                i += 9
            elif op == "-bptf":
                if i + 2 >= len(operations) or data.ndim != 4:
                    raise ValueError("-bptf requires two sigmas and a 4D image")
                hp, lp = map(float, operations[i + 1:i + 3])
                if hp > 0:
                    from ..feat.temporal import gaussian_highpass_matrix
                    transform = torch.as_tensor(
                        gaussian_highpass_matrix(data.shape[3], hp, preserve_mean=False),
                        device=computing_device, dtype=torch.float64)
                    flat = data.reshape(-1, data.shape[3])
                    chunks = []
                    for block in flat.split(8192):
                        centered = block.double() - block.double().mean(dim=1, keepdim=True)
                        chunks.append((centered @ transform.T).to(dtype))
                    data = torch.cat(chunks).reshape(data.shape)
                if lp > 0:
                    length = data.shape[3]
                    t = torch.arange(length, device=computing_device, dtype=torch.float64)
                    offsets = t[:, None] - t[None, :]
                    radius = int(lp * 20) + 2
                    weights = torch.exp(-0.5 * (offsets / lp).square()) * (offsets.abs() <= radius)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                    flat = data.reshape(-1, data.shape[3])
                    chunks = [(block.double() @ weights.T).to(dtype)
                              for block in flat.split(8192)]
                    data = torch.cat(chunks).reshape(data.shape)
                i += 3
            elif op == "-range":
                display_range = True
                i += 1
            else:
                raise NotImplementedError(f"fslmaths option not implemented: {op}")
    return _save(data, source, output_path, result_dtype, display_range, bool(operations))
