"""Run the repository's Conda-built ITK N4 filter on a 3D MGH volume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from .mgh_compat import save_same_dtype_mgh


def _to_uchar(values: np.ndarray) -> np.ndarray:
    """Match FreeSurfer's nonnegative clipping and nearest-integer conversion."""
    return np.floor(np.clip(values, 0, 255) + 0.5).astype(np.uint8)


def correct_volume(input_file: str | Path, output_file: str | Path,
                   *, binary: str | Path, reconstruction_threads: int = 1,
                   profile_path: str | Path | None = None) -> None:
    """Conda ITK N4：3D MGH/MGZ→同网格uint8 nu0，RAS/mm不变。

    binary为源码构建程序；拟合固定1线程，reconstruction_threads默认1，
    正整数只控制全分辨率空间重建。profile_path=None；指定时记录实际子段。
    旧二进制不支持新参数时仍使用原单线程，并如实记录；不改变拟合参数。
    输入/程序缺失、非法线程、维度或字节数错误抛异常；程序失败传播异常。
    对应独立benchmark N4BiasFieldCorrection，原参数见中文N4说明。
    """
    if isinstance(reconstruction_threads, bool) or not isinstance(reconstruction_threads, int) or reconstruction_threads < 1:
        raise ValueError("reconstruction_threads must be a positive integer")
    executable = Path(binary).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"N4 executable not found: {executable}")
    source = nib.load(str(input_file))
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3:
        raise ValueError("expected a 3D MGH/MGZ input")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fnit-n4-", dir=output.parent) as scratch:
        input_raw, output_raw = (Path(scratch) / name for name in ("input.raw", "output.raw"))
        np.asarray(source.dataobj, dtype=np.float32).ravel(order="F").tofile(input_raw)
        command = [str(executable), str(input_raw), str(output_raw),
                   *(str(size) for size in source.shape),
                   *(repr(float(size)) for size in source.header.get_zooms()[:3])]
        native_profile = Path(scratch) / "profile.json"
        supported = False
        if reconstruction_threads != 1 or profile_path is not None:
            probe = subprocess.run([str(executable), "--capabilities"], capture_output=True, text=True)
            if probe.returncode == 0:
                try:
                    supported = bool(json.loads(probe.stdout).get("reconstruction_threads"))
                except (ValueError, AttributeError):
                    pass
            if supported:
                command.extend([str(reconstruction_threads), str(native_profile)])
        subprocess.run(command, check=True)
        result = np.fromfile(output_raw, dtype=np.float32)
        if result.size != np.prod(source.shape):
            raise ValueError("N4 output byte count does not match the input shape")
        uchar = _to_uchar(result.reshape(source.shape, order="F"))
        profile = json.loads(native_profile.read_text()) if native_profile.exists() else {}
        profile.update(requested_reconstruction_threads=reconstruction_threads,
                       reconstruction_threads=reconstruction_threads if supported else 1,
                       native_profile_available=native_profile.exists(), fitting_threads=1)
    if source.get_data_dtype() == np.dtype("uint8"):
        save_same_dtype_mgh(input_file, output, uchar)
    else:
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.MGHImage(uchar, source.affine, header), str(output))
    if profile_path is not None:
        path = Path(profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", required=True, dest="input_file")
    parser.add_argument("--o", required=True, dest="output_file")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--reconstruction-threads", type=int, default=1)
    parser.add_argument("--profile", dest="profile_path")
    args = parser.parse_args(argv)
    correct_volume(args.input_file, args.output_file, binary=args.binary,
                   reconstruction_threads=args.reconstruction_threads, profile_path=args.profile_path)


if __name__ == "__main__":
    main()
