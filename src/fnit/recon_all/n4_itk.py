"""Run the repository's Conda-built ITK N4 filter on a 3D MGH volume."""

from __future__ import annotations

import argparse
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
                   *, binary: str | Path) -> None:
    """Write ``nu0.mgz`` from ``orig.mgz`` using the Conda-built N4 filter."""
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
        subprocess.run(command, check=True)
        result = np.fromfile(output_raw, dtype=np.float32)
        if result.size != np.prod(source.shape):
            raise ValueError("N4 output byte count does not match the input shape")
        uchar = _to_uchar(result.reshape(source.shape, order="F"))
    if source.get_data_dtype() == np.dtype("uint8"):
        save_same_dtype_mgh(input_file, output, uchar)
    else:
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.MGHImage(uchar, source.affine, header), str(output))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", required=True, dest="input_file")
    parser.add_argument("--o", required=True, dest="output_file")
    parser.add_argument("--binary", required=True)
    args = parser.parse_args(argv)
    correct_volume(args.input_file, args.output_file, binary=args.binary)


if __name__ == "__main__":
    main()
