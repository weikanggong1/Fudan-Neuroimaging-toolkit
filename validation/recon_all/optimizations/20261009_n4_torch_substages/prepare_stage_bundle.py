"""打包已授权公开 T1 的冻结 N4 输入、几何及显式参考，不修改源数据。

--config 是 JSON 数组；每项必须有 id/input_image/public_source_url，
可指定 reference_raw、reference_kind 与 source_t1。--output-directory 必须
不存在；--archive 指定新 tar.gz。只复制显式文件，不扫描目录或资产。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import nibabel as nib
import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(*, config: Path, output_directory: Path, archive: Path) -> dict:
    cases = json.loads(config.read_text())
    if output_directory.exists() or archive.exists():
        raise FileExistsError("stage bundle refuses to overwrite an existing directory/archive")
    if not isinstance(cases, list) or not cases:
        raise ValueError("config must be a nonempty case list")
    identifiers = [case["id"] for case in cases]
    if len(set(identifiers)) != len(identifiers) or any(x in ("", ".", "..") or Path(x).name != x for x in identifiers):
        raise ValueError("case ids must be unique filename components")
    if archive.resolve().is_relative_to(output_directory.resolve()):
        raise ValueError("archive must be outside the source bundle directory")
    output_directory.mkdir(parents=True)
    result = {"kind": "public_frozen_n4_inputs_not_raw_t1_recon_pipeline",
              "config_sha256": digest(config), "cases": []}
    for case in cases:
        directory = output_directory / case["id"]
        directory.mkdir()
        source = Path(case["input_image"])
        image = nib.load(str(source))
        array = np.asarray(image.dataobj, dtype=np.float32)
        if array.ndim != 3 or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("frozen N4 input must be finite nonnegative 3D")
        raw = directory / "input.raw"
        array.ravel(order="F").tofile(raw)
        copied = directory / ("input" + "".join(source.suffixes))
        shutil.copyfile(source, copied)
        entry = {"id": case["id"], "public_source_url": case["public_source_url"],
            "input_kind": case.get("input_kind", "frozen conformed stage input"),
            "source_image_path": str(source), "source_image_sha256": digest(source),
            "input_image": str(copied.relative_to(output_directory)),
            "input_raw": str(raw.relative_to(output_directory)), "input_raw_sha256": digest(raw),
            "shape": list(array.shape), "spacing_mm": list(map(float, image.header.get_zooms()[:3])),
            "affine": image.affine.tolist(), "source_dtype": str(image.get_data_dtype()),
            "raw_dtype": "float32", "raw_order": "Fortran x-fast", "interpolation": False,
            "reference_kind": case.get("reference_kind", "fresh same-host native reference required")}
        if case.get("source_t1"):
            entry["source_t1_sha256"] = digest(Path(case["source_t1"]))
        if case.get("reference_raw"):
            reference = Path(case["reference_raw"])
            if reference.stat().st_size != raw.stat().st_size:
                raise ValueError("FP32 raw reference size differs from frozen input")
            destination = directory / "historical_reference.raw"
            shutil.copyfile(reference, destination)
            entry["historical_reference_raw"] = str(destination.relative_to(output_directory))
            entry["historical_reference_sha256"] = digest(destination)
        (directory / "geometry.json").write_text(json.dumps(entry, indent=2) + "\n")
        result["cases"].append(entry)
    result["files"] = {str(path.relative_to(output_directory)): {
        "sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(output_directory.rglob("*")) if path.is_file()}
    (output_directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(output_directory, arcname=output_directory.name)
    receipt = {"archive": str(archive), "archive_sha256": digest(archive),
               "archive_bytes": archive.stat().st_size, "cases": identifiers}
    archive.with_suffix(archive.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(prepare(config=arguments.config, output_directory=arguments.output_directory,
                             archive=arguments.archive)))


if __name__ == "__main__":
    main()
