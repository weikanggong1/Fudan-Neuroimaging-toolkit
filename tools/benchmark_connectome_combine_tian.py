"""用同一真实皮层/Tian 体积对照原 UKB 图谱合并脚本。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from connectome_benchmark_common import _sha256
from fnit.connectome import ConnectomeNode, combine_cortical_tian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cortical", "tian", "tian-names", "left-annot", "right-annot",
                 "official-combined", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    left_names = nib.freesurfer.read_annot(str(args.left_annot))[2]
    right_names = nib.freesurfer.read_annot(str(args.right_annot))[2]
    names = tuple(name.decode() for name in (*left_names[1:], *right_names[1:]))
    nodes = tuple(ConnectomeNode(
        index=i, original_label=i, hemisphere="L" if i <= len(left_names) - 1 else "R",
        name=name,
    ) for i, name in enumerate(names, 1))
    start = time.perf_counter()
    combined, combined_nodes = combine_cortical_tian(
        cortical_t1=nib.load(str(args.cortical)), cortical_nodes=nodes,
        tian_t1=nib.load(str(args.tian)),
        tian_names=tuple(args.tian_names.read_text().splitlines()),
    )
    seconds = time.perf_counter() - start
    reference = nib.load(str(args.official_combined))
    expected = nib.as_closest_canonical(reference)
    actual = nib.as_closest_canonical(combined)
    if actual.shape != expected.shape or not np.allclose(actual.affine, expected.affine, atol=1e-5):
        raise ValueError("combined atlas grids differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(combined, str(args.output_dir / "combined_schaefer200_tian_s1_t1.nii.gz"))
    report = {
        "input_sha256": {name: _sha256(path) for name, path in (
            ("cortical", args.cortical), ("tian", args.tian),
            ("official_combined", args.official_combined))},
        "fnit_seconds": seconds,
        "nodes": len(combined_nodes),
        "voxel_xor": int(np.count_nonzero(np.asarray(actual.dataobj) !=
                                          np.asarray(expected.dataobj))),
        "voxel_count": int(np.prod(actual.shape)),
        "label_min_max": [int(np.min(np.asarray(actual.dataobj))),
                          int(np.max(np.asarray(actual.dataobj)))],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
