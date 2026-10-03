"""Read ten declared real inputs and prepare one subject; no tractography."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.template_inputs import TemplateSpec, prepare_template


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    bindings = json.loads(args.bindings.read_text())
    report = {"kind": "read_only_real_input_validation", "threads": 4,
              "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),
              "scope": "ten real subject input preflight and CON01 preparation; no end-to-end tracking benchmark",
              "source_sha256": {str(path): sha(path) for path in (
                  Path("src/fnit/connectome/template_inputs.py"), Path("src/fnit/connectome/assignment.py"),
                  Path("src/fnit/connectome/paired_assignment.py"))}, "cases": {}, "prepared": {}}
    for case, entry in bindings["cases"].items():
        root = Path(entry["anatomy"]["directory"])
        case_data = {}
        for name in ("aparc", "aparc.a2009s"):
            for hemi in ("lh", "rh"):
                path = root / f"label/{hemi}.{name}.annot"
                labels, _, names = nib.freesurfer.read_annot(path)
                vertices = nib.freesurfer.read_geometry(root / f"surf/{hemi}.white")[0]
                assert len(labels) == len(vertices)
                case_data[f"{hemi}.{name}"] = {"sha256": sha(path), "vertices": len(labels),
                                                "declared_nonbackground_nodes": len(names) - 1}
        for name in ("aparc+aseg", "aparc.a2009s+aseg"):
            path = root / f"mri/{name}.mgz"
            image = nib.load(path)
            values = np.asarray(image.dataobj)
            assert len(image.shape) == 3 and np.isfinite(values).all() and np.equal(values, np.round(values)).all()
            case_data[name] = {"sha256": sha(path), "shape": [int(v) for v in image.shape],
                               "nonbackground_rois": int((np.unique(values) > 0).sum())}
        report["cases"][case] = case_data
    first = bindings["cases"]["sub-CON01"]
    root = Path(first["anatomy"]["directory"])
    dwi = nib.load(first["selected_inputs"]["dwi"])
    transform = np.loadtxt(Path(first["prior_output_dir"]) / "dwi_to_t1_world.csv", delimiter=",")
    specs = [TemplateSpec(name, "surface", "native", left_path=root / f"label/lh.{name}.annot",
                          right_path=root / f"label/rh.{name}.annot") for name in ("aparc", "aparc.a2009s")]
    specs += [TemplateSpec(name, "volume", "t1", volume_path=root / f"mri/{name}.mgz")
              for name in ("aparc+aseg", "aparc.a2009s+aseg")]
    for spec in specs:
        start = time.perf_counter()
        result = prepare_template(spec, subject_dir=root, dwi_shape=dwi.shape[:3],
                                  dwi_affine=dwi.affine, dwi_to_t1_world=transform,
                                  device="cpu")
        report["prepared"][spec.name] = {"seconds": time.perf_counter() - start,
                                        "node_count": len(result.nodes),
                                        "represented_nodes": len(result.labels.unique()) - 1,
                                        "shape": list(result.labels.shape), "dtype": str(result.labels.dtype),
                                        "nonbackground_voxels": int((result.labels > 0).sum()),
                                        "declared_nodes_retained": True,
                                        "labels_sha256": hashlib.sha256(result.labels.numpy().tobytes()).hexdigest()}
        assert result.labels.shape == dwi.shape[:3] and result.labels.dtype == torch.int32
        assert torch.equal(result.affine, torch.as_tensor(dwi.affine))
        assert int(result.labels.max()) <= len(result.nodes)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["cuda_initialized"] = torch.cuda.is_initialized()
    assert not report["cuda_initialized"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["prepared"], indent=2))


if __name__ == "__main__":
    main()
