"""Complete real spherical-registration comparison, not a synthetic benchmark.

Run each arm in a separate process with the same frozen FNIT tree. Only the
ordered averaging module changes. Native FreeSurfer is run separately by the
isolated reference launcher; it is never imported by this diagnostic.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sphere", "smoothwm", "sulc", "atlas", "average-source"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    import fnit.recon_all
    module_name = "fnit.recon_all.mris_register_average_numba"
    spec = importlib.util.spec_from_file_location(module_name, args.average_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    original_call = module.RegistrationGradientAverager.__call__
    averaging = []

    def measured_call(self, gradient, iterations):
        started = time.perf_counter()
        result = original_call(self, gradient, iterations)
        averaging.append({"iterations": int(iterations),
                          "seconds": time.perf_counter() - started})
        return result

    module.RegistrationGradientAverager.__call__ = measured_call
    from fnit.recon_all.mris_register_run import run_register_sphere
    started = time.perf_counter()
    registration = run_register_sphere(
        args.sphere, args.smoothwm, args.sulc, args.atlas,
        args.output / "lh.sphere.reg", overlap_device="cpu",
        averaging_device="cpu")
    elapsed = time.perf_counter() - started
    import numba
    import numpy as np
    import nibabel as nib
    xyz, faces = nib.freesurfer.read_geometry(args.output / "lh.sphere.reg")
    source_root = Path(fnit.recon_all.__file__).resolve().parent
    report = {
        "scope": "complete_frozen_real_LH_registration_same_input_no_raw_T1_rerun",
        "hostname": socket.gethostname(), "threads": args.threads,
        "affinity": sorted(os.sched_getaffinity(0)),
        "versions": {"torch": torch.__version__, "numba": numba.__version__,
                     "numpy": np.__version__, "nibabel": nib.__version__},
        "averaging_source_sha256": digest(args.average_source),
        "common_source_sha256": {p.name: digest(p) for p in sorted(source_root.glob("*.py"))
                                  if p.name != "mris_register_average_numba.py"},
        "registration": registration,
        "api_seconds_including_io": elapsed,
        "averaging_calls": averaging,
        "averaging_seconds_including_first_JIT": sum(r["seconds"] for r in averaging),
        "vertices": len(xyz), "faces": len(faces),
        "coordinate_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
        "ordered_faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
        "finite_coordinates": bool(np.isfinite(xyz).all()),
    }
    (args.output / "record.private.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("scope", "api_seconds_including_io",
                                            "vertices", "faces", "finite_coordinates")}))


if __name__ == "__main__":
    main()
