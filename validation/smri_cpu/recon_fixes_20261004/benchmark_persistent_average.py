"""Finite real-mesh diagnostic; no change to the production averaging backend."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sphere", "smoothwm", "compiler", "cpp-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    library_path = args.output / "persistent_average.private.so"
    argv = [str(args.compiler), "-O3", "-std=c++17", "-ffp-contract=off",
            "-fno-fast-math", "-fopenmp", "-fPIC", "-shared",
            str(args.cpp_source), "-o", str(library_path)]
    start = time.perf_counter()
    compile_result = subprocess.run(argv, capture_output=True, text=True)
    compile_seconds = time.perf_counter() - start
    (args.output / "compile.private.log").write_text(compile_result.stdout + compile_result.stderr)
    compile_result.check_returncode()
    import nibabel as nib
    import numba
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from fnit.recon_all.mris_register_average_numba import RegistrationGradientAverager
    from fnit.recon_all.mris_register_nonlinear import (
        first_area_gradient, first_distance_gradient, prepare_registration_force_cache,
    )
    sphere, faces = nib.freesurfer.read_geometry(args.sphere)
    original, original_faces = nib.freesurfer.read_geometry(args.smoothwm)
    assert sphere.shape == original.shape and np.array_equal(faces, original_faces)
    positions = torch.from_numpy(sphere.copy()).float()
    smoothwm = torch.from_numpy(original.copy()).float()
    topology = torch.from_numpy(faces.astype(np.int64, copy=True))
    cache = prepare_registration_force_cache(positions, smoothwm, topology)
    gradient = first_distance_gradient(positions, smoothwm, positions, topology, cache=cache)
    gradient = first_area_gradient(positions, smoothwm, positions, topology, gradient, cache=cache)
    assert torch.isfinite(gradient).all() and torch.any(gradient != 0)
    source_gradient = gradient.numpy().copy()
    neighbors = np.ascontiguousarray(cache.neighbors.numpy())
    degrees = np.ascontiguousarray(cache.degrees.numpy())
    reciprocals = (1.0 / (cache.degrees + 1).float()).numpy().copy()
    reference = RegistrationGradientAverager(cache.neighbors, cache.degrees, device="cpu")
    library = ctypes.CDLL(str(library_path.resolve()))
    function = library.fnit_average_persistent
    pointer = ctypes.c_void_p
    function.argtypes = [pointer, pointer, pointer, pointer, ctypes.c_int64,
                         ctypes.c_int64, ctypes.c_int64, ctypes.c_int, pointer]
    function.restype = ctypes.c_int

    def candidate(rounds):
        result = np.empty_like(source_gradient)
        error = function(source_gradient.ctypes.data, neighbors.ctypes.data,
                         degrees.ctypes.data, reciprocals.ctypes.data,
                         len(degrees), neighbors.shape[1], rounds, args.threads,
                         result.ctypes.data)
        if error:
            raise RuntimeError("prototype status " + str(error))
        return result

    report = {"scope": "persistent_team_averaging_prototype_real_complete_LH_only",
              "hostname": socket.gethostname(), "affinity": sorted(os.sched_getaffinity(0)),
              "threads": args.threads, "vertices": len(sphere), "faces": len(faces),
              "versions": {"torch": torch.__version__, "numba": numba.__version__},
              "input_sha256": {"sphere": digest(args.sphere), "smoothwm": digest(args.smoothwm)},
              "cpp_source_sha256": digest(args.cpp_source),
              "worker_sha256": digest(__file__), "compiler_sha256": digest(args.compiler),
              "compiler_version": subprocess.check_output([str(args.compiler), "--version"], text=True).splitlines()[0],
              "compile_seconds": compile_seconds,
              "compiler_flags": argv[1:-3],
              "runtime_linkage": subprocess.check_output(["ldd", str(library_path)], text=True),
              "gradient_sha256": hashlib.sha256(source_gradient.tobytes()).hexdigest(),
              "records": [], "production_adopted": False}
    for name, call in (("current_numba", lambda n: reference(gradient, n).numpy()),
                       ("persistent_cpp", candidate)):
        start = time.perf_counter()
        value = call(1)
        report.setdefault("first_call_seconds", {})[name] = time.perf_counter() - start
    for rounds in (0, 1, 16, 256, 16384):
        rows, outputs = [], {}
        for name, call in (("current_numba", lambda n: reference(gradient, n).numpy()),
                           ("persistent_cpp", candidate), ("persistent_cpp", candidate),
                           ("current_numba", lambda n: reference(gradient, n).numpy())):
            start = time.perf_counter()
            value = call(rounds)
            rows.append({"arm": name, "seconds": time.perf_counter() - start})
            if name in outputs:
                assert np.array_equal(value, outputs[name])
            outputs[name] = value.copy()
        equal = np.array_equal(outputs["current_numba"], outputs["persistent_cpp"])
        bit_exact = outputs["current_numba"].tobytes() == outputs["persistent_cpp"].tobytes()
        unchanged = np.array_equal(gradient.numpy(), source_gradient)
        row = {"iterations": rounds, "abba": rows, "outputs_exact": bool(equal),
               "outputs_bit_exact": bit_exact,
               "input_unchanged": bool(unchanged),
               "output_sha256": hashlib.sha256(outputs["persistent_cpp"].tobytes()).hexdigest()}
        report["records"].append(row)
        (args.output / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
        assert equal and bit_exact and unchanged
    report["status"] = "complete_operator_only_not_full_registration"
    (args.output / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "compile_seconds", "records")}))


if __name__ == "__main__":
    main()
