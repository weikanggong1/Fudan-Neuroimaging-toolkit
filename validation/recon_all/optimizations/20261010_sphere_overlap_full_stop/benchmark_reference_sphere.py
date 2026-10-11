"""固定源码原生参考同输入双重复；冻结自产网格只用于诊断。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import nibabel.freesurfer.io as fsio
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quality(vertices, faces):
    wide = np.asarray(vertices, np.float32).astype(np.float64)
    triangles = wide[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    negative = np.sum(triangles.sum(axis=1) * cross, axis=1) < 0
    area = np.linalg.norm(cross, axis=1) * .5
    bad = area[negative]
    return {"vertices": len(vertices), "faces": len(faces), "finite": bool(np.isfinite(wide).all()),
            "negative_radial_faces_double": int(negative.sum()),
            "negative_area_mm2_sum": float(bad.sum()),
            "negative_area_mm2_max": float(bad.max(initial=0)),
            "negative_area_mm2_p99": float(np.percentile(bad, 99)) if len(bad) else 0,
            "zero_area_faces": int((area == 0).sum()),
            "coordinate_sha256": hashlib.sha256(np.asarray(vertices, np.float32).tobytes()).hexdigest(),
            "ordered_faces_sha256": hashlib.sha256(np.asarray(faces, np.int32).tobytes()).hexdigest()}


def comparison(reference, candidate):
    x, f = reference
    y, g = candidate
    corresponding = x.shape == y.shape and f.shape == g.shape and np.array_equal(f, g)
    result = {"same_ordered_faces_and_vertex_count": corresponding}
    if corresponding:
        distance = np.linalg.norm(x.astype(np.float64) - y.astype(np.float64), axis=1)
        result.update(coordinate_elements_differing=int(np.count_nonzero(x != y)),
                      mean_vertex_mm=float(distance.mean()), p99_vertex_mm=float(np.percentile(distance, 99)),
                      max_vertex_mm=float(distance.max(initial=0)))
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--inflated", type=Path, required=True)
    parser.add_argument("--smoothwm", type=Path, required=True)
    parser.add_argument("--candidate-sphere", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    binary = args.reference_prefix / "bin/mris_sphere"
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    environment["LD_LIBRARY_PATH"] = str(args.reference_prefix / "lib")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]:
        environment[key] = str(args.threads)
    hemi = args.inflated.name.split(".")[0]
    report = {"status": "running", "scope": "fixed-source Conda reference twice on identical frozen FNIT inputs; not raw T1 E2E",
              "script_sha256": sha(Path(__file__)), "reference_program_sha256": sha(binary),
              "threads": args.threads, "affinity": sorted(os.sched_getaffinity(0)), "random_seed": args.seed,
              "inputs_sha256": {"inflated": sha(args.inflated), "smoothwm": sha(args.smoothwm)},
              "candidate_sphere_sha256": sha(args.candidate_sphere), "runs": [],
              "child_library_policy": "private reference/lib plus system base; no LD_PRELOAD",
              "overall_metric_equivalence": "not_assessed"}
    def save():
        temp = args.output / "report.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output / "report.json")
    save()
    candidate = fsio.read_geometry(args.candidate_sphere)
    report["frozen_candidate_quality"] = quality(*candidate)
    geometries = []
    for index in [1, 2]:
        folder = args.output / f"repeat-{index}/surf"
        folder.mkdir(parents=True)
        inflated, smoothwm = folder / (hemi + ".inflated"), folder / (hemi + ".smoothwm")
        shutil.copyfile(args.inflated, inflated)
        shutil.copyfile(args.smoothwm, smoothwm)
        if sha(inflated) != report["inputs_sha256"]["inflated"] or sha(smoothwm) != report["inputs_sha256"]["smoothwm"]:
            raise ValueError("reference inputs changed while copying")
        output = folder / (hemi + ".sphere")
        command = [str(binary), "-seed", str(args.seed), "-threads", str(args.threads), str(inflated), str(output)]
        tick = time.perf_counter()
        with (args.output / f"repeat-{index}/native.log").open("w") as log:
            process = subprocess.run(command, env=environment, cwd=folder.parent, stdout=log, stderr=subprocess.STDOUT)
        row = {"repeat": index, "argv": command, "returncode": process.returncode,
               "full_cli_seconds_including_loader_and_IO": time.perf_counter() - tick}
        if process.returncode != 0:
            row["status"] = "failed_execution"
            report["runs"].append(row)
            report["status"] = "failed_execution"
            save()
            raise RuntimeError("reference sphere failed; see own native.log")
        geometry = fsio.read_geometry(output)
        geometries.append(geometry)
        row.update(status="complete", output_sha256=sha(output), quality=quality(*geometry),
                   comparison_to_frozen_candidate=comparison(geometry, candidate))
        report["runs"].append(row)
        save()
    report["repeat_comparison"] = comparison(geometries[0], geometries[1])
    report.update(status="complete", full_harness_seconds=time.perf_counter() - started,
                  candidate_reference_scope="single hemisphere fixed same-input stage; file headers contain different command paths")
    save()
    print("REFERENCE_SPHERE_PAIR_COMPLETE")


if __name__ == "__main__":
    main()
