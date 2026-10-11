"""完整同输入N4：独立Python库与固定native浮点输出及uint8比较。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output-directory", "module-directory", "reference-binary", "reference-source"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat-reference", action="store_true")
    args = parser.parse_args()
    if sha(args.reference_binary) != args.reference_sha256:
        raise ValueError("reference N4 program hash mismatch")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import nibabel as nib
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(args.module_directory.resolve()))
    from fnit.recon_all import n4_simpleitk as candidate
    from fnit.recon_all.n4_itk import _to_uchar
    image = nib.load(str(args.source))
    values = np.asarray(image.dataobj,dtype=np.float32)
    raw = args.output_directory/"input.f32"
    values.ravel(order="F").tofile(raw)
    report = {"status": "running", "scope": "complete frozen same-input N4; not continuous recon-all",
        "source_sha256": sha(args.source), "code_version": args.code_version, "driver_sha256": sha(__file__),
        "module_sha256": sha(candidate.__file__), "reference_program_sha256": sha(args.reference_binary),
        "reference_source_sha256": sha(args.reference_source), "threads": args.threads,
        "host": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "input_shape": list(values.shape), "input_dtype": str(image.get_data_dtype()),
        "input_affine_mm": image.affine.tolist(), "comparisons": {}}
    def native(index):
        path = args.output_directory/("native-"+str(index)+".f32")
        cmd = [str(args.reference_binary),str(raw),str(path),
            *[str(s) for s in image.shape],*[repr(float(s)) for s in image.header.get_zooms()[:3]],str(args.threads),
            str(args.output_directory/("native-"+str(index)+"-profile.json"))]
        started=time.perf_counter()
        with (args.output_directory/("native-"+str(index)+".log")).open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        report["native_"+str(index)+"_seconds"] = time.perf_counter()-started
        result=np.fromfile(path,dtype=np.float32)
        if result.size != values.size:raise ValueError("native byte count differs")
        return result.reshape(values.shape,order="F")
    first = native(0)
    started = time.perf_counter()
    corrected, detail = candidate.correct_array(values,spacing=tuple(float(s) for s in image.header.get_zooms()[:3]),
        reconstruction_threads=args.threads)
    report["python_array_call_seconds_including_metadata"] = time.perf_counter()-started
    report["python_detail"] = detail
    def compare(a,b):
        absolute=np.abs(a.astype(np.float64)-b.astype(np.float64))
        return {"different_voxels": int(np.count_nonzero(a!=b)), "maximum_absolute_error": float(absolute.max()),
            "p99_absolute_error": float(np.percentile(absolute,99)), "mean_absolute_error": float(absolute.mean()),
            "actual_dtype": str(a.dtype), "reference_dtype": str(b.dtype)}
    report["comparisons"]["float_python_vs_native"] = compare(corrected,first)
    report["comparisons"]["uint8_python_vs_native"] = compare(_to_uchar(corrected),_to_uchar(first))
    np.save(args.output_directory/"python_float.npy",corrected)
    report["python_float_sha256"] = sha(args.output_directory/"python_float.npy")
    report["native_float_sha256"] = sha(args.output_directory/"native-0.f32")
    if args.repeat_reference:
        second=native(1)
        report["comparisons"]["native_repeated_float"] = compare(second,first)
        report["comparisons"]["native_repeated_uint8"] = compare(_to_uchar(second),_to_uchar(first))
    report["status"] = "complete"
    (args.output_directory/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["comparisons"]),flush=True)


if __name__ == "__main__":
    main()
