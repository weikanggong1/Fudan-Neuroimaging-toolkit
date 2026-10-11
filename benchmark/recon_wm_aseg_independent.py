"""冻结自产输入：完整CPU WM编辑、已有Torch混合和隔离原生三方配对。"""
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
    for name in ("mri-directory", "output-directory", "module-directory", "reference-binary", "reference-assets"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if sha(args.reference_binary) != args.reference_sha256:
        raise ValueError("reference program hash mismatch")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import nibabel as nib
    import torch
    import numba
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.module_directory.resolve()))
    from fnit.recon_all import edit_wm_aseg_core_python as cpu
    from fnit.recon_all import edit_wm_aseg_torch as gpu
    from fnit.recon_all import edit_wm_aseg_late_python as late
    from fnit.recon_all import wm_aseg_io
    torch.set_num_threads(args.threads); numba.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    files = {name: args.mri_directory/(name+".mgz") for name in ("wm.seg", "brain", "aseg.presurf", "entowm")}
    target = torch.device(args.device)
    torch.zeros(1, device=target); torch.cuda.synchronize(target)
    torch.cuda.reset_peak_memory_stats(target)
    report = {"status": "running", "scope": "complete frozen same-input WM editing; not raw T1 recon-all",
        "code_version": args.code_version, "driver_sha256": sha(__file__),
        "modules_sha256": {Path(m.__file__).name: sha(m.__file__) for m in (cpu, gpu, late, wm_aseg_io)},
        "inputs_sha256": {k: sha(v) for k,v in files.items()},
        "reference_program_sha256": sha(args.reference_binary),
        "host": platform.node(), "threads": args.threads, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "device": str(target), "gpu": torch.cuda.get_device_name(target),
        "precision": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32, "no_half": True},
        "order": ["native", "cpu", "hybrid", "hybrid", "cpu", "native"], "records": [], "comparisons": {}}
    def comparison(a, b):
        x,y = [nib.load(str(p)) for p in (a,b)]
        xa,ya = [np.asarray(v.dataobj) for v in (x,y)]
        difference = np.abs(xa.astype(np.int16)-ya.astype(np.int16))
        dices = {}
        for label in np.union1d(np.unique(xa),np.unique(ya)):
            xx, yy = xa == label, ya == label
            dices[str(int(label))] = float(2*np.count_nonzero(xx&yy)/(np.count_nonzero(xx)+np.count_nonzero(yy)))
        return {"different_voxels": int(np.count_nonzero(difference)),
            "maximum_absolute_error": int(difference.max()), "p99_absolute_error": float(np.percentile(difference,99)),
            "per_label_dice": dices, "affine_exact": bool(np.array_equal(x.affine,y.affine)),
            "dtype_exact": x.get_data_dtype() == y.get_data_dtype(), "output_sha256": sha(a), "reference_sha256": sha(b)}
    outputs = []
    for index, backend in enumerate(report["order"]):
        output = args.output_directory/(str(index)+"-"+backend+".mgz")
        torch.cuda.synchronize(target); started = time.perf_counter()
        if backend == "native":
            command = [str(args.reference_binary), "-keep-in", "-fix-ento-wm", str(files["entowm"]), "3", "255", "255",
                "-fix-acj", str(files["aseg.presurf"]), "255", "255", "-fill-seg-wm", "-fix-scm-ha", "1",
                str(files["wm.seg"]), str(files["brain"]), str(files["aseg.presurf"]), str(output)]
            with (args.output_directory/(str(index)+"-native.log")).open("w") as log:
                subprocess.run(command, env=dict(os.environ, FREESURFER_HOME=str(args.reference_assets)),
                    cwd=args.output_directory, stdout=log, stderr=subprocess.STDOUT, check=True)
            detail = None
        else:
            parameters = dict(wm_file=files["wm.seg"], brain_file=files["brain"], aseg_file=files["aseg.presurf"],
                entowm_file=files["entowm"], output_file=output, fill_seg_wm=True)
            detail = (cpu.write_wm_asegedit_profile(**parameters) if backend == "cpu" else
                      gpu.write_wm_asegedit_hybrid_diagnostic(**parameters, device=str(target)))
        torch.cuda.synchronize(target)
        report["records"].append({"backend": backend, "file_api_wall_seconds": time.perf_counter()-started,
            "detail": detail, "output_sha256": sha(output)})
        outputs.append(output)
    for index, output in enumerate(outputs[1:],1):
        report["comparisons"][str(index)+"_vs_native0"] = comparison(output,outputs[0])
    report["torch_allocated_peak_bytes"] = torch.cuda.max_memory_allocated(target)
    report["torch_reserved_peak_bytes"] = torch.cuda.max_memory_reserved(target)
    report["memory_process_ownership"] = None
    report["timing_scope"] = "validated complete file API/command, including input reads/transfers/writes; imports/context precede measurement; first CPU includes JIT"
    report["status"] = "complete"
    (args.output_directory/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v["different_voxels"] for k,v in report["comparisons"].items()}),flush=True)


if __name__ == "__main__":
    main()
