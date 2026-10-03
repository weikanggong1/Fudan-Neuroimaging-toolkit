"""Private real-data diagnostic; never imported by production.

Fix official TOPUP/EDDY parameters and replacement data, then compare the
existing EDDY sampler and FNIT's mature official-precision TOPUP sampler.
Only aggregate metrics are exported. Original images are never rewritten.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

import nibabel as nib
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(candidate, reference, mask):
    x = np.asarray(candidate)[mask].astype(np.float64)
    y = np.asarray(reference)[mask].astype(np.float64)
    d = x-y
    a = np.abs(d)
    return {"values": int(x.size), "rmse": float(np.sqrt(np.mean(d*d))),
            "max_abs": float(a.max()), "p99_abs": float(np.percentile(a,99)),
            "neq": int(np.count_nonzero(d))}


class Monitor:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            row = {"time": time.monotonic()}
            try:
                proc = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits"], text=True)
                rows = [x.strip().split(", ") for x in proc.splitlines()]
                row["own_process_tree_bytes"] = sum(
                    int(x[2])*1024**2 for x in rows if int(x[0]) == os.getpid())
                row["gpu"] = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True).strip().splitlines()
            except Exception as exc:
                row["error"] = repr(exc)
            self.samples.append(row)
            self.stop.wait(.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["CON01", "CON03"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lock = open("/tmp/fnit-recon-five-20261002-gongwk.gpu.lock", "a+")
    started = time.perf_counter()
    fcntl.flock(lock, fcntl.LOCK_EX)
    lock_wait = time.perf_counter()-started
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(
        20e9/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from fnit.eddy.fsl2111_strict import warp
    from fnit.eddy.fsl2111_strict import spline
    from fnit.eddy.topup_field import _load_topup_field
    from fnit.topup._sampling_cuda import sample_cubic_cuda
    from fnit.topup.core import _render_fixed_topup
    legacy = spline.sample_cubic_periodic_fast
    legacy_jacobian = warp.jacobian_from_pe_displacement
    legacy_rotation = warp.fsl_rotation_matrix

    def precise_rotation(angles):
        old = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            return legacy_rotation(angles)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    def mirror_jacobian(disp, axis):
        d = disp[...,axis,:,:,:]
        spatial_axis = d.ndim - 3 + axis
        original_dtype = d.dtype
        c = d.movedim(spatial_axis,-1).double().clone()
        length = c.shape[-1]
        pole = math.sqrt(3)-2
        taps = min(int(math.log(1e-8)/math.log(abs(pole))+1.5),length)
        initial = c[...,0].clone()
        for i in range(1,taps):
            initial += pole**i*c[...,i]
        c[...,0] = initial
        last_original = c[...,-1].clone()
        for i in range(1,length):
            c[...,i] += pole*c[...,i-1]
        c[...,-1] = -pole/(1-pole*pole)*(2*c[...,-1]-last_original)
        for i in range(length-2,-1,-1):
            c[...,i] = pole*(c[...,i+1]-c[...,i])
        c = (c*6).to(original_dtype)
        derivative = .5*(c.roll(-1,-1)-c.roll(1,-1))
        derivative[...,0]=0; derivative[...,-1]=0
        return 1+derivative.movedim(-1,spatial_axis)

    def fused(coeff, coordinates, boundary="periodic", *, padded_coeff=None):
        if coeff.ndim == 3:
            coeff = coeff[None]
        if coordinates.ndim == 4:
            coordinates = coordinates[None]
        if coeff.shape[0] == 1 and coordinates.shape[0] > 1:
            coeff = coeff.expand(coordinates.shape[0], -1, -1, -1)
        if boundary != "periodic":
            raise ValueError("fixed scan-to-model diagnostic uses periodic sampling")
        return torch.stack([sample_cubic_cuda(c, xyz, 1)[0]
                            for c, xyz in zip(coeff, coordinates)])

    report = {"scope": "fixed official parameters/replacement frames, render only",
              "baseline_commit": "7af34e6d072e843fb2558c931bb2781f1d4b0be9",
              "lock_wait_seconds": lock_wait, "pid": os.getpid(),
              "host": os.uname().nodename, "subjects": {},
              "source_sha256": {str(Path(__file__)): sha(__file__),
                                str(Path(warp.__file__)): sha(warp.__file__),
                                str(Path(spline.__file__)): sha(spline.__file__)}}
    monitor = Monitor(); monitor.thread.start()
    try:
        for subject in args.subjects:
            root = args.reference_root / f"sub-{subject}"
            raw_image = nib.load(str(root / "raw/AP.nii.gz"))
            data = np.asarray(nib.load(str(root / "eddy/data.eddy_outlier_free_data.nii.gz")).dataobj,
                              dtype=np.float32)
            target = np.asarray(nib.load(str(root / "eddy/data.nii.gz")).dataobj, dtype=np.float32)
            mask = np.asarray(nib.load(str(root / "mask/nodif_brain_mask.nii.gz")).dataobj) > 0
            bvals = np.loadtxt(root / "raw/AP.bval")
            frames = np.flatnonzero(bvals < 100).tolist()
            params = torch.as_tensor(np.loadtxt(root / "eddy/data.eddy_parameters"),
                                     device="cuda", dtype=torch.float32)
            acq = np.loadtxt(root / "topup/acqparams.txt")
            phase = torch.as_tensor(acq[0,:3],device="cuda",dtype=torch.float32)
            readout = torch.as_tensor(acq[0,3],device="cuda",dtype=torch.float32)
            voxel_sizes = raw_image.header.get_zooms()[:3]
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            field, _, _ = _load_topup_field(root/"topup/fieldmap_out", data.shape[:3], "cuda", 1)
            torch.backends.cuda.matmul.allow_tf32 = tf32
            dense_field = np.asarray(nib.load(str(root/"topup/fieldmap_fout.nii.gz")).dataobj)
            scale = np.float32(100.0 / float(data[..., frames[0]][mask].mean()))
            inputs = [torch.as_tensor(data[..., f].copy(),device="cuda")*scale for f in frames]
            def render(sampler, jacobian=legacy_jacobian, rotation=legacy_rotation):
                warp.sample_cubic_periodic_fast = sampler
                warp.jacobian_from_pe_displacement = jacobian
                warp.fsl_rotation_matrix = rotation
                out = [warp.unwarp_scan_to_model(x,params[f,:6],params[f,6:],field,
                         phase,readout,voxel_sizes,pe_extrapolation_valid=True)[0]/scale
                       for f,x in zip(frames,inputs)]
                return torch.stack(out,dim=-1)
            # Compilation/warmup excluded from the four matched warm calls.
            render(legacy); render(fused); torch.cuda.synchronize()
            rows = []; arrays = {}
            for name, function, jac, rotation in [("legacy",legacy,legacy_jacobian,legacy_rotation),
                    ("rotation_fp32",legacy,legacy_jacobian,precise_rotation),
                    ("rotation_fp32_fused",fused,legacy_jacobian,precise_rotation),
                    ("rotation_fp32_fused",fused,legacy_jacobian,precise_rotation),
                    ("rotation_fp32",legacy,legacy_jacobian,precise_rotation),
                    ("legacy",legacy,legacy_jacobian,legacy_rotation),
                    ("fused",fused,legacy_jacobian,legacy_rotation),
                    ("mirror",legacy,mirror_jacobian,legacy_rotation)]:
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                t = time.perf_counter(); out = render(function,jac,rotation); torch.cuda.synchronize()
                elapsed = time.perf_counter()-t
                arrays[name] = out.cpu().numpy()
                rows.append({"sampler": name,"wall_seconds":elapsed,
                             "allocated_peak_bytes":torch.cuda.max_memory_allocated(),
                             "reserved_peak_bytes":torch.cuda.max_memory_reserved()})
            topup = _render_fixed_topup(root/"topup/B0_AP_PA.nii.gz",root/"topup/acqparams.txt",
                      root/"topup/fieldmap_out_fieldcoef.nii.gz",root/"topup/fieldmap_out_movpar.txt",device="cuda")
            official_topup = np.asarray(nib.load(str(root/"topup/fieldmap_iout.nii.gz")).dataobj)
            paths = [root/"raw/AP.nii.gz",root/"eddy/data.eddy_outlier_free_data.nii.gz",
                     root/"eddy/data.eddy_parameters",root/"eddy/data.nii.gz",
                     root/"topup/fieldmap_out_fieldcoef.nii.gz",root/"topup/fieldmap_out_movpar.txt",
                     root/"topup/acqparams.txt",root/"mask/nodif_brain_mask.nii.gz"]
            report["subjects"][subject] = {"frames":frames, "input_sha256":{str(p):sha(p) for p in paths},
                "acqparams":acq.tolist(), "paired_render":rows,
                "decoded_field_vs_official_fout":metrics(field.cpu().numpy(),dense_field,mask),
                "fixed_topup_render_vs_official_iout":metrics(topup["corrected"],official_topup,mask),
                "fixed_eddy_render_vs_official":{name:metrics(x,target[...,frames],mask) for name,x in arrays.items()},
                "per_frame_errors":{name:{str(f):metrics(x[...,i],target[...,f],mask) for i,f in enumerate(frames)}
                                    for name,x in arrays.items()},
                "official_nonzero_errors":{name:metrics(x,target[...,frames],mask & np.all(target[...,frames]!=0,axis=3))
                                           for name,x in arrays.items()},
                "legacy_vs_fused":metrics(arrays["legacy"],arrays["fused"],mask)}
            print(subject, json.dumps(report["subjects"][subject]["fixed_eddy_render_vs_official"]),flush=True)
    finally:
        warp.sample_cubic_periodic_fast = legacy
        warp.jacobian_from_pe_displacement = legacy_jacobian
        warp.fsl_rotation_matrix = legacy_rotation
        monitor.stop.set(); monitor.thread.join()
        report["GPU_monitor"] = {"interval_seconds":.5,"samples":monitor.samples,
                  "own_process_tree_peak_bytes":max((s.get("own_process_tree_bytes",0) for s in monitor.samples),default=None),
                  "max_sample_gap_seconds":max((b["time"]-a["time"] for a,b in zip(monitor.samples,monitor.samples[1:])),default=None),
                  "failed_samples":sum("error" in s for s in monitor.samples),
                  "process_scope":"single Python process; no GPU child process"}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        fcntl.flock(lock,fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
