"""Independent real BOLD-reference benchmark; never starts volume/reconstruction.

Official NiWorkflows/AFNI runs only in an explicitly selected reference SIF.
FNIT imports occur after acquiring the requested physical-GPU benchmark lock.
Every invocation needs a fresh output directory; failed attempts are retained.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import nibabel as nib
import numpy as np


REFERENCE_SHA = "ac885355a286ff6799aaeafc9735de1d0c0264b8afba55041ea4e94b1ddc3484"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_path(entry):
    path = Path(entry["path"]).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError("bound file size/SHA does not match")
    return path


def output_record(path):
    image = nib.load(path)
    return {"sha256": sha256(path), "bytes": Path(path).stat().st_size,
            "shape": list(image.shape), "affine": image.affine.tolist()}


def source_hashes(source):
    return {str(path.relative_to(source)): sha256(path) for path in sorted((source / "src").rglob("*.py"))}


def protect_output(output, allowed_root, protected):
    output, allowed_root = output.resolve(), allowed_root.resolve(strict=True)
    if output == allowed_root or allowed_root not in output.parents:
        raise ValueError("output must be a fresh descendant of the declared run root")
    for item in protected:
        item = Path(item).resolve(strict=True)
        if item == output or item in output.parents or output in item.parents:
            raise ValueError("output overlaps a protected input/source/resource")
    if output.exists():
        raise FileExistsError("attempt already exists; retain it and choose a new output directory")


def compare_reference(candidate, reference):
    candidate_image, reference_image = nib.load(candidate), nib.load(reference)
    if candidate_image.ndim != 3 or candidate_image.shape != reference_image.shape:
        raise ValueError("references must be matching 3D grids")
    if not np.allclose(candidate_image.affine, reference_image.affine, rtol=0, atol=1e-4):
        raise ValueError("reference physical grids differ")
    candidate_data = np.asarray(candidate_image.dataobj, dtype=np.float64)
    reference_data = np.asarray(reference_image.dataobj, dtype=np.float64)
    if not np.isfinite(candidate_data).all() or not np.isfinite(reference_data).all():
        raise ValueError("reference images contain nonfinite values")
    domains = {"full_FOV": np.ones(reference_data.shape, dtype=bool),
               "reference_nonzero": reference_data != 0}
    metrics = {}
    for name, domain in domains.items():
        x, y = candidate_data[domain], reference_data[domain]
        if not len(x):
            metrics[name] = {"voxels": 0, "spatial_pearson": None, "NRMSE": None,
                             "status": "empty_domain"}
            continue
        difference = x - y
        rmse, rms = float(np.sqrt(np.mean(difference**2))), float(np.sqrt(np.mean(y**2)))
        centered_x, centered_y = x - x.mean(), y - y.mean()
        denominator = float(np.sqrt(np.sum(centered_x**2) * np.sum(centered_y**2)))
        metrics[name] = {"voxels": len(x), "spatial_pearson":
                         float(np.sum(centered_x * centered_y) / denominator) if denominator else None,
                         "spatial_pearson_defined": denominator != 0,
                         "RMSE": rmse, "reference_RMS": rms,
                         "NRMSE": rmse / rms if rms else None, "NRMSE_defined": rms != 0,
                         "max_absolute_difference": float(np.max(np.abs(difference))),
                         "mean_bias": float(difference.mean()), "different_values": int(np.count_nonzero(difference))}
    return metrics


@contextlib.contextmanager
def gpu_lock(args):
    if not args.device.startswith("cuda"):
        yield
        return
    if not args.gpu_lock:
        raise ValueError("CUDA candidate requires --gpu-lock before importing FNIT/Torch")
    # The caller provides the existing physical GPU lock; do not remove it.
    with Path(args.gpu_lock).resolve(strict=True).open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not visible.startswith("GPU-") or "," in visible or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                raise ValueError("CUDA benchmark requires one explicit GPU UUID and PCI_BUS_ID ordering")
            while True:
                fields = subprocess.check_output(["nvidia-smi", "--id=" + visible,
                    "--query-gpu=uuid,memory.free,pci.bus_id", "--format=csv,noheader,nounits"], text=True).strip().split(",")
                if fields[0].strip() != visible:
                    raise ValueError("physical GPU UUID differs from the configured device")
                if int(fields[1].strip()) >= 20480:
                    args.cuda_admission = {"gpu_uuid": visible, "free_memory_MiB": int(fields[1].strip()),
                        "PCI_bus_ID": fields[2].strip(), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "minimum_free_memory_MiB": 20480}
                    break
                time.sleep(2)
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


OFFICIAL_SCRIPT = r'''
import hashlib,json,pathlib,time,os
import nibabel as nib
import fmriprep,niworkflows
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.bold import NonsteadyStatesDetector
from niworkflows.interfaces.images import RobustAverage
config=json.loads(pathlib.Path(__import__('sys').argv[1]).read_text())
root=pathlib.Path(config['output']); before={}
for name,entry in config['expected_sources'].items():
 p=pathlib.Path(entry['container_path']); before[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 if before[name]!=entry['sha256']:raise ValueError('installed official source SHA mismatch: '+name)
if fmriprep.__version__!='25.2.4' or niworkflows.__version__!='1.14.4':raise ValueError('wrong official version')
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('official reference must hide CUDA')
timing={}; begin=time.perf_counter()
for name in ('validate','nss','average'):(root/name).mkdir()
t=time.perf_counter(); val=ValidateImage(in_file=config['bold']).run(cwd=str(root/'validate')); timing['validation']=time.perf_counter()-t
t=time.perf_counter(); nss=NonsteadyStatesDetector(in_file=val.outputs.out_file).run(cwd=str(root/'nss')); timing['NSS']=time.perf_counter()-t
t=time.perf_counter(); avg=RobustAverage(in_file=val.outputs.out_file,t_mask=nss.outputs.t_mask,mc_method='AFNI',two_pass=True,nonnegative=True,num_threads=config['threads']).run(cwd=str(root/'average')); timing['RobustAverage_AFNI']=time.perf_counter()-t
timing['official_internal_call_seconds']=time.perf_counter()-begin
after={name:hashlib.sha256(pathlib.Path(entry['container_path']).read_bytes()).hexdigest() for name,entry in config['expected_sources'].items()}
if before!=after:raise ValueError('official source changed during call')
payload={'reference':str(avg.outputs.out_file),'selected_volumes':str(avg.outputs.out_volumes),'selected_indices':[i for i,v in enumerate(nss.outputs.t_mask) if v],'algorithm_dummy_scans':int(nss.outputs.n_dummy),'drift':list(avg.outputs.out_drift),'timing_seconds':timing,'installed_source_sha256_before':before,'installed_source_sha256_after':after,'installed_source_guards_equal':True,'actual_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'versions':{'fmriprep':fmriprep.__version__,'niworkflows':niworkflows.__version__}}
(root/'execution.private.json').write_text(json.dumps(payload,indent=2)+'\n')
'''


def official(args, raw, tools):
    image = record_path(tools["official_container"])
    config = {"bold": str(raw), "output": str(args.output), "threads": args.threads,
              "expected_sources": tools["official_sources"]}
    config_path = args.output / "config.private.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    command = [args.singularity, "exec", "--cleanenv", "--env",
               f"CUDA_VISIBLE_DEVICES=,OMP_NUM_THREADS={args.threads},ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={args.threads}",
               "-B", str(raw.parent) + ":" + str(raw.parent) + ":ro",
               "-B", str(args.output) + ":" + str(args.output),
               str(image), "python", "-c", OFFICIAL_SCRIPT, str(config_path)]
    # Independent container process clock; preparation/hash checks are outside.
    start = time.perf_counter()
    with (args.output / "official.private.log").open("xb") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    process_wall = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f"official reference process exited {completed.returncode}")
    if sha256(image) != tools["official_container"]["sha256"]:
        raise RuntimeError("official container changed during reference call")
    execution = json.loads((args.output / "execution.private.json").read_text())
    execution["container_process_wall_seconds"] = process_wall
    execution["exit_code"] = completed.returncode
    execution["container_sha256"] = tools["official_container"]["sha256"]
    execution["container_guards_equal"] = True
    execution["cpu_threads_requested"] = args.threads
    return execution


def prepare_cuda_measurement(torch, device):
    """Use the verified continuous-run startup and a 20 decimal GB cap."""
    target = torch.device(device)
    if not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(target)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 20e9 / properties.total_memory), target)
    torch.cuda.reset_peak_memory_stats(target)
    torch.cuda.synchronize(target)
    return properties


def candidate(args, raw):
    with gpu_lock(args):
        source = args.source_root.resolve(strict=True)
        src = source / "src"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(src))
        import torch
        from fnit.fmri.reference import prepare_bold_reference
        import fnit.fmri.reference as module
        if Path(module.__file__).resolve() != src / "fnit/fmri/reference.py":
            raise ValueError("FNIT imported from an unbound source root")
        if sha256(module.__file__) != args.expected_reference_sha256:
            raise ValueError("FNIT reference module SHA differs from frozen candidate")
        torch.set_num_threads(args.threads)
        gpu_properties = prepare_cuda_measurement(torch, args.device) if args.device.startswith("cuda") else None
        gpu_uuid = None
        if gpu_properties:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            ordinal = torch.device(args.device).index or 0
            if not visible or ordinal >= len(visible.split(",")):
                raise ValueError("CUDA benchmark requires explicit physical CUDA_VISIBLE_DEVICES mapping")
            token = visible.split(",")[ordinal]
            gpu_uuid = subprocess.check_output(
                ["nvidia-smi", "--id=" + token, "--query-gpu=uuid", "--format=csv,noheader"],
                text=True,
            ).strip()
            if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
                raise ValueError("could not verify the actual physical GPU UUID")
        start = time.perf_counter()
        result = prepare_bold_reference(raw, args.output / "robust", device=args.device)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))
        api_wall = time.perf_counter() - start
        raw_image = nib.load(raw)
        index = raw_image.shape[3] // 2
        start = time.perf_counter()
        header = raw_image.header.copy()
        header.set_data_dtype(np.float32)
        middle = args.output / "middle.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(raw_image.dataobj[..., index], dtype=np.float32),
                              raw_image.affine, header), middle)
        middle_wall = time.perf_counter() - start
        return {"reference": str(result.reference), "selected_volumes": str(result.selected_volumes),
                "metadata": str(result.metadata), "middle": str(middle), "middle_index": index,
                "selected_indices": list(result.selected_indices), "algorithm_dummy_scans": result.algorithm_dummy_scans,
                "timing_seconds": result.timing_seconds, "candidate_reference_API_wall_seconds": api_wall,
                "middle_load_and_save_seconds": middle_wall, "device": args.device,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": gpu_properties.name if gpu_properties else None,
                "gpu_uuid": gpu_uuid,
                "gpu_peak_scope": "PyTorch allocator; whole process-tree GPU peak measured separately by launcher",
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)) if args.device.startswith("cuda") else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device(args.device)) if args.device.startswith("cuda") else None,
                "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
                "cuda_admission": getattr(args, "cuda_admission", None),
                "allocator_cap_decimal_GB": 20 if gpu_properties else None,
                "TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "TF32_cudnn": torch.backends.cudnn.allow_tf32}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--phase", choices=("official", "candidate"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allowed-run-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gpu-lock", type=Path)
    p.add_argument("--expected-reference-sha256", default=REFERENCE_SHA)
    p.add_argument("--singularity", default="/public/software/apps/singularity/4.2.2/bin/singularity")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    manifest_path = args.manifest.resolve(strict=True)
    manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    case = manifest["cases"][args.case]
    raw = record_path(case["raw"]["bold"])
    original = record_path(case["official"]["bold_reference"]["output"])
    original_node_report = record_path(case["official"]["bold_reference"]["report"])
    original_node_input = record_path(case["official"]["bold_reference"]["input"])
    for entry in case["raw"].values():
        record_path(entry)
    image = nib.load(raw)
    if image.ndim != 4 or image.shape[3] != case["frames"] or case["frames"] != 180:
        raise ValueError("benchmark requires the bound complete180-frame BOLD")
    source = args.source_root.resolve(strict=True)
    sources_before = source_hashes(source)
    if not sources_before or sources_before.get("src/fnit/fmri/reference.py") != args.expected_reference_sha256:
        raise ValueError("source root/reference SHA does not match frozen candidate")
    inputs_before = {name: entry["sha256"] for name, entry in case["raw"].items()}
    inputs_before["original_average"] = sha256(original)
    inputs_before["original_node_report"] = sha256(original_node_report)
    inputs_before["original_node_input"] = sha256(original_node_input)
    script_before = sha256(__file__)
    protected = [source, manifest_path, Path(__file__).resolve().parent,
                 *[entry["path"] for entry in case["raw"].values()], original,
                 original_node_report, original_node_input,
                 manifest["tools"]["official_container"]["path"]]
    protect_output(args.output, args.allowed_run_root, protected)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "case_id": args.case, "phase": args.phase,
              "input_frames": case["frames"], "tr_seconds": case["tr_seconds"],
              "manifest_sha256": manifest_sha, "driver_sha256": script_before,
              "host": socket.gethostname(), "original_reference_runtime":
              {"duration": case["official"]["bold_reference"]["runtime"]["duration"],
               "hostname": case["official"]["bold_reference"]["runtime"]["hostname"]},
              "time_comparison_boundary": "historical node, fresh official process, and FNIT API are separate scopes"}
    try:
        execution = official(args, raw, manifest["tools"]) if args.phase == "official" else candidate(args, raw)
        produced = Path(execution["reference"])
        comparison = compare_reference(produced, original)
        inputs_after = {name: sha256(entry["path"]) for name, entry in case["raw"].items()}
        inputs_after["original_average"] = sha256(original)
        inputs_after["original_node_report"] = sha256(original_node_report)
        inputs_after["original_node_input"] = sha256(original_node_input)
        sources_after = source_hashes(source)
        if inputs_before != inputs_after or sources_before != sources_after or manifest_sha != sha256(manifest_path) or script_before != sha256(__file__):
            raise RuntimeError("source/input/manifest/driver changed during benchmark")
        public_execution = {name: value for name, value in execution.items()
                            if name not in ("reference", "selected_volumes", "metadata", "middle")}
        report.update(status="complete", execution=public_execution, comparison_to_original_reference=comparison,
                      input_sha256_before=inputs_before, input_sha256_after=inputs_after,
                      input_guards_equal=True, source_guards_equal=True, manifest_driver_guards_equal=True,
                      source_sha256_before=sources_before, source_sha256_after=sources_after,
                      outputs={"reference": output_record(produced)},
                      original_selected_indices=case["official"]["bold_reference"]["selected_frames_zero_based"],
                      selected_indices_match_original=(execution["selected_indices"] ==
                          case["official"]["bold_reference"]["selected_frames_zero_based"]),
                      scientific_equivalence="not_assessed",
                      metric_scope="3D spatial Pearson and unscaled NRMSE; not BOLD time correlation")
        if "middle" in execution:
            report["middle_comparison_to_original_reference"] = compare_reference(execution["middle"], original)
            report["outputs"]["middle"] = output_record(execution["middle"])
        (args.output / "files.private.json").write_text(json.dumps(execution, indent=2) + "\n")
        (args.output / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__,
                      error="See retained failure.private.txt for the original exception")
        (args.output / "failure.private.txt").write_text(traceback.format_exc())
        (args.output / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
