"""Prepared one-case benchmark worker. Not executed without root authorization.

Paths/credentials are supplied by a separate private frozen plan, not here.
Production APIs never import this oracle controller or call native programs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def atomic_report(path, report):
    # Encode BEFORE any file creation, then no-clobber via hard link. A
    # failed serializer cannot create a truncated result JSON as in A v1.
    payload = (json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False)+"\n").encode()
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name+".writing."+str(os.getpid()))
    try:
        fd = os.open(str(temporary), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def check_bindings(plan):
    actual = {}
    for key, item in plan["bindings"].items():
        path = Path(item["path"])
        value = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if value != {k: item[k] for k in ("bytes", "sha256")}:
            raise ValueError("binding changed: "+key)
        actual[key] = value
    return actual


def configure(plan):
    allowed = tuple(plan["physical_cores"])
    if len(allowed) != 8 or len(set(allowed)) != 8:
        raise ValueError("the declared eight physical cores must be distinct")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "8":
            raise ValueError("CPU eight-thread setting differs: "+key)
    for key in ("LD_LIBRARY_PATH","LD_PRELOAD","PYTHONPATH","OPENBLAS_CORETYPE",
                "FS_SetVoxToRasXform_Change_VoxSize"):
        if key in os.environ:
            raise ValueError("expected clean environment: "+key)
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("frozen source cannot write Python caches")
    os.sched_setaffinity(0, allowed)
    resource.setrlimit(resource.RLIMIT_AS, (20000000000, 20000000000))
    os.umask(0o077)
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU benchmark must hide CUDA devices")
    if torch.cuda.is_initialized():
        raise ValueError("CPU benchmark unexpectedly initialized CUDA")
    return {"CPU_affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "Torch_version": str(torch.__version__), "CUDA_initialized": False,
            "matmul_TF32_CPU_unused":torch.backends.cuda.matmul.allow_tf32,
            "Torch_GPU_allocated_bytes":0,"Torch_GPU_reserved_bytes":0,
            "CPU_thread_environment":{key:os.environ[key] for key in
                ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")},
            "system_load":list(os.getloadavg()),
            "address_space_cap_bytes": resource.getrlimit(resource.RLIMIT_AS)[0]}


def official(plan, directory):
    # Both commands read the original immutable inputs or their OWN mapped
    # rigid output. Native affine/fine GEMS references are never inputs.
    times = {}
    for mode in ("rigid", "affine"):
        source = plan["moving"] if mode == "rigid" else str(directory/"rigid.header.mgz")
        command = [plan["native_binary"], "--mov", source, "--dst", plan["fixed"],
                   "--lta", str(directory/(mode+".lta")),
                   "--mapmovhdr", str(directory/(mode+".header.mgz")), "--sat", "50", "-verbose", "0"]
        if mode == "affine":
            command.append("--affine")
        env = os.environ.copy()
        env["FS_LICENSE"] = plan["license_path"]
        env["FREESURFER_HOME"] = plan["native_home"]
        started = time.monotonic()
        with (directory/(mode+".stdout")).open("xb") as stream:
            child = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=plan["official_each_command_timeout_seconds"], check=False)
        times[mode] = {"wall_seconds": time.monotonic()-started, "returncode": child.returncode}
        atomic_report(directory/(mode+".command.json"), times[mode])
        if child.returncode:
            raise RuntimeError("official "+mode+" command failed, rc="+str(child.returncode))
    return {"commands": times, "official_commands_executed": 2}


def fnit(plan, directory):
    from fnit.robust_register import robust_rigid_affine
    stage = directory/"stages"
    started = time.monotonic()
    result = robust_rigid_affine(plan["moving"], plan["fixed"], stage_directory=stage,
                                **plan["parameters"])
    payload = {"rigid": result["rigid"].report, "affine": result["affine"].report,
               "combined_RAS_matrix": result["combined_RAS_matrix"].tolist(),
               "two_stage_with_save_seconds": result["seconds"],
               "call_wall_seconds": time.monotonic()-started,
               "official_commands_executed": 0, "no_native_GEMS_function": True}
    atomic_report(directory/"stages.json", payload)
    return payload


def header_affine(image):
    # Do not use the registration input's orthogonality guard on a final
    # affine mapmovhdr output, whose directions may contain scale/shear.
    import numpy as np
    from fnit.robust_register.registration import _compose_native
    return _compose_native(image.shape, np.asarray(image.header["delta"], np.float32),
                           np.asarray(image.header["Mdc"], np.float32).T,
                           np.asarray(image.header["Pxyz_c"], np.float32))


def score(plan, directory):
    import nibabel as nib
    import numpy as np
    import torch
    from fnit._transforms import load_lta
    from fnit.robust_register._sampling import native_inverse, native_matmul, resample
    out = Path(plan["run_directory"])
    old, new = out/"official", out/"fnit/stages"
    original, target = nib.load(plan["moving"]), nib.load(plan["fixed"])
    values = np.asanyarray(original.dataobj)
    fixed_mask = torch.from_numpy(np.array(target.dataobj, dtype=np.float32, copy=True)) > 0
    if not bool(fixed_mask.any()) or float(values.max()) <= 0:
        raise ValueError("original moving and fixed masks must be nonempty")
    mask_threshold = float(values.max()) * .5

    def overlap(warped):
        moving_mask = warped > mask_threshold
        nsource, ntarget = int(moving_mask.sum()), int(fixed_mask.sum())
        intersection = int((moving_mask & fixed_mask).sum())
        return {"moving_threshold": mask_threshold, "fixed_threshold": 0.,
                "moving_voxels": nsource, "fixed_voxels": ntarget, "intersection_voxels": intersection,
                "Dice": 2*intersection/(nsource+ntarget), "scoring_only_not_optimization_feedback": True}

    initial_pull = native_matmul(native_inverse(header_affine(original)), header_affine(target))
    initial_warp = resample(torch.from_numpy(values.astype(np.float32)), target.shape, initial_pull,
                            chunk_size=plan["parameters"]["spatial_chunk_size"])
    initial_overlap = overlap(initial_warp)
    shape = np.asarray(original.shape)
    origin = header_affine(original).astype(float)
    corners = np.array([[x,y,z] for x in (0,shape[0]-1) for y in (0,shape[1]-1) for z in (0,shape[2]-1)])
    lattice = np.array([[x,y,z] for x in np.linspace(0,shape[0]-1,5)
                        for y in np.linspace(0,shape[1]-1,5) for z in np.linspace(0,shape[2]-1,5)])
    points = origin @ np.c_[np.r_[corners,lattice], np.ones(133)].T
    records, gates = {}, []
    previous = {"official": np.eye(4), "fnit": np.eye(4)}
    for mode in ("rigid", "affine"):
        entries, warped = {}, {}
        for name, folder in (("official",old),("fnit",new)):
            lta_path, image_path = folder/(mode+".lta"), folder/(mode+".header.mgz")
            transform, image = load_lta(lta_path), nib.load(image_path)
            if transform.space != "world":
                raise ValueError("expected RAS-world LTA")
            matrix = transform.matrix
            combined = native_matmul(matrix, previous[name]).astype(float)
            previous[name] = combined
            exact = tuple(image.shape)==tuple(original.shape) and np.array_equal(np.asanyarray(image.dataobj),values)
            exact = exact and image.header.get_data_dtype()==original.header.get_data_dtype()
            expected_source = original if mode=="rigid" else nib.load(folder/"rigid.header.mgz")
            shape_ok = transform.source.shape==tuple(expected_source.shape) and transform.target.shape==tuple(target.shape)
            ideal = native_matmul(matrix,header_affine(expected_source))
            internal = float(np.max(np.abs(header_affine(image).astype(float)-ideal.astype(float))))
            pull = native_matmul(native_inverse(header_affine(image)),header_affine(target))
            warped[name] = resample(torch.from_numpy(values.astype(np.float32)), target.shape, pull,
                                     chunk_size=plan["parameters"]["spatial_chunk_size"])
            entries[name] = {"LTA_sha256":digest(lta_path),"mapped_MGZ_sha256":digest(image_path),
                "source_voxels_shape_dtype_exact":bool(exact),"LTA_source_target_shape_exact":bool(shape_ok),
                "mapped_voxel_to_RAS_internal_max_mm":internal,"RAS_matrix":matrix.tolist(),
                "combined_RAS_matrix":combined.tolist(),"stored_delta":image.header["delta"].tolist(),
                "stored_Mdc":image.header["Mdc"].tolist(),"stored_Pxyz_c":image.header["Pxyz_c"].tolist()}
            gates.extend([bool(exact),bool(shape_ok),internal<=plan["gates"]["mapped_geometry_consistency_max_mm"]])
        pold=np.asarray(entries["official"]["combined_RAS_matrix"])@points
        pnew=np.asarray(entries["fnit"]["combined_RAS_matrix"])@points
        displacement=np.linalg.norm((pnew-pold)[:3],axis=0)
        diff=(warped["fnit"]-warped["official"]).double()
        reference=warped["official"].double()
        if float(torch.linalg.vector_norm(reference)) == 0:
            raise ValueError("official saved geometry warps to an empty target support")
        rel=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(reference))
        support=int(torch.count_nonzero((warped["fnit"]!=0) ^ (warped["official"]!=0)))
        rms=float(np.sqrt(np.mean(displacement**2)));maximum=float(displacement.max())
        gates.extend([rms<=plan["gates"]["physical_displacement_RMS_mm_max"],
                      maximum<=plan["gates"]["physical_displacement_max_mm_max"],
                      rel<=plan["gates"]["warped_same_target_grid_relative_L2_max"],support==0])
        header_fields = {}
        old_header = nib.load(old/(mode+".header.mgz")).header
        new_header = nib.load(new/(mode+".header.mgz")).header
        for field in ("version","dims","type","dof","goodRASFlag","delta","Mdc","Pxyz_c",
                      "tr","flip_angle","te","ti","fov"):
            a,b=np.asarray(old_header[field]),np.asarray(new_header[field])
            header_fields[field]={"exact":bool(np.array_equal(a,b)),
                                  "max_absolute":float(np.max(np.abs(a.astype(float)-b.astype(float))))}
        records[mode]={"outputs":entries,"point_count":133,"displacement_RMS_mm":rms,
            "displacement_max_mm":maximum,"warp_relative_L2":rel,"warp_max_absolute":float(diff.abs().max()),
            "warp_P99_absolute":float(torch.quantile(diff.abs().flatten(),.99)),
            "warp_nonzero_support_difference_voxels":support,
            "world_LTA_matrix_max_absolute":float(np.max(np.abs(np.asarray(entries["fnit"]["RAS_matrix"])-
                                                                np.asarray(entries["official"]["RAS_matrix"])))),
            "combined_world_matrix_max_absolute":float(np.max(np.abs(np.asarray(entries["fnit"]["combined_RAS_matrix"])-
                                                                      np.asarray(entries["official"]["combined_RAS_matrix"])))),
            "mapped_MGH_13_fields":header_fields,
            "fixed_actual_mask_overlap":{"official":overlap(warped["official"]),"fnit":overlap(warped["fnit"])},
            "warp_scope":"shared FNIT sampler on official/new saved mapped geometry; not an independent official resampler oracle"}
    return {"all_declared_gates_pass":bool(all(gates)),"checks":len(gates),"stages":records,
            "initial_header_fixed_mask_overlap":initial_overlap,
            "native_GEMS_or_optimizer_called":False,"MRI_output_arrays_published":False}


def main():
    args=argparse.ArgumentParser()
    args.add_argument("--plan",required=True)
    args.add_argument("--approved-plan-sha",required=True)
    args.add_argument("--phase",choices=("score",),required=True)
    args.add_argument("--recovery-bindings",required=True)
    options=args.parse_args()
    if digest(options.plan)!=options.approved_plan_sha:
        raise ValueError("explicit approved frozen plan SHA differs")
    plan=json.loads(Path(options.plan).read_text())
    sys.path.insert(0,str(Path(plan["source_directory"])/"src"))
    directory=Path(plan["run_directory"])/"score_byteorder_recovery_v2"
    directory.mkdir(mode=0o700,parents=False,exist_ok=False)
    report={"phase":options.phase,"status":"starting","plan_sha256":options.approved_plan_sha,
            "PID":os.getpid(),"scope":"one standalone CPU registration pair or read-only score; no GEMS/whole"}
    code=1;started=time.monotonic()
    try:
        report["bindings_before"]=check_bindings(plan)
        recovery_plan=json.loads(Path(options.recovery_bindings).read_text())
        report["recovery_bindings_before"]=check_bindings(recovery_plan)
        report["actual_resources"]=configure(plan)
        report["result"]={"official":official,"fnit":fnit,"score":score}[options.phase](plan,directory)
        report["status"]="completed"
        if options.phase=="score" and not report["result"]["all_declared_gates_pass"]:
            report["status"]="completed_gates_failed";code=2
        else:
            code=0
    except BaseException as error:
        report["status"]="failed";report["exception"]={"type":type(error).__name__,"message":str(error)}
        (directory/"exception.txt").write_text(traceback.format_exc())
    finally:
        report["wall_seconds"]=time.monotonic()-started
        report["maxrss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if options.phase=="official":
            report["native_children_maxrss_bytes"]=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
        try:
            report["bindings_after"]=check_bindings(plan)
            report["recovery_bindings_after"]=check_bindings(recovery_plan)
            report["recovery_bindings_before_after_exact"]=report.get("recovery_bindings_before")==report["recovery_bindings_after"]
            report["bindings_before_after_exact"]=report.get("bindings_before")==report["bindings_after"]
        except BaseException as error:
            report["post_binding_error"]={"type":type(error).__name__,"message":str(error)};code=1
        report["science_returncode"]=code
        if "torch" in sys.modules:
            import torch
            report["flags_after"]={"CUDA_initialized":torch.cuda.is_initialized(),
                "Torch_GPU_allocated_bytes":0 if not torch.cuda.is_initialized() else torch.cuda.memory_allocated(),
                "Torch_GPU_reserved_bytes":0 if not torch.cuda.is_initialized() else torch.cuda.memory_reserved(),
                "torch_threads":torch.get_num_threads(),"torch_interop_threads":torch.get_num_interop_threads(),
                "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
                "matmul_TF32_CPU_unused":torch.backends.cuda.matmul.allow_tf32,
                "system_load":list(os.getloadavg()),
                "default_GPU_implementation_modified":False}
            if report["flags_after"]["CUDA_initialized"]:
                code=1;report["science_returncode"]=1;report["GPU_scope_violation"]=True
        atomic_report(directory/"report.private.json",report)
        print(json.dumps({"phase":options.phase,"status":report["status"],"science_returncode":code}))
    raise SystemExit(code)


if __name__=="__main__":
    main()
