"""只在声明的真实冻结状态上检查候选；不修改生产网格或调用原阶段程序。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    tick = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root",type=Path,required=True,help="已授权私有FNIT根目录")
    parser.add_argument("--output",type=Path,required=True,help="新的独立诊断目录")
    parser.add_argument("--threads",type=int,default=4,help="总CPU线程预算，默认4")
    args=parser.parse_args();root=args.fnit_root
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    import numba
    import nibabel.freesurfer.io as fsio
    from fnit.recon_all.normal_pending_python import repair_vertex_normals
    import fnit.recon_all.normal_pending_python as candidate_module
    import fnit.recon_all.place_surface_normals as normal_module
    import fnit.recon_all.inflate_topology as topology_module
    import fnit.recon_all.mris_register_nonlinear as reference_module
    torch.set_num_threads(args.threads);numba.set_num_threads(args.threads)
    source_files={name:sha(Path(module.__file__)) for name,module in
                  {"candidate":candidate_module,"unit_CSR_reference":normal_module,
                   "one_ring_reference":topology_module,"repaired_Torch_normal_reference":reference_module}.items()}
    assert source_files["repaired_Torch_normal_reference"]=="47107b4862ae695e06398615534158b8dfcfecd8654dabe83301160c5eb17688"
    cases=[("corner_state",root/"runs/diagnose_registration_nonfinite_20261010_v1/sub06-lh/first_nonfinite.npz"),
           ("face_state",root/"runs/diagnose_registration_nonfinite_20261010_v2/sub06-lh/first_nonfinite.npz"),
           ("v4_registered_final",root/"runs/recon_registration_zero_corner_20261010_v2/v4-new2/sphere.reg"),
           ("normal79_sphere",root/"runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1/sub06-candidate/subject/surf/lh.sphere"),
           ("v4_sphere_input",root/"runs/recon_python_profile_20261010_v4/sub06-candidate/subject/surf/lh.sphere")]
    rows=[]
    for name,path in cases:
        started=time.perf_counter();digest=sha(path)
        if path.suffix==".npz":
            with np.load(path) as checkpoint:
                positions=np.array(checkpoint["arg0"],dtype=np.float32,copy=True)
                faces=np.array(checkpoint["arg1"],dtype=np.int64,copy=True)
        else:
            positions,faces=fsio.read_geometry(str(path));positions=positions.astype(np.float32);faces=faces.astype(np.int64)
        original=positions.copy()
        current=torch.from_numpy(positions.copy());triangles=torch.from_numpy(faces)
        baseline_tick=time.perf_counter()
        try:
            baseline=reference_module.sphere_vertex_normals(current,triangles).numpy()
            baseline_error=None
        except ValueError as error:
            baseline=None;baseline_error=repr(error)
        baseline_seconds=time.perf_counter()-baseline_tick
        api_tick=time.perf_counter();result=repair_vertex_normals(positions=positions,faces=faces)
        api_seconds=time.perf_counter()-api_tick
        repeat_tick=time.perf_counter();repeat=repair_vertex_normals(positions=positions,faces=faces)
        repeat_seconds=time.perf_counter()-repeat_tick
        assert np.array_equal(original,positions) and sha(path)==digest
        assert np.array_equal(result.positions,repeat.positions) and np.array_equal(result.normals,repeat.normals)
        assert np.array_equal(result.pending_history,repeat.pending_history)
        diff=None if baseline is None else np.abs(result.normals.astype(np.float64)-baseline.astype(np.float64))
        row={"name":name,"input":str(path),"input_file_sha256":digest,
             "input_positions_sha256":array_sha(original),"input_faces_sha256":array_sha(faces),
             "vertices":len(positions),"faces":len(faces),"pending_counts":result.pending_counts.tolist(),
             "unresolved_vertices":result.unresolved_vertices.tolist(),"pending_branch_triggered":bool(result.pending_counts.max(initial=0)>0),
             "changed_vertices":len(result.changed_vertices),"positions_unchanged":np.array_equal(result.positions,original),
             "normals_finite":bool(np.isfinite(result.normals).all()),"baseline_error":baseline_error,
             "different_normal_elements":None if baseline is None else int(np.count_nonzero(diff)),
             "max_normal_error":None if diff is None else float(diff.max(initial=0)),
             "p99_normal_error":None if diff is None else float(np.quantile(diff,.99)),
             "result_positions_sha256":array_sha(result.positions),"result_normals_sha256":array_sha(result.normals),
             "first_candidate_API_seconds_including_topology":api_seconds,
             "repeated_candidate_API_seconds_including_topology":repeat_seconds,
             "Torch_CPU_normal_reference_seconds":baseline_seconds,
             "case_wall_seconds_including_read_hash_repeat_compare":time.perf_counter()-started}
        rows.append(row)
        np.savez_compressed(args.output/(name+".npz"),positions=result.positions,normals=result.normals,
                            areas=result.areas,original_areas=result.original_areas,face_normals=result.face_normals,
                            border=result.border,pending_history=result.pending_history,
                            unresolved_vertices=result.unresolved_vertices)
        row["private_output_arrays_sha256"]=sha(args.output/(name+".npz"))
        print(json.dumps(row),flush=True)
    report={"status":"complete","scope":"five frozen real same-input CPU states; not whole recon-all",
            "benchmark_script_sha256":sha(Path(__file__)),"source_module_sha256":source_files,
            "hostname":platform.node(),"CPU_affinity":sorted(os.sched_getaffinity(0)),
            "threads":args.threads,"numpy":np.__version__,"numba":numba.__version__,"torch":torch.__version__,
            "CUDA_initialized":torch.cuda.is_initialized(),"GPU_memory":"not_measured; CPU candidate only",
            "unitize":True,"fix_vertex_area":True,"status_parameter":"sphere","rips":"allFalse",
            "precision":"ordered float32 NumPy/Numba; noTF32 operation changed/nohalf",
            "cases":rows,"whole_harness_wall_seconds_including_imports_IO_and_hash":time.perf_counter()-tick,
            "original_FS_stage_programs":[],"original_FS_normal_same_input_reference":"not_available_or_run",
            "real_pending_branch_validation":"not_triggered" if not any(r["pending_branch_triggered"] for r in rows) else "triggered; seecases",
            "overall_equivalence":"not_assessed","production_connected":False}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
