"""同一真实 white/MRI 输入的完整 Python pial CPU/Torch 正则梯度回归。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import os
import shutil
import time
import traceback

import nibabel.freesurfer.io as fs
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--code-base-commit", required=True)
    parser.add_argument("--python-order", nargs="+", choices=("cpu", "torch"), default=["cpu", "torch"])
    parser.add_argument("--candidate-backend", choices=("tree", "snapshot", "torch_snapshot"), default="tree")
    parser.add_argument("--candidate-grid-cells-per-axis", type=int, choices=(2,3), default=2)
    parser.add_argument("--retained-mht-backend", choices=("tree", "compiled"), default="tree")
    parser.add_argument("--candidate-regularization-backend", choices=("cpu", "torch"), default="torch")
    parser.add_argument("--cleanup-marking-backend", choices=("legacy", "source_numba", "source_torch"), default="legacy")
    parser.add_argument("--cleanup-grid-cells-per-axis", type=int, choices=(2,3), default=2)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--assets-directory", type=Path)
    parser.add_argument("--native-repeat", type=int, default=2)
    args = parser.parse_args()
    if len(args.python_order)!=2 or set(args.python_order)!={"cpu","torch"}:
        raise ValueError("one CPU control and one GPU candidate, either order, required")
    if args.threads<1 or args.native_repeat<1:
        raise ValueError("positive threads/repeat required")
    if args.native_binary is not None and args.assets_directory is None:
        raise ValueError("native reference requires declared assets directory")
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory))
    from fnit.recon_all import place_pial_python as stage
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    hemi = args.hemisphere
    inputs = [args.subject / f"surf/{hemi}.white", args.subject / f"surf/autodet.gw.stats.{hemi}.dat",
              args.subject / f"label/{hemi}.cortex.label", args.subject / f"label/{hemi}.cortex+hipamyg.label"]
    inputs += [args.subject / f"mri/{name}.mgz" for name in ("brain.finalsurfs", "wm", "aseg.presurf")]
    report = {"scope": "frozen_same_input_complete_python_pial", "hostname": platform.node(),
              "code_base_commit": args.code_base_commit, "candidate_snapshot": "source_sha256",
              "torch": torch.__version__, "threads": args.threads, "device": args.device,
              "tf32_matmul": True, "tf32_cudnn": True, "half_precision": False,
              "gpu": None, "input_sha256": {path.name: sha256(path) for path in inputs},
              "stages": {}, "order": args.python_order, "repetitions": 1,
              "admission_requirement": "identical ordered final geometry and pass/trial decisions",
              "overall_metric_equivalence": "not_assessed", "source_sha256": {},
              "process_gpu_memory_sampling": "external sampler; allocator peaks separately recorded",
              "cpu_affinity_count":len(os.sched_getaffinity(0)),
              "cuda_allocator_environment":{key:os.environ.get(key) for key in
                  ("PYTORCH_NO_CUDA_MEMORY_CACHING","PYTORCH_CUDA_ALLOC_CONF","PYTORCH_ALLOC_CONF")},
              "candidate_strategy":{"candidate_backend":args.candidate_backend,
                  "grid_cells_per_axis":args.candidate_grid_cells_per_axis,
                  "retained_mht_backend":args.retained_mht_backend,
                  "regularization_backend":args.candidate_regularization_backend},
              "cleanup_strategy":{"backend":args.cleanup_marking_backend,
                  "grid_cells_per_axis":args.cleanup_grid_cells_per_axis},
              "native_reference_kind":"current independently Conda source-built program, fresh same-host same-input; not installed official",
              "native_runs":[],"status":"started"}
    # 所有实际参与模块分别绑定；不把服务器旧repo提交当成候选提交。
    import sys
    for name, module in list(sys.modules.items()):
        if name.startswith("fnit.recon_all.place_") and getattr(module, "__file__", None):
            report["source_sha256"][Path(module.__file__).name] = sha256(module.__file__)
    report["source_sha256"][Path(__file__).name] = sha256(__file__)
    def save():
        for name, module in list(sys.modules.items()):
            if name.startswith("fnit.recon_all.place_") and getattr(module, "__file__", None):
                report["source_sha256"][Path(module.__file__).name] = sha256(module.__file__)
        temporary=args.output_directory/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        temporary.replace(args.output_directory/'report.json')
    save()
    outputs, traces = {}, {}
    for backend in report["order"]:
        output = args.output_directory / f"{hemi}.pial.{backend}"
        trace = []
        def callback(step, outer_pass, coordinates, diagnostics):
            row = {"step": step, "pass": outer_pass,
                   "coordinate_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
                   "diagnostics": diagnostics}
            trace.append(row)
            print(json.dumps({"backend": backend, **row}, ensure_ascii=False), flush=True)
        tick = time.perf_counter()
        try:
            uses_gpu=backend=="torch" or args.cleanup_marking_backend=="source_torch"
            setup_seconds=0.0
            if uses_gpu:
                setup=time.perf_counter();torch.cuda.synchronize(torch.device(args.device))
                setup_seconds=time.perf_counter()-setup
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            tick=time.perf_counter()
            result = stage.place_pial_t1(subject=args.subject, hemisphere=hemi, output=output,
                max_steps=args.max_steps, sampling_backend="cpu",
                regularization_backend="cpu" if backend=="cpu" else args.candidate_regularization_backend,
                candidate_backend="tree" if backend=="cpu" else args.candidate_backend,
                candidate_grid_cells_per_axis=2 if backend=="cpu" else args.candidate_grid_cells_per_axis,
                retained_mht_backend="tree" if backend=="cpu" else args.retained_mht_backend,
                cleanup_marking_backend=args.cleanup_marking_backend,
                cleanup_candidate_grid_cells_per_axis=args.cleanup_grid_cells_per_axis,
                device=args.device if uses_gpu else None,
                trace_callback=callback, profile=True)
            if uses_gpu:
                torch.cuda.synchronize(torch.device(args.device))
                report["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
            report["stages"][backend] = {"status": "complete", "wall_seconds": time.perf_counter()-tick,
                "CUDA_setup_seconds":setup_seconds,
                "stage": result, "output_sha256": sha256(output), "trace": trace,
                "peak_allocated_bytes": None if not uses_gpu else torch.cuda.max_memory_allocated(torch.device(args.device)),
                "peak_reserved_bytes": None if not uses_gpu else torch.cuda.max_memory_reserved(torch.device(args.device))}
            outputs[backend] = output
            traces[backend] = trace
        except Exception as exc:
            report["stages"][backend] = {"status": "failed", "wall_seconds": time.perf_counter()-tick,
                                         "error": str(exc), "traceback": traceback.format_exc(), "trace": trace}
            report["status"]="failed_python"
            save()
            raise
        save()
    a, af = fs.read_geometry(str(outputs["cpu"]))
    b, bf = fs.read_geometry(str(outputs["torch"]))
    ordered = a.shape == b.shape and np.array_equal(af, bf)
    difference = None if not ordered else np.linalg.norm(a-b, axis=1)
    report["comparison"] = {"same_vertex_count_and_ordered_faces": ordered,
        "same_file_bytes": sha256(outputs["cpu"]) == sha256(outputs["torch"]),
        "same_trace": traces["cpu"] == traces["torch"],
        "different_coordinate_elements": None if not ordered else int(np.count_nonzero(a != b)),
        "max_vertex_distance_mm": None if not ordered else float(difference.max()),
        "p99_vertex_distance_mm": None if not ordered else float(np.percentile(difference, 99)),
        "speed_ratio_cpu_over_torch": report["stages"]["cpu"]["wall_seconds"] / report["stages"]["torch"]["wall_seconds"]}
    report["strict_backend_reproduction"]="passed" if (ordered and np.array_equal(a,b) and
        traces["cpu"]==traces["torch"]) else "failed"
    report["new_degradation_under_declared_exact_backend_gate"]=report["strict_backend_reproduction"]
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    report['mesh_quality_not_in_stage_timing']={}
    for backend,path in outputs.items():
        vertices,faces=fs.read_geometry(str(path))
        _,count=mark_source_intersections(vertices=vertices,faces=faces,
            predicate_backend='torch',device=args.device,candidate_grid_cells_per_axis=3)
        report['mesh_quality_not_in_stage_timing'][backend]={'finite_coordinates':bool(np.isfinite(vertices).all()),
            'source_directional_intersecting_face_count':int(count)}
    report['mesh_quality_status']='passed' if all(row['finite_coordinates'] and
        row['source_directional_intersecting_face_count']==0 for row in report['mesh_quality_not_in_stage_timing'].values()) else 'failed'
    save()
    if args.native_binary is not None:
        from fnit.recon_all.pial_t1_conda import run_pial_t1
        report["native_program_sha256"]=sha256(args.native_binary)
        native_outputs=[]
        for repetition in range(args.native_repeat):
            isolated=args.output_directory/f'native-{repetition}-subject'
            for directory in ('surf','mri','label'):(isolated/directory).mkdir(parents=True)
            for source in inputs+[args.subject/f'label/{hemi}.aparc.annot']:
                destination=isolated/source.relative_to(args.subject)
                shutil.copy2(source,destination)
            native_inputs={str(source.relative_to(args.subject)):sha256(isolated/source.relative_to(args.subject))
                for source in inputs+[args.subject/f'label/{hemi}.aparc.annot']}
            started=time.perf_counter()
            try:
                with (args.output_directory/f'native-{repetition}.private.log').open('w') as stream:
                    # Keep the existing native wrapper; capture only our child
                    # output, without global subprocess monkeypatching.
                    saved_out,saved_err=os.dup(1),os.dup(2)
                    try:
                        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
                        native=run_pial_t1(subject_dir=isolated,hemi=hemi,binary=args.native_binary,
                            assets_dir=args.assets_directory,threads=args.threads)
                    finally:
                        os.dup2(saved_out,1);os.dup2(saved_err,2);os.close(saved_out);os.close(saved_err)
                elapsed=time.perf_counter()-started
                native_output=Path(native['output']);native_outputs.append(native_output)
                n,nf=fs.read_geometry(str(native_output))
                same=n.shape==a.shape and np.array_equal(nf,af)
                errors=None if not same else np.linalg.norm(n-a,axis=1)
                _,intersection_count=mark_source_intersections(vertices=n,faces=nf,
                    predicate_backend='torch',device=args.device,candidate_grid_cells_per_axis=3)
                first,first_faces=fs.read_geometry(str(native_outputs[0]))
                report['native_runs'].append({'status':'complete','seconds_cold_CLI_including_read_compute_write':elapsed,
                    'native_input_sha256':native_inputs,'output_sha256':sha256(native_output),
                    'same_vertex_count_and_ordered_faces':same,
                    'different_coordinate_elements':None if not same else int(np.count_nonzero(n!=a)),
                    'p99_vertex_distance_mm':None if errors is None else float(np.percentile(errors,99)),
                    'max_vertex_distance_mm':None if errors is None else float(errors.max()),
                    'source_directional_intersecting_face_count':int(intersection_count),
                    'repeat_same_ordered_faces':np.array_equal(nf,first_faces),
                    'repeat_different_coordinate_elements':None if first.shape!=n.shape else int(np.count_nonzero(first!=n))})
            except Exception as exc:
                report['native_runs'].append({'status':'failed','wall_seconds':time.perf_counter()-started,
                    'error':str(exc),'traceback':traceback.format_exc()})
                report['status']='failed_native';save();raise
            save()
    report['input_sha256_after']={path.name:sha256(path) for path in inputs}
    if report['input_sha256_after']!=report['input_sha256']:
        report['status']='failed_input_changed';save();raise RuntimeError('frozen inputs changed')
    report['status']='complete'
    save()
    if report['strict_backend_reproduction']!='passed':raise SystemExit(1)


if __name__ == "__main__":
    main()
