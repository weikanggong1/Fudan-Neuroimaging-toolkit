"""Real stage T1 native thalamus diagnostics; no upstream executable is called.

By default this only observes production parameters. The optional synthetic-only
scope and accepted-step tracing are diagnostics, not end-to-end benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scope", choices=("synthetic", "full"), default="full")
    parser.add_argument("--optimization", choices=("fast", "balanced"), default="fast")
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-fraction", type=float, default=.14)
    parser.add_argument("--own-memory-limit-mib", type=int, default=19073)
    parser.add_argument("--alignment-report", type=Path,
                        help="optional saved report.json whose affine replaces recomputing alignment; diagnostic")
    parser.add_argument("--trace-steps", action="store_true",
                        help="save accepted-step vertices and gradient norms; adds host synchronization")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label requires only letters/digits/_/-")
    args.root = args.root.resolve()
    args.source = args.source.resolve()
    if args.threads < 1 or not 0 < args.memory_fraction < 1:
        parser.error("positive threads and memory fraction in (0,1) required")
    return args


def launch(args):
    output = args.root / args.label
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.physical_gpu),
               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads),
               PYTHONPATH=str(args.source / "src"), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    status = {"state": "running", "started_unix": time.time(), "command": command,
              "source": str(args.source), "physical_gpu": args.physical_gpu,
              "own_memory_limit_mib": args.own_memory_limit_mib,
              "max_own_memory_mib": 0, "driver": digest(__file__)}
    with (output / "worker.log").open("w") as log, (output / "gpu_load.jsonl").open("w") as load:
        child = subprocess.Popen(command, env=env, cwd=args.root, stdout=log, stderr=subprocess.STDOUT)
        status["pid"] = child.pid
        write_json(output / "status.json", status)
        while child.poll() is None:
            sample = {"unix_time": time.time(), "own_pid": child.pid}
            try:
                sample["gpus"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=3).strip().splitlines()
                rows = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"], text=True, timeout=3).strip().splitlines()
                own = max([int(row.split(",")[1]) for row in rows
                           if row.split(",")[0].strip() == str(child.pid)] or [0])
                sample["own_memory_mib"] = own
                status["max_own_memory_mib"] = max(status["max_own_memory_mib"], own)
                if own > args.own_memory_limit_mib:
                    status["memory_limit_exceeded_mib"] = own
                    child.terminate()
            except (subprocess.SubprocessError, ValueError) as error:
                sample["sampling_error"] = type(error).__name__
            load.write(json.dumps(sample) + "\n")
            load.flush()
            write_json(output / "status.json", status)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if "memory_limit_exceeded_mib" in status:
                    child.kill()
        status.update(state="completed" if child.returncode == 0 else "failed",
                      exit_code=child.returncode, finished_unix=time.time())
        status["process_wall_seconds"] = status["finished_unix"] - status["started_unix"]
        write_json(output / "status.json", status)
    print(json.dumps(status), flush=True)
    return child.returncode


def worker(args):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    import numpy as np
    import torch
    import fnit
    from fnit.gems.context import SubregionContext
    from fnit.gems.core import TorchGEMS
    from fnit.gems.optim import CachedArmijoLBFGS, CachedLBFGS
    from fnit.gems.recipes.thalamus import ThalamusRecipe
    from fnit.gems.initialize import estimate_mask_affine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output = args.root / args.label
    mesh_dir = output / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)
    mri = args.root.parent / "reconall_reference_gpucw1/fs_sub01/mri"
    paths = {name: mri / (name + ".mgz") for name in ("norm", "aseg", "wmparc")}
    reference = args.root.parent / "fnit_subregions_plus_20260928/official_thalamus_gpucw1_full_sub01_20260929/ThalamicNuclei.FSvoxelSpace.mgz"
    atlas_dir = args.root / "atlases/thalamus"
    source_root = Path(fnit.__file__).resolve().parent
    source_facts = {str(path.relative_to(source_root)): digest(path)
                    for path in sorted((source_root / "gems").rglob("*.py"))}
    write_json(output / "inputs.json", {
        "inputs": {name: digest(path) for name, path in paths.items()},
        "reference": digest(reference),
        "atlas": {name: digest(atlas_dir / name)
                  for name in ("AtlasMesh.gz", "AtlasDump.mgz", "compressionLookupTable.txt")},
        "source": source_facts, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device), "parameters": vars(args)})

    fits, trace_steps = [], []
    active_fit = {}
    original_call = TorchGEMS.__call__
    original_step = CachedLBFGS.step
    original_armijo_step = CachedArmijoLBFGS.step
    original_segmentation = ThalamusRecipe.fit_segmentation_mesh

    def observed_segmentation(self, atlas, context, device):
        atlas.save_npz(mesh_dir / "aligned_atlas.npz")
        fitted, report = original_segmentation(self, atlas, context, device)
        fitted.save_npz(mesh_dir / "synthetic_final_atlas.npz")
        return fitted, report

    def observed_call(self, image, **kwargs):
        number = len(fits)
        synthetic = kwargs.get("fixed_gaussians") is not None
        active_fit.update(number=number, synthetic=synthetic, step=0)
        initial_vertices = self.atlas.vertices.copy()
        atlas_alphas = self.atlas.alphas.copy()
        stage_arrays, stage_metadata = {}, []
        for stage_index, (stage_alphas, steps) in enumerate(kwargs.get("fit_alpha_stages") or []):
            values = (stage_alphas.detach().cpu().numpy() if torch.is_tensor(stage_alphas)
                      else np.asarray(stage_alphas)).copy()
            key = f"fit_alpha_stage_{stage_index:02d}"
            stage_arrays[key] = values
            stage_metadata.append({"npz_key": key, "steps": int(steps),
                                   "shape": list(values.shape), "dtype": str(values.dtype),
                                   "array_sha256": hashlib.sha256(values.tobytes()).hexdigest()})
        torch.cuda.synchronize(device)
        before = time.monotonic()
        result = original_call(self, image, **kwargs)
        torch.cuda.synchronize(device)
        fit_seconds = time.monotonic() - before
        line_search = result.optimization_stats.get("mesh_line_search",
                                                   kwargs.get("mesh_line_search", "strong_wolfe"))
        optimizer_class = ("CachedArmijoLBFGS" if line_search == "backtracking" else
                           "CachedLBFGS" if kwargs.get("cache_mesh_evaluations", True) else
                           "PrecisionLBFGS")
        vertices = result.vertices.detach().cpu().numpy()
        path = mesh_dir / f"fit_{number:02d}_{'synthetic' if synthetic else 'intensity'}.npz"
        np.savez_compressed(path, initial_vertices=initial_vertices, vertices=vertices,
                            atlas_alphas=atlas_alphas, **stage_arrays,
                            reference_vertices=self.atlas.reference_vertices,
                            tetrahedra=self.atlas.tetrahedra,
                            means=result.gaussian_parameters.means.detach().cpu().numpy(),
                            covariances=result.gaussian_parameters.covariances.detach().cpu().numpy())
        fits.append({"number": number, "synthetic": synthetic, "fit_seconds": fit_seconds,
                     "min_jacobian": result.min_jacobian,
                     "objective_history": result.objective_history,
                     "solver": result.optimization_stats, "mesh": digest(path),
                     "input_shape": list(image.shape),
                     "mesh_sampling_stride": kwargs.get("mesh_sampling_stride"),
                     "data_cost_fp64": kwargs.get("double_data_cost_accumulation"),
                     "stable_mesh_fitting": kwargs.get("stable_mesh_fitting", False),
                     "optimizer_state_precision": result.optimization_stats.get("optimizer_state_precision"),
                     "optimizer_class": optimizer_class,
                     "line_search": line_search,
                     "optimizer_class_evidence": "Recorded solver line search and unchanged TorchGEMS optimizer selection; per-step trace records the runtime class when enabled",
                     "fit_alpha_stages": stage_metadata,
                     "relative_cost_stop": kwargs.get("relative_cost_stop"),
                     "deformation_stop": kwargs.get("deformation_stop")})
        write_json(output / "fit_trace.json", fits)
        return result

    def observed_step(self, closure, **kwargs):
        before_vertices = (self._params[0].detach().clone()
                           if active_fit.get("synthetic") else None)
        original_method = (original_armijo_step if isinstance(self, CachedArmijoLBFGS)
                           else original_step)
        result = original_method(self, closure, **kwargs)
        if active_fit.get("synthetic"):
            number, step = active_fit["number"], active_fit["step"]
            active_fit["step"] += 1
            vertices = self._params[0].detach().cpu().numpy().copy()
            last_closure_gradient = self._gather_flat_grad().detach()
            accepted_gradient = None if self._cached is None else self._cached[1]
            displacement = torch.linalg.vector_norm(
                self._params[0].detach() - before_vertices, dim=1)
            state = self.state[self._params[0]]
            previous_gradient, direction = state.get("prev_flat_grad"), state.get("d")
            gtd = (None if previous_gradient is None or direction is None else
                   float(previous_gradient.dot(direction)))
            path = mesh_dir / f"synthetic_{number:02d}_step_{step:04d}.npz"
            np.savez_compressed(path, vertices=vertices)
            trace_steps.append({"fit_number": number, "step": step,
                "optimizer_class": type(self).__name__,
                "line_search": "backtracking" if isinstance(self, CachedArmijoLBFGS) else "strong_wolfe",
                "accepted_objective": None if self.accepted_objective is None else
                                      float(self.accepted_objective),
                "returned_objective": float(result),
                "accepted_displacement_max_voxels": float(displacement.max()),
                "accepted_displacement_mean_voxels": float(displacement.mean()),
                "optimizer_step_t": None if state.get("t") is None else float(state["t"]),
                "optimizer_gtd_previous_gradient_dot_direction": gtd,
                "optimizer_iterations": state.get("n_iter"),
                "accepted_gradient_l2": None if accepted_gradient is None else
                                        float(torch.linalg.vector_norm(accepted_gradient)),
                "accepted_gradient_abs_max": None if accepted_gradient is None else
                                             float(accepted_gradient.abs().max()),
                "accepted_gradient_precision": None if accepted_gradient is None else str(accepted_gradient.dtype),
                "last_closure_gradient_l2": float(torch.linalg.vector_norm(last_closure_gradient)),
                "last_closure_gradient_abs_max": float(last_closure_gradient.abs().max()),
                "gradient_observation": "Accepted projected gradient uses CachedLBFGS._cached[1]; the parameter gradient from the last closure need not be the accepted trial gradient. gtd uses optimizer prev_flat_grad and d after this step.",
                "closure_evaluations": self.closure_evaluations,
                "last_step_evaluations": self.last_step_evaluations,
                "cache_hits": self.cache_hits, "mesh": digest(path)})
            write_json(output / "accepted_step_trace.json", trace_steps)
        return result

    TorchGEMS.__call__ = observed_call
    ThalamusRecipe.fit_segmentation_mesh = observed_segmentation
    if args.trace_steps:
        CachedLBFGS.step = observed_step
        CachedArmijoLBFGS.step = observed_step
    try:
        started = time.monotonic()
        context = SubregionContext.prepare(paths["norm"], need_coarse=True, need_parc=False,
                   coarse_segmentation=paths["aseg"], wmparc=paths["wmparc"], device=device)
        recipe = ThalamusRecipe("thalamus", atlas_dir)
        recipe.set_optimization_profile(args.optimization)
        if args.scope == "full" and args.alignment_report is None:
            # Use the production run method unchanged.
            outcome = recipe.run(context, device)
            report = dict(outcome.report)
        else:
            atlas = recipe.atlas()
            if args.alignment_report is not None:
                saved = json.loads(args.alignment_report.read_text())
                saved = saved.get("initialization", saved)["thalamus"]
                matrix = np.asarray(saved["atlas_to_native_voxel"], dtype=np.float64)
                score = saved["alignment_dice"]
            else:
                matrix, score = estimate_mask_affine(recipe.alignment_image(), context.image,
                            context.coarse_segmentation, recipe.alignment_ids, device=device)
            aligned = atlas.transformed(matrix, transform_reference=True)
            aligned.save_npz(mesh_dir / "aligned_atlas.npz")
            fitted, synthetic_report = recipe.fit_segmentation_mesh(aligned, context, device)
            fitted.save_npz(mesh_dir / "synthetic_final_atlas.npz")
            report = {"alignment_dice": score, "atlas_to_native_voxel": matrix.tolist(),
                      "segmentation_fit": synthetic_report}
            if args.scope == "full":
                fit, _, work_report = recipe.fit_intensity_mesh(fitted, context, device)
                outcome = recipe.postprocess(fit, context, fitted)
                report["working_image"] = work_report
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        if args.scope == "full":
            native = nib.Nifti1Image(outcome.native_labels.astype(np.int32), context.image.affine)
            native.set_qform(context.image.affine, code=0)
            native.set_sform(context.image.affine, code=2)
            nib.save(native, output / "thalamus_native.nii.gz")
            nib.save(outcome.highres_labels, output / "thalamus_highres.nii.gz")
            truth_image = nib.load(reference)
            if truth_image.shape != native.shape or not np.allclose(truth_image.affine, native.affine, atol=1e-5):
                truth_image = resample_from_to(truth_image, (native.shape, native.affine), order=0)
            truth = np.asarray(truth_image.dataobj, dtype=np.int32)
            got = outcome.native_labels
            labels = sorted(set(int(v) for v in np.unique(truth) if 8100 <= v < 8300)
                            | set(int(v) for v in np.unique(got) if 8100 <= v < 8300))
            rows = []
            for label in labels:
                mask_r, mask_g = truth == label, got == label
                nr, ng = int(mask_r.sum()), int(mask_g.sum())
                dice = 2 * int(np.count_nonzero(mask_r & mask_g)) / max(nr + ng, 1)
                rows.append({"label": label, "reference_voxels": nr, "fnit_voxels": ng,
                             "dice": dice, "soft_volume_mm3": outcome.soft_volumes_mm3.get(label)})
            report["official_comparison"] = {
                "reference": str(reference), "regions": rows,
                "reference_voxel_weighted_dice": sum(row["dice"] * row["reference_voxels"] for row in rows)
                     / max(sum(row["reference_voxels"] for row in rows), 1),
                "mean_dice": sum(row["dice"] for row in rows) / max(len(rows), 1)}
            report["comparisons"] = {"thalamus": report["official_comparison"]}
            report["soft_volumes_mm3"] = outcome.soft_volumes_mm3
            report["comparison_baselines"] = {}
            for name, baseline in {
                "c6": args.root / "full_stage_v16_c6_gpu1_20261001/report.json",
                "regressed": args.root / "segment4_full_stage_20261001/report.json",
            }.items():
                if baseline.is_file():
                    old_rows = json.loads(baseline.read_text())["comparisons"]["thalamus"]["regions"]
                    old = {row["label"]: row for row in old_rows}
                    common = [row for row in rows if row["label"] in old]
                    denominator = sum(row["reference_voxels"] for row in common)
                    expected = sum(old[row["label"]]["dice"] * row["reference_voxels"] for row in common) / max(denominator, 1)
                    actual = sum(row["dice"] * row["reference_voxels"] for row in common) / max(denominator, 1)
                    report["comparison_baselines"][name] = {
                        "baseline_report": digest(baseline), "common_labels": len(common),
                        "reference_voxel_weighted_baseline": expected,
                        "reference_voxel_weighted_current": actual, "signed_delta": actual - expected,
                        "per_label": [{"label": row["label"], "dice_baseline": old[row["label"]]["dice"],
                            "dice_current": row["dice"], "reference_voxels": row["reference_voxels"]}
                            for row in common]}
            report["fit_min_jacobian"] = outcome.fit.min_jacobian
        report.update(scope=args.scope, optimization=args.optimization,
            wall_seconds=elapsed, peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            observation_overhead_included=True, accepted_step_tracing=args.trace_steps,
            frozen_affine=args.alignment_report is not None,
            production_parameters_unchanged=True, context_metadata=context.metadata,
            fit_trace=str(output / "fit_trace.json"))
        write_json(output / "report.json", report)
        print(json.dumps({"scope": args.scope, "seconds": elapsed,
                          "official": report.get("official_comparison", {}).get("reference_voxel_weighted_dice"),
                          "synthetic": report["segmentation_fit"]}), flush=True)
    finally:
        TorchGEMS.__call__ = original_call
        CachedLBFGS.step = original_step
        CachedArmijoLBFGS.step = original_armijo_step
        ThalamusRecipe.fit_segmentation_mesh = original_segmentation


if __name__ == "__main__":
    opts = arguments()
    if opts.worker:
        worker(opts)
    else:
        raise SystemExit(launch(opts))
