#!/usr/bin/env python3
"""Official CBIG/Workbench adapter for complete convert_space IO benchmarks.

Only ``run_case`` imports FNIT.  Reference commands call unmodified CBIG MATLAB
functions and Workbench.  The bridge subcommand converts their returned arrays
between MAT, NIfTI and GIFTI; it performs no image registration or interpolation.

Required case keys: id, source, source_space, target_space, assets_dir.
Optional: source_density, target_density, reference, label, wb_command.
For surface input, source is [left_path, right_path].
Resources: wb_command, matlab_command, cbig_source_dir, freesurfer_home,
reference_python (optional), rf_prop_map (optional).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FSAVERAGE = {"3k": "fsaverage4", "10k": "fsaverage5", "41k": "fsaverage6", "164k": "fsaverage"}


def run_case(case: dict, output_dir: Path, device: str) -> dict[str, str]:
    from fnit.space_conversion import convert_space

    source = case["source"]
    if isinstance(source, list):
        source = tuple(source)
    result = convert_space(
        source, case["source_space"], case["target_space"], output_dir,
        case["assets_dir"], source_density=case.get("source_density"),
        target_density=case.get("target_density"), reference=case.get("reference"),
        device=device, label=case.get("label", False),
        wb_command=case.get("wb_command", "wb_command"),
    )
    if isinstance(result, tuple):
        return {"left": str(result[0]), "right": str(result[1])}
    return {"volume": str(result)}


def _surface_paths(output_dir: Path, space: str, density: str, label: bool):
    suffix = "label" if label else "func"
    return [output_dir / f"{hemi}.{space}.{density}.{suffix}.gii" for hemi in ("L", "R")]


def reference_outputs(case: dict, output_dir: Path, resources: dict) -> dict[str, str]:
    if case["target_space"] == "MNI152":
        return {"volume": str(output_dir / "space-MNI152_cortex.nii.gz")}
    pair = _surface_paths(output_dir, case["target_space"], case["target_density"], case.get("label", False))
    return dict(zip(("left", "right"), map(str, pair)))


def _sphere_area(root: Path, space: str, density: str, hemisphere: str, other_space: str):
    # Paths specified by HCP's official 2017 resampling instructions.
    folder = root / "hcp_2017" / "resample_fsaverage"
    if space == "fsaverage":
        name = FSAVERAGE[density]
        return (folder / f"{name}_std_sphere.{hemisphere}.{density}_fsavg_{hemisphere}.surf.gii",
                folder / f"{name}.{hemisphere}.midthickness_va_avg.{density}_fsavg_{hemisphere}.shape.gii")
    if other_space == "fsaverage":
        sphere = folder / f"fs_LR-deformed_to-fsaverage.{hemisphere}.sphere.{density}_fs_LR.surf.gii"
    elif density == "164k":
        sphere = root / "hcp_2017" / f"fsaverage.{hemisphere}_LR.spherical_std.164k_fs_LR.surf.gii"
    else:
        sphere = root / "hcp_2017" / f"{hemisphere}.sphere.{density}_fs_LR.surf.gii"
    return sphere, folder / f"fs_LR.{hemisphere}.midthickness_va_avg.{density}_fs_LR.shape.gii"


def _resample_pair(case, resources, source_pair, source_space, source_density,
                   target_pair, target_space, target_density):
    commands = []
    for hemisphere, source, target in zip(("L", "R"), source_pair, target_pair):
        if source_space == target_space and source_density == target_density:
            if Path(source).resolve() != Path(target).resolve():
                commands.append(["/bin/cp", str(source), str(target)])
            continue
        sphere_in, area_in = _sphere_area(Path(case["assets_dir"]), source_space, source_density, hemisphere, target_space)
        sphere_out, area_out = _sphere_area(Path(case["assets_dir"]), target_space, target_density, hemisphere, source_space)
        commands.append([resources["wb_command"], "-label-resample" if case.get("label") else "-metric-resample",
                         str(source), str(sphere_in), str(sphere_out), "ADAP_BARY_AREA", str(target),
                         "-area-metrics", str(area_in), str(area_out)])
    return commands


def _matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def _matlab_command(resources, script):
    expression = "try,run(" + _matlab_string(script) + ");catch error,disp(getReport(error,'extended'));exit(1);end;exit(0);"
    return [resources["matlab_command"], "-nojvm", "-nodisplay", "-nosplash", "-nodesktop", "-r", expression]


def _matlab_setup(resources):
    return ["maxNumCompThreads(str2double(getenv('OMP_NUM_THREADS')));",
            "setenv('FREESURFER_HOME'," + _matlab_string(resources["freesurfer_home"]) + ");",
            "addpath(" + _matlab_string(Path(resources["freesurfer_home"]) / "matlab") + ");",
            "addpath(" + _matlab_string(resources["cbig_source_dir"]) + ");"]


def reference_command(case: dict, output_dir: Path, resources: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(case["assets_dir"])
    bridge = [resources.get("reference_python", sys.executable), str(Path(__file__).resolve()), "bridge"]
    commands = []
    label = bool(case.get("label", False))
    source_space, source_density = case["source_space"], case.get("source_density")
    if source_space == "MNI152":
        mat = output_dir / "cbig.forward.private.mat"
        script = output_dir / "cbig_forward_private.m"
        lines = _matlab_setup(resources)
        lines += ["reference_start=tic;",
                  "[left,right]=CBIG_RF_projectMNI2fsaverage(" + ",".join(map(_matlab_string, [
                      case["source"], "nearest" if label else "linear",
                      root / "rf_ants/lh.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.mat",
                      root / "rf_ants/rh.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.mat"])) + ");",
                  "save(" + _matlab_string(mat) + ",'left','right','-v7');",
                  "reference_seconds=toc(reference_start);disp(['FNIT_CBIG_API_SECONDS=' num2str(reference_seconds,17)]);"]
        script.write_text("\n".join(lines) + "\n")
        commands.append(_matlab_command(resources, script))
        source_pair = _surface_paths(output_dir, "fsaverage", "164k", label)
        commands.append([*bridge, "--kind", "forward", "--mat", str(mat),
                         "--left", str(source_pair[0]), "--right", str(source_pair[1]),
                         *(["--label"] if label else [])])
        source_space, source_density = "fsaverage", "164k"
    else:
        source_pair = [Path(path) for path in case["source"]]
    if case["target_space"] != "MNI152":
        target_pair = _surface_paths(output_dir, case["target_space"], case["target_density"], label)
        commands += _resample_pair(case, resources, source_pair, source_space, source_density,
                                   target_pair, case["target_space"], case["target_density"])
        return commands

    # The published CBIG reverse function recomputes nearest vertices from
    # .prop.mat coordinates.  The production .vertex.mat shortcut is not used.
    if source_space != "fsaverage" or source_density != "164k":
        target_pair = _surface_paths(output_dir, "fsaverage", "164k", label)
        commands += _resample_pair(case, resources, source_pair, source_space, source_density,
                                   target_pair, "fsaverage", "164k")
        source_pair = target_pair
    inputs = output_dir / "cbig.reverse.input.private.mat"
    result = output_dir / "cbig.reverse.output.private.mat"
    commands.append([*bridge, "--kind", "reverse-input", "--mat", str(inputs),
                     "--left", str(source_pair[0]), "--right", str(source_pair[1])])
    prop = resources.get("rf_prop_map", str(Path(resources["cbig_source_dir"]) /
                           "allSub_fsaverage_to_FSL_MNI152_FS4.5.0_RF_ANTs_avgMapping.prop.mat"))
    mask = root / "rf_ants/FSL_MNI152_FS4.5.0_cortex_estimate.nii.gz"
    script = output_dir / "cbig_reverse_private.m"
    lines = _matlab_setup(resources)
    lines += ["load(" + _matlab_string(inputs) + ");", "reference_start=tic;",
              "for frame=1:size(left,1)",
              "[projected,projected_seg]=CBIG_RF_projectfsaverage2Vol_single(left(frame,:),right(frame,:),'nearest',"
              + _matlab_string(prop) + "," + _matlab_string(mask) + ");",
              "if frame==1, volume=zeros([size(projected.vol),size(left,1)],'single');end;",
              "volume(:,:,:,frame)=single(projected.vol);", "end;",
              "save(" + _matlab_string(result) + ",'volume','-v7');",
              "reference_seconds=toc(reference_start);disp(['FNIT_CBIG_API_SECONDS=' num2str(reference_seconds,17)]);"]
    # scipy cannot read MATLAB v7.3; finite individual maps fit v7 unless the
    # caller requests hundreds of frames.  Such cases need a separate oracle.
    script.write_text("\n".join(lines) + "\n")
    commands.append(_matlab_command(resources, script))
    canonical = output_dir / ("canonical.cortex.nii.gz" if case.get("reference") else "space-MNI152_cortex.nii.gz")
    commands.append([*bridge, "--kind", "reverse-output", "--mat", str(result),
                     "--reference", str(mask), "--volume", str(canonical),
                     *(["--label"] if label else [])])
    if case.get("reference"):
        # A custom voxel grid is an FNIT extension; this explicit Workbench
        # nearest-voxel chain is reported separately from native CBIG output.
        commands.append([resources["wb_command"], "-volume-resample", str(canonical),
                         str(case["reference"]), "ENCLOSING_VOXEL", str(output_dir / "space-MNI152_cortex.nii.gz")])
    return commands


def _bridge(args):
    import nibabel as nib
    import numpy as np
    from scipy.io import loadmat, savemat

    if args.kind == "reverse-input":
        pairs = []
        for path in (args.left, args.right):
            image = nib.load(path)
            pairs.append(np.stack([np.asarray(frame.data) for frame in image.darrays], axis=0))
        savemat(args.mat, dict(zip(("left", "right"), pairs)))
    elif args.kind == "forward":
        maps = loadmat(args.mat)
        for key, path in (("left", args.left), ("right", args.right)):
            values = np.asarray(maps[key])
            image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
                np.asarray(frame, dtype=np.int32 if args.label else np.float32),
                intent="NIFTI_INTENT_LABEL" if args.label else "NIFTI_INTENT_SHAPE") for frame in values])
            nib.save(image, path)
    else:
        reference = nib.load(args.reference)
        volume = loadmat(args.mat)["volume"]
        if volume.ndim == 3:
            volume = volume[..., None]
        volume = volume.transpose(1, 0, 2, 3)
        if volume.shape[3] == 1:
            volume = volume[..., 0]
        volume = np.asarray(volume, dtype=np.int32 if args.label else np.float32)
        header = reference.header.copy()
        header.set_data_dtype(volume.dtype)
        nib.save(nib.Nifti1Image(volume, reference.affine, header), args.volume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bridge = sub.add_parser("bridge", help="IO-only conversion of official output")
    bridge.add_argument("--kind", choices=("forward", "reverse-input", "reverse-output"), required=True)
    bridge.add_argument("--mat", required=True)
    bridge.add_argument("--left")
    bridge.add_argument("--right")
    bridge.add_argument("--reference")
    bridge.add_argument("--volume")
    bridge.add_argument("--label", action="store_true")
    args = parser.parse_args()
    _bridge(args)


if __name__ == "__main__":
    main()
