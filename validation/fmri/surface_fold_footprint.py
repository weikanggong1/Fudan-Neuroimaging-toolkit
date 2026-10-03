"""Measure the fsLR32k footprint of saved-sphere folds; benchmark QC only.

Original Workbench resamples a private indicator of folded-face cortical
vertices through each run's actual sphere and area surfaces. This does not
alter either BOLD result or estimate a causal effect of repairing a fold.
Only aggregate counts and paired errors are written to the public report.
"""

import argparse
import json
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np

import compare_surface_e2e as comparison


def folded_vertices(registered, input_sphere, roi_file):
    def mesh(path):
        image = nib.load(str(path))
        return (np.asarray(image.get_arrays_from_intent(1008)[0].data, np.float64),
                np.asarray(image.get_arrays_from_intent(1009)[0].data, np.int64))
    actual, faces = mesh(registered)
    before, input_faces = mesh(input_sphere)
    if actual.shape != before.shape or not np.array_equal(faces, input_faces):
        raise ValueError("fold footprint requires identical registration input/output topology")
    def orientation(vertices):
        t = vertices[faces]
        return np.einsum("ij,ij->i", np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), t[:, 0])
    baseline, result = orientation(before), orientation(actual)
    usable = baseline != 0
    folded = usable & (result * baseline <= 0)
    roi = np.asarray(nib.load(str(roi_file)).darrays[0].data) > 0
    marker = np.zeros(actual.shape[0], np.float32)
    marker[np.unique(faces[folded])] = 1
    marker[~roi] = 0
    return marker, int(folded.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--hcp-assets-dir", type=Path, required=True)
    parser.add_argument("--wb-command", type=Path, required=True)
    parser.add_argument("--reference-wb-command", type=Path,
                        help="Optional reference-image launcher for the actual original Workbench")
    parser.add_argument("--reference-wb-sha256",
                        help="Actual binary SHA verified inside that reference-image launcher")
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if args.reference_wb_command and (not args.reference_wb_sha256 or len(args.reference_wb_sha256) != 64):
        parser.error("reference-wb-command requires the verified actual reference-wb-sha256")
    args.private_output.mkdir(parents=True, exist_ok=False)
    branches = {}
    for label, path in (("candidate", args.candidate_manifest), ("reference", args.reference_manifest)):
        manifest, inputs = comparison.read_manifest(path)
        msm = (comparison.read_json(manifest["msm_inputs_json"]) if "msm_inputs_json" in manifest
               else manifest["msm_inputs"])
        branches[label] = (manifest, inputs, msm)
    mesh = args.hcp_assets_dir / "global/templates/standard_mesh_atlases"
    report = {"schema_version": 1, "validation_complete": False,
              "definition": "A private 0/1 indicator marks native ROI vertices incident to saved-sphere "
                            "folded faces, defined relative to each actual rotated MSM input. Original "
                            "Workbench ADAP_BARY_AREA with each run's sphere, current ROI and native/32k "
                            "area surfaces resamples it. Positive 32k weights within the atlas ROI define "
                            "the footprint. Paired errors in the union of both footprints are descriptive; "
                            "they do not isolate a causal fold contribution from other pipeline differences.",
              "script_sha256": comparison.sha256(__file__),
              "candidate_workbench_sha256": comparison.sha256(args.wb_command),
              "reference_workbench_sha256": args.reference_wb_sha256
                  if args.reference_wb_command else comparison.sha256(args.wb_command),
              "reference_workbench_launcher_sha256": comparison.sha256(args.reference_wb_command)
                  if args.reference_wb_command else None,
              "hemispheres": {}}
    commands = []
    for index, hemi in enumerate("LR"):
        footprints, records = {}, {}
        atlas = np.asarray(nib.load(str(mesh / f"{hemi}.atlasroi.32k_fs_LR.shape.gii")).darrays[0].data) > 0
        for label, (manifest, inputs, msm) in branches.items():
            marker, count = folded_vertices(manifest["registered_spheres"][index],
                                            msm[hemi]["rotated_sphere"], inputs["native_rois"][index])
            source = args.private_output / f"{label}_{hemi}_native_fold.shape.gii"
            target = args.private_output / f"{label}_{hemi}_32k_fold.shape.gii"
            nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(marker, intent=2005)]), source)
            program = args.reference_wb_command if label == "reference" and args.reference_wb_command else args.wb_command
            command = [str(program), "-metric-resample", str(source),
                       manifest["registered_spheres"][index], str(mesh / f"{hemi}.sphere.32k_fs_LR.surf.gii"),
                       "ADAP_BARY_AREA", str(target), "-area-surfs", inputs["area_surfaces"]["native"][index],
                       inputs["area_surfaces"]["fsLR"][index], "-current-roi", inputs["native_rois"][index]]
            commands.append(command)
            with (args.private_output / "commands.private.log").open("ab") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            values = np.asarray(nib.load(str(target)).darrays[0].data)
            if values.shape != atlas.shape or not np.isfinite(values).all():
                raise ValueError("invalid fold indicator resampling")
            footprints[label] = (values > 0) & atlas
            records[label] = {"folded_faces": count, "native_roi_vertices_incident_to_folds": int(marker.sum()),
                              "affected_cortical_32k_vertices": int(footprints[label].sum()),
                              "registered_sphere_sha256": comparison.sha256(manifest["registered_spheres"][index])}
        union = footprints["candidate"] | footprints["reference"]
        field = "left" if hemi == "L" else "right"
        arrays = [np.stack([a.data for a in nib.load(manifest[field]).darrays])[:, union].astype(np.float64)
                  for manifest, _, _ in branches.values()]
        paired = None
        if union.any():
            x, y = arrays
            delta = x - y
            paired = {**comparison.temporal_summary(*comparison.temporal_statistics(x, y)),
                      "values": int(delta.size), "mean_absolute_error": float(np.abs(delta).mean()),
                      "rmse": float(np.sqrt(np.mean(delta * delta))),
                      "maximum_absolute_error": float(np.abs(delta).max())}
        report["hemispheres"][hemi] = {**records, "union_affected_cortical_32k_vertices": int(union.sum()),
                                       "paired_all_490_frames_in_union": paired}
    (args.private_output / "commands.private.json").write_text(json.dumps({"commands": commands}, indent=2) + "\n")
    report["validation_complete"] = True
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"validation_complete": True, "hemispheres": report["hemispheres"]}))


if __name__ == "__main__":
    main()
