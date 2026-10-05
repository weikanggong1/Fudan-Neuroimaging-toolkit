"""Adapt the frozen real-data bindings without altering their original manifest."""
import argparse
import json
from pathlib import Path

from run_same_input import SCHEMA, sha256, write_json, input_specs


def build(bindings_path, case_id, destination):
    source = Path(bindings_path).resolve()
    bindings = json.loads(source.read_text())
    case = bindings["cases"][case_id]
    fnit, raw, tools, assets = case["fnit"], case["raw"], bindings["tools"], bindings["assets"]
    native_files = {"orig_mgz": fnit["orig_mgz"], "raw_import_001_mgz": fnit["raw_import_001_mgz"],
                    "reconstruction_manifest": fnit["reconstruction_manifest"],
                    "surface_metadata": fnit["retained"]["metadata"],
                    "raw_bold_json": raw["bold_json"], "raw_t1w_json": raw["t1w_json"],
                    "source_bindings": {"path": str(source), "sha256": sha256(source)}}
    hemispheres = {}
    for hemi in ("L", "R"):
        native = fnit["hemispheres"][hemi]
        for name in ("white", "pial", "graymid", "sphere.reg", "thickness"):
            native_files[hemi + "." + name] = native[name]
        for name, entry in assets[hemi]["initialization_templates"].items():
            native_files[hemi + "." + name] = entry
        hemispheres[hemi] = {"registered_sphere": native["registered_sphere"],
                             "atlas_sphere": assets[hemi]["atlas_sphere"],
                             "atlas_roi": assets[hemi]["atlas_roi"]}
    manifest = {
        "schema": SCHEMA, "data_kind": "real_mri", "case_id": case_id,
        "expected_frames": case["frames"], "tr_seconds": case["tr_seconds"],
        "input_bindings_sha256": sha256(source), "manifest_builder_sha256": sha256(__file__),
        "inputs": {"raw_bold": raw["bold"], "raw_t1w": raw["t1w"],
                   "t1w_bold": fnit["retained"]["preproc_t1w"],
                   "mni_bold": fnit["retained"]["preproc_mni"],
                   "left_label": assets["L"]["atlas_roi"], "right_label": assets["R"]["atlas_roi"],
                   "hcp_dseg": assets["hcp_dseg"]},
        "hemispheres": hemispheres, "fnit_workbench": tools["fnit_workbench"],
        "reference": {
            "container": tools["official_container"],
            "container_prefix": [tools["singularity"]["path"], "exec", "--cleanenv", "--bind",
                                 "/cwStorage,/public,/home1", tools["official_container"]["path"]],
            "workbench": tools["official_workbench"]["container_path"],
            "workbench_sha256": tools["official_workbench"]["sha256"],
        },
        "preparation": {"subject_dir": fnit["subject_dir"], "hcp_assets_dir": fnit["hcp_assets_dir"],
                        "recon_source": "FNIT_own_frozen_reconstruction", "input_files": native_files},
        "protected_roots": [fnit["subject_dir"], fnit["hcp_assets_dir"],
                            str(Path(raw["bold"]["path"]).parent), str(Path(raw["t1w"]["path"]).parent),
                            str(Path(fnit["retained"]["preproc_t1w"]["path"]).parent)],
    }
    input_specs(manifest)
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError("Refusing to overwrite a bound diagnostic manifest")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, manifest)
    return {"case_id": case_id, "manifest_sha256": sha256(target),
            "input_bindings_sha256": manifest["input_bindings_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.bindings, args.case, args.output_manifest)))
