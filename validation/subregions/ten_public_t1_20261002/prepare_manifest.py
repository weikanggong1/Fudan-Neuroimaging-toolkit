"""Pin ten already verified public inputs and the current source for benchmarking."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    assets = args.assets_root.resolve()
    root = assets / "ten_public_t1_20261002"
    source = args.source.resolve()
    path = root / "cohort_manifest.json"
    if path.exists():
        raise ValueError("Preserve existing cohort manifest")
    data = json.loads((root / "data_manifest.json").read_text())
    if data["selected_subjects"] != [f"sub-{n:02d}" for n in range(1,11)]:
        raise ValueError("Not the preselected ten subjects")
    previous = assets / "source_precision_reproducibility_final_all_20261002/source_manifest.json"
    old = {r["path"]:r["sha256"] for r in json.loads(previous.read_text())["files"]}
    current = json.loads((source / "source_manifest.json").read_text())
    current_files = {r["path"]:r["sha256"] for r in current["files"]}
    gems = [p for p in old if p.startswith("src/fnit/gems/")]
    if not gems or any(current_files.get(p) != old[p] for p in gems):
        raise ValueError("Label metadata requires revalidation for changed GEMS source")
    previous_api = assets / "reproducibility_20261002/final_all_release_stage_r1/api_report.json"
    labels = json.loads(previous_api.read_text())["labels"]
    if len(labels) != 110:
        raise ValueError("Incorrect canonical label count")
    cases = []
    for record in data["inputs"]:
        name = record["subject_id"]
        case_root = root / "cases" / name
        official = case_root / "official"
        mri = official / "subjects" / name / "mri"
        subregions = {}
        filenames = {"brainstem": ("brainstem", "brainstemSsLabels"),
                     "thalamus": ("thalamus", "ThalamicNuclei"),
                     "hippo-amygdala-left": ("hippo-amygdala", "lh.hippoAmygLabels"),
                     "hippo-amygdala-right": ("hippo-amygdala", "rh.hippoAmygLabels")}
        for family, (directory, stem) in filenames.items():
            folder = official / "subregions" / directory
            soft = (["brainstemSsLabels.volumes.txt"] if family == "brainstem" else
                    ["ThalamicNuclei.volumes.txt"] if family == "thalamus" else
                    [f"{stem[:2]}.hippoSfVolumes.txt",f"{stem[:2]}.amygNucVolumes.txt"])
            subregions[family] = dict(native=str(folder/f"{stem}.FSvoxelSpace.mgz"),
                                     hr=str(folder/f"{stem}.mgz"),
                                     soft_volume_files=[str(folder/f) for f in soft])
        cases.append(dict(id=name, case_root=str(case_root), development_seen=record["development_seen"],
                          raw_t1=dict(path=record["server_path"], bytes=record["size_bytes"],sha256=record["sha256"]),
                          official=dict(status_file=str(official/"official_report.json"),
                                        subjects_dir=str(official/"subjects"), norm=str(mri/"norm.mgz"),
                                        aseg=str(mri/"aseg.mgz"),wmparc=str(mri/"wmparc.mgz"),subregions=subregions)))
    manifest = dict(schema_version=1,planned_subjects=10, source=dict(path=str(source),
                    manifest_sha256=sha(source/"source_manifest.json"),base_commit=current["base_commit"]),
                    data_manifest=dict(path=str(root/"data_manifest.json"),sha256=sha(root/"data_manifest.json")),
                    dataset=data["dataset"],snapshot=data["snapshot"],license=data["license"],dataset_doi=data["dataset_doi"],
                    selection_rule=data["selection_rule"],cases=cases, canonical_label_metadata=labels,
                    label_metadata_provenance=dict(path=str(previous_api),sha256=sha(previous_api),
                      validated_against_current_unchanged_gems_files=len(gems)),
                    fnit_configuration=dict(structures="all",optimization="fast",threads=4,precision="float32 with default TF32 and existing deterministic accumulations"),
                    native_grid_policy=dict(raw="actual raw T1 grid",stage="fresh official norm.mgz grid"),
                    highres_grid_policy="official axis, spacing, integer phase; union FOV, no score-dependent transform",
                    reference_used_for_raw_fnit_fitting=False,
                    official_configuration=dict(reconall="fresh recon-all -all -openmp 4 on CPU",subregion_threads=4),
                    figure_selection_rule="sub-01 continuity example; median and lowest aggregate raw_native Dice over evaluated labels, computed after all ten outcomes are known")
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(manifest=str(path),sha256=sha(path),subjects=len(cases),canonical_labels=len(labels))))


if __name__ == "__main__":
    main()
