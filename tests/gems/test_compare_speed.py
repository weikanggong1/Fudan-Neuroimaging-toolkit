import importlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "validation/subregions"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("compare_speed")
    metadata = {}
    groups = [(module.BRAINSTEM_IDS, "brainstem", None, "brainstem"),
              (range(8100, 8150), "thalamus", None, "thalamus"),
              (range(200, 219), "hippocampus", "left", "hippo-amygdala-left"),
              (range(7001, 7010), "amygdala", "left", "hippo-amygdala-left"),
              (range(10200, 10219), "hippocampus", "right", "hippo-amygdala-right"),
              (range(17001, 17010), "amygdala", "right", "hippo-amygdala-right")]
    for ids, parent, side, source in groups:
        for label in ids:
            metadata[str(label)] = {"id": label, "name": f"label{label}", "parent": parent,
                                    "hemisphere": side, "source": source, "family": source}
    old = np.zeros((5, 5, 5), np.int32)
    old[0, 0, :2] = [173, 174]
    old[0, 1, :2] = [175, 178]
    old[2, 1, 1:3] = [8100, 8101]
    old[4, 3, 3] = 200
    new = old.copy()
    new[2, 1, 1:3] = [8101, 8100]
    new[4, 3, 4] = 200
    paths = []
    for name, data, optimization in (("old", old, "balanced"), ("new", new, "fast")):
        path = tmp_path / name
        (path / "highres").mkdir(parents=True)
        nib.save(nib.Nifti1Image(data, np.eye(4)), path / "subregions_native.nii.gz")
        for structure in module.STRUCTURES:
            spacing = .33333 if structure.startswith("hippo-amygdala") else .5
            affine = np.diag([spacing] * 3 + [1.])
            if name == "new":
                affine[:3, 3] = 2
            shape = (3, 3, 3) if name == "old" else (4, 4, 4)
            nib.save(nib.Nifti1Image(np.zeros(shape, np.int32), affine), path / "highres" / f"{structure}.nii.gz")
        comparisons = {source: {"reference": "reference/" + source, "regions": [], "empty_hard_regions": []}
                       for source in module.STRUCTURES}
        for identifier, value in metadata.items():
            label = int(identifier)
            reference = int(np.count_nonzero(old == label)); got = int(np.count_nonzero(data == label))
            if reference or got:
                comparisons[value["source"]]["regions"].append({
                    **value, "label": label, "reference_voxels": reference, "fnit_voxels": got,
                    "dice": 2 * np.count_nonzero((old == label) & (data == label)) / (reference + got),
                    "accepted": reference == got, "hard_volume_difference": abs(got - reference) / reference,
                    "reference_soft_volume_mm3": reference, "fnit_soft_volume_mm3": got})
        report = {"validation_mode": "official_stage_inputs", "optimization": optimization,
                  "input": "/same/norm.mgz", "aseg": None, "wmparc": None,
                  "input_sha256": {"/same/norm.mgz": "abc"}, "label_metadata": metadata,
                  "volumes": {str(label): {"soft_volume_mm3": float(np.count_nonzero(data == int(label)))}
                              for label in metadata}, "comparisons": comparisons,
                  "families": {family: {"foreground_dice": .97, "accepted": 1} for family in module.FAMILIES},
                  "wall_seconds": 12 if name == "old" else 10, "peak_gpu_gib": 2.,
                  "torch_version": "fixture", "cuda_version": None,
                  "source_sha256": {"gems/core.py": "xyz"}, "output": "old/recorded/path.nii.gz"}
        (path / "report.json").write_text(json.dumps(report))
        paths.append(path)
    return module, paths


def test_controlled_structure_changes_and_different_highres_crop(comparison, tmp_path):
    module, (old, new) = comparison
    result = module.compare_runs(old, new)
    assert result["checks_passed"]
    foreground = result["old_new"]["foreground"]
    assert foreground["foreground_union_voxels"] == 8
    assert foreground["label_different_voxels_in_union"] == 3
    assert foreground["label_difference_percent_union"] == 37.5
    assert foreground["foreground_boundary_changed_voxels"] == 1
    assert foreground["foreground_internal_label_switched_voxels"] == 2
    thal = result["old_new"]["families"]["thalamus"]
    assert thal["old_new_foreground_dice"] == 1 and thal["label_difference_percent_union"] == 100
    hippo = result["old_new"]["families"]["hippocampus-left"]
    assert hippo["old_new_foreground_dice"] == pytest.approx(2 / 3)
    assert hippo["hard_volume_mm3"]["signed_change_percent"] == 100
    assert hippo["soft_volume_mm3"]["signed_change_percent"] == 100
    assert hippo["centroid_shift_mm"] == .5
    assert len(result["old_new"]["native"]["per_label"]) == 110
    assert len(result["official"]["inferred_empty_hard_labels"]["old"]) == 103
    assert not result["old_new"]["highres"]["thalamus"]["geometry"]["same_geometry"]
    tsv = tmp_path / "summary.tsv"
    module.write_tsv(result, tsv)
    assert len(tsv.read_text().splitlines()) == 7
    assert "official_new_foreground_dice" in tsv.read_text().splitlines()[0]
    with pytest.raises(FileExistsError):
        module.write_tsv(result, tsv)


def test_source_hash_and_highres_resolution_checks(comparison, tmp_path):
    module, (old, new) = comparison
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_commit": "claimed_commit", "files": [
        {"path": "/frozen/src/fnit/gems/core.py", "sha256": "xyz"},
        {"path": "/frozen/src/fnit/cli.py", "sha256": "unreported"}]}))
    result = module.compare_runs(old, new, new_source_manifest=manifest)
    assert result["checks_passed"]
    assert result["checks"]["new_manifest_source_hashes_match"]
    source = result["source_provenance"]["new"]
    assert source["complete_reported_source_coverage"]
    assert source["verified_file_count"] == 1
    assert source["not_reported_files"] == ["cli.py"] and source["not_reported_file_count"] == 1
    assert source["reported_files_without_manifest"] == []
    partial = module.source_provenance({"source_sha256": {"gems/core.py": "xyz", "gems/base.py": "other"}}, manifest)
    assert partial["listed_source_hashes_match"] and not partial["complete_reported_source_coverage"]
    assert partial["reported_files_without_manifest"] == ["gems/base.py"]
    manifest.write_text(manifest.read_text().replace('"xyz"', '"changed"'))
    result = module.compare_runs(old, new, new_source_manifest=manifest)
    assert not result["checks_passed"]
    assert result["source_provenance"]["new"]["mismatched_files"] == ["gems/core.py"]
    manifest.write_text(json.dumps({"source_commit": "claimed_commit", "files": [
        {"path": "/frozen/src/fnit/cli.py", "sha256": "unreported"}]}))
    result = module.compare_runs(old, new, new_source_manifest=manifest)
    source = result["source_provenance"]["new"]
    assert not result["checks"]["new_manifest_source_hashes_match"]
    assert source["verified_file_count"] == 0 and source["mismatched_files"] == []
    assert not source["complete_reported_source_coverage"]
    assert source["reported_files_without_manifest"] == ["gems/core.py"]
    path = new / "highres/thalamus.nii.gz"
    image = nib.load(path)
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), np.eye(4)), path)
    result = module.compare_runs(old, new)
    assert not result["checks"]["thalamus_highres_spacing_correct"]
