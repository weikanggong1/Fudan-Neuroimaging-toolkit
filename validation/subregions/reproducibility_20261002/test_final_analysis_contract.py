"""CPU contract fixtures for final analysis gates; not a real-data benchmark."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile

from analyze_final_full_pipeline import audit_snapshot, full_successes, separate_noise_and_accuracy


def main():
    valid = {"runs": [{"mode":mode,"repeat":repeat,"structure":"all","phase":"verified_full_run","exit_code":0}
                      for mode in ("stage","raw") for repeat in (1,2,3)]}
    assert len(full_successes(valid)) == 6
    partial = copy.deepcopy(valid)
    partial["runs"][-1]["structure"] = "thalamus"
    try:
        full_successes(partial)
    except ValueError:
        pass
    else:
        raise AssertionError("Partial/full scope mixture must fail")
    duplicate = copy.deepcopy(valid)
    duplicate["runs"].append(copy.deepcopy(valid["runs"][0]))
    try:
        full_successes(duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate successful repeat must fail")
    region = {"fnit_repeat_dice":{"min":1.},"official_repeat_dice":{"min":1.},
              "fnit_repeat_different_voxels":{"max":0},"official_repeat_different_voxels":{"max":0},
              "fnit_voxels":[9,9,9],"official_voxels":[10,10,10],"cross_method_dice":{"min":.7},
              "hard_geometry_status":"outside", "cross_not_below_official":False}
    result = separate_noise_and_accuracy({"regions":[region]})["regions"][0]
    assert result["own_repeat_status"] == "within_official_observed_repeat_noise"
    assert result["cross_accuracy_status"] == "measured_cross_method_bias" and result["accuracy_acceptance"] is None
    assert "cross_not_below_official" not in result and "hard_geometry_status" not in result
    empty = copy.deepcopy(region)
    empty.update(fnit_voxels=[0]*3,official_voxels=[0]*3,fnit_repeat_dice={"min":None},official_repeat_dice={"min":None},cross_method_dice={"min":None})
    empty = separate_noise_and_accuracy({"regions":[empty]})["regions"][0]
    assert empty["own_repeat_status"] == "stable_absent_hard_label"
    assert empty["cross_accuracy_status"] == "both_empty_hard_label"
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory)/"snapshot"
        root = source/"src/fnit"
        root.mkdir(parents=True)
        entries = []
        # Arbitrary count proves there is no hard-coded 415/424/431-module gate.
        for name in ("__init__.py","a.py","b.py"):
            path=root/name;path.write_text("# fixture\n")
            entries.append({"path":"src/fnit/"+name,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest=source/"source_manifest.json";manifest.write_text(json.dumps({"files":entries}))
        reviewed=Path(directory)/"reviewed.json";reviewed.write_bytes(manifest.read_bytes())
        runtime, audit=audit_snapshot(source,reviewed)
        assert len(runtime)==3 and audit["verified_runtime_python_files"]==3
        (root/"unlisted.py").write_text("# must fail\n")
        try:
            audit_snapshot(source,reviewed)
        except ValueError:
            pass
        else:
            raise AssertionError("Unlisted runtime module must fail")
    result={"scope":"CPU contract fixtures only; not fitting or a real-data benchmark","passed":6,
            "checks":["six full scopes","reject mixed scopes","reject duplicate success","separate own noise from cross bias","empty Dice remains NA","dynamic module count and reject unlisted module"],
            "analysis_sha256":hashlib.sha256(Path(__file__).with_name("analyze_final_full_pipeline.py").read_bytes()).hexdigest()}
    output=Path(__file__).parent/"release_tests/final_analysis_contract.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":main()
