"""Prepare exactly two default CPU memory API jobs from verified saved inputs.

Only writes a new private plan/config directory. Does not import FNIT, PyTorch,
read voxel arrays, run networks or alter the frozen source/environment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


IDENTITY = {"source_label": "task5_candidate_cpu_v5",
            "head_commit": "00fedf3544291b9c3e1db9b6c2d6fcef3917d958",
            "archive_sha256": "ab9a2d587d0fb9599fc2b6200011907b86e12b60f2630db131b62641ee755a72"}
SOURCE_SHA = {
    "src/fnit/_nib.py": "8fe4e897c96931e74ccd8288a4c4c67ebc7fac1b58babcd63b1acb50603326f6",
    "src/fnit/synthstrip/pipeline.py": "af4978d445f3438a82c06de2548918b0dcc049e746eba8190fc50b889f0a5a48",
    "src/fnit/synthstrip/model.py": "6afcff10848a8a2a57b9c00d4b637eda90e4c0e21f301d8a8031107d783a5f44",
    "src/fnit/synthstrip/geometry.py": "c31febaf93404d1b4b7e39f36bfbacd7adb659ddcf923ae6a0924bc4e0a00dc1",
    "src/fnit/synthsr/pipeline.py": "c896d8e2e4a1a214546e6674ce3aacdcf84906d1b2010b5397ec1b076f502943",
    "src/fnit/synthsr/model.py": "290d0ae911d8c99f4f87c22936ca497ea45bb57b8f50066b822d9cd10c4661b2",
    "src/fnit/synthsr/spatial.py": "962a2f0bd5b97db1f039efb67fa87d2f61aa5d8fc23512689f255b168b42dabe"}
RESOURCE = {"input": {"bytes": 20077109, "sha256": "afd1a20fe75fdea44313f0eda05020b916c87234e7a2045f7ccc6bb7c6e90b19"},
            "synthstrip": {"bytes": 30851709, "sha256": "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"},
            "synthsr": {"bytes": 106163752, "sha256": "a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b"}}
REFERENCES = {
    "synthstrip": {
        "official": {
            "image": ("official_strip_profile_case01_v2/image.nii.gz", "3ae8a207799bb1f45a19cc9ae20bec0ba5ec41da5ec5c8161d37a182d7172d9f"),
            "mask": ("official_strip_profile_case01_v2/mask.nii.gz", "b35980a9aba4f8a85da8557bea7221c504a4a5ac98587f5134f6980f9329af6d"),
            "distance": ("official_strip_profile_case01_v2/distance.nii.gz", "d50931140ceb91299e6c7ffbceca5309ccb5c7f6e03b386af5916c0d8ead7ae4")},
        "saved_fnit_path": {
            "image": ("production_strip_cli_v3/case01_synthstrip_default_2_baseline/artifacts/image.nii.gz", "fd8c3253a6960777715c85844a38e760959b6bf80849bd28b7e5b906dee0dea4"),
            "mask": ("production_strip_cli_v3/case01_synthstrip_default_2_baseline/artifacts/mask.nii.gz", "f6c99210d9617b2450c9d90da2f34a85d9b885adf911a57f03a7e402d3f27ad3"),
            "distance": ("production_strip_cli_v3/case01_synthstrip_default_2_baseline/artifacts/distance.nii.gz", "639b2f0e6b6bf6d2656440db6f3a5f94b8f0904bc4e91644c84d6ba6b949dde5")}},
    "synthsr": {
        "official": {"image": ("official_sr_profile_case01_v1/image.nii.gz", "dff4ef9d658243a500704c141290c636414e7e871e3d6fccf5b2f211a7328922")},
        "saved_fnit_path": {"image": ("paired_baseline/case01_synthsr_default_2_baseline/artifacts/image.nii.gz", "fbccec5cad5503d3c1f2589d79888571bf15e19607ced94502ae124624d821f6")},
        "official_float": {"float_image": ("official_sr_profile_case01_v1/image.npz", "c0c2488455db3005c0c54f56994e90e76657db83c263cd06b2ad3431f5e24199")},
        "saved_fnit_float": {"float_image": ("sr_network_control_v2/baseline/artifacts/image.npz", "f937f1370ab9639bc0fa286e981dfee9a53d23ba54e5ea4468e53b6dc9bbecb2")}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous job plans")
    base = args.server_root
    workspace = base / "workspaces/smri_cpu_20261004"
    code = workspace / "task1_memory_v1"
    run = base / "runs/smri_cpu_20261004/task1_memory_v1"
    source = workspace / "task5_candidate_cpu_v5"
    common = {"source_root": str(source), "source_identity": IDENTITY,
              "source_files_sha256": SOURCE_SHA,
              "source_manifest_sha256": "42ae32baba858d64f44ea095bc163d92d8ff142f5a7e78081c192356af99e8ef",
              "input": str(base / "runs/smri_cpu_20261004/inputs/ds003138/case01_T1w.nii.gz"),
              "input_resource": RESOURCE["input"],
              "comparator": str(code / "compare_outputs.py"),
              "comparator_sha256": "51239330ea955c3b26c99246cfba65b59f1d608e6ffadf36f3ae4d13e385148f",
              "canonical_index": str(base / "INDEX.json")}
    jobs = []
    for feature, filename in (("synthstrip", "synthstrip.1.pt"), ("synthsr", "synthsr_v20_230130.h5")):
        config = {**common, "feature": feature,
                  "weight": str(workspace / "assets/weights" / filename),
                  "weight_resource": RESOURCE[feature],
                  "loader_policy": (
                      "path and SpatialImage: load_image then asanyarray(dataobj); float32 conform/network after decoding; no proxy float32 request"
                      if feature == "synthstrip" else
                      "path: get_fdata default float64; SpatialImage: asanyarray(dataobj), then float64; final network tensor float32"),
                  "references": {arm: {name: {"path": str(base / "runs/smri_cpu_20261004/task1" / relative), "sha256": digest}
                                       for name, (relative, digest) in members.items()}
                                 for arm, members in REFERENCES[feature].items()}}
        path = code / (feature + ".config.private.json")
        if path.exists():
            raise FileExistsError("preserve config: " + path.name)
        path.write_text(json.dumps(config, indent=2) + "\n")
        result = run / feature
        jobs.append({"id": feature + "_memory_default_cpu8",
                     "argv": [str(base / "envs/default/bin/python"), str(code / "worker.py"),
                              "--config", str(path), "--output", str(result)],
                     "env": {"PYTHONPATH": str(source / "src")},
                     "timeout_seconds": 600,
                     "expected_outputs": [str(result / "report.private.json"), str(result / "image.nii.gz")]
                                          + ([str(result / "mask.nii.gz"), str(result / "distance.nii.gz")]
                                             if feature == "synthstrip" else [str(result / "image.npz")])})
    plan = {"schema": "fnit.smri.memory_api.jobs.v1", "source_identity": IDENTITY,
            "scope": "exactly one default case01 CPU API per feature; saved references only",
            "threads": 8, "cpu_list": "0,4,8,12,16,20,24,28", "jobs": jobs}
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
