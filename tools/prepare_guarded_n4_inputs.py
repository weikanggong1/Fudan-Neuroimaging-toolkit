"""只读核对两例自产orig及旧完整Torch N4回执，不执行N4或GPU计算。"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(candidate, reference):
    left, right = nib.load(str(candidate)), nib.load(str(reference))
    a, b = np.asarray(left.dataobj), np.asarray(right.dataobj)
    return {"same_shape": a.shape == b.shape,
            "same_affine": bool(np.array_equal(left.affine, right.affine)),
            "same_dtype": str(a.dtype) == str(b.dtype),
            "different_voxels": int(np.count_nonzero(a != b)) if a.shape == b.shape else None,
            "shape": list(a.shape), "dtype": str(a.dtype),
            "affine": left.affine.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.fnit_root
    raw_root = root / "runs/input_n4_raw_t1_a100_two_cases_20261009_v1"
    worker_root = root / "runs/n4_cached_exec_same_orig_a100_two_cases_20261009_v2"
    raw = json.loads((raw_root / "summary.json").read_text())
    old = json.loads((worker_root / "summary.json").read_text())
    assert raw["status"] == "complete_raw_input_to_nu_pair"
    assert old["status"] == "complete_cached_exec_two_modes"
    report = {"status": "input_receipts_verified_waiting_wheel_and_start",
              "source_commit": "9292a0a7be63f6bc96f2dfa358261cb1587e0f23",
              "scope": "two self-produced orig; no downloads, no upstream image input, no MRI execution",
              "raw_receipt_sha256": sha(raw_root / "summary.json"),
              "old_worker_receipt_sha256": sha(worker_root / "summary.json"), "cases": []}
    for raw_case in raw["rows"]:
        case = raw_case["case"]
        subject = "sub" + case.rsplit("-", 1)[1]
        input_path = raw_root / case / "native/subject/mri/orig.mgz"
        torch_input = raw_root / case / "torch/subject/mri/orig.mgz"
        direct_output = raw_root / case / "torch/subject/mri/tmp/nu0.mgz"
        prefix_root = root / "runs/recon_torch_n4_prefix_20261010_v1" / subject
        complete = json.loads((prefix_root / "COMPLETE.json").read_text())
        assert complete["status"] == "complete_prefixes"
        assert all(run["exit_code"] == 0 for run in complete["runs"])
        prefix_input = prefix_root / "1-torch-isolated/subject/mri/orig.mgz"
        same = compare(input_path, prefix_input)
        assert all(same[key] for key in ("same_shape", "same_affine", "same_dtype"))
        assert same["different_voxels"] == 0
        input_sha = sha(input_path)
        torch_run = next(run for run in raw_case["runs"] if run["backend"] == "torch")
        native_run = next(run for run in raw_case["runs"] if run["backend"] == "native")
        assert native_run["output_sha256"]["mri/orig.mgz"] == input_sha
        assert torch_run["output_sha256"]["mri/orig.mgz"] == sha(torch_input)
        torch_same = compare(input_path, torch_input)
        assert all(torch_same[key] for key in ("same_shape", "same_affine", "same_dtype"))
        assert torch_same["different_voxels"] == 0
        assert torch_run["output_sha256"]["mri/tmp/nu0.mgz"] == sha(direct_output)
        prior_rows = [row for row in old["rows"] if row["case"] == case]
        assert len(prior_rows) == 2
        references = []
        for row in prior_rows:
            receipt_path = worker_root / case / row["mode"] / "child.json"
            receipt = json.loads(receipt_path.read_text())
            output_path = worker_root / case / row["mode"] / "nu0.mgz"
            assert row["input_sha256"] == receipt["input_sha256"] == input_sha
            assert row["worker"]["output_sha256"] == receipt["output_sha256"] == sha(output_path)
            assert receipt["api"]["iterations"] == 200
            assert row["to_cache_disabled_complete_n4"]["whole"]["different_voxels"] == 0
            references.append({"mode": row["mode"], "output": str(output_path),
                               "output_sha256": sha(output_path), "receipt_sha256": sha(receipt_path),
                               "iterations": 200, "source_sha256": receipt["source_sha256"],
                               "historical_to_native": row["to_native"],
                               "historical_wall_seconds": row["wall_seconds_including_exec_load_transfer_io"]})
        report["cases"].append({"case": case, "public_source_url": raw_case["public_source_url"],
                                "raw_input_sha256": raw_case["sha256"],
                                "input": str(input_path), "input_sha256": input_sha,
                                "input_origin": "FNIT conform output in old native branch; not upstream official image",
                                "torch_orig_sha256": sha(torch_input), "torch_orig_comparison": torch_same,
                                "geometry": same, "prefix_orig_sha256": sha(prefix_input),
                                "prefix_complete_sha256": sha(prefix_root / "COMPLETE.json"),
                                "old_direct_output": str(direct_output),
                                "old_direct_output_sha256": sha(direct_output), "old_workers": references})
    assert len(report["cases"]) == 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": [
        {"case": row["case"], "input_sha256": row["input_sha256"],
         "prefix_orig_identical": row["geometry"]["different_voxels"] == 0} for row in report["cases"]]}))


if __name__ == "__main__":
    main()
