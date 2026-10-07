"""Compare full, completed AFNI-compatible and unchanged-default real outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from compare_completed import images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())["datasets"]["real_run_01"]
    source = nib.load(config["confounds_bold"])
    mask_image = nib.load(config["brain_mask"])
    if source.shape[-1] != 490 or mask_image.shape != source.shape[:3]:
        raise ValueError("Expected the complete recorded 490-frame input grid")
    if not np.allclose(mask_image.affine, source.affine, atol=1e-4, rtol=0):
        raise ValueError("Brain mask grid differs")
    mask = np.asarray(mask_image.dataobj) > 0
    records = {}
    cases = ("drift", "motion6", "motion12", "motion24", "tissue", "global", "all", "all_bandpass")
    for threads in (1, 8):
        for case in cases:
            candidate = args.run_dir / f"confounds_{case}_t{threads}_afni_compat"
            original = args.run_dir / f"confounds_{case}_t{threads}_official"
            reports = [json.loads((path / "report.public.json").read_text())
                       for path in (candidate, original)]
            if any(not report.get("executed") for report in reports):
                raise ValueError("A requested complete API has not executed")
            if reports[0]["threads"] != threads or reports[1]["threads"] != threads:
                raise ValueError("Native and FNIT thread budgets differ")
            if reports[0]["host"] != reports[1]["host"]:
                raise ValueError("Native and FNIT must run on the same host")
            if reports[0]["input_sha256"] != reports[1]["input_sha256"]:
                raise ValueError("Native and FNIT input provenance differs")
            records[f"{case}_t{threads}"] = {
                "comparison": images(candidate / "cleaned.nii.gz", original / "cleaned.nii.gz", mask),
                "api_wall_seconds": {"fnit": reports[0]["api_wall_seconds"],
                                     "original": reports[1]["api_wall_seconds"]},
                "threads": threads, "source_sha256": reports[0]["source_sha256"],
                "native_program_sha256": reports[1]["result"]["native_program_sha256"],
            }
        for case in ("all", "all_bandpass"):
            candidate = args.run_dir / f"confounds_{case}_t{threads}_strict/cleaned.nii.gz"
            baseline = args.baseline_dir / f"confounds_{case}_t{threads}_fnit/cleaned.nii.gz"
            records[f"default_old_new_{case}_t{threads}"] = images(candidate, baseline, mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": 1, "dataset_alias": "real_run_01", "frames": 490,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": records, "timing_status": "one paired complete call per case and thread budget",
    }, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete_comparisons": len(records), "private_images_exported": False}))


if __name__ == "__main__":
    main()
