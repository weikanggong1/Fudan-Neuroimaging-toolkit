"""Check complete saved temporal, slice-timing and sampling-reference outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fmri"))
from compare_matched_pipeline import correlations


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    reports, file_hashes, results = {}, {}, {}

    def recorded(folder, name):
        path = args.run_root / folder / name
        value = json.loads((path / "report.safe.json").read_text())
        if value["status"] != "complete" or not value["source_unchanged"]:
            raise ValueError("A complete, unchanged full-input invocation is required")
        for relative, expected in value["source_sha256"].items():
            source = args.source / relative
            if source not in file_hashes:
                file_hashes[source] = sha(source)
            if file_hashes[source] != expected:
                raise ValueError("The recorded frozen source changed")
        reports[folder + "/" + name] = {
            "function": value["function"], "backend": value["backend"],
            "source_revision": value["source_revision"],
            "threads": value["cpu_threads"], "affinity": value["cpu_affinity"],
            "adapter_sha256": value["adapter_sha256"],
            "input_sha256": value["input_sha256"],
            "calls": [{key: row[key] for key in (
                "repeat", "call_type", "api_wall_seconds_including_io",
                "official_wall_seconds_including_io") if key in row}
                for row in value["records"]]}
        return path / "repeat_0", value

    def compare(candidate, reference, frames=None):
        ci, ri = nib.load(candidate), nib.load(reference)
        if ci.shape != ri.shape or (frames is not None and ci.shape[-1] != frames):
            raise ValueError("Complete output dimensions differ")
        affine_error = float(np.max(np.abs(ci.affine - ri.affine)))
        if affine_error > 1e-4:
            raise ValueError("The full output grids differ")
        ca, ra = np.asarray(ci.dataobj, np.float32), np.asarray(ri.dataobj, np.float32)
        if not np.isfinite(ca).all() or not np.isfinite(ra).all():
            raise ValueError("All output values must be finite")
        return {"shape": list(ci.shape), "same_grid": True,
                "maximum_affine_difference": affine_error,
                "storage_dtypes": [str(ci.get_data_dtype()), str(ri.get_data_dtype())],
                "full_grid": correlations(ca, ra, temporal=frames is not None),
                "output_sha256": {"fnit": sha(candidate), "official": sha(reference)}}

    for threads in (1, 8):
        for prefix in ("temporal", "temporal_remove_mean"):
            candidate, cv = recorded("baseline_v3", f"{prefix}_fnit_t{threads}")
            reference, rv = recorded("reference_fixed_v2", f"{prefix}_official_t{threads}")
            if cv["input_sha256"] != rv["input_sha256"] or cv["cpu_affinity"] != rv["cpu_affinity"]:
                raise ValueError("Original and FNIT input bytes or CPU allocations differ")
            for filename in ("scaled.nii.gz", "highpass.nii.gz"):
                results[f"{prefix}_t{threads}_{filename}"] = compare(
                    candidate / filename, reference / filename, 490)
        candidate, cv = recorded("baseline_v3", f"sampling_reference_fnit_t{threads}")
        reference, rv = recorded("reference_fixed_v2", f"sampling_reference_official_t{threads}")
        results[f"sampling_reference_t{threads}"] = compare(
            candidate / "sampling_reference.nii.gz", reference / "sampling_reference.nii.gz")
    for prefix, threads in (("stc", 1), ("stc", 8), ("stc180_ignore4_start", 8)):
        candidate, cv = recorded("baseline_v3", f"{prefix}_fnit_t{threads}")
        reference, rv = recorded("reference_container_v3", f"{prefix}_official_t{threads}")
        if cv["input_sha256"] != rv["input_sha256"] or cv["cpu_affinity"] != rv["cpu_affinity"]:
            raise ValueError("Original and FNIT complete slice-timing inputs differ")
        value = compare(candidate / "stc.nii.gz", reference / "stc.nii.gz", 180)
        if prefix == "stc180_ignore4_start":
            raw = np.asarray(nib.load(inputs["public180"]["bold"]).dataobj)[..., :4]
            corrected = np.asarray(nib.load(candidate / "stc.nii.gz").dataobj)[..., :4]
            value["ignored_frames_unchanged"] = bool(np.array_equal(raw, corrected))
            if not value["ignored_frames_unchanged"]:
                raise ValueError("Ignored frames changed")
        results[f"{prefix}_t{threads}"] = value
    public = {"schema_version": 1, "full_comparisons": len(results),
              "driver_sha256": sha(__file__), "results": results,
              "completed_invocations": reports, "default_slice_timing": False,
              "scope": "Saved complete 490-frame temporal and 180-frame optional STC; full 3D sampling grids",
              "timing_scope": "Recorded original full API clocks; current comparisons are outside timing",
              "source_scope": "Frozen baseline revision in each invocation; not latest integrated pipeline",
              "private_images_exported": False}
    if args.output.exists():
        raise FileExistsError("Preserve previous analysis results")
    args.output.write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete_comparisons": len(results), "private_images_exported": False}))


if __name__ == "__main__":
    main()
