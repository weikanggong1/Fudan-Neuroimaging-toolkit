"""Compare completed full PICA threshold runs and real BIDS output contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

from compare_completed import pica


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requests = json.loads(args.requests.read_text())
    execution = json.loads((args.run_dir / "execution.public.json").read_text())
    names = [request["name"] for request in requests["requests"]]
    rows = execution["records"]
    if len(rows) != 5 or [row["name"] for row in rows] != names or any(
            row["returncode"] != 0 for row in rows):
        raise ValueError("All five full-input requests must complete successfully")
    for filename, expected in requests["frozen_sha256"].items():
        if digest(filename) != expected:
            raise ValueError("The frozen feature source or driver changed")
    reports, configs = {}, {}
    for request in requests["requests"]:
        argv = request["arguments"]
        manifest = json.loads(Path(argv[argv.index("--manifest") + 1]).read_text())
        config = manifest["datasets"][argv[argv.index("--dataset") + 1]]
        name = request["name"]
        report = json.loads((args.run_dir / name / "report.private.json").read_text())
        if not report["executed"] or report["device"] != "cpu" or (
                report["threads"] != request["threads"] or
                report["cpu_affinity"] != sorted(request["cpus"])):
            raise ValueError("Actual complete invocation or CPU budget differs")
        if report["shape"][-1] != config["n_frames"]:
            raise ValueError("The run did not retain every acquired frame")
        source = Path(request["source_path"]) / "fnit"
        for relative, recorded in report["source_sha256"].items():
            if digest(source / relative) != recorded:
                raise ValueError("Executed module differs from the current frozen tree")
        for key, recorded in report["input_sha256"].items():
            if digest(config[key]) != recorded:
                raise ValueError("An actual input changed after execution")
        reports[name], configs[name] = report, config
    comparisons = {}
    for threads in (1, 8):
        stem = f"pica_fixed95_mm75_t{threads}"
        candidate, reference = stem + "_fnit", stem + "_official"
        if reports[candidate]["input_sha256"] != reports[reference]["input_sha256"]:
            raise ValueError("Original and candidate inputs differ")
        mask = np.asarray(nib.load(configs[candidate]["brain_mask"]).dataobj) > 0
        comparisons[stem] = pica(args.run_dir / candidate / "ica",
                                 args.run_dir / reference / "melodic.ica", mask)
    name = "bids_public180_t8_fnit"
    report, config = reports[name], configs[name]
    result = report["result"]
    source_image = nib.load(config["bids_bold"])
    checked = {}
    for field in ("components", "posterior", "thresholded"):
        image = nib.load(result[field])
        values = np.asarray(image.dataobj)
        if image.shape != source_image.shape[:3] + (20,) or (
                not np.array_equal(image.affine, source_image.affine) or
                image.get_data_dtype() != np.dtype("float32") or
                not np.isfinite(values).all()):
            raise ValueError("A complete BIDS component output is invalid")
        checked[field] = {"shape": list(image.shape), "dtype": "float32",
                          "finite": True, "same_spatial_grid": True}
    mixing = np.loadtxt(result["mixing"], skiprows=1, ndmin=2)
    if mixing.shape != (180, 20) or not np.isfinite(mixing).all():
        raise ValueError("The full BIDS mixing table must retain 180 rows")
    metadata = json.loads(Path(result["metadata"]).read_text())
    if metadata["NumberOfComponents"] != 20 or len(metadata["Sources"]) != 2 or (
            any(not value.startswith("bids:preproc:") for value in metadata["Sources"])):
        raise ValueError("BIDS component metadata has incorrect source links")
    output_root = args.run_dir / name / "derivatives"
    files = sorted(path for path in output_root.rglob("*") if path.is_file())
    before = {path: digest(path) for path in files}
    sys.path.insert(0, str(Path(requests["requests"][-1]["source_path"])))
    from fnit.melodic import run_melodic_bids
    try:
        run_melodic_bids(config["source_derivatives_root"], output_root,
                         input_bold=config["bids_bold"], brain_mask=config["bids_brain_mask"],
                         n_components=20, device="cpu", overwrite=False)
    except FileExistsError:
        pass
    else:
        raise ValueError("Existing real BIDS outputs were not protected")
    if before != {path: digest(path) for path in files}:
        raise ValueError("The overwrite guard changed existing output bytes")
    public = {"schema_version": 1, "driver_sha256": digest(__file__),
              "requests_sha256": digest(args.requests), "full_requests": 5,
              "pica_frames": 490, "mm_threshold": 0.75,
              "results": comparisons, "execution": rows,
              "bids_public180": {"frames": 180, "components": 20,
                                 "outputs": checked, "mixing_shape": list(mixing.shape),
                                 "source_uris_verified": True,
                                 "existing_outputs_protected_without_recomputation": True,
                                 "existing_bytes_unchanged": True},
              "timing_scope": "Completed full API calls; overwrite guard is excluded",
              "private_images_exported": False}
    args.output.write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete_pica_comparisons": 2, "full_bids_contract": True}))


if __name__ == "__main__":
    main()
