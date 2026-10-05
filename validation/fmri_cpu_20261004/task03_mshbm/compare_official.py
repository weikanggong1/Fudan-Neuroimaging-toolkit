"""Compare completed original CBIG labels with full FNIT output labels.

Label identities 1--17 remain fixed: no permutation or cropped vertex subset
is used to improve the reported precision. Paths and individual maps remain
private; the output contains aggregate counts, Dice and source receipts.
"""

import argparse
import hashlib
import json
from pathlib import Path


CBIG_COMMIT = "b69b822a15e2a94f1e439606552fc44b6858cf3c"
CORE_SHA256 = {
    "fnit": "8cd2f6fb706a13c60be5070c575cdb77898e2d309cf3f958c4fdbfee355a27ea",
    "fnit_candidate": "eabb4d62c810fc180d71f41c0e783bcf2dfd9c1bb6ae42fcc699e44859a9aa9e",
}
COMMON_MODULE_SHA256 = {
    "__init__.py": "9c92d019e9522964b1bfd63aed0392dc55c8f79e8ae39392f710d48a3d6d5e04",
    "cli.py": "8c660616381340fcdf23b5dd74a57f2bec470debf11fcf2ce81e105cd858762e",
    "volume.py": "ac7c8de095bbe15f320150efeccb3e160df7ed8e436d6fdd47579037c98062fe",
    "output.py": "e006bd3d0e42a2cfabeeb5646dba970d46d4eeb3730618e7353e6a7985493feb",
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checked_labels(values, expected, np):
    values = np.asarray(values).ravel()
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError("Every original fsLR32k vertex label must be present and finite")
    if not np.isin(values, np.arange(18)).all():
        raise ValueError("Original network identities must remain 0--17")
    return values.astype(np.uint8)


def completed_job(queue, name):
    matches = [job for job in queue["jobs"] if job["name"] == name]
    if len(matches) != 1:
        raise ValueError("Exactly one recorded job must identify each comparison")
    job = matches[0]
    if job.get("state") != "complete" or job.get("returncode") != 0:
        raise ValueError("Only successfully completed original/FNIT jobs may be compared")
    return job


def check_official_binding(directory, queue, reader, official_report):
    """Bind the actual MATLAB output, input and resolved source to this queue."""
    from export_reference import verify_export

    binding = json.loads((directory.parent / "matlab_binding.private.json").read_text())
    timeseries = binding.get("timeseries")
    if (not isinstance(timeseries, list) or len(timeseries) != 1
            or sha256(timeseries[0]) != queue["input_sha256"]):
        raise ValueError("The actual original binding must use the gated complete input")
    if binding.get("w") != official_report["w"] or binding.get("c") != official_report["c"]:
        raise ValueError("The actual original binding and report parameters differ")
    clean = Path(binding["cbig_clean_dir"]).resolve()
    manifest = verify_export(clean)
    expected = {path: item["clean_sha256"] for path, item in manifest["files"].items()}
    if manifest["commit"] != queue["reference_commit"]:
        raise ValueError("The actual original binding must use the fixed pristine export")
    resolved = json.loads((directory / "resolved_functions.private.json").read_text())
    required = {"CBIG_MSHBM_generate_individual_parcellation", "CBIG_MSHBM_read_fmri",
                "CBIG_MSHBM_parcellation_single_subject", "CBIG_ComputeCorrelationProfile",
                "CBIG_corr", "ft_read_cifti", "ft_write_cifti", "gifti", "xmltree"}
    if not required.issubset(resolved):
        raise ValueError("Actual original function resolution receipts are incomplete")
    actual_hashes = {}
    for name in required:
        path = Path(resolved[name]).resolve()
        try:
            relative = path.relative_to(clean).as_posix()
        except ValueError as error:
            raise ValueError("An actual original function resolved outside the pristine export") from error
        digest = sha256(path)
        if expected.get(relative) != digest:
            raise ValueError("An actual original function differs from the pristine export")
        actual_hashes[name] = digest
    censor_files = binding.get("censor")
    censor_sha = queue.get("censor_sha256")
    if censor_sha is not None:
        if (not isinstance(censor_files, list) or len(censor_files) != 1
                or sha256(censor_files[0]) != censor_sha):
            raise ValueError("The actual original binding must use the exact optional censor")
    elif censor_files:
        raise ValueError("The primary original binding must remain uncensored")
    return binding, actual_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-results", type=Path, required=True)
    parser.add_argument("--fnit-results", type=Path, nargs="+", required=True)
    parser.add_argument("--queue-status", type=Path, required=True,
                        help="Coordinator receipt for the actual completed paired queue")
    parser.add_argument("--reader-values", type=Path, required=True,
                        help="Independent original-reader full-value gate for this exact input")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep previous official precision receipts")
    import numpy as np
    from scipy.io import loadmat

    queue = json.loads(args.queue_status.read_text())
    reader = json.loads(args.reader_values.read_text())
    queue_root = args.queue_status.resolve().parent
    if args.official_results.resolve().parent != queue_root:
        raise ValueError("Actual original outputs must belong to this paired queue directory")
    if queue.get("state") != "complete" or queue.get("reference_commit") != CBIG_COMMIT:
        raise ValueError("The original paired queue must be complete at the fixed CBIG commit")
    if (reader.get("compared_values") != 29111880
            or reader.get("different_values") != 0
            or reader.get("input_sha256") != queue.get("input_sha256")
            or not reader.get("cortical_vertex_identity_exact")):
        raise ValueError("The actual original reader must match every cortical value and vertex")
    official_job = completed_job(queue, args.official_results.name)
    if not official_job["name"].startswith("cbig_"):
        raise ValueError("The reference job must be the original CBIG implementation")
    official_report = json.loads((args.official_results / "report.public.json").read_text())
    if official_report.get("vertex_count") != 64984:
        raise ValueError("The actual completed original report must contain all vertices")
    if official_report.get("mode") != "full":
        raise ValueError("This comparison requires the completed original full CIFTI chain")
    if official_report.get("threads") != official_job["threads"]:
        raise ValueError("Actual MATLAB thread count differs from its paired queue receipt")
    native_binding, resolved_hashes = check_official_binding(
        args.official_results, queue, reader, official_report)
    original_path = args.official_results / "labels.mat"
    original = loadmat(original_path)
    reference = np.concatenate([checked_labels(original[key], 32492, np)
                                for key in ("lh_labels", "rh_labels")])
    report = {"scope": "completed original CBIG versus complete FNIT labels",
              "full_vertices": 64984, "network_identities_permuted": False,
              "reference_commit": queue["reference_commit"],
              "input_sha256": queue["input_sha256"],
              "reader_values_compared": reader["compared_values"],
              "coordinator_and_input_gates_passed": True,
              "official_affinity": official_job["affinity"],
              "actual_official_functions_sha256": resolved_hashes,
              "official_labels_sha256": sha256(original_path),
              "official_report": official_report, "comparisons": []}
    for directory in args.fnit_results:
        if directory.resolve().parent != queue_root:
            raise ValueError("Actual FNIT outputs must belong to this paired queue directory")
        fnit_job = completed_job(queue, directory.name)
        implementation = fnit_job["name"].rsplit("_cpu", 1)[0]
        if implementation not in CORE_SHA256:
            raise ValueError("FNIT job must identify the frozen or optimized source")
        fnit_report = json.loads((directory / "report.public.json").read_text())
        if not fnit_report.get("actual_import_matches_frozen_source"):
            raise ValueError("FNIT receipt lacks its actual frozen-source import gate")
        if fnit_report.get("labels_shape") != [64984]:
            raise ValueError("The complete FNIT label output is required")
        if (fnit_report.get("kind") != "surface" or fnit_report.get("device") != "cpu"
                or fnit_report.get("profiling_run")):
            raise ValueError("A complete unprofiled CPU CIFTI chain is required")
        if (fnit_report.get("threads") != official_report["threads"]
                or fnit_job["threads"] != official_job["threads"]
                or fnit_report["environment"]["affinity"] != official_job["affinity"]
                or fnit_job["affinity"] != official_job["affinity"]):
            raise ValueError("Original and FNIT must use the same actual CPU thread/affinity budget")
        if (fnit_report["environment"].get("hostname") != "nodecw10"
                or fnit_report["environment"].get("torch_threads") != official_job["threads"]
                or fnit_report["environment"].get("torch_interop_threads") != 1):
            raise ValueError("The actual FNIT process host and Torch thread count must match")
        if fnit_report["source_sha256"].get("src/fnit/mshbm/core.py") != CORE_SHA256[implementation]:
            raise ValueError("The actual imported FNIT core differs from its frozen source")
        if (len(fnit_report["inputs"]) != 1
                or fnit_report["inputs"][0]["sha256"] != queue["input_sha256"]
                or fnit_report.get("assets_sha256") != reader["assets_sha256"]):
            raise ValueError("Original and FNIT must consume the same complete input and fixed assets")
        fnit_binding = json.loads((queue_root / (implementation + "_binding.private.json")).read_text())
        if (fnit_binding.get("timeseries") != native_binding["timeseries"]
                or fnit_binding.get("censor") != native_binding.get("censor")
                or sha256(fnit_binding["assets"]) != reader["assets_sha256"]
                or sha256(Path(fnit_binding["source_root"]) / "src/fnit/mshbm/core.py")
                   != CORE_SHA256[implementation]):
            raise ValueError("Actual paired private bindings must match inputs, censor, assets and frozen code")
        for module, expected_hash in COMMON_MODULE_SHA256.items():
            relative = "src/fnit/mshbm/" + module
            if (fnit_report["source_sha256"].get(relative) != expected_hash
                    or sha256(Path(fnit_binding["source_root"]) / relative) != expected_hash):
                raise ValueError("An actual FNIT reader/output module differs from the frozen source")
        if fnit_report.get("w") != official_report.get("w") or fnit_report.get("c") != official_report.get("c"):
            raise ValueError("Original and FNIT weights must match")
        path = directory / "labels_fslr32k_64984.npy"
        if fnit_report["outputs_sha256"].get(path.name) != sha256(path):
            raise ValueError("The actual FNIT labels differ from their execution-time output hash")
        labels = checked_labels(np.load(path, allow_pickle=False), 64984, np)
        by_hemisphere = {}
        for name, vertex_slice in (("left", slice(0, 32492)), ("right", slice(32492, 64984)),
                                   ("both", slice(None))):
            expected = reference[vertex_slice]
            actual = labels[vertex_slice]
            rows = []
            for network in range(1, 18):
                left = expected == network
                right = actual == network
                expected_count = int(left.sum())
                actual_count = int(right.sum())
                intersection = int(np.count_nonzero(left & right))
                denominator = expected_count + actual_count
                rows.append({"network": network, "original_vertices": expected_count,
                             "fnit_vertices": actual_count, "intersection": intersection,
                             "dice": 2 * intersection / denominator if denominator else None})
            by_hemisphere[name] = {"vertices": int(expected.size),
                "different_labels": int(np.count_nonzero(expected != actual)),
                "background_different": int(np.count_nonzero((expected == 0) != (actual == 0))),
                "per_network": rows}
        report["comparisons"].append({
            "kind": fnit_report["kind"], "threads": fnit_report["threads"],
            "device": fnit_report["device"], "actual_source_sha256": fnit_report["source_sha256"],
            "fnit_labels_sha256": sha256(path), "fnit_input_receipts": fnit_report["inputs"],
            "function_chain_seconds": fnit_report["function_chain_seconds"],
            "stages_seconds": fnit_report["stages_seconds"], "precision": by_hemisphere})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"comparisons": len(report["comparisons"]),
        "different_labels": [row["precision"]["both"]["different_labels"]
                             for row in report["comparisons"]]}))


if __name__ == "__main__":
    main()
