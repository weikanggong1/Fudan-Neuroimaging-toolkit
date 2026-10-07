"""Audit complete helper outputs with an exact float32 AFNI input control.

This read-only collector preserves the earlier integer-storage comparison.
It launches no reference program and performs no production computation.
"""
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_values(path):
    image = nib.load(path)
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("The complete image must contain finite values")
    return image, values


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--float-input", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--stc-candidate-folder", default="candidate_stc_matched_v5")
    parser.add_argument("--stc-reference-folder", default="reference_stc_float32_v5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous analysis results")
    inputs = json.loads(args.inputs.read_text())
    prior = json.loads(args.prior_report.read_text())
    reports, file_hashes, results = {}, {}, {}

    def identity(path):
        path = Path(path)
        if path not in file_hashes:
            file_hashes[path] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        return file_hashes[path]

    actual_source = {str(path.relative_to(args.source)): identity(path)["sha256"]
                     for path in (args.source / "src/fnit").rglob("*.py")}
    if not actual_source:
        raise ValueError("The frozen production source must be readable")
    raw_path = Path(inputs["public180"]["bold"])
    raw_image, raw_values = image_values(raw_path)
    float_image, float_values = image_values(args.float_input)
    if (raw_image.shape != (64, 64, 42, 180)
            or float_image.shape != raw_image.shape
            or float_image.get_data_dtype() != np.dtype("float32")
            or not np.array_equal(raw_values, float_values)
            or not np.allclose(raw_image.affine, float_image.affine, atol=1e-6, rtol=0)
            or raw_image.header.get_xyzt_units() != float_image.header.get_xyzt_units()
            or raw_image.header.get_zooms() != float_image.header.get_zooms()):
        raise ValueError("The float32 control must preserve every raw value, grid and TR")
    conversion = {
        "original_input": identity(raw_path),
        "float32_input": identity(args.float_input),
        "shape": list(raw_image.shape),
        "storage_dtypes": [str(raw_image.get_data_dtype()), "float32"],
        "complete_numeric_values_exact": True, "same_grid_and_tr": True,
        "protocol": "Exact float32 input control; AFNI 3dTshift has no -float option",
    }
    del float_values
    expected_container = inputs["expected_fmriprep_sha256"]
    if identity(inputs["fmriprep_image"])["sha256"] != expected_container:
        raise ValueError("The actual reference container differs from its frozen identity")

    def recorded(folder, name, function, backend, threads, expected_inputs):
        path = args.run_root / folder / name
        value = json.loads((path / "report.safe.json").read_text())
        if (value.get("status") != "complete" or not value.get("source_unchanged")
                or value["function"] != function or value["backend"] != backend
                or value["cpu_threads"] != threads
                or len(set(value["cpu_affinity"])) != threads
                or value["source_sha256"] != actual_source
                or not value["records"] or value["records"][0]["repeat"] != 0
                or value["records"][0]["call_type"] != "first_call"):
            raise ValueError("A complete invocation of the unchanged frozen source and budget is required")
        for key, filename in expected_inputs.items():
            if value["input_sha256"].get(key) != identity(filename):
                raise ValueError("The complete actual input differs from its invocation record")
        if function == "stc":
            metadata_sha = hashlib.sha256(json.dumps(
                inputs["public180"]["metadata"], sort_keys=True).encode()).hexdigest()
            if value["raw_public_metadata_sha256"] != metadata_sha:
                raise ValueError("The actual slice-timing metadata differs")
        row = value["records"][0]
        if function in ("temporal", "stc") and backend == "official":
            if not row.get("native_exit_accepted"):
                raise ValueError("The original executable must have an accepted actual child exit")
        reports[folder + "/" + name] = {
            "function": function, "backend": backend,
            "source_revision": value["source_revision"],
            "source_files_verified": len(actual_source),
            "threads": threads, "affinity": value["cpu_affinity"],
            "adapter_sha256": value["adapter_sha256"],
            "input_sha256": value["input_sha256"],
            "calls": [{key: record[key] for key in (
                "repeat", "call_type", "api_wall_seconds_including_io",
                "official_wall_seconds_including_io", "native_exit_code", "native_exit_accepted")
                if key in record} for record in value["records"]],
        }
        return path / "repeat_0", value

    def paired(left, right):
        if (left["cpu_affinity"] != right["cpu_affinity"]
                or left["cpu_threads"] != right["cpu_threads"]):
            raise ValueError("Original and FNIT physical-core allocations differ")

    def compare(candidate, reference, expected_shape, temporal=False):
        ci, ca = image_values(candidate)
        ri, ra = image_values(reference)
        affine_error = float(np.max(np.abs(ci.affine - ri.affine)))
        if (ci.shape != expected_shape or ri.shape != expected_shape
                or affine_error > 1e-4
                or ci.get_data_dtype() != np.dtype("float32")
                or ri.get_data_dtype() != np.dtype("float32")):
            raise ValueError("Complete float32 outputs must use the intended full input grid")
        if temporal and (ci.header.get_zooms()[3] != ri.header.get_zooms()[3]
                         or ci.header.get_xyzt_units()[1] != ri.header.get_xyzt_units()[1]):
            raise ValueError("The complete temporal output TR or units differ")
        return {"shape": list(ci.shape), "same_grid": True,
                "maximum_affine_difference": affine_error,
                "storage_dtypes": ["float32", "float32"],
                "full_grid": correlations(ca, ra, temporal=temporal),
                "output_sha256": {"fnit": sha(candidate), "official": sha(reference)}}

    temporal_inputs = {key: inputs["temporal490"][key]
                       for key in ("official_corrected", "brain_mask")}
    temporal_shape = nib.load(temporal_inputs["official_corrected"]).shape
    if temporal_shape != (88, 88, 64, 490):
        raise ValueError("Use the complete fixed 490-frame temporal input")
    sampling_inputs = {key: inputs[key] for key in (
        "sampling_t1", "sampling_moving", "sampling_t1_mask")}
    stc_inputs = {"raw_BOLD": raw_path, "raw_T1w": inputs["public180"]["t1w"]}
    for threads in (1, 8):
        for prefix in ("temporal", "temporal_remove_mean"):
            candidate, cv = recorded("baseline_v3", f"{prefix}_fnit_t{threads}",
                                     "temporal", "fnit", threads, temporal_inputs)
            reference, rv = recorded("reference_fixed_v2", f"{prefix}_official_t{threads}",
                                     "temporal", "official", threads, temporal_inputs)
            paired(cv, rv)
            preserve_mean = prefix == "temporal"
            if any(value["records"][0]["preserve_mean"] != preserve_mean
                   or value["records"][0]["complete_frames"] != 490 for value in (cv, rv)):
                raise ValueError("The original and FNIT highpass settings differ")
            for filename in ("scaled.nii.gz", "highpass.nii.gz"):
                results[f"{prefix}_t{threads}_{filename}"] = compare(
                    candidate / filename, reference / filename, temporal_shape, True)
        candidate, cv = recorded("baseline_v3", f"sampling_reference_fnit_t{threads}",
                                 "sampling_reference", "fnit", threads, sampling_inputs)
        reference, rv = recorded("reference_fixed_v2", f"sampling_reference_official_t{threads}",
                                 "sampling_reference", "official", threads, sampling_inputs)
        paired(cv, rv)
        value = compare(candidate / "sampling_reference.nii.gz",
                        reference / "sampling_reference.nii.gz", (55, 72, 60))
        value["input_scope"] = "Complete T1/real BOLD geometry/WM FOV mask; not an anatomical brain-mask benchmark"
        value["qform_sform_codes"] = {}
        for kind, directory in (("fnit", candidate), ("official", reference)):
            image = nib.load(directory / "sampling_reference.nii.gz")
            codes = [int(image.header["qform_code"]), int(image.header["sform_code"])]
            if (codes != [2, 2]
                    or not np.allclose(image.get_qform(), image.affine, atol=1e-4, rtol=0)
                    or not np.allclose(image.get_sform(), image.affine, atol=1e-4, rtol=0)):
                raise ValueError("Sampling references must preserve the specified qform/sform codes")
            value["qform_sform_codes"][kind] = codes
        results[f"sampling_reference_t{threads}"] = value
    for prefix, threads, official_name in (
            ("stc", 1, "stc_official_t1"), ("stc", 8, "stc_official_t8"),
            ("stc180_ignore4_start", 8, "stc_ignore4_ref0_official_t8")):
        candidate, cv = recorded(args.stc_candidate_folder, f"{prefix}_fnit_t{threads}",
                                 "stc", "fnit", threads, stc_inputs)
        reference, rv = recorded(args.stc_reference_folder, official_name,
                                 "stc", "official", threads,
                                 dict(stc_inputs, stc_float32_input=args.float_input))
        paired(cv, rv)
        fraction, ignore = (0.0, 4) if prefix == "stc180_ignore4_start" else (0.5, 0)
        slice_times = np.asarray(inputs["public180"]["metadata"]["SliceTiming"])
        target = float(np.round(slice_times.min() + fraction * np.ptp(slice_times), 3))
        cr, rr = cv["records"][0], rv["records"][0]
        if (cr["SliceTimeReference"] != fraction
                or cr["IgnoredInitialVolumes"] != ignore
                or cr["StartTime"] != target or rr["StartTime"] != target
                or cr["complete_frames"] != 180 or rr["complete_frames"] != 180):
            raise ValueError("The original and FNIT complete STC settings differ")
        value = compare(candidate / "stc.nii.gz", reference / "stc.nii.gz",
                        raw_image.shape, True)
        if prefix == "stc180_ignore4_start":
            value["ignored_frames_unchanged"] = {}
            for kind, directory in (("fnit", candidate), ("official", reference)):
                values = np.asarray(nib.load(directory / "stc.nii.gz").dataobj,
                                    dtype=np.float32)[..., :4]
                exact = bool(np.array_equal(raw_values[..., :4], values))
                if not exact:
                    raise ValueError("Ignored frames changed")
                value["ignored_frames_unchanged"][kind] = exact
        key = f"{prefix}_t{threads}"
        value["reference_protocol"] = "Original AFNI, exact float32 input control"
        previous, pv = recorded("baseline_v3", f"{prefix}_fnit_t{threads}",
                                 "stc", "fnit", threads, stc_inputs)
        previous_image, previous_values = image_values(previous / "stc.nii.gz")
        current_image, current_values = image_values(candidate / "stc.nii.gz")
        previous_sha = sha(previous / "stc.nii.gz")
        if previous_sha != prior["results"][key]["output_sha256"]["fnit"]:
            raise ValueError("The earlier complete FNIT output changed after its recorded comparison")
        header_grid_equal = bool(
            previous_image.shape == current_image.shape
            and previous_image.get_data_dtype() == current_image.get_data_dtype()
            and np.array_equal(previous_image.affine, current_image.affine)
            and previous_image.header.get_zooms() == current_image.header.get_zooms()
            and previous_image.header.get_xyzt_units() == current_image.header.get_xyzt_units()
            and np.array_equal(previous_image.get_qform(), current_image.get_qform())
            and np.array_equal(previous_image.get_sform(), current_image.get_sform())
            and int(previous_image.header["qform_code"]) == int(current_image.header["qform_code"])
            and int(previous_image.header["sform_code"]) == int(current_image.header["sform_code"]))
        values_exact = bool(np.array_equal(previous_values, current_values))
        if not values_exact or not header_grid_equal:
            raise ValueError("Fresh matched-core FNIT STC must preserve the earlier complete values and header grid")
        value["fresh_vs_previous_fnit"] = {
            "complete_values_exact": values_exact, "header_grid_exact": header_grid_equal,
            "binary_nifti_header_exact": bool(
                previous_image.header.binaryblock == current_image.header.binaryblock),
            "previous_output_sha256": previous_sha,
            "fresh_output_sha256": sha(candidate / "stc.nii.gz"),
            "previous_affinity": pv["cpu_affinity"], "fresh_affinity": cv["cpu_affinity"],
            "scope": "Full numerical/header comparison across physical-core groups; no shared timing claim",
        }
        results[key] = value
    old_stc = {key: value for key, value in prior["results"].items()
               if key.startswith("stc")}
    if len(results) != 13 or len(old_stc) != 3 or any(
            value["storage_dtypes"] != ["float32", "int16"] for value in old_stc.values()):
        raise ValueError("Preserve all thirteen full comparisons and three earlier integer controls")
    if any(value["output_sha256"]["fnit"] != results[key]["fresh_vs_previous_fnit"]["previous_output_sha256"]
           or value["shape"] != results[key]["shape"] for key, value in old_stc.items()):
        raise ValueError("Earlier integer controls must use these exact FNIT outputs")
    public = {
        "schema_version": 2, "full_comparisons": len(results),
        "driver_sha256": sha(__file__), "results": results,
        "completed_invocations": reports, "float32_input_control": conversion,
        "reference_container_sha256": expected_container,
        "preserved_integer_storage_controls": {
            "original_report_sha256": sha(args.prior_report), "results": old_stc,
            "completed_invocations": {key: value for key, value
                                      in prior["completed_invocations"].items()
                                      if value["function"] == "stc"},
            "scope": "Original raw int16 storage protocol; includes output quantization",
        },
        "default_slice_timing": False,
        "scope": "Complete 490-frame temporal, complete 180-frame optional STC, full 3D sampling reference",
        "timing_scope": "Original normal-I/O API clocks; analysis and float32 conversion outside timers",
        "source_scope": "Unchanged frozen baseline, not latest integrated robust-reference volume pipeline",
        "private_images_exported": False,
    }
    args.output.write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete_comparisons": 13, "preserved_integer_controls": 3,
                      "private_images_exported": False}))


if __name__ == "__main__":
    main()
