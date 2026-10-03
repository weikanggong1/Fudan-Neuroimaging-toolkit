"""Read completed original fMRIPrep interfaces; never execute an imaging step.

Run inside the same fixed reference image after the full workflow finishes.
The public report contains operation names and scalar timings. File paths,
input/output fields and node identities remain in the optional private manifest.
MapNode parent runtimes are kept separately from their leaf executions to avoid
counting the same work twice. Interface intervals can overlap under MultiProc;
their sums do not replace the continuous workflow wall time.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fields(value):
    if isinstance(value, dict):
        return {str(key): fields(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [fields(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def path_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from path_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from path_values(item)
    elif isinstance(value, (str, Path)):
        yield Path(value)


def stage_name(path):
    """Classify observed original workflow nodes without changing execution."""
    text = str(path)
    for component, name in (
        ("msm_sulc_wf", "anatomical_msmsulc"),
        ("brain_extraction_wf", "anatomical_brain_extraction"),
        ("surface_recon_wf", "anatomical_existing_fs_processing"),
        ("register_template_wf", "anatomical_standard_normalization"),
        ("brain_seg_wf", "anatomical_tissue_segmentation"),
        ("bold_hmc_wf", "bold_motion_estimation"),
        ("bold_reg_wf", "bold_epi_to_t1_registration"),
        ("bold_anat_wf", "bold_t1_preproc_sampling"),
        ("bold_MNI6_wf", "bold_mni_preproc_sampling"),
        ("bold_std_wf", "bold_mni_preproc_sampling"),
        ("bold_t1_trans_wf", "bold_t1_preproc_sampling"),
        ("bold_std_trans_wf", "bold_mni_preproc_sampling"),
        ("bold_native_wf", "bold_native_preproc_sampling"),
        ("bold_fsLR_resampling_wf", "bold_cortical_fslr_projection"),
        ("bold_fsLR_wf", "bold_cortical_fslr_projection"),
        ("bold_surf_wf", "bold_native_surface_projection"),
        ("bold_grayords_wf", "bold_cifti_assembly"),
        ("bold_confounds_wf", "bold_confounds"),
        ("bold_boldref_wf", "bold_reference_preparation"),
        ("hmc_boldref_wf", "bold_reference_preparation"),
        ("enhance_and_skullstrip_bold_wf", "bold_reference_preparation"),
    ):
        if component in text:
            return name
    if "/anat_fit_wf/" in text and Path(path).name == "result_fast.pklz":
        return "anatomical_tissue_segmentation"
    return "other_original_workflow_nodes"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path)
    args = parser.parse_args()
    if importlib.metadata.version("fmriprep") != "25.2.4":
        raise ValueError("collect inside the fixed fMRIPrep 25.2.4 reference image")
    from nipype.utils.filemanip import loadpkl

    nodes, private, summary, resources, intervals = [], [], defaultdict(list), {}, []
    for path in sorted(args.work_root.rglob("result_*.pklz")):
        result = loadpkl(str(path))
        interface = getattr(result.interface, "__name__", str(result.interface))
        operation = path.name.removeprefix("result_").removesuffix(".pklz")
        runtime = result.runtime
        aggregate = isinstance(runtime, (list, tuple))
        runtimes = runtime if aggregate else [runtime]
        durations = [float(item.duration) if getattr(item, "duration", None) is not None
                     else None for item in runtimes]
        if any(value is not None and (not math.isfinite(value) or value < 0)
               for value in durations):
            raise ValueError("completed reference interface contains an invalid runtime duration")
        item = {"operation": operation, "interface": interface,
                "mapnode_parent": aggregate, "duration_seconds": durations,
                "completed_executions": len(runtimes), "stage": stage_name(path)}
        bounds = []
        for runtime_item in runtimes:
            start = getattr(runtime_item, "startTime", None)
            end = getattr(runtime_item, "endTime", None)
            bounds.append((datetime.fromisoformat(start), datetime.fromisoformat(end))
                          if isinstance(start, str) and isinstance(end, str) else None)
        intervals.extend(bound[0] for bound in bounds if bound is not None)
        item["_intervals"] = bounds
        nodes.append(item)
        if not aggregate:
            summary[interface].extend(value for value in durations if value is not None)
        outputs = (result.outputs.trait_get() if hasattr(result.outputs, "trait_get")
                   else vars(result.outputs) if hasattr(result.outputs, "__dict__")
                   else result.outputs)
        inputs = getattr(result, "inputs", None)
        for resource in path_values(inputs):
            text = str(resource)
            is_template = (text.startswith("/reference-templateflow/")
                           or "/smriprep/data/" in text
                           or text.startswith("/opt/freesurfer/subjects/fsaverage/"))
            if is_template and resource.is_file() and text not in resources:
                resources[text] = {"name": resource.name, "size_bytes": resource.stat().st_size,
                                   "sha256": sha256(resource)}
        private.append({"operation": operation, "interface": interface,
                        "result_file": str(path), "mapnode_parent": aggregate,
                        "input_fields": fields(inputs), "output_fields": fields(outputs),
                        "duration_seconds": durations,
                        "runtime_commands": [getattr(item, "cmdline", None) for item in runtimes]})
    if not nodes:
        raise ValueError("no completed reference interfaces found")
    origin = min(intervals) if intervals else None
    for item in nodes:
        bounds = item.pop("_intervals")
        item["interval_seconds_after_first_native_interface"] = [
            {"start": (bound[0] - origin).total_seconds(),
             "end": (bound[1] - origin).total_seconds()} if bound is not None else None
            for bound in bounds]
    stages = {}
    for name in sorted({item["stage"] for item in nodes}):
        leaves = [item for item in nodes if item["stage"] == name and not item["mapnode_parent"]]
        durations = [value for item in leaves for value in item["duration_seconds"] if value is not None]
        bounds = [value for item in leaves for value in item["interval_seconds_after_first_native_interface"]
                  if value is not None]
        stages[name] = {
            "leaf_result_files": len(leaves), "leaf_execution_duration_sum_seconds": sum(durations),
            "first_execution_start_seconds_after_first_interface": min(x["start"] for x in bounds) if bounds else None,
            "last_execution_end_seconds_after_first_interface": max(x["end"] for x in bounds) if bounds else None,
            "execution_interval_span_seconds": max(x["end"] for x in bounds) - min(x["start"] for x in bounds)
                                               if bounds else None,
            "boundary": "Observed original leaf interface times; span can include scheduling gaps and "
                        "other concurrent stages. Neither summed durations nor spans replace whole-job wall."}
    from smriprep import load_data
    msm_config = Path(load_data("msm/MSMSulcStrainFinalconf"))
    newmsm = Path("/reference-newmsm/bin/newmsm")
    report = {
        "schema_version": 1, "reference_version": "fMRIPrep 25.2.4 with explicit official newMSM",
        "collector_sha256": sha256(__file__), "completed_result_files": len(nodes),
        "leaf_interface_summary": {
            name: {"executions": len(values), "sum_seconds": sum(values),
                   "maximum_seconds": max(values), "minimum_seconds": min(values)}
            for name, values in sorted(summary.items()) if values},
        "completed_interfaces": nodes,
        "stage_groups": stages,
        "actual_template_input_files": list(resources.values()),
        "official_msm_configuration": {
            "name": msm_config.name, "sha256": sha256(msm_config),
            "size_bytes": msm_config.stat().st_size,
            "explicit_newmsm_binary_sha256": sha256(newmsm) if newmsm.is_file() else None,
            "configuration_text": msm_config.read_text()},
        "timing_boundary": "Each original completed interface runtime duration. MapNode parent "
                           "durations duplicate leaf executions and are excluded from summaries. "
                           "Parallel intervals overlap; sums do not equal continuous pipeline wall time.",
        "privacy": "Names and scalar timings are public; paths, node identities, commands, "
                   "source images and full values stay on the private reference server.",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.private_manifest is not None:
        args.private_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.private_manifest.write_text(json.dumps({"nodes": private}, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed_interface_files": len(nodes), "runtime_summary_complete": True}))


if __name__ == "__main__":
    main()
