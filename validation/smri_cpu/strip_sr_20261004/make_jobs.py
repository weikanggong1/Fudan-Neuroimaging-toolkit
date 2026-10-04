"""Create fixed real-input SynthStrip/SynthSR cold CLI comparison jobs.

Run the generated private plan only after the coordinator verifies the native
environment. Every child is timed by the shared queue runner, using its common
CPU affinity, thread budget and exclusive lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WEIGHTS = {
    "strip": "synthstrip.1.pt", "nocsf": "synthstrip.nocsf.1.pt",
    "sr_v2": "synthsr_v20_230130.h5", "sr_v1": "synthsr_v10_210712.h5",
    "sr_lowfield": "synthsr_lowfield_v20_230130.h5",
}


def scenarios(root):
    root = Path(root)
    inputs = root / "runs/smri_cpu_20261004/inputs/ds003138"
    flair = root / "legacy/freesurfer_synth/examples/wmh_data/sub-02_FLAIR.nii.gz"
    cases = []
    for feature in ("synthstrip", "synthsr"):
        for alias in ("case01", "case02"):
            cases.append({"id": alias + "_" + feature + "_default",
                          "feature": feature, "image": str(inputs / (alias + "_T1w.nii.gz")),
                          "weight": "strip" if feature == "synthstrip" else "sr_v2",
                          "candidate_options": [], "reference_options": [],
                          "repeat_order": ["reference", "baseline", "baseline", "reference"],
                          "data_description": "public raw ds003138 v1.0.1 T1w"})
    cases.extend([
        {"id": "flair_strip_default", "feature": "synthstrip", "image": str(flair),
         "weight": "strip", "candidate_options": [], "reference_options": [],
         "data_description": "existing public derived FLAIR, not raw scanner data"},
        {"id": "case01_strip_nocsf", "feature": "synthstrip", "weight": "nocsf",
         "candidate_options": ["--no-csf"], "reference_options": ["--no-csf"],
         "automatic_weight_selection": True},
        {"id": "case01_strip_border2", "feature": "synthstrip", "weight": "strip",
         "candidate_options": ["-b", "2"], "reference_options": ["-b", "2"]},
        {"id": "case01_strip_large_border", "feature": "synthstrip", "weight": "strip",
         "candidate_options": ["-b", "100"], "reference_options": ["-b", "100"],
         "data_description": "raw T1 functional distance-extension branch; not a recommended clinical border"},
        {"id": "case01_strip_fill", "feature": "synthstrip", "weight": "strip",
         "candidate_options": ["-f", "-1"], "reference_options": ["-f", "-1"]},
        {"id": "flair_sr_default", "feature": "synthsr", "image": str(flair),
         "weight": "sr_v2", "candidate_options": [], "reference_options": [],
         "data_description": "existing public derived FLAIR, not raw scanner data"},
        {"id": "case01_sr_v1", "feature": "synthsr", "weight": "sr_v1",
         "candidate_options": ["--v1"], "reference_options": ["--v1"],
         "automatic_weight_selection": True},
        {"id": "case01_sr_lowfield_model", "feature": "synthsr", "weight": "sr_lowfield",
         "candidate_options": ["--lowfield"], "reference_options": ["--lowfield"],
         "automatic_weight_selection": True,
         "data_description": "lowfield checkpoint on raw T1; the supplemental plan separately uses actual 64mT data"},
        {"id": "case01_sr_v1_precedence", "feature": "synthsr", "weight": "sr_v1",
         "candidate_options": ["--v1", "--lowfield"], "reference_options": ["--v1", "--lowfield"],
         "automatic_weight_selection": True},
        {"id": "case01_sr_no_flip", "feature": "synthsr", "weight": "sr_v2",
         "candidate_options": ["--disable_flipping"], "reference_options": ["--disable_flipping"]},
        {"id": "case01_sr_no_sharpen", "feature": "synthsr", "weight": "sr_v2",
         "candidate_options": ["--disable_sharpening"], "reference_options": ["--disable_sharpening"]},
        {"id": "case01_sr_no_flip_no_sharpen", "feature": "synthsr", "weight": "sr_v2",
         "candidate_options": ["--disable_flipping", "--disable_sharpening"],
         "reference_options": ["--disable_flipping", "--disable_sharpening"]},
    ])
    for case in cases:
        case.setdefault("image", str(inputs / "case01_T1w.nii.gz"))
        case.setdefault("repeat_order", ["reference", "baseline"])
        case.setdefault("data_description", "public raw ds003138 v1.0.1 T1w")
    return cases


def build_jobs(server_root, *, reference_wrapper=None, source_path=None, scenarios_override=None,
               run_directory=None, source_revision="1d31e7baaebbb644ab199471f7fe6282721455fd"):
    root = Path(server_root)
    work = root / "workspaces/smri_cpu_20261004"
    run = Path(run_directory) if run_directory else root / "runs/smri_cpu_20261004/task1/paired_baseline"
    source = str(source_path or work / "baseline/src")
    wrapper = str(reference_wrapper or work / "reference_env.sh")
    python = str(root / "envs/default/bin/python")
    weights = work / "assets/weights"
    jobs = []
    cases = scenarios_override or scenarios(root)
    for case in cases:
        for order_index, arm in enumerate(case["repeat_order"], 1):
            identifier = case["id"] + "_%d_" % order_index + arm
            output = run / identifier / "artifacts"
            suffix = case.get("output_suffix", ".nii.gz")
            image_out = str(output / ("image" + suffix))
            output_argument = image_out
            if case.get("output_directory"):
                source_name = Path(case["image"]).name
                source_suffix = ".nii.gz" if source_name.endswith(".nii.gz") else Path(source_name).suffix
                image_out = str(output / (source_name[:-len(source_suffix)] + "_synthsr" + source_suffix))
                output_argument = str(output)
            weight = str(weights / WEIGHTS[case["weight"]])
            candidate_weight_options = ["--weights", str(weights) if case.get("automatic_weight_selection") else weight]
            reference_weight_options = [] if case.get("automatic_weight_selection") else ["--model", weight]
            if case["feature"] == "synthstrip":
                outputs = [image_out, str(output / "mask.nii.gz"), str(output / "distance.nii.gz")]
                options = ["-i", case["image"], "-o", outputs[0], "-m", outputs[1], "-d", outputs[2]]
                if arm == "baseline":
                    argv = [python, "-m", "fnit.cli", "synthstrip", *options,
                            "--device", "cpu", "-j", "8", *candidate_weight_options,
                            *case["candidate_options"]]
                else:
                    argv = ["bash", wrapper, "mri_synthstrip", *options,
                            "-t", "8", *reference_weight_options, *case["reference_options"]]
            else:
                outputs = [image_out]
                options = ["--i", case["image"], "--o", output_argument, "--threads", "8", "--cpu"]
                if arm == "baseline":
                    argv = [python, "-m", "fnit.cli", "synthsr", *options,
                            "--device", "cpu", *candidate_weight_options, *case["candidate_options"]]
                else:
                    argv = ["bash", wrapper, "mri_synthsr", *options,
                            *reference_weight_options, *case["reference_options"]]
            # Native CLI does not create its output directory. Directory creation
            # is explicit, quick and within both arms' measured wrapper scope.
            argv = ["bash", "-c", 'mkdir -p "$1"; shift; exec "$@"',
                    "fnit-strip-sr-output", str(output), *argv]
            jobs.append({"id": identifier, "argv": argv,
                         "env": {"PYTHONPATH": source}, "expected_outputs": outputs,
                         "timeout_seconds": 1800})
    return {"schema": "fnit.smri.cpu.strip_sr.jobs.v1", "source_revision": source_revision,
            "thread_count": 8, "affinity": "0,4,8,12,16,20,24,28",
            "cases": cases, "jobs": jobs,
            "not_in_default_plan": ["actual HU CT and 64mT inputs in the supplemental plan",
                                     "real multi-frame and MGZ/NPZ inputs in the supplemental plan",
                                     "in-memory API and reused floating/format outputs in profile plans"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", default="/cwStorage/home/gongwk/Notebook_code/FNIT")
    parser.add_argument("--reference-wrapper")
    parser.add_argument("--source-path")
    parser.add_argument("--source-revision", default="1d31e7baaebbb644ab199471f7fe6282721455fd")
    parser.add_argument("--run-directory", type=Path,
                        help="fresh output root for a separately frozen candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new immutable job plan path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_jobs(args.server_root, reference_wrapper=args.reference_wrapper,
                                               source_path=args.source_path,
                                               source_revision=args.source_revision,
                                               run_directory=args.run_directory), indent=2) + "\n")


if __name__ == "__main__":
    main()
