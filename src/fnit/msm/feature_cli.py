"""Prepare MSMAll features from an explicit, private parameter manifest."""

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path


_PATH_FIELDS = {
    "clean_dtseries", "ica_timecourses", "noise_components", "output_file",
    "reference_maps", "output_dir", "variance_normalization", "vertex_area",
    "component_indices", "low_dimensional_maps", "left_midthickness", "right_midthickness",
    "source_sphere", "reference_sphere", "source_rsn", "reference_rsn",
    "source_rsn_weights", "reference_rsn_weights", "subject_myelin", "reference_myelin",
    "subject_myelin_bias", "source_roi", "reference_roi", "initial_sphere",
    "source_topography", "reference_topography", "source_topography_weights",
    "reference_topography_weights",
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-msm-features", description=__doc__)
    parser.add_argument("method", choices=("vn", "regression", "prepare"))
    parser.add_argument("--inputs-json", required=True,
                        help="JSON object with the selected Python function's named parameters")
    args = parser.parse_args(argv)
    manifest = Path(args.inputs_json).expanduser().resolve()
    parameters = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(parameters, dict):
        raise ValueError("feature manifest must be a parameter object")

    def resolve(value):
        if isinstance(value, str):
            item = Path(value).expanduser()
            return str(item.resolve() if item.is_absolute() else (manifest.parent / item).resolve())
        return value

    for field in _PATH_FIELDS & parameters.keys():
        value = parameters[field]
        parameters[field] = ([resolve(item) for item in value] if field == "low_dimensional_maps"
                             and isinstance(value, list) else resolve(value))
    if "wb_command" in parameters and "/" in parameters["wb_command"]:
        parameters["wb_command"] = resolve(parameters["wb_command"])
    from .features import (compute_msmall_variance_normalization,
                           prepare_msmall_inputs, run_msmall_regression)
    function = {"vn": compute_msmall_variance_normalization,
                "regression": run_msmall_regression, "prepare": prepare_msmall_inputs}[args.method]
    result = function(**parameters)
    print(json.dumps(asdict(result), default=str, indent=2) if is_dataclass(result) else str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
