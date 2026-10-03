"""Postprocess untouched measured ABBA NPZ and raw official values; no rerun."""
import argparse
import json
from pathlib import Path
import numpy as np
from sift2_reference_io import metrics, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-report", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_path = args.component_report.with_suffix(".npz")
    data = np.load(data_path, allow_pickle=False)
    timing = json.loads(args.component_report.read_text())["nested_stage_s"]
    keys = dict(A1="baseline_weights", A2="baseline_repeat_weights",
                B1="weights", B2="candidate_repeat_weights")
    densities = dict(A1="baseline_density", A2="baseline_repeat_density",
                     B1="candidate_density", B2="candidate_repeat_density")
    pairs = [("A1", "A2"), ("B1", "B2"), ("A1", "B1"),
             ("A1", "B2"), ("A2", "B1"), ("A2", "B2")]
    report = dict(scope="Postprocessing measured arrays only; original ABBA/official reports unchanged",
                  inputs={str(path): sha(path) for path in [args.component_report, data_path,
                      args.official_dir/"sift2_weights.txt", args.official_dir/"mean_fa.txt"]},
                  each_arm_wall_s={arm: timing["optimizer_"+arm] for arm in keys},
                  all_weight_pairs={left+"_"+right: metrics(data[keys[left]], data[keys[right]])
                                    for left, right in pairs},
                  reference_geometry_conversion=json.loads((args.official_dir.parent/"official_inputs/reference_input_conversion.json").read_text()),
                  official_geometry_metadata={name: dict(sha256=sha(args.official_dir/(name+"_mrinfo.json")),
                      contents=json.loads((args.official_dir/(name+"_mrinfo.json")).read_text()))
                      for name in ("wm_fod", "five_tissue", "fa")},
                  metadata_scope="mrinfo read with RealignTransform false; actual tcksift2/tcksample commands and input SHA are preserved in reference_report.json. Metadata introspection flag is not claimed to be a program execution flag.",
                  density_arrays_saved=all(key in data for key in densities.values()),
                  official_weights_each_arm={arm: metrics(np.loadtxt(args.official_dir/"sift2_weights.txt"), data[key])
                                             for arm, key in keys.items()},
                  official_FA=metrics(np.loadtxt(args.official_dir/"mean_fa.txt"), data["fa"]),
                  gate_changed=False, equivalence_assessed=False,
                  causality="Baseline repeat variability is observed. Existing duplicate-index FP64 CUDA index_add uses atomic accumulation; cached candidate dataflow may also change scheduling. No per-operation causal isolation was performed; do not assign all cross-arm differences to existing atomics.")
    if report["density_arrays_saved"]:
        report["all_density_pairs"] = {left+"_"+right: metrics(data[densities[left]], data[densities[right]])
                                       for left, right in pairs}
    else:
        report["density_scope"] = "Original NPZ did not retain per-arm density; only four density comparisons in original JSON are available. Missing cross density is not reconstructed."
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
