"""Compare fixed-TCK official outputs with component and original-chain exports."""
import argparse
import json
from pathlib import Path

import numpy as np
from sift2_reference_io import metrics, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--component-report", required=True, type=Path)
    p.add_argument("--official-dir", required=True, type=Path)
    p.add_argument("--track-metrics", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    data = np.load(args.component_report.with_suffix(".npz"), allow_pickle=False)
    official_report = json.loads((args.official_dir / "reference_report.json").read_text())
    if not official_report.get("completed"):
        raise RuntimeError("official reference incomplete")
    conversion_path = args.official_dir.parent / "official_inputs/reference_input_conversion.json"
    conversion = json.loads(conversion_path.read_text())
    weights = np.loadtxt(args.official_dir / "sift2_weights.txt")
    fa = np.loadtxt(args.official_dir / "mean_fa.txt")
    original = np.load(args.track_metrics, allow_pickle=False)
    endpoint_metrics = metrics(original["endpoints"], data["reloaded_endpoints"])
    endpoint_bit_neq = int(np.count_nonzero(
        original["endpoints"].astype(np.float32).view(np.uint32) !=
        data["reloaded_endpoints"].view(np.uint32)))
    report = dict(scope="same fixed TCK component versus official; never raw full pipeline benchmark",
                  component_report=sha(args.component_report),
                  official_reference_report=sha(args.official_dir / "reference_report.json"),
                  track_metrics=sha(args.track_metrics),
                  reference_input_conversion_sha256=sha(conversion_path),
                  reference_geometry_conversion=conversion,
                  official_full_4x4_affine_bit_identity=all(image["affine_exact"] for image in conversion["images"].values()),
                  official_vs_reloaded_TCK_baseline_weights=metrics(weights, data["baseline_weights"]),
                  official_vs_reloaded_TCK_candidate_weights=metrics(weights, data["weights"]),
                  same_map_baseline_vs_candidate_weights=metrics(data["baseline_weights"], data["weights"]),
                  baseline_repeat_weights=metrics(data["baseline_weights"], data["baseline_repeat_weights"]),
                  official_FA=metrics(fa, data["fa"]),
                  root_production_vs_reloaded_TCK_baseline_weights=metrics(original["weights"], data["baseline_weights"]),
                  root_production_vs_reloaded_TCK_candidate_weights=metrics(original["weights"], data["weights"]),
                  root_endpoints_vs_TCK_reload=endpoint_metrics,
                  root_endpoints_vs_TCK_reload_bit_neq=endpoint_bit_neq,
                  root_vs_reload_attribution="Separate from same-map optimizer A/B: original production padded backing is absent after TCK reload. Endpoint identity alone does not establish every original path point. CUDA repeated reductions also assessed separately; no causal attribution made.",
                  root_export_FA=metrics(original["mean_fa"], data["fa"]),
                  official_text_precision="Raw decimal outputs compared as parsed float64; no rounding or post-hoc tolerance applied",
                  equivalence_assessed=False)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
