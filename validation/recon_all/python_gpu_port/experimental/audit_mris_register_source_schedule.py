"""Audit source-derived registration stopping against exact saved trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fnit.recon_all.mris_register_schedule import next_smoothwm_scale


ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = {
    "lh": (
        "mris_register_lh_default_epoch0057_0068_continuous_control-host.json",
        "mris_register_lh_default_epoch0069_0080_resume_control-host.json",
        "mris_register_lh_default_epoch0081_0087_resume_control-host.json",
        "mris_register_lh_default_epoch0088_0094_resume_control-host.json",
        "mris_register_lh_default_epoch0095_0101_resume_control-host.json",
        "mris_register_lh_fold_cleanup_epoch0102_accelerator-a-1.json",
        "mris_register_lh_fold_cleanup_epoch0103_0107_accelerator-a-1.json",
    ),
    "rh": (
        "mris_register_rh_default_epoch0056_0058_area_scale_fixed_accelerator-a-1.json",
        "mris_register_rh_default_epoch0059_0066_area_scale_fixed_accelerator-a-1.json",
        "mris_register_rh_default_epoch0067_0083_first_difference_accelerator-a-1.json",
        "mris_register_rh_default_epoch0083_0097_lcorr_f32_accelerator-a-1.json",
    ),
}


def main() -> None:
    report = {"source_commit": "d932c45", "hemispheres": {}}
    for hemi, files in SEGMENTS.items():
        rows = {}
        inputs = {}
        for name in files:
            path = ROOT / name
            inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in json.loads(path.read_text())["continuation"]:
                parity = row["saved_surface"]
                if parity["exact_vertices"] == parity["total_vertices"]:
                    rows[row["epoch"]] = row
        epochs = sorted(rows)
        if epochs != list(range(epochs[0], epochs[-1] + 1)):
            raise ValueError(f"{hemi} exact saved epochs have a gap")
        state = ("smoothwm", 0, 1024, 1)
        projected = False
        checked = []
        for epoch in epochs:
            row = rows[epoch]
            stage, sigma_index, averages, steps = state
            scores = row["line_samples"]
            chosen = min(scores, key=lambda sample: abs(sample[0] - row["dt"]))
            if abs(chosen[0] - row["dt"]) > 1e-8:
                raise ValueError(f"{hemi} epoch {epoch}: selected trial is absent")
            observed_stage = "fold_cleanup" if hemi == "lh" and epoch >= 102 else "smoothwm"
            matches = (averages == row["gradient_averages"]
                       and projected == row["integration_start_projection"]
                       and stage == observed_stage)
            next_state = next_smoothwm_scale(
                *state, scores[0][1], chosen[1], row["dt"],
                negative_faces=193 if hemi == "lh" else 0)
            checked.append({"epoch": epoch, "predicted_stage": stage,
                            "predicted_sigma": (4.0, 2.0, 1.0, 0.5)[sigma_index],
                            "predicted_averages": averages,
                            "observed_averages": row["gradient_averages"],
                            "predicted_projection": projected,
                            "observed_projection": row["integration_start_projection"],
                            "match": matches})
            if next_state is None and epoch != epochs[-1]:
                raise ValueError(f"{hemi} source schedule ended before last saved epoch")
            if next_state is not None:
                projected = next_state[:3] != state[:3]
                state = next_state
        report["hemispheres"][hemi] = {
            "input_sha256": inputs, "first_epoch": epochs[0],
            "last_epoch": epochs[-1], "checked_updates": len(checked),
            "exact_schedule_decisions": sum(item["match"] for item in checked),
            "source_stopped_after_last": next_state is None,
            "last_state": state, "decisions": checked,
        }
    output = ROOT / "mris_register_source_schedule_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    for hemi, result in report["hemispheres"].items():
        print(hemi, result["exact_schedule_decisions"], "/",
              result["checked_updates"], "stopped", result["source_stopped_after_last"])
        if (result["exact_schedule_decisions"] != result["checked_updates"]
                or not result["source_stopped_after_last"]):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
