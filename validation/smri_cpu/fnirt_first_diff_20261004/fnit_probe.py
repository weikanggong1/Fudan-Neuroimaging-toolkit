"""Capture just the first GM-FNIRT level on fixed, saved official inputs."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket

import nibabel as nib
import numpy as np
import torch

from fnit.flirt.coordinates import flirt_to_world_affine
from fnit.fnirt import GMFNIRTConfig, TorchFNIRT
from fnit.fnirt import registration as registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=("reference", "optimized"), required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(8)
    moving, fixed = nib.load(args.moving), nib.load(args.fixed)
    initial = flirt_to_world_affine(np.loadtxt(args.affine), moving.affine, fixed.affine,
                                   moving.shape, fixed.shape, moving.header.get_zooms()[:3],
                                   fixed.header.get_zooms()[:3])
    # FullResKsp depends on the process' final configured subsampling. Preserve
    # the complete recipe; only disable iterations after the first accepted step.
    conf = replace(GMFNIRTConfig(), maximum_iterations=(1, 0, 0, 0))
    evaluate, linearize = registration._LevelSystem.evaluate, registration._LevelSystem.linearize
    state_log = []
    initial_saved = False
    candidate_saved = False

    def save(name, value):
        np.save(args.output / (name + ".npy"), value.detach().cpu().numpy())

    def capture_evaluate(system, coefficients, scale, **kwargs):
        nonlocal initial_saved, candidate_saved
        state = evaluate(system, coefficients, scale, **kwargs)
        state_log.append({"call": len(state_log), "derivatives": kwargs.get("derivatives", False),
                          "count": state["count"], "scale": float(scale),
                          "ssd": float(state["ssd"]), "cost": float(state["cost"]),
                          "effective_lambda": state["effective_lambda"],
                          "coefficient_norm": float(torch.linalg.vector_norm(coefficients))})
        if not initial_saved:
            for name in ("warped", "mask", "residual"):
                save("initial_" + name, state[name])
            save("initial_fixed", system.fixed)
            save("smoothed_moving", system.moving)
            initial_saved = True
        if not candidate_saved and coefficients.count_nonzero() and float(state["cost"]) < state_log[0]["cost"]:
            save("accepted_parameters", registration._pack(coefficients, scale))
            save("accepted_warped", state["warped"])
            candidate_saved = True
        return state

    def capture_linearize(system, coefficients, scale):
        state, gradient, matvec, diagonal = linearize(system, coefficients, scale)
        save("initial_derivative", state["gradient_fsl"].movedim(0, -1))
        save("initial_gradient_lm", 2 * gradient)
        _, full_gradient = system.gradient(coefficients, scale)
        save("initial_gradient_direct", full_gradient)
        save("initial_diagonal", 2 * diagonal)
        probe = torch.sin(torch.arange(1, gradient.numel() + 1, dtype=torch.float64) * .017)
        save("initial_probe", probe)
        save("initial_hessian_probe", 2 * matvec(probe))
        return state, gradient, matvec, diagonal

    registration._LevelSystem.evaluate = capture_evaluate
    registration._LevelSystem.linearize = capture_linearize
    fit = TorchFNIRT(config=conf, device="cpu", execution=args.execution)(
        moving, fixed, initial, reference_mask=nib.load(args.mask))
    for name, image in (("moving", moving), ("fixed", fixed)):
        data = np.asarray(image.dataobj, dtype=np.float32)
        mean = registration.spm_like_mean(data)
        np.save(args.output / ("normalized_" + name + ".npy"), data * np.float32(100 / mean))
    report = {"status": "complete", "scope": "first level, one accepted update; no upstream or full pipeline",
              "host": socket.gethostname(), "affinity": sorted(os.sched_getaffinity(0)),
              "torch_threads": torch.get_num_threads(), "execution": args.execution,
              "initial_world": initial.tolist(), "source": {
                  name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                  for name, module in (("registration", registration),)},
              "inputs": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in (("gm", args.moving), ("template", args.fixed),
                                            ("mask", args.mask), ("flirt", args.affine))},
              "state_log": state_log, "fit_qc": fit.qc}
    (args.output / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"execution": args.execution, "complete": True, "states": len(state_log)}))


if __name__ == "__main__":
    main()
