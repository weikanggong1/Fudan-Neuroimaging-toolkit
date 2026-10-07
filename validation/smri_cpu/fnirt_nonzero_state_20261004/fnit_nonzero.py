"""Read-only objective/gradient probe on saved official nonzero FP64 parameters."""
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
from fnit.fnirt import GMFNIRTConfig, TorchFNIRT, registration


class Captured(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gm", "template", "mask", "affine", "parameters", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(8)
    gm, template = nib.load(args.gm), nib.load(args.template)
    affine = flirt_to_world_affine(np.loadtxt(args.affine), gm.affine, template.affine,
                                   gm.shape, template.shape, gm.header.get_zooms()[:3],
                                   template.header.get_zooms()[:3])
    original = registration._LevelSystem.evaluate
    captured = {}

    def capture(system, coefficients, scale, **kwargs):
        captured.update(system=system, shape=tuple(coefficients.shape[1:]))
        raise Captured()

    registration._LevelSystem.evaluate = capture
    try:
        TorchFNIRT(device="cpu", config=replace(GMFNIRTConfig(), maximum_iterations=(0,0,0,0)))(
            gm, template, affine, reference_mask=nib.load(args.mask))
    except Captured:
        pass
    finally:
        registration._LevelSystem.evaluate = original
    system = captured["system"]
    vector = torch.from_numpy(np.loadtxt(args.parameters)).double()
    coefficients, scale = registration._unpack(vector, captured["shape"], True)
    state, gradient = system.gradient(coefficients, scale)
    _, gradient_lm, matvec, diagonal = system.linearize(coefficients, scale)
    probe = torch.sin(torch.arange(1, vector.numel()+1, dtype=torch.float64) * .017)

    def save(name, value):
        np.save(args.output / (name + ".npy"), value.detach().numpy())

    for name in ("warped", "mask", "residual"):
        save("shared_" + name, state[name])
    save("shared_fixed", system.fixed)
    save("shared_scaled_fixed", (scale * system.fixed.double()).float())
    save("shared_derivative", state["gradient_fsl"].movedim(0,-1))
    save("shared_gradient", gradient)
    save("shared_gradient_lm", 2 * gradient_lm)
    save("shared_diagonal", 2 * diagonal)
    save("shared_hessian_probe", 2 * matvec(probe))
    trial_scale = scale + 3.
    trial_state = system.evaluate(coefficients, trial_scale)
    _, stale_gradient = system.gradient(coefficients, scale,
                                       effective_lambda=trial_state["effective_lambda"])
    save("stale_gradient_override", stale_gradient)
    _, reset_gradient = system.gradient(coefficients, scale)
    save("reset_gradient", reset_gradient)
    report = {"scope": "shared nonzero official parameters; no coefficient solve or pipeline",
              "host": socket.gethostname(), "affinity": sorted(os.sched_getaffinity(0)),
              "torch_threads": torch.get_num_threads(), "shape": captured["shape"],
              "source_registration_sha256": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
              "inputs": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
                         (("gm",args.gm),("template",args.template),("mask",args.mask),
                          ("flirt",args.affine),("parameters",args.parameters))},
              "shared": {name: float(state[name]) if torch.is_tensor(state[name]) else state[name]
                         for name in ("count","ssd","cost","effective_lambda","bending_energy")},
              "trial": {name: float(trial_state[name]) for name in ("ssd","cost","effective_lambda")},
              "artificial_stale_sequence": "native state-order isolation only; not the actual LM retry schedule"}
    (args.output / "report.public.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["shared"]))


if __name__ == "__main__":
    main()
