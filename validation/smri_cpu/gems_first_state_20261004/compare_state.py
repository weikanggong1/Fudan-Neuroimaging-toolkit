"""Report the limited real GEMS state without publishing atlas arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fnit.gems.gaussian import GaussianParameters, gaussian_log_likelihood


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(left, right):
    delta = left.astype(np.float64) - right.astype(np.float64)
    absolute = np.abs(delta)
    return {"different_elements": int(np.count_nonzero(delta)),
            "max_abs": float(absolute.max()), "rms": float(np.sqrt(np.mean(delta**2))),
            "p99_abs": float(np.quantile(absolute, .99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = args.native or args.run / "native"
    probe = args.probe or args.run / "shared_probe"
    fnit = args.run / "fnit"
    args.output.mkdir(exist_ok=False,parents=True)
    data = np.load(fnit/"shared_input.npz")
    priors = np.load(fnit/"fnit_priors.npy")
    official_priors = np.load(native/"native_priors.npy")
    uint16 = np.load(native/"native_priors_uint16.npy")
    valid = data["image"] != 0
    report = {"structure": json.loads((fnit/"report.public.json").read_text())["structure"],
              "scope": "first synthetic-label objective only; no optimizer update, no intensity EM or regional segmentation gate",
              "run": str(args.run), "native": str(native),
              "source_binding": json.loads((fnit/"report.public.json").read_text()),
              "native_binding": json.loads((native/"report.public.json").read_text()),
              "autograd": json.loads((probe/"report.public.json").read_text()),
              "arrays": {k:{"dtype":str(data[k].dtype),"shape":list(data[k].shape),
                             "sha256_c_order":hashlib.sha256(np.ascontiguousarray(data[k]).tobytes()).hexdigest()}
                         for k in data.files},
              "finite_image": bool(np.isfinite(data["image"]).all()),
              "prior_comparison": metrics(priors,official_priors),
              "prior_vs_uint16_div65535": metrics(priors,uint16/65535),
              "coverage": {"valid_voxels":int(valid.sum()),
                           "fnit_covered":int(np.load(fnit/"fnit_coverage.npy").sum()),
                           "official_nonzero_mass":int(np.count_nonzero(official_priors.sum(0))),
                           "official_mass_min":float(official_priors.sum(0).min()),
                           "official_mass_max":float(official_priors.sum(0).max())},
              "classes": []}
    for group in range(priors.shape[0]):
        report["classes"].append({"class_index":group,"fixed_mean":float(data["means"][group,0]),
                                  "prior":metrics(priors[group],official_priors[group])})
    parameters = GaussianParameters(torch.from_numpy(data["means"].astype(np.float32)),
                                    torch.from_numpy(data["variances"].astype(np.float32)))
    image = torch.from_numpy(data["image"])
    ll = gaussian_log_likelihood(image[valid].reshape(-1,1,1),parameters).reshape(priors.shape)
    old_log = (torch.from_numpy(priors).clamp_min(torch.finfo(torch.float32).tiny).log()+ll).logsumexp(0)
    new_log = torch.logaddexp(old_log,old_log.new_tensor(np.log(1e-15)))
    correction = (new_log-old_log).numpy()
    report["epsilon_effect"] = {"voxels_density_below_1e_minus15":int((old_log<np.log(1e-15)).sum()),
        "max_point_cost_reduction":float(correction.max()),"total_data_cost_reduction":float(correction.astype(np.float64).sum()),
        "voxels_reduced_cost_over_1e_minus3":int(np.count_nonzero(correction>1e-3)),
        "changed_fnit_vertex_gradient_components":report["autograd"]["variants"]["epsilon"]["gradient_vs_saved_fnit"]["different"]}
    figure_data = np.zeros(data["image"].shape,dtype=np.float32)
    figure_data[valid] = correction
    plane = int(np.argmax(figure_data.sum(axis=(0,1))))
    # Only a real observation slice and derived scalar cost correction. No
    # atlas, mesh, alpha, fine reference label or full-volume prior is exported.
    np.savez_compressed(args.output/"figure_slice.public.npz",observation=data["image"][:,:,plane],
                        correction=figure_data[:,:,plane],plane=plane)
    report["figure"] = {"plane_axis":2,"plane_index":plane,
                       "data_sha256":sha(args.output/"figure_slice.public.npz")}
    (args.output/"comparison.public.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"structure":report["structure"],"prior":report["prior_comparison"],
                      "epsilon_effect":report["epsilon_effect"]}))


if __name__ == "__main__":
    main()
