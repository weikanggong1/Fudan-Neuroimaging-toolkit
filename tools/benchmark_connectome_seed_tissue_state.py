"""比较真实 Seedtest/FNIT 种子在同一精确 5TT 仿射下的 ACT 状态。"""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.tracking import _act_seed_direction, _five_tissue_mrtrix


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--five-tissue", type=Path, required=True)
parser.add_argument("--official", type=Path, required=True)
parser.add_argument("--official-repeat", type=Path, required=True)
parser.add_argument("--fnit", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--affine-mode", choices=("header-spacing", "exact"),
                    default="header-spacing")
args = parser.parse_args()
device = torch.device(args.device)
torch.backends.cuda.matmul.allow_tf32 = True
image = nib.load(str(args.five_tissue))
five = torch.as_tensor(np.asarray(image.dataobj, dtype=np.float32).copy(), device=device)
affine = torch.as_tensor(image.affine, device=device, dtype=torch.float64)
spacing = torch.as_tensor(image.header.get_zooms()[:3], device=device, dtype=torch.float64)
if args.affine_mode == "header-spacing":
    affine[:3, :3] *= spacing / torch.linalg.vector_norm(affine[:3, :3], dim=0)
inverse = torch.linalg.inv(affine)
files = {"official_0": args.official, "official_1": args.official_repeat, "fnit": args.fnit}
report = {"affine_mode": args.affine_mode,
          "input_sha256": {"five_tissue": hashlib.sha256(args.five_tissue.read_bytes()).hexdigest(),
                           **{name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in files.items()}}, "results": {}}
for name, path in files.items():
    data = np.load(path) if name == "fnit" else np.loadtxt(
        path, delimiter=",", comments="#", usecols=(2, 3, 4))
    positions = torch.as_tensor(data.copy(), device=device, dtype=torch.float32)
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1
    tissue = _five_tissue_mrtrix(five, positions, inverse)
    valid, one_way, _ = _act_seed_direction(five, positions, directions, inverse)
    difference = tissue[:, 0] + tissue[:, 1] - tissue[:, 2]
    report["results"][name] = {
        "count": len(data),
        "act_valid_fraction": float(valid.float().mean()),
        "one_way_fraction": float(one_way.float().mean()),
        "sgm_dominant_fraction": float((tissue[:, 1] > tissue[:, 0]).float().mean()),
        "gm_side_fraction": float((difference > 0).float().mean()),
        "gm_minus_wm_quantiles_0_10_25_50_75_90_100":
            torch.quantile(difference, torch.tensor([0., .1, .25, .5, .75, .9, 1.], device=device)).cpu().tolist(),
    }
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["results"], indent=2))
