"""Compare PyTorch MRtrix shell clustering and Dhollander labels on real DWI.

References: mrinfo corrected.mif -shell_bvalues -shell_sizes;
dwi2response dhollander corrected.mif wm.txt gm.txt csf.txt;
head -1 wm.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from fnit.connectome.response import mrtrix_shell_centres


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("grad", "dwi-mif", "wm-response", "mrinfo-bin", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    gradient = torch.as_tensor(np.loadtxt(args.grad), dtype=torch.float64)
    start = time.perf_counter()
    means, selection, header, sizes = mrtrix_shell_centres(gradient)
    torch_seconds = time.perf_counter() - start
    start = time.perf_counter()
    run = subprocess.run(
        [str(args.mrinfo_bin), str(args.dwi_mif), "-shell_bvalues", "-shell_sizes"],
        capture_output=True, text=True, check=True,
    )
    reference_seconds = time.perf_counter() - start
    lines = run.stdout.splitlines()
    reference_means = np.fromstring(lines[0], sep=" ")
    reference_sizes = np.fromstring(lines[1], sep=" ", dtype=int)
    first = args.wm_response.open().readline().strip()
    if not first.startswith("# Shells:"):
        raise ValueError("WM response has no shell header")
    reference_header = np.fromstring(first.split(":", 1)[1], sep=",")
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm corrected DWI",
        "scope": "105-volume MRtrix gradient shell clustering, mrinfo means/sizes, Dhollander rounded selection, amp2response truncated header",
        "reference_commands": __doc__.split("References: ", 1)[1].strip(),
        "input_sha256": {"grad": _sha(args.grad), "dwi_mif": _sha(args.dwi_mif),
                         "wm_response": _sha(args.wm_response)},
        "torch_seconds": torch_seconds,
        "mrinfo_wall_seconds": reference_seconds,
        "means": means.tolist(),
        "mrinfo_means": reference_means.tolist(),
        "mean_max_abs": float(np.max(np.abs(means.numpy() - reference_means))),
        "selection": selection.tolist(),
        "response_header": header.tolist(),
        "official_response_header": reference_header.tolist(),
        "shell_sizes": sizes.tolist(),
        "mrinfo_shell_sizes": reference_sizes.tolist(),
        "header_exact": bool(np.array_equal(header.numpy(), reference_header)),
        "sizes_exact": bool(np.array_equal(sizes.numpy(), reference_sizes)),
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
