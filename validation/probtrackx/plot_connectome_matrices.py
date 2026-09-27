"""Plot FSL and FNIT matrix1/2/3 from paired real-DWI runs."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))
from probtrackx_matrix_current import _mapped_edges, _plot_matrices  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Each tracking mode was timed in its own process. These directories only
    # collect links for the matched visualization; counts remain in run-dir.
    fsl = args.run_dir / "plot_fsl"
    fnit = args.run_dir / f"plot_fnit_{args.device.replace(':', '_')}"
    for target, prefix in ((fsl, "fsl_union_matrix"),
                           (fnit, f"fnit_runs/fnit_{args.device}_matrix")):
        target.mkdir(exist_ok=True)
        for number in (1, 2, 3):
            source = args.run_dir / f"{prefix}{number}"
            for filename in (f"fdt_matrix{number}.dot", f"coords_for_fdt_matrix{number}"):
                link = target / filename
                if not link.exists():
                    link.symlink_to(source / filename)
            if number == 2:
                name = "tract_space_coords_for_fdt_matrix2"
                link = target / name
                if not link.exists():
                    link.symlink_to(source / name)
        _mapped_edges(target, 1)
    _plot_matrices(fsl, fnit, [1, 2, 3], args.output)
    print(args.output)


if __name__ == "__main__":
    main()
