"""Place LH white.preaparc on a candidate subject and smooth it three times."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fnit.recon_all.smooth_surface_python import smooth_surface
from fnit.recon_all.white_preaparc_conda import run_white_preaparc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    subject = args.subject.resolve()
    if (subject / "surf/lh.white.preaparc").exists():
        raise FileExistsError(subject / "surf/lh.white.preaparc")
    result = run_white_preaparc(subject, "lh", args.binary, args.assets,
                                threads=4)
    start = time.perf_counter()
    smooth_surface(subject / "surf/lh.white.preaparc",
                   subject / "surf/lh.smoothwm", iterations=3, device="cpu")
    result["smoothwm_seconds"] = time.perf_counter() - start
    result["binary_realpath"] = str(args.binary.resolve())
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(args.report)


if __name__ == "__main__":
    main()
