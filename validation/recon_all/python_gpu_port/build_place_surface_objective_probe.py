"""Build copied-source diagnostic for full-precision first-step RMS and SSE."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--main-object", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture-steps", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = (args.source / "utils/mrisurf_mri.cpp").read_text()
    initial = "  showDtSSeRms(parms->fp, -1, 0.0, sse, rms, -1.0, __LINE__);"
    step = "    showDtSSeRms(parms->fp, n, delta_t, sse, rms, last_rms, __LINE__);"
    if source.count(initial) != 1 or source.count(step) != 2:
        raise RuntimeError("unexpected FreeSurfer placement source markers")
    source = source.replace(
        initial,
        '  fprintf(stderr, "PY_OBJ_REF initial rms=%.17g sse=%.17g orig_area=%.17g total_area=%.17g\\n", rms, sse, mris->orig_area, mris->total_area);\n' + initial,
    )
    source = source.replace(
        step,
        f'    if (n < {args.capture_steps}) fprintf(stderr, "PY_OBJ_REF step%d rms=%.17g sse=%.17g orig_area=%.17g total_area=%.17g\\n", n + 1, rms, sse, mris->orig_area, mris->total_area);\n' + step,
        1,
    )
    patched = args.out / "mrisurf_mri_objective_probe.cpp"
    patched.write_text(source)
    target = args.build / "utils/CMakeFiles/utils.dir"
    definitions = {}
    for line in (target / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            definitions[key] = shlex.split(value)
    compiled = args.out / "mrisurf_mri_objective_probe.o"
    compiler = "/home1/benchmark-user/anaconda3/bin/x86_64-conda-linux-gnu-g++"
    command = [compiler]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        command += definitions[key]
    command += ["-I", str(args.source / "utils"), "-c", str(patched), "-o", str(compiled)]
    subprocess.run(command, check=True)
    link = shlex.split((args.build / "mris_make_surfaces/CMakeFiles/mris_place_surface.dir/link.txt").read_text())
    link = [
        str(args.main_object) if token == "CMakeFiles/mris_place_surface.dir/mris_place_surface.cpp.o"
        else f"-Wl,-Map,{args.out / 'link.map'}" if token == "-Wl,-Map,ld_map.txt"
        else token
        for token in link
    ]
    link.insert(link.index("../utils/libutils.a"), str(compiled))
    link[link.index("-o") + 1] = str(args.out / "mris_place_surface_objective_probe")
    subprocess.run(link, cwd=args.build / "mris_make_surfaces", check=True)
    print(args.out / "mris_place_surface_objective_probe")


if __name__ == "__main__":
    main()
