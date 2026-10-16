"""Compile the pinned, unmodified mris_place_surface source with frozen build libraries."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.build / "mris_make_surfaces/CMakeFiles/mris_place_surface.dir"
    if not (args.build / "mris_make_surfaces/mris_place_surface.help.xml.h").exists():
        (args.out / "mris_place_surface.help.xml.h").write_text(
            'static const unsigned char mris_place_surface_help_xml[] = "";\n'
            'static const unsigned int mris_place_surface_help_xml_len = 0;\n'
        )
    definitions = dict(
        line.split(" = ", 1)
        for line in (target / "flags.make").read_text().splitlines()
        if " = " in line
    )
    compiler = "/home1/benchmark-user/anaconda3/bin/x86_64-conda-linux-gnu-g++"
    source = args.source / "mris_make_surfaces/mris_place_surface.cpp"
    compiled = args.out / "mris_place_surface.cpp.o"
    command = [compiler]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        command += shlex.split(definitions[key])
    command += ["-I", str(args.out), "-c", str(source), "-o", str(compiled)]
    subprocess.run(command, cwd=args.out, check=True)
    link = shlex.split((target / "link.txt").read_text())
    link = [
        str(compiled) if token == "CMakeFiles/mris_place_surface.dir/mris_place_surface.cpp.o"
        else f"-Wl,-Map,{args.out / 'link.map'}" if token == "-Wl,-Map,ld_map.txt"
        else token for token in link
    ]
    link[link.index("-o") + 1] = str(args.out / "mris_place_surface_pristine")
    subprocess.run(link, cwd=args.build / "mris_make_surfaces", check=True)
    print(args.out / "mris_place_surface_pristine")


if __name__ == "__main__":
    main()
