"""Instrument copied pinned source at the first collision decision (diagnostic only)."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


BLOCKED_TO_TRACE = (52, 1781, 3045, 9016, 49048, 62101, 66825, 71757, 74856, 87179, 92155)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--main-object", type=Path, required=True)
    parser.add_argument("--gradient-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original = (args.source / "utils/mrisurf_timeStep.cpp").read_text()
    needle = "  if (MHTisVectorFilled(mht, vno, v->odx, v->ody, v->odz)) {"
    if original.count(needle) != 1:
        raise RuntimeError("missing single native collision call")
    condition = " || ".join(f"vno == {vertex}" for vertex in BLOCKED_TO_TRACE)
    replacement = f'''  int place_collision = MHTisVectorFilled(mht, vno, v->odx, v->ody, v->odz);
  if (place_collision && getenv("PLACE_COLLISION_PREFIX") && ({condition})) {{
    char path[STRLEN];
    snprintf(path, STRLEN, "%s.%d", getenv("PLACE_COLLISION_PREFIX"), vno);
    FILE *snapshot = fopen(path, "wb");
    if (!snapshot) exit(1);
    float step[] = {{v->odx, v->ody, v->odz}};
    fwrite(step, sizeof(step), 1, snapshot);
    for (int i = 0; i < mris->nvertices; i++) {{
      const VERTEX &other = mris->vertices[i];
      float xyz[] = {{other.x, other.y, other.z}};
      fwrite(xyz, sizeof(xyz), 1, snapshot);
    }}
    fclose(snapshot);
  }}
  if (place_collision) {{'''
    original = original.replace(needle, replacement)
    marker = "  // Merge the per thread subvolumes into the tid==0 subvolume"
    if original.count(marker) != 1:
        raise RuntimeError("missing native subvolume merge")
    dump = r'''  if (getenv("PLACE_SUBVOLUME_PREFIX")) {
    char path[STRLEN];
    snprintf(path, STRLEN, "%s.subvols", getenv("PLACE_SUBVOLUME_PREFIX"));
    FILE *output = fopen(path, "wb");
    if (!output) exit(1);
    int counts[] = {mris->nvertices, mris->nfaces};
    fwrite(counts, sizeof(counts), 1, output);
    float geometry[] = {
      allVertexsContext.xLo, allVertexsContext.xHi,
      allVertexsContext.yLo, allVertexsContext.yHi,
      allVertexsContext.zLo, allVertexsContext.zHi,
      allVertexsContext.xSubvolLen, allVertexsContext.ySubvolLen,
      allVertexsContext.zSubvolLen,
      allVertexsContext.xSubvolVerge, allVertexsContext.ySubvolVerge,
      allVertexsContext.zSubvolVerge,
    };
    fwrite(geometry, sizeof(geometry), 1, output);
    fwrite(vnoToSvi, sizeof(int), mris->nvertices, output);
    for (int fno = 0; fno < mris->nfaces; fno++) {
      int value = faceInfos[fno].svi;
      fwrite(&value, sizeof(value), 1, output);
    }
    for (int fno = 0; fno < mris->nfaces; fno++) {
      int value = mris->faces[fno].ripflag;
      fwrite(&value, sizeof(value), 1, output);
    }
    fclose(output);
  }
'''
    original = original.replace(marker, dump + marker)
    patched = args.out / "mrisurf_timeStep_collision_probe.cpp"
    patched.write_text(original)
    target = args.build / "utils/CMakeFiles/utils.dir"
    definitions = {}
    for line in (target / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            definitions[key] = shlex.split(value)
    compiler = "/home1/benchmark-user/anaconda3/bin/x86_64-conda-linux-gnu-g++"
    compiled = args.out / "mrisurf_timeStep_collision_probe.o"
    command = [compiler]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        command += definitions[key]
    command += ["-I", str(args.source / "utils"), "-c", str(patched), "-o", str(compiled)]
    subprocess.run(command, check=True)
    link = shlex.split((args.build / "mris_make_surfaces/CMakeFiles/mris_place_surface.dir/link.txt").read_text())
    replacements = {
        "CMakeFiles/mris_place_surface.dir/mris_place_surface.cpp.o": str(args.main_object),
        "-Wl,-Map,ld_map.txt": f"-Wl,-Map,{args.out / 'link.map'}",
    }
    link = [replacements.get(token, token) for token in link]
    for name in (
        "mrisurf_mri_gradient_probe.o",
        "mrisurf_compute_dxyz_spring_probe.o",
        "mrisurf_metricProperties_average_probe.o",
    ):
        link.insert(link.index("../utils/libutils.a"), str(args.gradient_probe / name))
    link.insert(link.index("../utils/libutils.a"), str(compiled))
    output = args.out / "mris_place_surface_collision_probe"
    link[link.index("-o") + 1] = str(output)
    subprocess.run(link, cwd=args.build / "mris_make_surfaces", check=True)
    print(output)


if __name__ == "__main__":
    main()
