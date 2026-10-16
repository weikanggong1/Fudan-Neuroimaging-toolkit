"""Build a read-only first-pass border search probe from pinned FreeSurfer source.

The copied main program writes the MRI and vertex state immediately around the
first MRIScomputeBorderValues call. It is a validation tool, not a dependency.
"""

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

    original = (args.source / "mris_make_surfaces/mris_place_surface.cpp").read_text()
    before = "MRIScomputeBorderValues(surf, involCBV, NULL, inside_hi,border_hi,border_low,outside_low,outside_hi,"
    if original.count(before) != 1:
        raise RuntimeError("expected one border search call")
    help_include = '#include "mris_place_surface.help.xml.h"'
    if original.count(help_include) != 1:
        raise RuntimeError("expected one generated help header")
    original = original.replace(
        help_include,
        'static const unsigned char mris_place_surface_help_xml[] = "";\n'
        'static const unsigned int mris_place_surface_help_xml_len = 0;',
    )
    classifier_call = "MRI *mri_labeled = MRIfindBrightNonWM(invol, wm) ;"
    if original.count(classifier_call) != 1:
        raise RuntimeError("expected one bright non-WM classifier")
    original = original.replace(
        classifier_call,
        classifier_call + r'''
    if (getenv("PLACE_CBV_PREFIX")) {
      char path[STRLEN];
      snprintf(path, STRLEN, "%s.bright_labels.mgz", getenv("PLACE_CBV_PREFIX"));
      MRIwrite(mri_labeled, path);
    }''',
    )
    dump = r'''auto dump_cbv = [&](const char *suffix) {
          const char *prefix = getenv("PLACE_CBV_PREFIX");
          if (!prefix || i != 0) return;
          char path[STRLEN];
          snprintf(path, STRLEN, "%s.%s", prefix, suffix);
          FILE *output = fopen(path, "wb");
          if (!output) exit(1);
          for (int vertex = 0; vertex < surf->nvertices; vertex++) {
            const VERTEX &v = surf->vertices[vertex];
            float values[] = {v.x, v.y, v.z, v.nx, v.ny, v.nz,
                              v.origx, v.origy, v.origz, v.val, v.d,
                              v.mean, v.targx, v.targy, v.targz, v.val2};
            int flags[] = {v.ripflag, v.marked};
            fwrite(values, sizeof(values), 1, output);
            fwrite(flags, sizeof(flags), 1, output);
          }
          fclose(output);
        };
        dump_cbv("before");
        if (i == 0 && getenv("PLACE_CBV_PREFIX")) {
          char path[STRLEN];
          snprintf(path, STRLEN, "%s.volume.mgz", getenv("PLACE_CBV_PREFIX"));
          MRIwrite(involCBV, path);
          snprintf(path, STRLEN, "%s.transform", getenv("PLACE_CBV_PREFIX"));
          FILE *matrix_output = fopen(path, "wb");
          if (!matrix_output) exit(1);
          MRIS_SurfRAS2VoxelMap *map = MRIS_makeRAS2VoxelMap(involCBV, surf);
          for (int row = 1; row <= 4; row++) {
            for (int col = 1; col <= 4; col++) {
              float entry = *MATRIX_RELT(map->sras2vox, row, col);
              fwrite(&entry, sizeof(entry), 1, matrix_output);
            }
          }
          fclose(matrix_output);
          MRIS_freeRAS2VoxelMap(&map);
        }
        '''
    original = original.replace(before, dump + before)
    after = "current_sigma, 2*max_cbv_dist, parms.fp, surftype, stopmask, 0.5, parms.flags,seg,-1,-1) ;"
    if original.count(after) != 1:
        raise RuntimeError("expected one border search call ending")
    original = original.replace(after, after + '\n        dump_cbv("after");')
    rip_entry = "int RIP_MNGR::RipVertices(void)\n{"
    if original.count(rip_entry) != 1:
        raise RuntimeError("expected one RIP_MNGR::RipVertices")
    original = original.replace(rip_entry, rip_entry + r'''
  static int rip_call = 0;
  rip_call++;
  auto dump_rip = [&](const char *stage) {
    const char *prefix = getenv("PLACE_RIP_PREFIX");
    if (!prefix) return;
    char path[STRLEN];
    snprintf(path, STRLEN, "%s.call%d.%s", prefix, rip_call, stage);
    FILE *output = fopen(path, "wb");
    if (!output) exit(1);
    for (int vertex = 0; vertex < surf->nvertices; vertex++) {
      const VERTEX &v = surf->vertices[vertex];
      int flags[] = {v.ripflag, v.marked, v.marked2};
      fwrite(flags, sizeof(flags), 1, output);
      fwrite(&v.val, sizeof(v.val), 1, output);
    }
    fclose(output);
  };
  dump_rip("entry");''')
    for needle, stage in (
        ("  int ripsurfneeded = 0;", "after_label"),
        ("  if(RipBG){\n    // probably want to use white for this", "after_midline"),
        ("  if(nRipSegs){\n    // probably want to use white for this", "after_bg"),
        ("  if(ripsurffile){\n    // Copy ripflags back into the input surface", "after_seg"),
    ):
        if original.count(needle) != 1:
            raise RuntimeError(f"expected one rip stage {stage}")
        original = original.replace(needle, f'  dump_rip("{stage}");\n' + needle)
    patched = args.out / "mris_place_surface_border_probe.cpp"
    patched.write_text(original)

    target = args.build / "mris_make_surfaces/CMakeFiles/mris_place_surface.dir"
    definitions = {}
    for line in (target / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            definitions[key] = shlex.split(value)
    compiler = "/home1/benchmark-user/anaconda3/bin/x86_64-conda-linux-gnu-g++"
    compiled = args.out / "mris_place_surface_border_probe.o"
    command = [compiler]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        command += definitions[key]
    command += ["-c", str(patched), "-o", str(compiled)]
    subprocess.run(command, check=True)

    link = shlex.split((target / "link.txt").read_text())
    link = [
        str(compiled) if token == "CMakeFiles/mris_place_surface.dir/mris_place_surface.cpp.o"
        else f"-Wl,-Map,{args.out / 'link.map'}" if token == "-Wl,-Map,ld_map.txt"
        else token
        for token in link
    ]
    link[link.index("-o") + 1] = str(args.out / "mris_place_surface_border_probe")
    subprocess.run(link, cwd=args.build / "mris_make_surfaces", check=True)
    print(args.out / "mris_place_surface_border_probe")


if __name__ == "__main__":
    main()
