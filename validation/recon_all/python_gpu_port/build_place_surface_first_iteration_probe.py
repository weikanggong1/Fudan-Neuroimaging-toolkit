"""Build an isolated FreeSurfer source probe that writes one optimizer step.

This is validation instrumentation, not a product dependency. It reuses the
already built pinned source libraries without changing their objects or source.
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
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--no-surface-snapshots", action="store_true")
    parser.add_argument("--stop-after-target-pass", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.build / "mris_make_surfaces/mris_place_surface.help.xml.h").exists():
        (args.out / "mris_place_surface.help.xml.h").write_text(
            'static const unsigned char mris_place_surface_help_xml[] = "";\n'
            'static const unsigned int mris_place_surface_help_xml_len = 0;\n'
        )
    original = (args.source / "mris_make_surfaces/mris_place_surface.cpp").read_text()
    if args.iterations > 1 and not args.no_surface_snapshots:
        show = "Gdiag |= DIAG_SHOW ;"
        if original.count(show) != 1:
            raise RuntimeError("missing source diagnostic flags")
        original = original.replace(show, "Gdiag |= DIAG_SHOW | DIAG_WRITE ;")
    for before, after in (
        ("parms.write_iterations = 0 /*WRITE_ITERATIONS */;", "parms.write_iterations = 1;"),
        ("parms.niterations = 100;", f'parms.niterations = {args.iterations}; strcpy(parms.base_name, "probe");'),
    ):
        if original.count(before) != 1:
            raise RuntimeError(f"expected one source occurrence: {before}")
        original = original.replace(before, after)
    marked_values_call = "MRISaverageMarkedVals(surf, vavgs) ;"
    if original.count(marked_values_call) != 1:
        raise RuntimeError("expected one marked-value averaging call")
    original = original.replace(
        marked_values_call,
        """auto dump_marked_values = [&](const char *suffix) {
            const char *prefix = getenv("PLACE_VALUES_PREFIX");
            if (!prefix || i != 0) return;
            char path[STRLEN];
            snprintf(path, STRLEN, "%s.%s", prefix, suffix);
            FILE *output = fopen(path, "wb");
            if (!output) exit(1);
            for (int vertex = 0; vertex < surf->nvertices; vertex++) {
              float value = surf->vertices[vertex].val;
              int marked = surf->vertices[vertex].marked;
              int ripped = surf->vertices[vertex].ripflag;
              fwrite(&value, sizeof(value), 1, output);
              fwrite(&marked, sizeof(marked), 1, output);
              fwrite(&ripped, sizeof(ripped), 1, output);
            }
            fclose(output);
          };
          dump_marked_values("before");
          MRISaverageMarkedVals(surf, vavgs) ;
          dump_marked_values("after");""",
    )
    if args.stop_after_target_pass is not None:
        outer_loop = "  for (i = 0 ;  n_averages >= n_min_averages ; n_averages /= 2, current_sigma /= 2, i++) {"
        if original.count(outer_loop) != 1:
            raise RuntimeError("missing outer placement loop")
        diagnostic = """
    auto dump_outer_state = [&](const char *stage) {
      const char *prefix = getenv("PLACE_OUTER_PREFIX");
      if (!prefix || i != STOP_PASS) return;
      char path[STRLEN];
      snprintf(path, STRLEN, "%s.%s", prefix, stage);
      FILE *output = fopen(path, "wb");
      if (!output) exit(1);
      for (int vertex = 0; vertex < surf->nvertices; vertex++) {
        const VERTEX &v = surf->vertices[vertex];
        float values[] = {v.x, v.y, v.z, v.nx, v.ny, v.nz,
                          v.val, v.val2, v.d, v.mean, v.targx, v.targy, v.targz};
        int flags[] = {v.ripflag, v.marked, v.border, v.cropped};
        fwrite(values, sizeof(values), 1, output);
        fwrite(flags, sizeof(flags), 1, output);
      }
      fclose(output);
    };
    dump_outer_state("entry");
""".replace("STOP_PASS", str(args.stop_after_target_pass))
        original = original.replace(outer_loop, outer_loop + diagnostic)
        after = '          dump_marked_values("after");'
        if original.count(after) != 1:
            raise RuntimeError("missing averaged target checkpoint")
        original = original.replace(
            after,
            after + '\n          dump_outer_state("after_target");'
                  + f'\n          if (i == {args.stop_after_target_pass}) exit(0);',
        )
    patched = args.out / "mris_place_surface_first_iteration.cpp"
    patched.write_text(original)

    target = args.build / "mris_make_surfaces/CMakeFiles/mris_place_surface.dir"
    definitions = {}
    for line in (target / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            definitions[key] = shlex.split(value)
    definitions["CXX_INCLUDES"] = [
        token.replace("/tmp/fs_full_source_d932", str(args.source))
        for token in definitions["CXX_INCLUDES"]
    ]
    compiler = "/home1/gongwk/anaconda3/bin/x86_64-conda-linux-gnu-g++"
    compiled = args.out / "mris_place_surface_first_iteration.o"
    command = [compiler]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        command += definitions[key]
    command += ["-I", str(args.source / "include"), "-I", str(args.source / "mris_make_surfaces"),
                "-c", str(patched), "-o", str(compiled)]
    subprocess.run(command, check=True)

    link = shlex.split((target / "link.txt").read_text())
    link = [
        str(compiled) if token == "CMakeFiles/mris_place_surface.dir/mris_place_surface.cpp.o"
        else f"-Wl,-Map,{args.out / 'link.map'}" if token == "-Wl,-Map,ld_map.txt"
        else token
        for token in link
    ]
    link[link.index("-o") + 1] = str(args.out / "mris_place_surface_first_iteration")
    subprocess.run(link, cwd=args.build / "mris_make_surfaces", check=True)
    print(args.out / "mris_place_surface_first_iteration")


if __name__ == "__main__":
    main()
