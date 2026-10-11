"""在独立参考 prefix 中构建仅观察初始化的探针；不供生产使用。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError("fixed-source probe anchor is absent or ambiguous")
    return text.replace(anchor, replacement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    recipe = json.loads((args.reference_prefix / "build.json").read_text())
    program = recipe["programs"]["mris_sphere"]
    compile_args = list(program["compile_argv"])
    link_args = list(program["link_argv"])
    source = Path(compile_args[compile_args.index("-c") + 1])
    original = source.read_text()
    helper = r'''
static void fnit_initialization_observer(MRIS* mris, const char* phase) {
  const char* directory = getenv("FNIT_SPHERE_INITIALIZATION_OBSERVER");
  if (!directory) return;
  char output[4096];
  snprintf(output, sizeof(output), "%s/%s", directory, phase);
  MRISwrite(mris, output);
  printf("FNIT_INITIALIZATION_OBSERVER phase=%s xlo=%a xhi=%a ylo=%a yhi=%a zlo=%a zhi=%a xctr=%a yctr=%a zctr=%a abs_size=%zu abs_span=%a\n",
    phase, (double)mris->xlo, (double)mris->xhi, (double)mris->ylo, (double)mris->yhi,
    (double)mris->zlo, (double)mris->zhi, (double)mris->xctr, (double)mris->yctr,
    (double)mris->zctr, sizeof(abs(mris->xhi-mris->xlo)), (double)abs(mris->xhi-mris->xlo));
  fflush(stdout);
}
'''
    text = once(original, "int\nmain(int argc, char *argv[])", helper + "\nint\nmain(int argc, char *argv[])")
    text = once(text, "  MRISaddCommandLine(mris, cmdline) ;", "  fnit_initialization_observer(mris, \"01-read-inflated\");\n  MRISaddCommandLine(mris, cmdline) ;")
    text = once(text, "  if (smooth_avgs > 0)", "  fnit_initialization_observer(mris, \"02-original-properties\");\n  if (smooth_avgs > 0)")
    text = once(text, "  if (target_radius < 0)", "  fnit_initialization_observer(mris, \"03-size-cap\");\n  if (target_radius < 0)")
    anchor = "  MRISprojectOntoSphere(mris, mris, target_radius) ;"
    text = once(text, anchor, "  fnit_initialization_observer(mris, \"04-before-projection\");\n" + anchor
                + "\n  fnit_initialization_observer(mris, \"05-after-projection\");\n"
                + "  if (getenv(\"FNIT_SPHERE_INITIALIZATION_OBSERVER\")) return 0;")
    probe_source = args.output / "initialization_probe.cpp"
    probe_source.write_text(text)
    old_object = compile_args[compile_args.index("-o") + 1]
    obj = args.output / "initialization_probe.o"
    compile_args[compile_args.index("-c") + 1] = str(probe_source)
    for flag, value in [("-o", str(obj)), ("-MF", str(obj) + ".d"), ("-MT", str(obj))]:
        compile_args[compile_args.index(flag) + 1] = value
    link_args[link_args.index("-o") + 1] = str(args.output / "mris_sphere_initialization_probe")
    for index, token in enumerate(link_args):
        if token == old_object:
            link_args[index] = str(obj)
        elif token.startswith("-Wl,-Map,"):
            link_args[index] = "-Wl,-Map," + str(args.output / "initialization_probe.map")
    started = time.perf_counter()
    with (args.output / "build.log").open("w") as log:
        subprocess.run(compile_args, cwd=args.output, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(link_args, cwd=args.output, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = args.output / "mris_sphere_initialization_probe"
    report = {"status": "complete", "scope": "benchmark-only initialization observer with early exit before unfold",
              "fixed_original_source_sha256": sha(source), "instrumented_private_source_sha256": sha(probe_source),
              "builder_sha256": sha(Path(__file__)), "reference_recipe_sha256": sha(args.reference_prefix / "build.json"),
              "program_sha256": sha(binary), "compile_argv": compile_args, "link_argv": link_args,
              "build_seconds": time.perf_counter() - started}
    (args.output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print("INITIALIZATION_PROBE_BUILT", report["program_sha256"])


if __name__ == "__main__":
    main()
