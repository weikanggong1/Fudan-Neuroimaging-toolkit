"""追加同一固定源码 MRI 采样观察，不修改生产或已完成观察版本。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, native_build, base_observer, output):
    native_build, base_observer, output = map(Path, (native_build, base_observer, output))
    if output.exists():
        raise FileExistsError("fresh benchmark output required")
    receipt = json.loads((base_observer / "build.private.json").read_text())
    base = base_observer / "numeric_reference/mrisurf_defect.cpp"
    base_archive = base_observer / "numeric_reference/libutils.a"
    if sha(base) != receipt["patched_defect_sha256"] or sha(base_archive) != receipt["archive_sha256"]:
        raise ValueError("frozen preparation observer source/archive changed")
    output.mkdir(parents=True)
    text = base.read_text()
    start = text.index("static int mrisTessellateDefect_wkr(MRI_SURFACE *mris,", text.index("static int compare_edge_length"))
    begin = text.index("  n = 0;\n  ROMP_PF_begin", start)
    end = text.index("qsort(et,", begin)
    body = text[begin:end]
    body = body.replace("  n = 0;", '''  FILE* fnit_samples_file = NULL;
  const char* fnit_samples_path = getenv("FNIT_TOPOLOGY_SAMPLE_CAPTURE");
  if (fnit_samples_path) {
    fnit_samples_file = fopen(fnit_samples_path, "wb");
    if (!fnit_samples_file) abort();
  }
  n = 0;''', 1)
    pattern = re.compile(r"(MRISsurfaceRASToVoxelCached\(mris, mri, (x [+-] \.5 \* nx), (y [+-] \.5 \* ny), (z [+-] \.5 \* nz), &xv, &yv, &zv\);\n#endif\n      )MRIsampleVolume\(mri, xv, yv, zv, &val\);")

    def observe(match):
        coordinates = ", ".join(match.group(i) for i in [2, 3, 4])
        return match.group(1) + '''MRIsampleVolume(mri, xv, yv, zv, &val);
      if (fnit_samples_file) {
        int endpoints[2] = {vlist[i], vlist[j]};
        double observation[7] = {''' + coordinates + ''', xv, yv, zv, val};
        if (fwrite(endpoints, sizeof(int), 2, fnit_samples_file) != 2 ||
            fwrite(observation, sizeof(double), 7, fnit_samples_file) != 7) abort();
      }'''

    # Interior indentation differs but the six complete sampling sites share code.
    body = body.replace("\n        MRIsampleVolume(mri, xv, yv, zv, &val);", "\n      MRIsampleVolume(mri, xv, yv, zv, &val);")
    body, count = pattern.subn(observe, body)
    if count != 6:
        raise ValueError(f"expected all 6 endpoint/interior sample sites, found {count}")
    body += "  if (fnit_samples_file && fclose(fnit_samples_file) != 0) abort();\n  "
    patched = output / "mrisurf_defect.cpp"
    patched.write_text(text[:begin] + body + text[end:])
    command = list(receipt["compile_argv"])
    obj = output / "mrisurf_defect.cpp.o"
    for flag, value in [("-o", str(obj)), ("-c", str(patched)), ("-MF", str(obj) + ".d"), ("-MT", str(obj))]:
        command[command.index(flag) + 1] = value
    archive = output / "libutils.a"
    shutil.copyfile(base_archive, archive)
    link = [str(archive) if token == str(base_archive) else token for token in receipt["link_argv"]]
    binary = output / "mris_fix_topology_sample_observer"
    link[link.index("-o") + 1] = str(binary)
    compiler = Path(command[0])
    started = time.perf_counter()
    with (output / "build.log").open("w") as log:
        subprocess.run(command, cwd=native_build / "build", stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ar")), "r", str(archive), str(obj)], check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ranlib")), str(archive)], check=True)
        subprocess.run(link, cwd=native_build / "build", stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(status="complete_benchmark_sample_observer", scope="read-only 22-sample trace; no production/GA change",
                  base_observer_receipt_sha256=sha(base_observer / "build.private.json"),
                  builder_sha256=sha(__file__), patched_defect_sha256=sha(patched), archive_sha256=sha(archive),
                  observer_program_sha256=sha(binary), compile_argv=command, link_argv=link,
                  seconds=time.perf_counter() - started,
                  binary_format="64-byte records: int32 source endpoints(2), float64 RAS(3)/voxel(3)/sample(1)")
    (output / "build.private.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["native-build", "base-observer", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
