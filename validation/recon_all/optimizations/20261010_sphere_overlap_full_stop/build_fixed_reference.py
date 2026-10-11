"""从已有固定源码Conda build新建参考程序；只供独立benchmark。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

COMMIT = "d932c45b7941662ea380a05efef580568b98d41a"
EXPECTED = {
    "mris_sphere/mris_sphere.cpp": "a3c9540ba926ead9541ff3703fb1ad9a2cdd431ff09804c234f5328054edcb2d",
    "mris_register/mris_register.cpp": "f728018bef459b65b70e40d2b789e15cbdf9584fd9389e8fa0ce145850c0a7f5",
    "utils/mrisurf_timeStep.cpp": "402bfd0bfdb5cf0bc59600f30592facf6a60890e78cdff3fa696f7fcec4ca553",
    "utils/mrisurf_metricProperties.cpp": "458f896e7d2bdcbcdd7c9e73b2a396823d2015c723117d604bb06e3e8f27ba17",
    "include/mrisurf.h": "810f60f832a19d07580f7ce2e4d8c04e899069b03726eb9cca4e8553491907be",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--help-build", type=Path, default=None,
                        help="optional completed Ninja build with matching generated XML help headers")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native, output = args.native_build.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("new reference output must be empty and absent")
    provenance = (native / "build-provenance.txt").read_text()
    if f"SOURCE_COMMIT={COMMIT}" not in provenance:
        raise ValueError("reference source commit mismatch")
    source, build = native / "source", native / "build"
    hashes = {name: sha(source / name) for name in EXPECTED}
    if hashes != EXPECTED:
        raise ValueError("fixed source SHA-256 mismatch")
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    compiler = Path(next(line.split("=", 1)[1] for line in cache
                         if line.startswith("CMAKE_CXX_COMPILER:"))).resolve()
    prefix = compiler.parent.parent
    ninja = prefix / "bin/ninja"
    compiler.relative_to(prefix)
    output.mkdir(parents=True)
    (output / "bin").mkdir()
    (output / "archives").mkdir()
    (output / "generated").mkdir()
    report = {"status": "running", "scope": "benchmark-only; fresh mains with copied existing fixed-source Conda archives",
              "source_commit": COMMIT, "source_selected_sha256": hashes,
              "build_provenance": provenance, "compiler": str(compiler),
              "compiler_sha256": sha(compiler), "builder_sha256": sha(Path(__file__)),
              "affinity": sorted(os.sched_getaffinity(0)), "programs": {}, "archives": {},
              "generated_help": {}}
    started = time.perf_counter()
    for target in ["mris_sphere", "mris_register"]:
        lines = subprocess.check_output([str(ninja), "-t", "commands", target],
                                        cwd=build, text=True).splitlines()
        main_source = source / target / (target + ".cpp")
        compile_line = next(line for line in lines if " -c " in line
                            and line.endswith(str(main_source)))
        command = shlex.split(compile_line)
        Path(command[0]).resolve().relative_to(prefix)
        help_build = args.help_build.resolve() if args.help_build is not None else build
        help_header = help_build / target / (target + ".help.xml.h")
        if args.help_build is not None:
            current_xml = source / target / (target + ".help.xml")
            previous_xml = help_build.parent / "source" / target / (target + ".help.xml")
            if sha(current_xml) != sha(previous_xml):
                raise ValueError("generated help comes from different XML source")
            report["generated_help"][target] = {"xml_sha256": sha(current_xml),
                                                 "header_sha256": sha(help_header)}
        shutil.copy2(help_header, output / "generated" / help_header.name)
        command.insert(1, "-I" + str(output / "generated"))
        old_object = command[command.index("-o") + 1]
        obj = output / (target + ".o")
        for flag, value in [("-o", str(obj)), ("-MF", str(obj) + ".d"), ("-MT", str(obj))]:
            command[command.index(flag) + 1] = value
        link = shlex.split(lines[-1].split("&&")[1].strip())
        Path(link[0]).resolve().relative_to(prefix)
        binary = output / "bin" / target
        link[link.index("-o") + 1] = str(binary)
        for index, token in enumerate(link):
            if token == old_object:
                link[index] = str(obj)
            elif token.endswith(".a"):
                original = (build / token).resolve()
                original.relative_to(build.resolve())
                archive = output / "archives" / original.name
                if not archive.exists():
                    shutil.copy2(original, archive)
                if sha(original) != sha(archive):
                    raise ValueError("archive changed while copying")
                report["archives"][token] = {"sha256": sha(archive), "bytes": archive.stat().st_size}
                link[index] = str(archive)
            elif token.startswith("-Wl,-Map,"):
                link[index] = "-Wl,-Map," + str(output / (target + ".map"))
        tick = time.perf_counter()
        with (output / (target + ".build.log")).open("w") as log:
            subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(link, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
        dependencies = subprocess.check_output(["ldd", str(binary)], text=True)
        if "not found" in dependencies:
            raise RuntimeError("reference has unresolved dynamic dependency")
        (output / (target + ".ldd")).write_text(dependencies)
        report["programs"][target] = {"sha256": sha(binary), "bytes": binary.stat().st_size,
                                     "seconds": time.perf_counter() - tick,
                                     "compile_argv": command, "link_argv": link}
        (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    report.update(status="complete", total_seconds=time.perf_counter() - started)
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print("REFERENCE_BUILD_COMPLETE")


if __name__ == "__main__":
    main()
