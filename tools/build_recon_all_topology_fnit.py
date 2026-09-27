"""Build an isolated FreeSurfer 8.2 topology binary with two numeric fixes.

This edits neither the pinned source tree nor its existing Conda build.  It
requires a completed Ninja build of ``mris_fix_topology`` and the same Conda
compiler that produced that build.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build, source, output = (path.resolve() for path in
                             (args.build, args.source, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    commands = subprocess.check_output(
        ["ninja", "-t", "commands", "mris_fix_topology"],
        cwd=build, text=True).splitlines()
    # Reject source drift: these two source files were validated from the
    # pinned FreeSurfer 8.2 commit, not arbitrary compatible source.
    expected_sha256 = {
        "mrisurf_metricProperties.cpp":
            "458f896e7d2bdcbcdd7c9e73b2a396823d2015c723117d604bb06e3e8f27ba17",
        "mrisurf_defect.cpp":
            "55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187",
    }
    for filename, expected in expected_sha256.items():
        actual = hashlib.sha256((source / "utils" / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"source drift in {filename}: {actual}")
    replacements = {
        "mrisurf_metricProperties.cpp": (
            "  MRIScenterSphere_old(mris);",
            '  MRIScenterSphere_old(mris);\n'
            '  const char* centered = getenv("FNIT_CENTERED_COORDS");\n'
            '  if (!centered || !*centered) {\n'
            '    fprintf(stderr, "FNIT_CENTERED_COORDS is required\\n");\n'
            '    abort();\n'
            '  }\n'
            '  if (MRISreadVertexPositions(mris, centered) != NO_ERROR) abort();\n'
            '  fprintf(stderr, "FNIT_CENTERED_SUBSTITUTION_USED\\n");'),
        "mrisurf_defect.cpp": (
            "tanh(F * (1 / r - E))",
            "tanh((double)(F * (1 / r - E)))"),
    }
    compiler = None
    archive = output / "libutils.a"
    shutil.copy2(build / "utils/libutils.a", archive)
    objects = []
    for filename, (needle, replacement) in replacements.items():
        original = source / "utils" / filename
        content = original.read_text()
        if content.count(needle) != 1:
            raise ValueError(f"expected one patch site in {original}")
        patched = output / filename
        patched.write_text(content.replace(needle, replacement))
        line = next(line for line in commands if " -c " in line
                    and line.rstrip().endswith(str(original)))
        cmd = shlex.split(line)
        cmd.insert(1, "-I" + str(original.parent))
        obj = output / (filename + ".o")
        cmd[cmd.index("-o") + 1] = str(obj)
        cmd[cmd.index("-c") + 1] = str(patched)
        for flag, suffix in (("-MF", ".d"), ("-MT", "")):
            if flag in cmd:
                cmd[cmd.index(flag) + 1] = str(obj) + suffix
        subprocess.run(cmd, cwd=build, check=True)
        compiler = Path(cmd[0])
        objects.append(obj)
    assert compiler is not None
    ar = compiler.with_name("x86_64-conda-linux-gnu-ar")
    ranlib = compiler.with_name("x86_64-conda-linux-gnu-ranlib")
    subprocess.run([str(ar), "r", str(archive), *map(str, objects)], check=True)
    subprocess.run([str(ranlib), str(archive)], check=True)
    link = commands[-1].split("&&")[1].strip()
    tokens = shlex.split(link)
    tokens = [str(archive) if token == "utils/libutils.a" else token for token in tokens]
    binary = output / "mris_fix_topology_fnit"
    tokens[tokens.index("-o") + 1] = str(binary)
    subprocess.run(tokens, cwd=build, check=True)
    print(binary)


if __name__ == "__main__":
    main()
