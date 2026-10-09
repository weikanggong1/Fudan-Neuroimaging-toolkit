"""读取固定 Conda CMake 构建命令，在独立目录编译诊断副本。

支持 Unix Makefiles 和 Ninja；不修改上游源码、构建对象或安装程序。
source/build 是已有固定源码及构建目录；patched/output 是私有诊断路径。
命令和库来源必须另存收据，诊断程序需通过完整输出回归后才能用作参考。
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def _compiler(build: Path) -> str:
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith("CMAKE_CXX_COMPILER:"):
            return line.split("=", 1)[1]
    raise RuntimeError("fixed build lacks CMAKE_CXX_COMPILER")


def _ninja_variables(build: Path, target: str) -> dict[str, str]:
    lines = (build / "build.ninja").read_text().splitlines()
    prefix = f"build {target}:"
    matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one Ninja target: {target}")
    result = {}
    for line in lines[matches[0] + 1 :]:
        if not line.startswith("  "):
            break
        if " = " in line:
            key, value = line.strip().split(" = ", 1)
            if "$" in value:
                raise RuntimeError("unresolved Ninja variable; use expanded compile commands")
            result[key] = value
    return result


def compile_probe(*, source: Path, build: Path, target: str,
                  patched: Path, output: Path) -> list[str]:
    """沿用原编译器、宏、include 和优化参数，编译一个复制的翻译单元。"""
    if (build / "build.ninja").exists():
        values = _ninja_variables(build, target)
        flags = [token for key in ("DEFINES", "INCLUDES", "FLAGS")
                 for token in shlex.split(values.get(key, ""))]
    else:
        values = {}
        for line in (build / Path(target).parent / "flags.make").read_text().splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                values[key] = value
        flags = [token for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS")
                 for token in shlex.split(values.get(key, ""))]
    flags = [token.replace("/tmp/fs_full_source_d932", str(source)) for token in flags]
    command = [_compiler(build), *flags, "-I", str(source / "include"),
               "-I", str(patched.parent), "-I", str(source / "utils"),
               "-I", str(source / "mris_make_surfaces"),
               "-c", str(patched), "-o", str(output)]
    subprocess.run(command, cwd=build, check=True)
    return command


def link_probe(*, build: Path, main_object: Path, output: Path,
               extra_objects: tuple[Path, ...] = ()) -> list[str]:
    """诊断对象先于固定 libutils.a 链接；其他原生库及链接参数原样保留。"""
    if (build / "build.ninja").exists():
        values = _ninja_variables(build, "mris_make_surfaces/mris_place_surface")
        command = [_compiler(build), *shlex.split(values.get("FLAGS", "")),
                   *shlex.split(values.get("LINK_FLAGS", "")), str(main_object),
                   "-o", str(output), *[str(path) for path in extra_objects],
                   *shlex.split(values["LINK_LIBRARIES"])]
        cwd = build
    else:
        target = build / "mris_make_surfaces/CMakeFiles/mris_place_surface.dir"
        command = shlex.split((target / "link.txt").read_text())
        original = "CMakeFiles/mris_place_surface.dir/mris_place_surface.cpp.o"
        if command.count(original) != 1 or command.count("../utils/libutils.a") != 1:
            raise RuntimeError("unrecognized fixed Make link command")
        command[command.index(original)] = str(main_object)
        index = command.index("../utils/libutils.a")
        command[index:index] = [str(path) for path in extra_objects]
        command[command.index("-o") + 1] = str(output)
        cwd = build / "mris_make_surfaces"
    command = [f"-Wl,-Map,{output.parent / 'link.map'}"
               if token == "-Wl,-Map,ld_map.txt" else token for token in command]
    subprocess.run(command, cwd=cwd, check=True)
    return command
