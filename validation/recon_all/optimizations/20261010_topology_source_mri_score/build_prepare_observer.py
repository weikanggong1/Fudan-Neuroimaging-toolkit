"""构建仅 benchmark 使用的固定源码准备态观察器；不修改已有 build。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

FIXED_COMMIT = "d932c45b7941662ea380a05efef580568b98d41a"
FIXED_DEFECT_SHA = "55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187"

# 仅写出当时实际字段，不改变原几何/强度/标记/排序/随机状态。
# 此记录器不属于 FNIT 生产算法；原程序仍在后续 qsort interposer 停下。
OBSERVE = r'''
  static bool fnit_preparation_captured = false;
  const char* fnit_preparation_path = getenv("FNIT_TOPOLOGY_PREPARATION_CAPTURE");
  if (fnit_preparation_path && !fnit_preparation_captured) {
    fnit_preparation_captured = true;
    FILE* capture = fopen(fnit_preparation_path, "wb");
    if (!capture) abort();
    int header[5] = {mris->nvertices, mris->nfaces, mri->width, mri->height, mri->depth};
    if (fwrite(header, sizeof(int), 5, capture) != 5) abort();
    MRIS_SurfRAS2VoxelMap* map = MRIS_makeRAS2VoxelMap(mri, mris);
    for (int row = 1; row <= 4; row++)
      for (int col = 1; col <= 4; col++) {
        float entry = *MATRIX_RELT(map->sras2vox, row, col);
        if (fwrite(&entry, sizeof(float), 1, capture) != 1) abort();
      }
    MRIS_freeRAS2VoxelMap(&map);
    for (int vertex = 0; vertex < mris->nvertices; vertex++) {
      VERTEX* vv = &mris->vertices[vertex];
      float original_normal[3] = {0, 0, 0};
      mrisComputeOrigNormal(mris, vertex, original_normal);
      float fields[16] = {vv->origx, vv->origy, vv->origz,
        original_normal[0], original_normal[1], original_normal[2],
        vv->val2, vv->val2bak, vv->x, vv->y, vv->z,
        vv->nx, vv->ny, vv->nz, (float)vv->marked, (float)vv->ripflag};
      if (fwrite(fields, sizeof(float), 16, capture) != 16) abort();
    }
    for (int face = 0; face < mris->nfaces; face++)
      if (fwrite(mris->faces[face].v, sizeof(int), 3, capture) != 3) abort();
    if (fclose(capture) != 0) abort();
    fprintf(stderr, "FNIT_TOPOLOGY_PREPARATION_CAPTURE_USED\n");
  }
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_observer(*, native_build: Path, fixed_builder: Path, output: Path) -> dict:
    """完整独立编译；输入固定 build/现有 FNIT 三补丁脚本，输出新二进制。

    native_build 内含 source、build、build-provenance.txt；fixed_builder 是
    tools/build_recon_all_topology_fnit.py，不从安装目录复制程序。output 必须
    不存在。先以同一 Conda 编译器生成已有数值适配，再仅添加字段记录。
    输出 observer 与 build.private.json，记录代码/编译器/程序/命令 SHA。
    此程序只供隔离 benchmark；无生产 API、MRI 产物或 GA 完成声明。
    来源/输出冲突/编译/依赖检查失败均抛异常，保留已有记录。
    """
    native_build = native_build.resolve()
    source, build = native_build / "source", native_build / "build"
    if output.exists():
        raise FileExistsError("observer requires a fresh output directory")
    if "SOURCE_COMMIT=" + FIXED_COMMIT not in (native_build / "build-provenance.txt").read_text():
        raise ValueError("fixed source commit mismatch")
    original = source / "utils/mrisurf_defect.cpp"
    if sha(original) != FIXED_DEFECT_SHA:
        raise ValueError("fixed defect source SHA mismatch")
    compiler = Path(next(line.split("=", 1)[1] for line in (build / "CMakeCache.txt").read_text().splitlines()
                         if line.startswith("CMAKE_CXX_COMPILER:"))).resolve()
    ninja = compiler.parent / "ninja"
    output.mkdir(parents=True)
    started = time.perf_counter()
    numeric = output / "numeric_reference"
    with (output / "numeric.build.log").open("w") as log:
        subprocess.run([sys.executable, str(fixed_builder), "--build", str(build),
                        "--source", str(source), "--output-dir", str(numeric)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    text = (numeric / "mrisurf_defect.cpp").read_text()
    start = text.index("static int mrisTessellateDefect_wkr(MRI_SURFACE *mris,", text.index("static int compare_edge_length"))
    site = text.index("  n = 0;\n  ROMP_PF_begin", start)
    patched = numeric / "mrisurf_defect.cpp"
    patched.write_text(text[:site] + OBSERVE + text[site:])
    lines = subprocess.check_output([str(ninja), "-t", "commands", "mris_fix_topology"],
                                    cwd=build, text=True).splitlines()
    command = shlex.split(next(line for line in lines if " -c " in line
                              and line.rstrip().endswith("/utils/mrisurf_defect.cpp")))
    command.insert(1, "-I" + str(original.parent))
    obj = numeric / "mrisurf_defect.cpp.o"
    for flag, value in [("-o", str(obj)), ("-c", str(patched)),
                        ("-MF", str(obj) + ".d"), ("-MT", str(obj))]:
        command[command.index(flag) + 1] = value
    archive = numeric / "libutils.a"
    link = shlex.split(lines[-1].split("&&")[1].strip())
    link = [str(archive) if token == "utils/libutils.a" else token for token in link]
    binary = output / "mris_fix_topology_preparation_observer"
    link[link.index("-o") + 1] = str(binary)
    with (output / "observer.build.log").open("w") as log:
        subprocess.run(command, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ar")), "r", str(archive), str(obj)], check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ranlib")), str(archive)], check=True)
        subprocess.run(link, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
    dependencies = subprocess.check_output(["ldd", str(binary)], text=True)
    if "not found" in dependencies:
        raise RuntimeError("observer has unresolved libraries")
    report = dict(status="complete_benchmark_observer", scope="source preparation observation only; no production change",
                  source_commit=FIXED_COMMIT, fixed_defect_sha256=sha(original),
                  fixed_builder_sha256=sha(fixed_builder), observer_builder_sha256=sha(Path(__file__)),
                  patched_defect_sha256=sha(patched), compiler_sha256=sha(compiler),
                  numeric_reference_sha256=sha(numeric / "mris_fix_topology_fnit"),
                  observer_program_sha256=sha(binary), archive_sha256=sha(archive),
                  compile_argv=command, link_argv=link, library_dependencies=dependencies,
                  seconds=time.perf_counter() - started,
                  binary_format="little endian native int32 header(N,F,X,Y,Z), float32 matrix(4,4), float32 vertex fields(N,16), int32 ordered faces(F,3)",
                  vertex_fields=["origx", "origy", "origz", "orig_nx", "orig_ny", "orig_nz", "white", "gray",
                                 "x", "y", "z", "nx", "ny", "nz", "marked", "ripped"])
    (output / "build.private.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["native-build", "fixed-builder", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    build_observer(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
