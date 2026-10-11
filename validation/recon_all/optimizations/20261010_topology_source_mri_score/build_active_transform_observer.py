"""只读记录 topology 实际 MATRIX_ALLOCATION 分支变换；不改生产程序。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, native_build, base_observer, output):
    """具名固定build、已测采样观察器、新目录→观察器/完整编译收据。"""
    native_build, base_observer, output = map(Path, (native_build, base_observer, output))
    if output.exists():
        raise FileExistsError("fresh benchmark output required")
    receipt = json.loads((base_observer / "build.private.json").read_text())
    base = base_observer / "mrisurf_defect.cpp"
    base_archive = base_observer / "libutils.a"
    if sha(base) != receipt["patched_defect_sha256"] or sha(base_archive) != receipt["archive_sha256"]:
        raise ValueError("frozen sample observer source/archive changed")
    text = base.read_text()
    anchor = '  static bool fnit_preparation_captured = false;'
    if text.count(anchor) != 1:
        raise ValueError("actual first-candidate preparation site mismatch")
    observation = r'''
  static bool fnit_active_matrix_captured=false;
  const char* fnit_active_matrix_path=getenv("FNIT_TOPOLOGY_ACTIVE_MATRIX_CAPTURE");
  if(fnit_active_matrix_path && !fnit_active_matrix_captured) {
    fnit_active_matrix_captured=true;
    #if MATRIX_ALLOCATION
    FILE* capture=fopen(fnit_active_matrix_path,"wb");
    if(!capture) abort();
    for(int row=1;row<=4;row++) for(int col=1;col<=4;col++) {
      float entry=*MATRIX_RELT(VoxelFromSRASmatrix,row,col);
      if(fwrite(&entry,sizeof(float),1,capture)!=1) abort();
    }
    if(fclose(capture)!=0) abort();
    fprintf(stderr,"FNIT_TOPOLOGY_ACTIVE_MATRIX_ALLOCATION_USED\n");
    #else
    fprintf(stderr,"FNIT_TOPOLOGY_ACTIVE_MATRIX_ALLOCATION_NOT_USED\n");
    #endif
  }
'''
    output.mkdir(parents=True)
    patched = output / "mrisurf_defect.cpp"
    patched.write_text(text.replace(anchor, observation + anchor, 1))
    command = list(receipt["compile_argv"])
    obj = output / "mrisurf_defect.cpp.o"
    for flag, value in [("-o", str(obj)), ("-c", str(patched)), ("-MF", str(obj) + ".d"), ("-MT", str(obj))]:
        command[command.index(flag) + 1] = value
    archive = output / "libutils.a"
    shutil.copyfile(base_archive, archive)
    link = [str(archive) if token == str(base_archive) else token for token in receipt["link_argv"]]
    binary = output / "mris_fix_topology_active_transform_observer"
    link[link.index("-o") + 1] = str(binary)
    compiler = Path(command[0])
    started = time.perf_counter()
    with (output / "build.log").open("w") as log:
        subprocess.run(command, cwd=native_build / "build", stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ar")), "r", str(archive), str(obj)], check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ranlib")), str(archive)], check=True)
        subprocess.run(link, cwd=native_build / "build", stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(status="complete_benchmark_active_transform_observer", scope="actual MATRIX_ALLOCATION transform observation; no production/GA change",
                  base_receipt_sha256=sha(base_observer / "build.private.json"), observer_builder_sha256=sha(__file__),
                  patched_defect_sha256=sha(patched), archive_sha256=sha(archive), observer_program_sha256=sha(binary),
                  compile_argv=command, link_argv=link, seconds=time.perf_counter() - started,
                  binary_format="16 float32 matrix entries in row-major order")
    (output / "build.private.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["native-build", "base-observer", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
