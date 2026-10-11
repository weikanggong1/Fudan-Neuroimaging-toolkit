"""只读观察固定参考的实际静态坐标缓存；不修改生产源码或已有参考。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

FIXED_MRI_SHA = "16740fce04268f8d4e7841592acf3ff89e394bf211e6c0ab0a7b541527a7c092"

# 只记录已有字段。该日志没有独立算法/生产 API；输出含机器私有位置，
# 发布前必须使用本目录脱敏导出器，不发布程序、MRI 或坐标二进制。
OBSERVE = r'''
    const char* fnit_cache_path = getenv("FNIT_TOPOLOGY_CACHE_CAPTURE");
    if (fnit_cache_path) {
      double sig[40] = {
        (double)mri->width, (double)mri->height, (double)mri->depth,
        mri->xsize, mri->ysize, mri->zsize,
        mri->x_r,mri->x_a,mri->x_s,mri->y_r,mri->y_a,mri->y_s,mri->z_r,mri->z_a,mri->z_s,
        mri->c_r,mri->c_a,mri->c_s,
        (double)mris->vg.valid,(double)mris->vg.width,(double)mris->vg.height,(double)mris->vg.depth,
        mris->vg.xsize,mris->vg.ysize,mris->vg.zsize,
        mris->vg.x_r,mris->vg.x_a,mris->vg.x_s,mris->vg.y_r,mris->vg.y_a,mris->vg.y_s,
        mris->vg.z_r,mris->vg.z_a,mris->vg.z_s,
        mris->vg.c_r,mris->vg.c_a,mris->vg.c_s,0,0,0};
      static double fnit_last_sig[40];
      static long long fnit_calls=0;
      long long fnit_call;
      #pragma omp atomic capture
      fnit_call=fnit_calls++;
      if (fnit_call < 64 || memcmp(sig, fnit_last_sig, sizeof(sig)) != 0) {
        #pragma omp critical(fnit_topology_cache_observation)
        {
          if (fnit_call < 64 || memcmp(sig, fnit_last_sig, sizeof(sig)) != 0) {
            memcpy(fnit_last_sig,sig,sizeof(sig));
            FILE* fnit_file=fopen(fnit_cache_path,"a");
            if (!fnit_file) abort();
            fprintf(fnit_file,"{\"call\":%lld,\"caller\":\"%p\",\"signature\":[",fnit_call,__builtin_return_address(0));
            for(int k=0;k<40;k++) fprintf(fnit_file,"%s%.17g",k?",":"",sig[k]);
            fprintf(fnit_file,"],\"cached_matrix\":[");
            for(int row=1;row<=4;row++) for(int col=1;col<=4;col++)
              fprintf(fnit_file,"%s%.17g",row==1&&col==1?"":",",(double)*MATRIX_RELT(map.sras2vox,row,col));
            fprintf(fnit_file,"]}\n");
            if(fclose(fnit_file)!=0) abort();
          }
        }
      }
    }
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, native_build, base_observer, output):
    """具名输入固定 build、已测采样观察器、新目录；输出程序及完整编译收据。"""
    native_build, base_observer, output = map(Path, (native_build, base_observer, output))
    if output.exists():
        raise FileExistsError("fresh benchmark output required")
    original = native_build / "source/utils/mrisurf_mri.cpp"
    if sha(original) != FIXED_MRI_SHA:
        raise ValueError("fixed MRI source SHA mismatch")
    receipt = json.loads((base_observer / "build.private.json").read_text())
    base_archive = base_observer / "libutils.a"
    if sha(base_archive) != receipt["archive_sha256"]:
        raise ValueError("frozen sample observer archive changed")
    compiler = Path(receipt["compile_argv"][0])
    ninja = compiler.parent / "ninja"
    lines = subprocess.check_output([str(ninja), "-t", "commands", "mris_fix_topology"],
                                    cwd=native_build / "build", text=True).splitlines()
    command = shlex.split(next(line for line in lines if " -c " in line
                              and line.rstrip().endswith("/utils/mrisurf_mri.cpp")))
    command.insert(1, "-I" + str(original.parent))
    text = original.read_text()
    anchor = "    MRIS_loadRAS2VoxelMap(&map, mri, mris);\n"
    if text.count(anchor) != 1:
        raise ValueError("actual static map insertion site mismatch")
    output.mkdir(parents=True)
    patched = output / "mrisurf_mri.cpp"
    patched.write_text(text.replace(anchor, anchor + OBSERVE, 1))
    obj = output / "mrisurf_mri.cpp.o"
    for flag, value in [("-o", str(obj)), ("-c", str(patched)), ("-MF", str(obj) + ".d"), ("-MT", str(obj))]:
        command[command.index(flag) + 1] = value
    archive = output / "libutils.a"
    shutil.copyfile(base_archive, archive)
    link = [str(archive) if token == str(base_archive) else token for token in receipt["link_argv"]]
    binary = output / "mris_fix_topology_cache_observer"
    link[link.index("-o") + 1] = str(binary)
    started = time.perf_counter()
    with (output / "build.log").open("w") as log:
        subprocess.run(command, cwd=native_build / "build", stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ar")), "r", str(archive), str(obj)], check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ranlib")), str(archive)], check=True)
        subprocess.run(link, cwd=native_build / "build", stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(status="complete_benchmark_cache_observer", scope="actual static cached map observation; no production/GA change",
                  fixed_mri_sha256=sha(original), base_receipt_sha256=sha(base_observer / "build.private.json"),
                  observer_builder_sha256=sha(__file__), patched_mri_sha256=sha(patched), archive_sha256=sha(archive),
                  observer_program_sha256=sha(binary), compile_argv=command, link_argv=link,
                  seconds=time.perf_counter() - started,
                  signature_fields=["mri_width", "mri_height", "mri_depth", "mri_xsize", "mri_ysize", "mri_zsize",
                    "mri_x_r","mri_x_a","mri_x_s","mri_y_r","mri_y_a","mri_y_s","mri_z_r","mri_z_a","mri_z_s",
                    "mri_c_r","mri_c_a","mri_c_s","vg_valid","vg_width","vg_height","vg_depth",
                    "vg_xsize","vg_ysize","vg_zsize","vg_x_r","vg_x_a","vg_x_s","vg_y_r","vg_y_a","vg_y_s",
                    "vg_z_r","vg_z_a","vg_z_s","vg_c_r","vg_c_a","vg_c_s","reserved0","reserved1","reserved2"])
    (output / "build.private.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["native-build", "base-observer", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
