"""隔离编译完整放置流程的逐轮边界目标探针。

仅增加状态写出，不更改迭代、目标平滑、接受或清理算法。诊断程序必须先
与同输入未插桩程序完成最终几何录取；不是生产依赖，也不复制参考结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from native_probe_build import compile_probe, link_probe


def patch_source(original: str) -> str:
    """在固定命令的四轮入口及目标平滑前后写出 float32/有符号标记。"""
    outer = "  for (i = 0 ;  n_averages >= n_min_averages ; n_averages /= 2, current_sigma /= 2, i++) {"
    average = "MRISaverageMarkedVals(surf, vavgs) ;"
    if original.count(outer) != 1 or original.count(average) != 1:
        raise RuntimeError("fixed source placement anchors changed")
    capture = r'''
    auto dump_target_pass_state = [&](const char *stage) {
      const char *prefix = getenv("PLACE_TARGET_PASS_PREFIX");
      if (!prefix) return;
      char path[STRLEN];
      snprintf(path, STRLEN, "%s.pass%02d.%s", prefix, i, stage);
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
      if (fclose(output)) exit(1);
    };
    dump_target_pass_state("entry");
'''
    original = original.replace(outer, outer + capture)
    return original.replace(average,
        'dump_target_pass_state("before_average");\n          '
        + average + '\n          dump_target_pass_state("after_average");')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, mode=0o700)
    if not (args.build / "mris_make_surfaces/mris_place_surface.help.xml.h").exists():
        (args.out / "mris_place_surface.help.xml.h").write_text(
            'static const unsigned char mris_place_surface_help_xml[] = "";\n'
            'static const unsigned int mris_place_surface_help_xml_len = 0;\n')
    source = args.source / "mris_make_surfaces/mris_place_surface.cpp"
    patched = args.out / "mris_place_surface_target_pass.cpp"
    patched.write_text(patch_source(source.read_text()))
    object_file = args.out / "mris_place_surface_target_pass.o"
    compile_command = compile_probe(source=args.source, build=args.build,
        target="mris_make_surfaces/CMakeFiles/mris_place_surface.dir/mris_place_surface.cpp.o",
        patched=patched, output=object_file)
    executable = args.out / "mris_place_surface_target_pass_probe"
    link_command = link_probe(build=args.build, main_object=object_file, output=executable)
    receipt = dict(scope="transparent full placement target diagnostic; admission pending",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        patched_sha256=hashlib.sha256(patched.read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        compiler_command=compile_command, link_command=link_command)
    (args.out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(executable)


if __name__ == "__main__":
    main()
