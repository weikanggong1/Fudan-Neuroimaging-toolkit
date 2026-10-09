"""用固定原生 TU 编译参数检验步长范数表达式的浮点重载；仅诊断。

输入二进制为逐顶点 float32 x/y/z/dx/dy/dz，输出为 float32 位移与
float64 magnitude；没有影像、拓扑或碰撞算法，不是阶段替代程序。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from native_probe_build import compile_probe


SOURCE = r'''
#include "mrisurf.h"
#include <cstdio>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  FILE* input = fopen(argv[1], "rb");
  FILE* output = fopen(argv[2], "wb");
  if (!input || !output) return 3;
  float fields[6];
  const float delta_t = 0.5f, momentum = 0.0f, max_mag = 0.3f;
  while (fread(fields, sizeof(fields), 1, input) == 1) {
    float odx = delta_t * fields[3] + momentum * 0.0f;
    float ody = delta_t * fields[4] + momentum * 0.0f;
    float odz = delta_t * fields[5] + momentum * 0.0f;
    double mag = sqrt(odx * odx + ody * ody + odz * odz);
    const double original_mag = mag;
    if (mag > max_mag) {
      mag = max_mag / mag;
      odx *= mag; ody *= mag; odz *= mag;
    }
    float offsets[] = {odx, ody, odz};
    fwrite(offsets, sizeof(offsets), 1, output);
    fwrite(&original_mag, sizeof(original_mag), 1, output);
  }
  fclose(input); fclose(output);
  printf("sqrt_float_return_bytes=%zu\n", sizeof(sqrt(float(0))));
  return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / "placement_norm_expression.cpp"
    source.write_text(SOURCE)
    obj = args.out / "placement_norm_expression.o"
    command = compile_probe(source=args.source, build=args.build,
        target="utils/CMakeFiles/utils.dir/mrisurf_timeStep.cpp.o", patched=source, output=obj)
    executable = args.out / "placement_norm_expression"
    link = [command[0], str(obj), "-o", str(executable)]
    subprocess.run(link, check=True)
    receipt = {"scope": "C++ float-overload diagnostic only; not native stage",
        "compile_command": command, "link_command": link,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}
    (args.out / "build-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(executable)


if __name__ == "__main__":
    main()
