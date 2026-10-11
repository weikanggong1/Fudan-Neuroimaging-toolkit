"""沿已冻结TU头文件与编译参数检查sqrt(float)的实际返回类型。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-receipt", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    recipe = json.loads(args.integration_receipt.read_text())
    code = r'''
#include "mrisurf_sseTerms.h"
#include "mrisurf_project.h"
#include "mrisurf_base.h"
#include "mrisurf_MRIS_MP.h"
#include "mrisurf_SurfaceFromMRIS_MP_generated.h"
#include "mrishash_SurfaceFromMRIS.h"
#include <type_traits>
int main(int argc, char** argv) {
  if (argc!=4) return 2;
  float a=strtod(argv[1],nullptr), b=strtod(argv[2],nullptr), c=strtod(argv[3],nullptr);
  auto ratio=a/(b-c);
  auto root=sqrt(a/(b-c));
  double assigned=root;
  printf("ratio=%a root=%a assigned=%a ratio_is_float=%d root_is_float=%d\n",(double)ratio,(double)root,assigned,(int)std::is_same<decltype(ratio),float>::value,(int)std::is_same<decltype(root),float>::value);
}
'''
    source = args.output / "sse_expression_probe.cpp"
    source.write_text(code)
    command = list(recipe["compile_argv"])
    obj = args.output / "sse_expression_probe.o"
    command[command.index("-c")+1] = str(source)
    for key,value in [("-o",str(obj)),("-MT",str(obj)),("-MF",str(obj)+".d")]:
        command[command.index(key)+1] = value
    command.insert(1,"-I"+str(args.source_directory))
    compiler=Path(command[0])
    binary=args.output/"sse_expression_probe"
    subprocess.run(command,check=True)
    subprocess.run([str(compiler),str(obj),"-o",str(binary),"-Wl,-rpath,"+str(compiler.parent.parent/"lib")],check=True)
    report={"status":"complete","scope":"own expression only; original headers remain private",
            "builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "own_probe_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "program_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),"compile_argv":command}
    (args.output/"build.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
