"""在独立诊断 workspace 编译固定 ITK cubic refinement 常数检查。"""
import argparse
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--prefix", type=Path, required=True, help="已有固定 ITK 5.4.7 Conda prefix")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = Path(__file__).with_name("dump_refinement.cpp").resolve()
code = args.workspace / "refinement_diagnostic"
code.mkdir(parents=True, exist_ok=True)
(code / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.20)
project(fnit_refinement_diagnostic LANGUAGES CXX)
find_package(ITK 5.4 REQUIRED)
include(${{ITK_USE_FILE}})
add_executable(fnit_dump_refinement "{source}")
target_compile_features(fnit_dump_refinement PRIVATE cxx_std_17)
target_link_libraries(fnit_dump_refinement PRIVATE ${{ITK_LIBRARIES}})
''')
compiler = args.prefix / "bin/x86_64-conda-linux-gnu-g++"
cmake = args.prefix / "bin/cmake"
subprocess.run([str(cmake), "-S", str(code), "-B", str(code / "build"), "-G", "Ninja",
                f"-DCMAKE_PREFIX_PATH={args.prefix}", f"-DCMAKE_CXX_COMPILER={compiler}",
                "-DCMAKE_BUILD_TYPE=Release"], check=True)
subprocess.run([str(cmake), "--build", str(code / "build"), "--parallel", "1"], check=True)
data = subprocess.check_output([str(code / "build/fnit_dump_refinement")])
args.output.write_bytes(data)
