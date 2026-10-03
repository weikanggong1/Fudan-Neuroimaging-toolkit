"""从固定公开FNIT N4源码生成独立诊断；不复制第三方软件源码。"""
import argparse, hashlib, json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
source=Path(__file__).resolve().parents[4]/'tools/n4_itk/n4_itk.cpp'
expected='c281204beb21ba756257449c564c23a3f0efc0a15387b7d3527cf858042fca8f'
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('expected frozen 816e5610 N4 source')
args.output.mkdir(parents=True,exist_ok=True)
src=source.read_text().replace('#include <fstream>','#include <fstream>\n#include <cstdlib>\n#include "itkVersion.h"').replace('#include "itkMultiThreaderBase.h"','#if ITK_VERSION_MAJOR >= 5\n#include "itkMultiThreaderBase.h"\n#else\n#include "itkMultiThreader.h"\nnamespace itk { using MultiThreaderBase = MultiThreader; }\n#endif')
src=src.replace('const auto fit_end = Clock::now();','''const auto fit_end = Clock::now();
    const char *diagnostic_prefix = std::getenv("FNIT_N4_DIAGNOSTIC_PREFIX");
    const auto dump = [&](const std::string &name, const float *data, size_t n) {
      if (!diagnostic_prefix) return;
      std::ofstream file(std::string(diagnostic_prefix) + "." + name + ".raw", std::ios::binary);
      file.write(reinterpret_cast<const char *>(data), n * sizeof(float));
      if (!file) throw std::runtime_error("diagnostic write failed");
    };
    auto lattice = corrector->GetLogBiasFieldControlPointLattice();
    dump("lattice", reinterpret_cast<const float *>(lattice->GetBufferPointer()),
         lattice->GetLargestPossibleRegion().GetNumberOfPixels());''')
src=src.replace('using Exp = itk::ExpImageFilter<Image, Image>;','dump("logfield", log_field->GetBufferPointer(), count);\n    using Exp = itk::ExpImageFilter<Image, Image>;')
src=src.replace('const auto reconstruction_end = Clock::now();','const auto reconstruction_end = Clock::now();\n    dump("expfield", exp_field->GetOutput()->GetBufferPointer(), count);')
cmake='''cmake_minimum_required(VERSION 3.20)
project(fnit_n4_diagnostic LANGUAGES C CXX)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
add_executable(fnit_n4_diagnostic n4_diagnostic.cpp)
target_compile_features(fnit_n4_diagnostic PRIVATE cxx_std_17)
target_link_libraries(fnit_n4_diagnostic PRIVATE ${ITK_LIBRARIES})
'''
(args.output/'n4_diagnostic.cpp').write_text(src);(args.output/'CMakeLists.txt').write_text(cmake)
legacy=args.output/'gcc48';legacy.mkdir(exist_ok=True)
(legacy/'n4_diagnostic.cpp').write_text(src.replace('[](auto a, auto b)','[](const Clock::time_point &a, const Clock::time_point &b)').replace('using Mask = itk::Image<unsigned char, 3>;','using Mask = Image;'))
(legacy/'CMakeLists.txt').write_text(cmake.replace('cxx_std_17','cxx_std_11'))
print(json.dumps({str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.rglob('*') if p.is_file()},indent=2))
