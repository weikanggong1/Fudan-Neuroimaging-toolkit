"""在独立 ITK 计时构建中导出最终 lattice、log field、exp field。

参数与 prepare_diagnostic.py 相同；输出目录必须不存在。仅增加已完成
输出的导出，不改变方程、精度或线程。完整输出须与普通 native 比较。
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itk-include", type=Path, required=True)
    parser.add_argument("--n4-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = Path(__file__).with_name("prepare_diagnostic.py")
    subprocess.run([sys.executable, str(base), "--itk-include", str(args.itk_include),
                    "--n4-source", str(args.n4_source), "--output", str(args.output)], check=True)
    helper = args.output / "include/fnit_n4_profile.h"
    helper_source = helper.read_text()
    ending = "}  // namespace fnit_n4_profile\n"
    if helper_source.count(ending) != 1:
        raise ValueError("profile helper anchor changed")
    addition = r'''
template<class Image> void capture_final(const Image *image, const std::string &stem) {
  static_assert(sizeof(typename Image::PixelType) == sizeof(float), "diagnostic expects one FP32 scalar per voxel");
  Scope timer("final_diagnostic_export");
  raw(image, stem + ".raw");
  const auto size = image->GetLargestPossibleRegion().GetSize();
  const auto spacing = image->GetSpacing();
  const auto origin = image->GetOrigin();
  const auto direction = image->GetDirection();
  std::ofstream out(stem + ".json"); out.precision(17);
  out << "{\"shape\":[" << size[0] << ',' << size[1] << ',' << size[2]
      << "],\"spacing_mm\":[" << spacing[0] << ',' << spacing[1] << ',' << spacing[2]
      << "],\"origin_mm\":[" << origin[0] << ',' << origin[1] << ',' << origin[2]
      << "],\"direction\":[";
  for (unsigned int a=0; a<3; ++a) {
    if (a) out << ',';
    out << '[' << direction[a][0] << ',' << direction[a][1] << ',' << direction[a][2] << ']';
  }
  out << "],\"dtype\":\"float32\",\"order\":\"Fortran x-fast\"}\n";
  if (!out) throw std::runtime_error("final diagnostic geometry export failed");
}
'''
    helper.write_text(helper_source.replace(ending, addition + ending))
    source = args.output / "n4_diagnostic.cpp"
    text = source.read_text()
    anchor = "    out.close();\n"
    if text.count(anchor) != 1:
        raise ValueError("native output anchor changed")
    text = text.replace(anchor, anchor + '''    fnit_n4_profile::capture_final(corrector->GetLogBiasFieldControlPointLattice(), fnit_n4_profile::prefix + ".final_lattice");
    fnit_n4_profile::capture_final(log_field.GetPointer(), fnit_n4_profile::prefix + ".logfield");
    fnit_n4_profile::capture_final(exp_field->GetOutput(), fnit_n4_profile::prefix + ".expfield");
''')
    source.write_text(text)
    cmake = args.output / "CMakeLists.txt"
    cmake_source = cmake.read_text()
    package_anchor = "find_package(ITK 5.4 REQUIRED)"
    if cmake_source.count(package_anchor) != 1:
        raise ValueError("fixed diagnostic CMake anchor changed")
    # Raw FP32 I/O does not use ITK image readers. Avoid loading unrelated
    # HDF5/image-IO CMake discovery paths from other Conda installations.
    cmake.write_text(cmake_source.replace(package_anchor,
        "find_package(ITK 5.4 REQUIRED COMPONENTS ITKBiasCorrection ITKImageGrid ITKImageIntensity)"))
    manifest_path = args.output / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["final_capture"] = "raw final lattice/logfield/expfield plus exact geometry; no equations changed"
    manifest["capture_write_time_is_not_native_performance"] = True
    manifest["generated_sha256"] = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, helper, cmake)}
    manifest["final_generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
