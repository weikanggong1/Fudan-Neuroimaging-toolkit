"""Clock-only ITK N4 diagnostic build and frozen single-level fit captures.

Reads the installed fixed ITK header; writes private override copies only in
the independent benchmark build. No upstream header is bundled in this repo.
"""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one fixed-source anchor: {old[:70]!r}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itk-include", type=Path, required=True)
    parser.add_argument("--n4-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    headers = out / "include"
    headers.mkdir(parents=True, exist_ok=True)
    name = "itkN4BiasFieldCorrectionImageFilter"
    header_path, body_path = (args.itk_include / (name + suffix) for suffix in (".h", ".hxx"))
    header, body = header_path.read_text(), body_path.read_text()
    body = replace_once(body, "#define itkN4BiasFieldCorrectionImageFilter_hxx",
                        '#define itkN4BiasFieldCorrectionImageFilter_hxx\n#include "fnit_n4_profile.h"')
    for anchor, stage in (
        ("  RealImageType *       sharpenedImage) const\n{", "sharpen_total"),
        ("  const size_t    numberOfIncludedPixels)\n{", "update_bias_total"),
        ("  const BiasFieldControlPointLatticeType * controlPointLattice)\n{", "small_reconstruction"),
        ("  const RealImageType * fieldEstimate2) const\n{", "convergence"),
    ):
        body = replace_once(body, anchor, anchor + f'\n  fnit_n4_profile::Scope fnitTimer("{stage}");')
    body = replace_once(body, "  bspliner->Update();", '{ fnit_n4_profile::Scope fnitTimer("bspline_fit"); bspliner->Update(); }')
    body = replace_once(body, "  typename BiasFieldControlPointLatticeType::Pointer phiLattice = bspliner->GetPhiLattice();",
        '  typename BiasFieldControlPointLatticeType::Pointer phiLattice = bspliner->GetPhiLattice();\n  fnit_n4_profile::capture(fieldEstimate, phiLattice.GetPointer());')
    for call, stage in (("      residualBiasField->Update();", "residual_subtract"),
                        ("      logUncorrectedImage->Update();", "uncorrected_subtract"),
                        ("    this->m_LogBiasFieldControlPointLattice = reconstructer->RefineControlPointLattice(numberOfLevels);", "refine_lattice")):
        body = replace_once(body, call, '{ fnit_n4_profile::Scope fnitTimer("' + stage + '"); ' + call.strip() + ' }')
    (headers / (name + ".h")).write_text(header)
    (headers / (name + ".hxx")).write_text(body)
    helper = Path(__file__).with_name("fnit_n4_profile.h")
    (headers / helper.name).write_bytes(helper.read_bytes())
    source = args.n4_source.read_text()
    source = replace_once(source, "    const auto fit_start = Clock::now();",
                          '    if (argc != 11) throw std::runtime_error("diagnostic requires profile path");\n    fnit_n4_profile::prefix = argv[10];\n    const auto fit_start = Clock::now();')
    source = replace_once(source, "  } catch (const std::exception &error) {", '    fnit_n4_profile::write_report();\n  } catch (const std::exception &error) {')
    (out / "n4_diagnostic.cpp").write_text(source)
    (out / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(fnit_n4_profile LANGUAGES CXX)
find_package(ITK 5.4 REQUIRED)
include(${ITK_USE_FILE})
add_executable(fnit_n4_profile n4_diagnostic.cpp)
target_compile_features(fnit_n4_profile PRIVATE cxx_std_17)
target_include_directories(fnit_n4_profile BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fnit_n4_profile PRIVATE ${ITK_LIBRARIES})
''')
    files = (header_path, body_path, args.itk_include / "itkBSplineScatteredDataPointSetToImageFilter.hxx",
             args.itk_include / "itkBSplineControlPointImageFilter.hxx", args.n4_source, helper, Path(__file__))
    report = {"source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "instrumentation": "clock scopes and first frozen phi/residual exports; equations unchanged",
              "production_changed": False, "output_equivalence": "requires actual same-input regression"}
    (out / "source_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
