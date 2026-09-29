#include "command.h"
#include "dwi/tractography/algorithms/iFOD2.h"
#include "dwi/tractography/seeding/basic.h"
#include <chrono>
#include <iomanip>

using namespace MR;
using namespace MR::DWI::Tractography;
using namespace MR::DWI::Tractography::Algorithms;
using namespace App;

void usage () {
  AUTHOR = "FNIT benchmark calling the official MRtrix3 iFOD2 calibrator";
  SYNOPSIS = "Write iFOD2 rejection calibration for a real WM FOD image";
  ARGUMENTS + Argument ("fod", "normalised WM FOD image").type_image_in();
}

void run () {
  Properties properties;
  properties.seeds.add(new Seeding::Sphere("0,0,0,1"));
  properties["samples_per_step"] = "3";
  properties["fod_power"] = "0.5";
  properties["step_size"] = "1.00639975";
  properties["angle"] = "45";
  properties["cutoff"] = "0.1";
  iFOD2::Shared shared(argument[0], properties);
  const auto started = std::chrono::steady_clock::now();
  iFOD2 method(shared);
  const auto& directions = method.oracle_calibration_directions();
  std::cout << std::setprecision(9) << method.oracle_calibration_ratio() << ' '
            << directions.size() << '\n';
  for (const auto& direction : directions)
    std::cout << direction[0] << ' ' << direction[1] << ' ' << direction[2] << '\n';
  std::cerr << "calibration_core_seconds=" << std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count() << '\n';
}
