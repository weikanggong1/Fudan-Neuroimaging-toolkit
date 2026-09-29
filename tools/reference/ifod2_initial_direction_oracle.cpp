#include "command.h"
#include "dwi/tractography/algorithms/iFOD2.h"
#include "dwi/tractography/seeding/basic.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace MR;
using namespace MR::DWI::Tractography;
using namespace MR::DWI::Tractography::Algorithms;
using namespace App;

void usage () {
  AUTHOR = "FNIT reference benchmark using unmodified MRtrix3 iFOD2";
  SYNOPSIS = "Run official iFOD2 initial direction sampling at fixed real FOD points";
  ARGUMENTS
    + Argument ("fod", "normalised WM FOD image").type_image_in()
    + Argument ("seeds", "space-separated world-mm x y z rows").type_file_in();
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
  std::cerr << "max_seed_attempts=" << shared.max_seed_attempts
            << " init_threshold=" << shared.init_threshold
            << " tracking_threshold=" << shared.threshold << '\n';
  iFOD2 method(shared);
  std::ifstream input(argument[1]);
  std::string line;
  std::cout << std::setprecision(9);
  const auto begin = std::chrono::steady_clock::now();
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    float x, y, z;
    if (!(row >> x >> y >> z)) throw Exception("invalid seed row");
    method.pos = {x, y, z};
    method.dir = {NaN, NaN, NaN}; // tracking/exec.h resets direction before GMWMI seeding
    const bool valid = method.init();
    if (valid) {
      const float amplitude = method.get_metric(method.pos, method.dir);
      std::cout << 1 << ' ' << method.dir[0] << ' ' << method.dir[1]
                << ' ' << method.dir[2] << ' ' << amplitude << '\n';
    } else {
      std::cout << "0 nan nan nan nan\n";
    }
  }
  std::cerr << "oracle_core_seconds=" << std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count() << '\n';
}
