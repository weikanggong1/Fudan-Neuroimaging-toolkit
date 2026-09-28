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
  AUTHOR = "FNIT benchmark instrumenting MRtrix3 iFOD2 source";
  SYNOPSIS = "Evaluate deterministic iFOD2 single arcs from a real FOD";
  ARGUMENTS
    + Argument ("fod", "real normalised WM FOD image").type_image_in()
    + Argument ("cases", "space-separated x y z dx dy dz ex ey ez rows").type_file_in();
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
  iFOD2 method(shared);
  std::ifstream input(argument[1]);
  std::string line;
  std::cout << std::setprecision(9);
  const auto begin = std::chrono::steady_clock::now();
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    float x,y,z,dx,dy,dz,ex,ey,ez;
    if (!(row >> x >> y >> z >> dx >> dy >> dz >> ex >> ey >> ez))
      throw Exception("invalid single-arc case");
    vector<Eigen::Vector3f> positions(shared.num_samples), tangents(shared.num_samples);
    float start_amp = NaN, start_amp_direct = NaN, probability = NaN;
    bool valid = method.oracle_arc({x,y,z}, {dx,dy,dz}, {ex,ey,ez},
                                   positions, tangents, start_amp, start_amp_direct, probability);
    std::cout << valid << ' ' << shared.step_size << ' ' << shared.threshold
              << ' ' << start_amp << ' ' << start_amp_direct << ' ' << probability;
    for (size_t i = 0; i < shared.num_samples; ++i)
      std::cout << ' ' << positions[i][0] << ' ' << positions[i][1] << ' ' << positions[i][2]
                << ' ' << tangents[i][0] << ' ' << tangents[i][1] << ' ' << tangents[i][2];
    std::cout << '\n';
  }
  std::cerr << "oracle_core_seconds=" << std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count() << '\n';
}
