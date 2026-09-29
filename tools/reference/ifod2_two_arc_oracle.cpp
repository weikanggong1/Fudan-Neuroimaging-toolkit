#include "command.h"
#include "dwi/tractography/algorithms/iFOD2.h"
#include "dwi/tractography/seeding/basic.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace MR;
using namespace MR::DWI::Tractography;
using namespace MR::DWI::Tractography::Algorithms;
using namespace App;

void usage () {
  AUTHOR = "FNIT benchmark using the official MRtrix3 iFOD2 state machine";
  SYNOPSIS = "Evaluate two fixed successive real-FOD iFOD2 arcs";
  ARGUMENTS + Argument ("fod", "normalised WM FOD image").type_image_in()
            + Argument ("cases", "x y z start_xyz first_end_xyz rows").type_file_in();
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
  const auto started = std::chrono::steady_clock::now();
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    float x,y,z,dx,dy,dz,ex,ey,ez;
    if (!(row >> x >> y >> z >> dx >> dy >> dz >> ex >> ey >> ez))
      throw Exception("invalid two-arc case");
    float first = NaN, second = NaN;
    Eigen::Vector3f first_end = {NaN,NaN,NaN};
    Eigen::Vector3f second_end = {NaN,NaN,NaN};
    bool valid = method.oracle_two_arcs({x,y,z}, {dx,dy,dz}, {ex,ey,ez},
                                        first, second, first_end, second_end);
    std::cout << valid << ' ' << first << ' ' << second;
    for (const auto& point : {first_end, second_end})
      std::cout << ' ' << point[0] << ' ' << point[1] << ' ' << point[2];
    std::cout << '\n';
  }
  std::cerr << "oracle_core_seconds=" << std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count() << '\n';
}
