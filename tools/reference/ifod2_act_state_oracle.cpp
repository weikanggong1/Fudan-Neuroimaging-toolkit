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
  AUTHOR = "FNIT benchmark calling official MRtrix3 ACT state methods";
  SYNOPSIS = "Check ACT structural state on real 5TT and fixed world-mm paths";
  ARGUMENTS + Argument ("fod", "normalised WM FOD image").type_image_in()
            + Argument ("five", "five-tissue image").type_image_in()
            + Argument ("cases", "S seed_xyz, P point_xyz, R reverse rows").type_file_in();
}

void run () {
  Properties properties;
  properties.seeds.add(new Seeding::Sphere("0,0,0,1"));
  properties["act"] = std::string(argument[1]);
  properties["samples_per_step"] = "3";
  properties["fod_power"] = "0.5";
  properties["step_size"] = "1.00639975";
  properties["angle"] = "45";
  properties["cutoff"] = "0.1";
  iFOD2::Shared shared(argument[0], properties);
  iFOD2 method(shared);
  std::ifstream input(argument[2]);
  std::string line;
  std::cout << std::setprecision(9);
  const auto started = std::chrono::steady_clock::now();
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    char kind;
    row >> kind;
    if (kind == 'R') {
      method.act().reverse_track();
      std::cout << "R 0 " << method.act().sgm_depth << ' '
                << method.act().seed_in_sgm << ' ' << method.act().sgm_seed_to_wm << '\n';
      continue;
    }
    float x,y,z;
    if (!(row >> x >> y >> z)) throw Exception("invalid ACT state case");
    method.pos = {x,y,z};
    int result;
    if (kind == 'S') {
      method.act().seed_in_sgm = false;
      method.act().sgm_seed_to_wm = false;
      result = method.check_seed();
    }
    else if (kind == 'P') result = method.act().check_structural(method.pos);
    else throw Exception("unknown ACT state case kind");
    std::cout << kind << ' ' << result << ' ' << method.act().sgm_depth << ' '
              << method.act().seed_in_sgm << ' ' << method.act().sgm_seed_to_wm;
    const auto& tissue = method.act().tissues();
    std::cout << ' ' << tissue.get_cgm() << ' ' << tissue.get_sgm()
              << ' ' << tissue.get_wm() << ' ' << tissue.get_csf()
              << ' ' << tissue.get_path() << '\n';
  }
  std::cerr << "oracle_core_seconds=" << std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count() << '\n';
}
