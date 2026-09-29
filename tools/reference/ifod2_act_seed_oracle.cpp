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
  AUTHOR = "FNIT reference benchmark calling official MRtrix3 ACT seed functions";
  SYNOPSIS = "Check ACT seed validity and one-way direction on fixed real 5TT and FOD";
  ARGUMENTS
    + Argument ("fod", "normalised WM FOD image").type_image_in()
    + Argument ("five", "five-tissue image").type_image_in()
    + Argument ("seeds", "three-column RAS world-mm seed points").type_file_in()
    + Argument ("directions", "five-column initial direction oracle output").type_file_in();
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
  std::ifstream seeds(argument[2]), initial(argument[3]);
  std::string seed_line, initial_line;
  std::cout << std::setprecision(9);
  const auto started = std::chrono::steady_clock::now();
  while (std::getline(seeds, seed_line) && std::getline(initial, initial_line)) {
    std::istringstream p(seed_line), d(initial_line);
    float x,y,z,dx,dy,dz,amp;
    int success;
    if (!(p >> x >> y >> z) || !(d >> success))
      throw Exception("invalid seed or direction row");
    if (!success) {
      std::cout << "0 0 nan nan nan nan nan nan nan nan nan\n";
      continue;
    }
    if (!(d >> dx >> dy >> dz >> amp))
      throw Exception("invalid successful initial direction row");
    method.pos = {x,y,z};
    method.dir = {dx,dy,dz};
    const bool valid = method.check_seed();
    float gradient = NaN;
    if (valid) {
      const Eigen::Vector3f seed_pos = method.pos, seed_dir = method.dir;
      method.act().fetch_tissue_data(seed_pos + seed_dir * 0.001f);
      const float positive = method.act().tissues().get_gm() - method.act().tissues().get_wm();
      method.act().fetch_tissue_data(seed_pos - seed_dir * 0.001f);
      const float negative = method.act().tissues().get_gm() - method.act().tissues().get_wm();
      gradient = positive - negative;
      method.act().fetch_tissue_data(seed_pos);
    }
    const bool unidirectional = valid && method.act().seed_is_unidirectional(method.pos, method.dir);
    const auto& tissue = method.act().tissues();
    std::cout << valid << ' ' << unidirectional << ' ' << method.dir[0]
              << ' ' << method.dir[1] << ' ' << method.dir[2]
              << ' ' << tissue.get_cgm() << ' ' << tissue.get_sgm()
              << ' ' << tissue.get_wm() << ' ' << tissue.get_csf()
              << ' ' << tissue.get_path() << ' ' << gradient << '\n';
  }
  std::cerr << "oracle_core_seconds=" << std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count() << '\n';
}
