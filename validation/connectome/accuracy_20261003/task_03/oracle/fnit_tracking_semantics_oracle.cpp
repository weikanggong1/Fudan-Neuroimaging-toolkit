#include "command.h"
#include "dwi/tractography/algorithms/iFOD2.h"
#include "dwi/tractography/seeding/basic.h"
#include "dwi/tractography/ACT/gmwmi.h"
#include <fstream>
#include <iomanip>
#include <sstream>
using namespace MR;
using namespace MR::DWI::Tractography;
using namespace MR::DWI::Tractography::Algorithms;
using namespace App;
class ProbeFinder: public ACT::GMWMI_finder {
public:
 using ACT::GMWMI_finder::GMWMI_finder;
 Eigen::Vector3f step(const Eigen::Vector3f& p) const { Interp interp(interp_template);return get_cf_min_step(p,interp); }
 ACT::Tissues tissue(const Eigen::Vector3f& p) const { Interp interp(interp_template);return get_tissues(p,interp); }
};
void usage() {
 AUTHOR="FNIT fixed-input diagnostic of MRtrix3 026e850d source";
 SYNOPSIS="Probe real FOD arcs and GMWMI fixed candidate projection";
 ARGUMENTS + Argument("fod","WM SH image").type_image_in()
           + Argument("five","5TT image").type_image_in()
           + Argument("cases","A xyz priorXYZ endXYZ, G xyz").type_file_in()
           + Argument("step","iFOD2 step millimetres").type_float();
}
void run() {
 Properties properties;
 properties.seeds.add(new Seeding::Sphere("0,0,0,1"));
 properties["act"]=std::string(argument[1]);
 properties["samples_per_step"]="3";
 properties["fod_power"]="0.5";
 properties["step_size"]=std::string(argument[3]);
 properties["angle"]="45";
 properties["threshold"]="0.1";
 properties["init_threshold"]="0.1";
 iFOD2::Shared shared(argument[0],properties);
 iFOD2 method(shared);
 auto five=Image<float>::open(argument[1]);
 ProbeFinder finder(five);
 std::ifstream input(argument[2]); std::string line;
 std::cout << std::setprecision(9);
 while(std::getline(input,line)) {
  if(line.empty()||line[0]=='#')continue;
  std::istringstream row(line); char kind; float x,y,z; row>>kind>>x>>y>>z;
  Eigen::Vector3f p(x,y,z);
  if(kind=='T') {
   auto tissue=finder.tissue(p);auto step=finder.step(p);
   std::cout<<"T "<<tissue.valid()<<' '<<tissue.get_cgm()<<' '<<tissue.get_sgm()<<' '<<tissue.get_wm()<<' '<<tissue.get_csf()<<' '<<tissue.get_path()<<' '<<step[0]<<' '<<step[1]<<' '<<step[2]<<'\n';
  } else if(kind=='G') {
   bool valid=finder.find_interface(p); auto normal=finder.normal(p);
   std::cout<<"G "<<valid<<' '<<p[0]<<' '<<p[1]<<' '<<p[2];
   std::cout<<' '<<normal[0]<<' '<<normal[1]<<' '<<normal[2]<<'\n';
  } else {
   float dx,dy,dz,ex,ey,ez; row>>dx>>dy>>dz>>ex>>ey>>ez;
   vector<Eigen::Vector3f> positions(2),tangents(2); Eigen::Vector3f metrics;
   float probability=NaN, start_amplitude=NaN; bool valid;
   valid=method.fnit_oracle_arc(p,{dx,dy,dz},{ex,ey,ez},positions,tangents,start_amplitude,probability,metrics);
   std::cout<<"A "<<valid<<' '<<start_amplitude<<' '<<probability;
   for(size_t i=0;i<2;++i) {
    std::cout<<' '<<positions[i][0]<<' '<<positions[i][1]<<' '<<positions[i][2];
    std::cout<<' '<<tangents[i][0]<<' '<<tangents[i][1]<<' '<<tangents[i][2];
    auto chord=(positions[i]-(i?positions[0]:p)).normalized();
    std::cout<<' '<<method.get_metric(positions[i],tangents[i])<<' '<<method.get_metric(positions[i],chord);
   }
   std::cout<<'\n';
  }
 }
}
