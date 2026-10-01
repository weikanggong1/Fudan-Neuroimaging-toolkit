#include <iostream>
#include <iomanip>
#include "newimage/newimageall.h"
#include "newimage/costfns.h"
int main(int argc, char** argv) {
 NEWIMAGE::volume4D<float> data; NEWIMAGE::volume<float> ref;
 NEWIMAGE::read_volume4D(data,argv[1]); NEWIMAGE::read_volume(ref,argv[2]);
 int frame=atoi(argv[3]);
 for(float scale: {8.f,4.f}) {
  auto iso=NEWIMAGE::isotropic_resample(ref,scale);
  NEWIMAGE::Costfn cost(iso,data[frame]);cost.set_costfn(NEWIMAGE::NormCorr);cost.smoothsize=1.0;
  cost.set_no_bins(256/int(scale));
  std::cout << "scale " << scale << " shape " << iso.xsize()<<" "<<iso.ysize()<<" "<<iso.zsize()<<" cog ";
  for(int i=1;i<=3;i++)std::cout<<std::setprecision(17)<<cost.testCog(i)<<" ";std::cout<<"\n";
  for(float dx: {0.f,.1f,-.1f,1.f}) {
   NEWMAT::Matrix mat=NEWMAT::IdentityMatrix(4); mat(1,4)=dx;
   std::cout<<"cost "<<std::setprecision(17)<<dx<<" "<<cost.cost(mat)<<"\n";
  }
  NEWIMAGE::save_volume(iso,std::string(argv[4])+"_ref"+std::to_string(int(scale)));
 }
}
